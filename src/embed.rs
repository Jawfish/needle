mod openai;
mod voyage;

#[cfg(feature = "local")]
mod local;

use std::{sync::Arc, time::Duration};

use serde::Deserialize;

use crate::{error::NeedleError, types::EmbedConfig};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_RETRIES: u32 = 3;
const CHUNK_TARGET_CHARS: usize = 4000; // ~1000 tokens

type SendFuture<'a> = std::pin::Pin<
    Box<
        dyn std::future::Future<Output = anyhow::Result<(reqwest::StatusCode, Vec<u8>)>>
            + Send
            + 'a,
    >,
>;

/// Port for HTTP request dispatch.  Concrete providers receive this via their
/// constructors so tests can substitute a fake without a live network.
///
/// The return type uses a boxed future so the trait is object-safe and can be
/// stored as `Arc<dyn HttpTransport>`.
pub trait HttpTransport: Send + Sync {
    fn send(&self, request: reqwest::Request) -> SendFuture<'_>;
}

/// Production implementation backed by a real `reqwest::Client`.
pub struct ReqwestTransport {
    client: reqwest::Client,
}

impl ReqwestTransport {
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            client: reqwest::Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .build()?,
        })
    }
}

impl HttpTransport for ReqwestTransport {
    fn send(&self, request: reqwest::Request) -> SendFuture<'_> {
        Box::pin(async move {
            let response = self.client.execute(request).await?;
            let status = response.status();
            let body = response.bytes().await?.to_vec();
            Ok((status, body))
        })
    }
}

pub enum Embedder {
    Voyage(voyage::VoyageProvider),
    OpenAi(openai::OpenAiProvider),
    #[cfg(feature = "local")]
    Local(local::LocalProvider),
    #[cfg(test)]
    Null {
        dim: usize,
    },
}

impl Embedder {
    pub fn from_config(config: &EmbedConfig) -> anyhow::Result<Self> {
        let kind = match config.provider.as_deref() {
            Some(name) => parse_provider_name(name)?,
            None => infer_from_keys(config)?,
        };

        match kind {
            ProviderKind::Voyage => {
                let api_key = config
                    .voyage_api_key
                    .as_deref()
                    .ok_or_else(|| NeedleError::MissingApiKey("VOYAGE_API_KEY".to_owned()))?;
                let transport = Arc::new(ReqwestTransport::new()?);
                Ok(Self::Voyage(voyage::VoyageProvider::new(
                    api_key,
                    config.model.as_deref(),
                    config.dim,
                    transport,
                )?))
            }
            ProviderKind::OpenAi => {
                let transport = Arc::new(ReqwestTransport::new()?);
                Ok(Self::OpenAi(openai::OpenAiProvider::new(
                    config.openai_api_key.as_deref(),
                    config.needle_api_key.as_deref(),
                    config.api_base.as_deref(),
                    config.model.as_deref(),
                    config.dim,
                    transport,
                )?))
            }
            #[cfg(feature = "local")]
            ProviderKind::Local => {
                let (model, dim) = local::init_model(config.model.as_deref())?;
                Ok(Self::Local(local::LocalProvider::new(model, dim)))
            }
            #[cfg(not(feature = "local"))]
            ProviderKind::Local => Err(NeedleError::NoEmbeddingProvider.into()),
        }
    }

    #[cfg(test)]
    pub const fn create_null(dim: usize) -> Self {
        Self::Null { dim }
    }

    pub const fn dim(&self) -> usize {
        match self {
            Self::Voyage(p) => p.dim(),
            Self::OpenAi(p) => p.dim(),
            #[cfg(feature = "local")]
            Self::Local(p) => p.dim(),
            #[cfg(test)]
            Self::Null { dim } => *dim,
        }
    }

    pub async fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        match self {
            Self::Voyage(p) => p.embed_documents(texts).await,
            Self::OpenAi(p) => p.embed_documents(texts).await,
            #[cfg(feature = "local")]
            Self::Local(p) => p.embed_documents(texts).await,
            #[cfg(test)]
            Self::Null { dim } => Ok(texts.iter().map(|_| vec![0.0; *dim]).collect()),
        }
    }

    pub async fn embed_query(&self, query: &str) -> anyhow::Result<Vec<f32>> {
        match self {
            Self::Voyage(p) => p.embed_query(query).await,
            Self::OpenAi(p) => p.embed_query(query).await,
            #[cfg(feature = "local")]
            Self::Local(p) => p.embed_query(query).await,
            #[cfg(test)]
            Self::Null { dim } => Ok(vec![0.0; *dim]),
        }
    }
}

enum ProviderKind {
    Voyage,
    OpenAi,
    Local,
}

fn parse_provider_name(name: &str) -> anyhow::Result<ProviderKind> {
    match name {
        "voyage" => Ok(ProviderKind::Voyage),
        "openai" => Ok(ProviderKind::OpenAi),
        "local" => Ok(ProviderKind::Local),
        _ => anyhow::bail!("unknown provider: {name} (expected: voyage, openai, local)"),
    }
}

fn infer_from_keys(config: &EmbedConfig) -> anyhow::Result<ProviderKind> {
    if config.voyage_api_key.is_some() {
        return Ok(ProviderKind::Voyage);
    }
    if config.openai_api_key.is_some() {
        return Ok(ProviderKind::OpenAi);
    }
    if cfg!(feature = "local") {
        return Ok(ProviderKind::Local);
    }
    Err(NeedleError::NoEmbeddingProvider.into())
}

// --- Shared HTTP helpers ---

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

async fn send_with_retry(
    transport: &dyn HttpTransport,
    build_request: impl Fn() -> anyhow::Result<reqwest::Request>,
) -> anyhow::Result<Vec<u8>> {
    let mut last_err: Option<anyhow::Error> = None;

    for attempt in 0..=MAX_RETRIES {
        if attempt > 0 {
            let delay_secs = 1u64 << (attempt - 1);
            tokio::time::sleep(Duration::from_secs(delay_secs)).await;
            tracing::warn!(attempt, "retrying embedding request");
        }

        let request = build_request()?;
        let (status, body) = match transport.send(request).await {
            Ok(r) => r,
            Err(e) => {
                last_err = Some(e);
                continue;
            }
        };

        if status.is_success() {
            return Ok(body);
        }

        let body_text = String::from_utf8_lossy(&body).into_owned();

        if status.is_client_error() && status != reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(NeedleError::EmbeddingApi(format!("{status}: {body_text}")).into());
        }

        last_err = Some(NeedleError::EmbeddingApi(format!("{status}: {body_text}")).into());
    }

    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("embedding request failed")))
}

// --- Text chunking (provider-independent) ---

pub fn chunk_text(content: &str) -> Vec<String> {
    let content = strip_frontmatter(content);
    let paragraphs: Vec<&str> = content.split("\n\n").collect();
    let mut chunks = Vec::new();
    let mut current = String::new();

    for paragraph in paragraphs {
        if current.len() + paragraph.len() > CHUNK_TARGET_CHARS && !current.is_empty() {
            chunks.push(std::mem::take(&mut current));
        }

        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(paragraph);
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    if chunks.is_empty() {
        chunks.push(String::new());
    }

    chunks
}

fn strip_frontmatter(content: &str) -> &str {
    if !content.starts_with("---") {
        return content;
    }
    let after_open = &content[3..];
    let Some(close_pos) = after_open.find("\n---") else {
        return content;
    };

    let body = after_open[..close_pos].trim();
    if !is_yaml_frontmatter(body) {
        return content;
    }

    content[close_pos + 7..].trim_start()
}

fn is_yaml_frontmatter(block: &str) -> bool {
    block.is_empty() || block.lines().any(|line| line.trim().contains(": "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_file_produces_single_chunk() {
        let chunks = chunk_text("hello world");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }

    #[test]
    fn large_file_splits_on_paragraph_boundaries() {
        let paragraph = "a".repeat(3000);
        let content = format!("{paragraph}\n\n{paragraph}\n\n{paragraph}");
        let chunks = chunk_text(&content);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(
                !chunk.is_empty(),
                "no chunk should be empty after splitting"
            );
        }
    }

    #[test]
    fn empty_content_produces_one_chunk() {
        let chunks = chunk_text("");
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn chunk_preserves_all_content() {
        let content = "paragraph one\n\nparagraph two\n\nparagraph three";
        let chunks = chunk_text(content);
        let reassembled = chunks.join("\n\n");
        assert_eq!(reassembled, content);
    }

    #[test]
    fn chunk_never_splits_mid_paragraph() {
        let short = "short paragraph";
        let long = "x".repeat(3500);
        let content = format!("{short}\n\n{long}\n\n{short}");
        let chunks = chunk_text(&content);
        for chunk in &chunks {
            assert!(
                !chunk.contains("\n\n") || chunk.len() <= CHUNK_TARGET_CHARS,
                "should only keep multiple paragraphs in one chunk if under target size"
            );
        }
    }

    #[test]
    fn strip_frontmatter_removes_yaml() {
        let content = "---\ntitle: Test\ntags: [a, b]\n---\n\n# Actual content";
        let stripped = strip_frontmatter(content);
        assert_eq!(stripped, "# Actual content");
    }

    #[test]
    fn strip_frontmatter_leaves_content_without_frontmatter() {
        let content = "# Just a heading\n\nSome text";
        let stripped = strip_frontmatter(content);
        assert_eq!(stripped, content);
    }

    #[test]
    fn strip_frontmatter_handles_unclosed_frontmatter() {
        let content = "---\ntitle: Test\nno closing delimiter";
        let stripped = strip_frontmatter(content);
        assert_eq!(
            stripped, content,
            "unclosed frontmatter should be left as-is"
        );
    }

    #[test]
    fn strip_frontmatter_handles_empty_frontmatter() {
        let content = "---\n---\n\nBody text";
        let stripped = strip_frontmatter(content);
        assert_eq!(stripped, "Body text");
    }

    #[test]
    fn chunk_text_strips_frontmatter_before_chunking() {
        let content = "---\ntitle: Note\n---\n\n# Heading\n\nBody text";
        let chunks = chunk_text(content);
        assert_eq!(chunks.len(), 1);
        assert!(!chunks[0].contains("---"), "frontmatter should be stripped");
        assert!(chunks[0].contains("Heading"));
    }

    #[test]
    fn strip_frontmatter_ignores_dashes_within_yaml_values() {
        let content = "---\nfoo: a---b\n---\n\nBody";
        let stripped = strip_frontmatter(content);
        assert_eq!(stripped, "Body");
    }

    #[test]
    fn strip_frontmatter_preserves_leading_horizontal_rule() {
        let content = "---\n\nSome content after a horizontal rule";
        let stripped = strip_frontmatter(content);
        assert_eq!(
            stripped, content,
            "horizontal rule should not be treated as frontmatter"
        );
    }

    #[test]
    fn strip_frontmatter_preserves_non_yaml_block() {
        let content = "---\nthis is not yaml\n---\n\nBody";
        let stripped = strip_frontmatter(content);
        assert_eq!(
            stripped, content,
            "block without key: value pairs should not be stripped"
        );
    }

    #[test]
    fn strip_frontmatter_with_dashes_in_body() {
        let content = "No frontmatter\n\n---\n\nSome divider";
        let stripped = strip_frontmatter(content);
        assert_eq!(
            stripped, content,
            "--- not at start should not be treated as frontmatter"
        );
    }

    #[test]
    fn infer_voyage_from_api_key() {
        let config = EmbedConfig {
            provider: None,
            model: None,
            api_base: None,
            dim: None,
            voyage_api_key: Some("vk-test".to_owned()),
            openai_api_key: None,
            needle_api_key: None,
        };
        let kind = infer_from_keys(&config).expect("should infer");
        assert!(matches!(kind, ProviderKind::Voyage));
    }

    #[test]
    fn infer_openai_from_api_key() {
        let config = EmbedConfig {
            provider: None,
            model: None,
            api_base: None,
            dim: None,
            voyage_api_key: None,
            openai_api_key: Some("sk-test".to_owned()),
            needle_api_key: None,
        };
        let kind = infer_from_keys(&config).expect("should infer");
        assert!(matches!(kind, ProviderKind::OpenAi));
    }

    #[test]
    fn voyage_takes_precedence_when_both_keys_set() {
        let config = EmbedConfig {
            provider: None,
            model: None,
            api_base: None,
            dim: None,
            voyage_api_key: Some("vk-test".to_owned()),
            openai_api_key: Some("sk-test".to_owned()),
            needle_api_key: None,
        };
        let kind = infer_from_keys(&config).expect("should infer");
        assert!(matches!(kind, ProviderKind::Voyage));
    }

    #[test]
    fn parse_provider_name_accepts_valid_names() {
        assert!(matches!(
            parse_provider_name("voyage").expect("valid"),
            ProviderKind::Voyage
        ));
        assert!(matches!(
            parse_provider_name("openai").expect("valid"),
            ProviderKind::OpenAi
        ));
        assert!(matches!(
            parse_provider_name("local").expect("valid"),
            ProviderKind::Local
        ));
    }

    #[test]
    fn parse_provider_name_rejects_unknown() {
        assert!(parse_provider_name("gemini").is_err());
    }

    #[test]
    fn null_embedder_returns_correct_dimension() {
        let embedder = Embedder::create_null(384);
        assert_eq!(embedder.dim(), 384);
    }

    #[tokio::test]
    async fn null_embedder_returns_zero_vectors() {
        let embedder = Embedder::create_null(128);
        let docs = embedder
            .embed_documents(&["hello", "world"])
            .await
            .expect("should succeed");
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].len(), 128);
        assert!(docs[0].iter().all(|&v| v == 0.0));

        let query = embedder.embed_query("hello").await.expect("should succeed");
        assert_eq!(query.len(), 128);
    }
}
