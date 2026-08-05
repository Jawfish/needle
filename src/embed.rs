mod openai;
mod voyage;

#[cfg(feature = "local")]
mod local;

use std::{sync::Arc, time::Duration};

use serde::Deserialize;

use crate::{
    error::NeedleError,
    types::{EmbedConfig, EmbedderProfile},
};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_RETRIES: u32 = 3;

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
                let (model, model_name, dim) = local::init_model(config.model.as_deref())?;
                Ok(Self::Local(local::LocalProvider::new(
                    model, model_name, dim,
                )))
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

    pub fn profile(&self) -> EmbedderProfile {
        match self {
            Self::Voyage(p) => EmbedderProfile {
                provider: "voyage".to_owned(),
                endpoint: None,
                model: p.model().to_owned(),
                dimension: p.dim(),
            },
            Self::OpenAi(p) => EmbedderProfile {
                provider: "openai".to_owned(),
                endpoint: Some(p.api_base().to_owned()),
                model: p.model().to_owned(),
                dimension: p.dim(),
            },
            #[cfg(feature = "local")]
            Self::Local(p) => EmbedderProfile {
                provider: "local".to_owned(),
                endpoint: None,
                model: p.model().to_owned(),
                dimension: p.dim(),
            },
            #[cfg(test)]
            Self::Null { dim } => EmbedderProfile {
                provider: "test-null".to_owned(),
                endpoint: None,
                model: "test-null".to_owned(),
                dimension: *dim,
            },
        }
    }

    pub fn identity(&self) -> EmbedderProfile {
        self.profile()
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn openai_profile_resolves_defaults_and_normalizes_endpoint() {
        let config = EmbedConfig {
            provider: Some("openai".to_owned()),
            model: None,
            api_base: Some("https://example.test/v1///".to_owned()),
            dim: Some(768),
            voyage_api_key: None,
            openai_api_key: None,
            needle_api_key: None,
        };
        let profile = Embedder::from_config(&config).expect("embedder").profile();
        assert_eq!(profile.provider, "openai");
        assert_eq!(profile.endpoint.as_deref(), Some("https://example.test/v1"));
        assert_eq!(profile.model, "text-embedding-3-small");
        assert_eq!(profile.dimension, 768);
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
