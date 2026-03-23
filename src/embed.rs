use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::error::NeedleError;

const VOYAGE_API_URL: &str = "https://api.voyageai.com/v1/embeddings";
const VOYAGE_MODEL: &str = "voyage-4";
const MAX_BATCH_SIZE: usize = 128;
const CHUNK_TARGET_CHARS: usize = 4000; // ~1000 tokens
#[cfg(test)]
const EMBEDDING_DIM: usize = 1024;

pub struct VoyageClient {
    inner: ClientMode,
}

enum ClientMode {
    Live {
        client: reqwest::Client,
        api_key: String,
    },
    #[cfg(test)]
    Null,
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
    input_type: &'a str,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl VoyageClient {
    pub fn new(api_key: &str) -> Self {
        Self {
            inner: ClientMode::Live {
                client: reqwest::Client::new(),
                api_key: api_key.to_owned(),
            },
        }
    }

    #[cfg(test)]
    pub const fn create_null() -> Self {
        Self {
            inner: ClientMode::Null,
        }
    }

    pub async fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut all_embeddings = Vec::with_capacity(texts.len());

        for batch in texts.chunks(MAX_BATCH_SIZE) {
            let embeddings = self.embed_batch(batch, "document").await?;
            all_embeddings.extend(embeddings);
        }

        Ok(all_embeddings)
    }

    pub async fn embed_query(&self, query: &str) -> anyhow::Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[query], "query").await?;
        embeddings
            .into_iter()
            .next()
            .context("voyage API returned no embeddings")
    }

    async fn embed_batch(&self, texts: &[&str], input_type: &str) -> anyhow::Result<Vec<Vec<f32>>> {
        let (client, api_key) = match &self.inner {
            ClientMode::Live { client, api_key } => (client, api_key),
            #[cfg(test)]
            ClientMode::Null => {
                return Ok(texts.iter().map(|_| vec![0.0; EMBEDDING_DIM]).collect());
            }
        };

        let request = EmbeddingRequest {
            model: VOYAGE_MODEL,
            input: texts,
            input_type,
        };

        let response = client
            .post(VOYAGE_API_URL)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(NeedleError::VoyageApi(format!("{status}: {body}")).into());
        }

        let parsed: EmbeddingResponse = response.json().await?;
        let embeddings: Vec<Vec<f32>> = parsed.data.into_iter().map(|d| d.embedding).collect();
        if embeddings.len() != texts.len() {
            return Err(NeedleError::EmbeddingCountMismatch {
                expected: texts.len(),
                actual: embeddings.len(),
            }
            .into());
        }
        Ok(embeddings)
    }
}

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
    content[3..]
        .find("\n---")
        .map_or(content, |end| content[end + 7..].trim_start())
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
    fn strip_frontmatter_with_dashes_in_body() {
        let content = "No frontmatter\n\n---\n\nSome divider";
        let stripped = strip_frontmatter(content);
        assert_eq!(
            stripped, content,
            "--- not at start should not be treated as frontmatter"
        );
    }
}
