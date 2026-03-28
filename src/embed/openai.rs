use anyhow::Context;
use serde::Serialize;

use super::{EmbeddingResponse, send_with_retry};
use crate::error::NeedleError;

const DEFAULT_API_BASE: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "text-embedding-3-small";
const DEFAULT_DIM: usize = 1536;
const MAX_BATCH_SIZE: usize = 128;

pub struct OpenAiProvider {
    client: reqwest::Client,
    api_key: Option<String>,
    api_base: String,
    model: String,
    dim: usize,
}

#[derive(Serialize)]
struct OpenAiRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

impl OpenAiProvider {
    pub fn new(
        openai_api_key: Option<&str>,
        needle_api_key: Option<&str>,
        api_base: Option<&str>,
        model: Option<&str>,
        dim_override: Option<usize>,
    ) -> anyhow::Result<Self> {
        let base = api_base.unwrap_or(DEFAULT_API_BASE);
        let is_custom_base = base.trim_end_matches('/') != DEFAULT_API_BASE;
        let model_name = model.unwrap_or(DEFAULT_MODEL);
        let dim = dim_override
            .or_else(|| lookup_dim(model_name))
            .ok_or_else(|| NeedleError::UnknownModelDimension {
                model: model_name.to_owned(),
            })?;

        let api_key = if is_custom_base {
            if needle_api_key.is_none() && openai_api_key.is_some() {
                tracing::warn!(
                    "NEEDLE_API_BASE is set but only OPENAI_API_KEY is present; \
                     OPENAI_API_KEY will not be sent to the custom endpoint. \
                     Set NEEDLE_API_KEY to authenticate with the custom base URL."
                );
            }
            needle_api_key.map(str::to_owned)
        } else {
            openai_api_key.map(str::to_owned)
        };

        Ok(Self {
            client: super::build_http_client()?,
            api_key,
            api_base: base.trim_end_matches('/').to_owned(),
            model: model_name.to_owned(),
            dim,
        })
    }

    pub const fn dim(&self) -> usize {
        self.dim
    }

    pub async fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut all = Vec::with_capacity(texts.len());
        for batch in texts.chunks(MAX_BATCH_SIZE) {
            let embeddings = self.embed_batch(batch).await?;
            all.extend(embeddings);
        }
        Ok(all)
    }

    pub async fn embed_query(&self, query: &str) -> anyhow::Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[query]).await?;
        embeddings
            .into_iter()
            .next()
            .context("OpenAI API returned no embeddings")
    }

    async fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let body = OpenAiRequest {
            model: &self.model,
            input: texts,
        };
        let json_body = serde_json::to_value(&body)?;
        let url = format!("{}/embeddings", self.api_base);

        let api_key = self.api_key.clone();
        let response = send_with_retry(|| {
            let mut req = self.client.post(&url).json(&json_body);
            if let Some(key) = &api_key {
                req = req.header("Authorization", format!("Bearer {key}"));
            }
            req
        })
        .await?;

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

#[cfg(test)]
impl OpenAiProvider {
    fn selected_api_key(&self) -> Option<&str> {
        self.api_key.as_deref()
    }
}

fn lookup_dim(model: &str) -> Option<usize> {
    match model {
        "text-embedding-3-small" => Some(DEFAULT_DIM),
        "text-embedding-3-large" => Some(3072),
        "text-embedding-ada-002" => Some(1536),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(
        openai_key: Option<&str>,
        needle_key: Option<&str>,
        api_base: Option<&str>,
    ) -> OpenAiProvider {
        OpenAiProvider::new(openai_key, needle_key, api_base, None, Some(1536))
            .expect("provider construction failed")
    }

    #[test]
    fn no_base_uses_openai_key() {
        let p = provider(Some("sk-openai"), None, None);
        assert_eq!(p.selected_api_key(), Some("sk-openai"));
    }

    #[test]
    fn explicit_default_base_url_uses_openai_key() {
        let p = provider(Some("sk-openai"), None, Some(DEFAULT_API_BASE));
        assert_eq!(
            p.selected_api_key(),
            Some("sk-openai"),
            "explicit default base must not suppress OPENAI_API_KEY"
        );
    }

    #[test]
    fn explicit_default_base_url_with_trailing_slash_uses_openai_key() {
        let with_slash = format!("{DEFAULT_API_BASE}/");
        let p = provider(Some("sk-openai"), None, Some(&with_slash));
        assert_eq!(
            p.selected_api_key(),
            Some("sk-openai"),
            "trailing slash on default base must not suppress OPENAI_API_KEY"
        );
    }

    #[test]
    fn custom_base_uses_needle_key() {
        let p = provider(
            Some("sk-openai"),
            Some("nk-needle"),
            Some("http://localhost:11434/v1"),
        );
        assert_eq!(p.selected_api_key(), Some("nk-needle"));
    }

    #[test]
    fn custom_base_with_only_openai_key_sends_no_auth() {
        let p = provider(Some("sk-openai"), None, Some("http://localhost:11434/v1"));
        assert_eq!(
            p.selected_api_key(),
            None,
            "OPENAI_API_KEY must not be forwarded to a custom endpoint"
        );
    }

    #[test]
    fn custom_base_with_no_keys_sends_no_auth() {
        let p = provider(None, None, Some("http://localhost:11434/v1"));
        assert_eq!(p.selected_api_key(), None);
    }

    #[test]
    fn no_base_no_keys_sends_no_auth() {
        let p = provider(None, None, None);
        assert_eq!(p.selected_api_key(), None);
    }
}
