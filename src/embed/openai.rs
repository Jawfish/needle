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
        api_key: Option<&str>,
        api_base: Option<&str>,
        model: Option<&str>,
        dim_override: Option<usize>,
    ) -> anyhow::Result<Self> {
        let base = api_base.unwrap_or(DEFAULT_API_BASE);
        let model_name = model.unwrap_or(DEFAULT_MODEL);
        let dim = dim_override
            .or_else(|| lookup_dim(model_name))
            .ok_or_else(|| NeedleError::UnknownModelDimension {
                model: model_name.to_owned(),
            })?;

        Ok(Self {
            client: super::build_http_client()?,
            api_key: api_key.map(str::to_owned),
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

fn lookup_dim(model: &str) -> Option<usize> {
    match model {
        "text-embedding-3-small" => Some(DEFAULT_DIM),
        "text-embedding-3-large" => Some(3072),
        "text-embedding-ada-002" => Some(1536),
        _ => None,
    }
}
