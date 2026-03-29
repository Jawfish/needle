use std::sync::Arc;

use anyhow::Context;
use serde::Serialize;

use super::{EmbeddingResponse, HttpTransport, send_with_retry};
use crate::error::NeedleError;

const DEFAULT_API_URL: &str = "https://api.voyageai.com/v1/embeddings";
const DEFAULT_MODEL: &str = "voyage-4";
const DEFAULT_DIM: usize = 1024;
const MAX_BATCH_SIZE: usize = 128;

pub struct VoyageProvider {
    transport: Arc<dyn HttpTransport>,
    api_key: String,
    model: String,
    dim: usize,
}

#[derive(Serialize)]
struct VoyageRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
    input_type: &'a str,
}

impl VoyageProvider {
    pub fn new(
        api_key: &str,
        model: Option<&str>,
        dim_override: Option<usize>,
        transport: Arc<dyn HttpTransport>,
    ) -> anyhow::Result<Self> {
        let model_name = model.unwrap_or(DEFAULT_MODEL);
        let dim = dim_override
            .or_else(|| lookup_dim(model_name))
            .ok_or_else(|| NeedleError::UnknownModelDimension {
                model: model_name.to_owned(),
            })?;

        Ok(Self {
            transport,
            api_key: api_key.to_owned(),
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
            let embeddings = self.embed_batch(batch, "document").await?;
            all.extend(embeddings);
        }
        Ok(all)
    }

    pub async fn embed_query(&self, query: &str) -> anyhow::Result<Vec<f32>> {
        let embeddings = self.embed_batch(&[query], "query").await?;
        embeddings
            .into_iter()
            .next()
            .context("voyage API returned no embeddings")
    }

    async fn embed_batch(&self, texts: &[&str], input_type: &str) -> anyhow::Result<Vec<Vec<f32>>> {
        let body = VoyageRequest {
            model: &self.model,
            input: texts,
            input_type,
        };
        let json_body = serde_json::to_vec(&body)?;
        let api_key = self.api_key.clone();

        // A throwaway client is used only to build the `reqwest::Request`
        // value; actual dispatch goes through the injected transport.
        let builder_client = reqwest::Client::new();

        let body_bytes = send_with_retry(self.transport.as_ref(), || {
            Ok(builder_client
                .post(DEFAULT_API_URL)
                .header("Content-Type", "application/json")
                .header(
                    reqwest::header::AUTHORIZATION,
                    reqwest::header::HeaderValue::from_str(&format!("Bearer {api_key}"))
                        .context("invalid API key characters")?,
                )
                .body(json_body.clone())
                .build()?)
        })
        .await?;

        let parsed: EmbeddingResponse = serde_json::from_slice(&body_bytes)?;
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
        "voyage-4" | "voyage-3" | "voyage-3-large" | "voyage-code-3" => Some(DEFAULT_DIM),
        "voyage-3-lite" => Some(512),
        _ => None,
    }
}
