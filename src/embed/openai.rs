use std::sync::Arc;

use anyhow::Context;
use serde::Serialize;

use super::{EmbeddingResponse, HttpTransport, send_with_retry};
use crate::error::NeedleError;

const DEFAULT_API_BASE: &str = "https://api.openai.com/v1";
const DEFAULT_MODEL: &str = "text-embedding-3-small";
const DEFAULT_DIM: usize = 1536;
const MAX_BATCH_SIZE: usize = 128;

pub struct OpenAiProvider {
    transport: Arc<dyn HttpTransport>,
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
        transport: Arc<dyn HttpTransport>,
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
            transport,
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
        let json_body = serde_json::to_vec(&body)?;
        let url = format!("{}/embeddings", self.api_base);
        let api_key = self.api_key.clone();

        // A throwaway client is used only to build the `reqwest::Request`
        // value; actual dispatch goes through the injected transport.
        let builder_client = reqwest::Client::new();

        let body_bytes = send_with_retry(self.transport.as_ref(), || {
            let mut builder = builder_client
                .post(&url)
                .header("Content-Type", "application/json")
                .body(json_body.clone());
            if let Some(key) = &api_key {
                builder = builder.header(
                    reqwest::header::AUTHORIZATION,
                    reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
                        .context("invalid API key characters")?,
                );
            }
            Ok(builder.build()?)
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
    use std::sync::Mutex;

    use super::*;
    use crate::embed::HttpTransport;

    struct FakeTransport {
        responses: Mutex<Vec<(reqwest::StatusCode, Vec<u8>)>>,
    }

    impl FakeTransport {
        fn new(responses: Vec<(reqwest::StatusCode, Vec<u8>)>) -> Arc<Self> {
            Arc::new(Self {
                responses: Mutex::new(responses),
            })
        }
    }

    impl HttpTransport for FakeTransport {
        fn send(&self, _request: reqwest::Request) -> crate::embed::SendFuture<'_> {
            Box::pin(async move {
                let mut queue = self
                    .responses
                    .lock()
                    .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
                if queue.is_empty() {
                    anyhow::bail!("FakeTransport: no more responses queued");
                }
                Ok(queue.remove(0))
            })
        }
    }

    fn embedding_response_body(dim: usize) -> Vec<u8> {
        let vec: Vec<f32> = vec![0.0; dim];
        let json = serde_json::json!({
            "data": [{ "embedding": vec }]
        });
        serde_json::to_vec(&json).expect("valid json")
    }

    fn provider_with_transport(transport: Arc<dyn HttpTransport>) -> OpenAiProvider {
        OpenAiProvider::new(Some("sk-openai"), None, None, None, Some(1536), transport)
            .expect("provider construction failed")
    }

    fn key_test_provider(
        openai_key: Option<&str>,
        needle_key: Option<&str>,
        api_base: Option<&str>,
        transport: Arc<dyn HttpTransport>,
    ) -> OpenAiProvider {
        OpenAiProvider::new(
            openai_key,
            needle_key,
            api_base,
            None,
            Some(1536),
            transport,
        )
        .expect("provider construction failed")
    }

    fn noop_transport() -> Arc<dyn HttpTransport> {
        FakeTransport::new(vec![])
    }

    #[test]
    fn no_base_uses_openai_key() {
        let p = key_test_provider(Some("sk-openai"), None, None, noop_transport());
        assert_eq!(p.selected_api_key(), Some("sk-openai"));
    }

    #[test]
    fn explicit_default_base_url_uses_openai_key() {
        let p = key_test_provider(
            Some("sk-openai"),
            None,
            Some(DEFAULT_API_BASE),
            noop_transport(),
        );
        assert_eq!(
            p.selected_api_key(),
            Some("sk-openai"),
            "explicit default base must not suppress OPENAI_API_KEY"
        );
    }

    #[test]
    fn explicit_default_base_url_with_trailing_slash_uses_openai_key() {
        let with_slash = format!("{DEFAULT_API_BASE}/");
        let p = key_test_provider(Some("sk-openai"), None, Some(&with_slash), noop_transport());
        assert_eq!(
            p.selected_api_key(),
            Some("sk-openai"),
            "trailing slash on default base must not suppress OPENAI_API_KEY"
        );
    }

    #[test]
    fn custom_base_uses_needle_key() {
        let p = key_test_provider(
            Some("sk-openai"),
            Some("nk-needle"),
            Some("http://localhost:11434/v1"),
            noop_transport(),
        );
        assert_eq!(p.selected_api_key(), Some("nk-needle"));
    }

    #[test]
    fn custom_base_with_only_openai_key_sends_no_auth() {
        let p = key_test_provider(
            Some("sk-openai"),
            None,
            Some("http://localhost:11434/v1"),
            noop_transport(),
        );
        assert_eq!(
            p.selected_api_key(),
            None,
            "OPENAI_API_KEY must not be forwarded to a custom endpoint"
        );
    }

    #[test]
    fn custom_base_with_no_keys_sends_no_auth() {
        let p = key_test_provider(
            None,
            None,
            Some("http://localhost:11434/v1"),
            noop_transport(),
        );
        assert_eq!(p.selected_api_key(), None);
    }

    #[test]
    fn no_base_no_keys_sends_no_auth() {
        let p = key_test_provider(None, None, None, noop_transport());
        assert_eq!(p.selected_api_key(), None);
    }

    #[tokio::test]
    async fn successful_response_returns_embeddings() {
        let transport: Arc<dyn HttpTransport> = FakeTransport::new(vec![(
            reqwest::StatusCode::OK,
            embedding_response_body(1536),
        )]);
        let p = provider_with_transport(transport);
        let result = p.embed_query("hello").await.expect("should succeed");
        assert_eq!(result.len(), 1536);
    }

    #[tokio::test]
    async fn rate_limit_response_retries_and_succeeds() {
        let transport: Arc<dyn HttpTransport> = FakeTransport::new(vec![
            (
                reqwest::StatusCode::TOO_MANY_REQUESTS,
                b"rate limited".to_vec(),
            ),
            (reqwest::StatusCode::OK, embedding_response_body(1536)),
        ]);
        let p = provider_with_transport(transport);
        let result = p
            .embed_query("hello")
            .await
            .expect("should retry and succeed");
        assert_eq!(result.len(), 1536);
    }

    #[tokio::test]
    async fn client_error_does_not_retry() {
        let fake = FakeTransport::new(vec![
            (reqwest::StatusCode::UNAUTHORIZED, b"unauthorized".to_vec()),
            // A second response is queued; it must never be consumed.
            (reqwest::StatusCode::OK, embedding_response_body(1536)),
        ]);
        let p = provider_with_transport(Arc::clone(&fake) as Arc<dyn HttpTransport>);
        let err = p
            .embed_query("hello")
            .await
            .expect_err("401 must not retry");
        assert!(
            err.to_string().contains("401"),
            "error should mention status code"
        );
        // Verify the second response was never consumed (no retry happened).
        let remaining = fake.responses.lock().expect("lock").len();
        assert_eq!(remaining, 1, "only one response should have been consumed");
    }

    #[tokio::test]
    async fn server_error_exhausts_retries() {
        let responses: Vec<_> = (0..=crate::embed::MAX_RETRIES)
            .map(|_| (reqwest::StatusCode::INTERNAL_SERVER_ERROR, b"oops".to_vec()))
            .collect();
        let transport: Arc<dyn HttpTransport> = FakeTransport::new(responses);
        let p = provider_with_transport(transport);
        let err = p
            .embed_query("hello")
            .await
            .expect_err("exhausted retries must fail");
        assert!(err.to_string().contains("500"));
    }

    #[tokio::test]
    async fn network_error_retries_and_eventually_fails() {
        // All attempts fail with a transport error.
        struct AlwaysFailTransport;
        impl HttpTransport for AlwaysFailTransport {
            fn send(&self, _request: reqwest::Request) -> crate::embed::SendFuture<'_> {
                Box::pin(async { anyhow::bail!("simulated network failure") })
            }
        }
        let transport: Arc<dyn HttpTransport> = Arc::new(AlwaysFailTransport);
        let p = provider_with_transport(transport);
        let err = p
            .embed_query("hello")
            .await
            .expect_err("all retries exhausted must fail");
        assert!(
            err.to_string().contains("network failure"),
            "error should surface the transport message"
        );
    }
}
