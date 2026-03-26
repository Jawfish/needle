use std::sync::{Arc, Mutex};

use anyhow::Context;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

const DEFAULT_MODEL: EmbeddingModel = EmbeddingModel::AllMiniLML6V2;
const DEFAULT_DIM: usize = 384;

pub struct LocalProvider {
    model: Arc<Mutex<TextEmbedding>>,
    dim: usize,
}

impl LocalProvider {
    pub fn new(model_name: Option<&str>) -> anyhow::Result<Self> {
        let (model_enum, dim) = resolve_model(model_name)?;
        let model =
            TextEmbedding::try_new(InitOptions::new(model_enum).with_show_download_progress(true))
                .context("failed to initialize local embedding model")?;
        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            dim,
        })
    }

    pub const fn dim(&self) -> usize {
        self.dim
    }

    pub async fn embed_documents(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let owned: Vec<String> = texts.iter().map(|s| (*s).to_owned()).collect();
        let model = Arc::clone(&self.model);
        tokio::task::spawn_blocking(move || {
            let mut model = model
                .lock()
                .map_err(|e| anyhow::anyhow!("model lock poisoned: {e}"))?;
            model
                .embed(owned, None)
                .map_err(|e| anyhow::anyhow!("local embedding failed: {e}"))
        })
        .await?
    }

    pub async fn embed_query(&self, query: &str) -> anyhow::Result<Vec<f32>> {
        let results = self.embed_documents(&[query]).await?;
        results
            .into_iter()
            .next()
            .context("local model returned no embeddings")
    }
}

fn resolve_model(name: Option<&str>) -> anyhow::Result<(EmbeddingModel, usize)> {
    match name {
        None | Some("all-MiniLM-L6-v2") => Ok((DEFAULT_MODEL, DEFAULT_DIM)),
        Some("nomic-embed-text-v1.5") => Ok((EmbeddingModel::NomicEmbedTextV15, 768)),
        Some("bge-small-en-v1.5") => Ok((EmbeddingModel::BGESmallENV15, 384)),
        Some("bge-base-en-v1.5") => Ok((EmbeddingModel::BGEBaseENV15, 768)),
        Some("bge-large-en-v1.5") => Ok((EmbeddingModel::BGELargeENV15, 1024)),
        Some(other) => anyhow::bail!("unknown local model: {other}"),
    }
}
