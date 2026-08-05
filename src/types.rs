use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedderProfile {
    pub provider: String,
    pub endpoint: Option<String>,
    pub model: String,
    pub dimension: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexProfile {
    pub embedder: EmbedderProfile,
    pub preparer: String,
}

#[derive(Debug)]
pub struct RrfWeights {
    pub semantic: f64,
    pub fts: f64,
    pub filename: f64,
}

impl Default for RrfWeights {
    fn default() -> Self {
        Self {
            semantic: 1.5,
            fts: 1.0,
            filename: 0.7,
        }
    }
}

#[derive(Debug)]
pub struct EmbedConfig {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub api_base: Option<String>,
    pub dim: Option<usize>,
    pub voyage_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub needle_api_key: Option<String>,
}
