#[derive(Debug, thiserror::Error)]
pub enum NeedleError {
    #[error("docs directories not found:\n{0}")]
    MissingDirectories(String),

    #[error("docs directories overlap (configure non-overlapping paths):\n{0}")]
    OverlappingDirectories(String),

    #[error("missing API key: {0}")]
    MissingApiKey(String),

    #[error("embedding API error: {0}")]
    EmbeddingApi(String),

    #[error("embedding count mismatch: expected {expected}, got {actual}")]
    EmbeddingCountMismatch { expected: usize, actual: usize },

    #[error("note has no embeddings: {0}")]
    NoteNotEmbedded(String),

    #[error(
        "embedding dimension mismatch: database has {db}, provider gives {provider} (reindex required)"
    )]
    DimensionMismatch { db: usize, provider: usize },

    #[error(
        "no embedding provider available: set VOYAGE_API_KEY, OPENAI_API_KEY, or compile with --features local"
    )]
    NoEmbeddingProvider,

    #[error("unknown model {model}: set dim in config or NEEDLE_DIM")]
    UnknownModelDimension { model: String },
}
