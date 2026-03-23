use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum NeedleError {
    #[error("notes directory not found: {0}")]
    NotesDirectoryNotFound(PathBuf),

    #[error("VOYAGE_API_KEY not set")]
    MissingApiKey,

    #[error("voyage API error: {0}")]
    VoyageApi(String),

    #[error("embedding count mismatch: expected {expected}, got {actual}")]
    EmbeddingCountMismatch { expected: usize, actual: usize },
}
