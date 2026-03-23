use std::path::PathBuf;

#[derive(clap::Parser)]
#[command(name = "needle", about = "Semantic search for markdown notes")]
pub struct Cli {
    #[arg(long, env = "ZK_NOTEBOOK_DIR")]
    pub notes_dir: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::Subcommand)]
pub enum Command {
    /// Watch for file changes and index automatically
    Watch,
    /// Search notes using fused ranking (semantic + FTS + filename)
    Search {
        query: String,
        #[arg(short, long, default_value = "10")]
        limit: usize,
        #[arg(long, env = "NEEDLE_W_SEMANTIC")]
        w_semantic: Option<f64>,
        #[arg(long, env = "NEEDLE_W_FTS")]
        w_fts: Option<f64>,
        #[arg(long, env = "NEEDLE_W_FILENAME")]
        w_filename: Option<f64>,
    },
    /// Reindex all notes
    Reindex,
}
