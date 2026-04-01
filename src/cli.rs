use std::path::PathBuf;

#[derive(clap::Parser)]
#[command(name = "needle", about = "Semantic search for markdown notes")]
pub struct Cli {
    #[arg(long = "docs-dir", action = clap::ArgAction::Append)]
    pub docs_dirs: Vec<PathBuf>,

    #[arg(long, env = "NEEDLE_PROVIDER")]
    pub provider: Option<String>,

    #[arg(long, env = "NEEDLE_MODEL")]
    pub model: Option<String>,

    #[arg(long, env = "NEEDLE_API_BASE")]
    pub api_base: Option<String>,

    #[arg(long, global = true)]
    pub json: bool,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::Subcommand)]
pub enum Command {
    /// Watch for file changes and index automatically
    Watch,
    /// Search notes using fused ranking (semantic + FTS + filename)
    Search {
        query: Option<String>,
        #[arg(short, long, default_value = "10")]
        limit: usize,
        #[arg(short, long)]
        paths_only: bool,
        #[arg(long, env = "NEEDLE_W_SEMANTIC")]
        w_semantic: Option<f64>,
        #[arg(long, env = "NEEDLE_W_FTS")]
        w_fts: Option<f64>,
        #[arg(long, env = "NEEDLE_W_FILENAME")]
        w_filename: Option<f64>,
    },
    /// Find similar document pairs based on embeddings
    Similar {
        #[arg(long, default_value = "0.85")]
        threshold: f64,
        #[arg(short, long, default_value = "50")]
        limit: usize,
        #[arg(long)]
        group: bool,
        #[arg(short, long)]
        paths_only: bool,
    },
    /// Find documents related to a specific note
    Related {
        path: String,
        #[arg(short, long, default_value = "10")]
        limit: usize,
        #[arg(short, long)]
        paths_only: bool,
    },
    /// Reindex all notes
    Reindex,
}
