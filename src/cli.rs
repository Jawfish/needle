use std::net::IpAddr;

#[derive(clap::Parser)]
#[command(
    name = "needle",
    about = "Semantic search for Markdown, text, PDF, EPUB, HTML, and Word documents"
)]
pub struct Cli {
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
    /// Serve indexed documents in a browser
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: IpAddr,
        #[arg(long, default_value_t = 8080)]
        port: u16,
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
    /// Reindex all supported documents
    Reindex,
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn docs_dir_is_not_a_supported_option() {
        assert!(Cli::try_parse_from(["needle", "--docs-dir", "/tmp", "search", "query"]).is_err());
    }

    #[test]
    fn serve_uses_default_host_and_port() {
        let cli = Cli::try_parse_from(["needle", "serve"]).expect("parse");
        let parsed = match cli.command {
            Command::Serve { host, port } => Some((host, port)),
            _ => None,
        };
        assert_eq!(
            parsed,
            Some(("127.0.0.1".parse::<IpAddr>().expect("IP"), 8080))
        );
    }

    #[test]
    fn serve_accepts_host_and_port_overrides() {
        let cli = Cli::try_parse_from(["needle", "serve", "--host", "0.0.0.0", "--port", "9090"])
            .expect("parse");
        let parsed = match cli.command {
            Command::Serve { host, port } => Some((host, port)),
            _ => None,
        };
        assert_eq!(
            parsed,
            Some(("0.0.0.0".parse::<IpAddr>().expect("IP"), 9090))
        );
    }
}
