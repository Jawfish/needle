use std::net::IpAddr;

use crate::service;

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

    /// Increase needle's log verbosity (repeatable)
    #[arg(short = 'v', long = "verbose", global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::Args)]
pub struct ServiceArgs {
    #[arg(value_enum)]
    pub role: service::Role,
    #[arg(long, value_enum, default_value_t = service::Backend::host())]
    pub backend: service::Backend,
    #[arg(long)]
    pub exec_path: Option<std::path::PathBuf>,
    #[arg(long, default_value = "15m")]
    pub interval: service::Interval,
    #[arg(long, default_value = "info")]
    pub log_level: String,
    #[arg(long, default_value = "127.0.0.1")]
    pub host: IpAddr,
    #[arg(long, default_value_t = 8080)]
    pub port: u16,
}

#[derive(clap::Subcommand)]
pub enum Command {
    /// List configured documentation namespaces
    Namespaces,
    /// Watch for file changes and index automatically
    Watch,
    /// Search notes using fused ranking (semantic + FTS + filename)
    Search {
        query: Option<String>,
        #[arg(long = "namespace")]
        namespaces: Vec<String>,
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
    /// Print a service definition for the selected backend
    Service(ServiceArgs),
    /// Find similar document pairs based on embeddings
    Similar {
        #[arg(long = "namespace")]
        namespaces: Vec<String>,
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
    /// List remembered document preparation failures
    Failures,
    /// Reindex all supported documents
    Reindex {
        #[arg(long)]
        retry_failed: bool,
    },
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn namespaces_command_parses() {
        let cli = Cli::try_parse_from(["needle", "namespaces"]).expect("parse");
        assert!(matches!(cli.command, Command::Namespaces));
    }

    #[test]
    fn namespaces_command_accepts_json_output() {
        let cli = Cli::try_parse_from(["needle", "--json", "namespaces"]).expect("parse");
        assert!(cli.json);
        assert!(matches!(cli.command, Command::Namespaces));
    }

    #[test]
    fn failures_command_accepts_json_output() {
        let cli = Cli::try_parse_from(["needle", "--json", "failures"]).expect("parse");
        assert!(cli.json);
        assert!(matches!(cli.command, Command::Failures));
    }

    #[test]
    fn reindex_retry_failed_flag_parses_and_defaults_to_false() {
        let default = Cli::try_parse_from(["needle", "reindex"]).expect("parse");
        assert!(matches!(
            default.command,
            Command::Reindex {
                retry_failed: false
            }
        ));

        let retry = Cli::try_parse_from(["needle", "reindex", "--retry-failed"]).expect("parse");
        assert!(matches!(
            retry.command,
            Command::Reindex { retry_failed: true }
        ));
    }

    #[test]
    fn search_accepts_repeated_namespace_options() {
        let cli = Cli::try_parse_from([
            "needle",
            "search",
            "query",
            "--namespace",
            "alpha",
            "--namespace",
            "shared",
        ])
        .expect("parse");
        let namespaces = match cli.command {
            Command::Search { namespaces, .. } => Some(namespaces),
            _ => None,
        };
        assert_eq!(
            namespaces,
            Some(vec!["alpha".to_owned(), "shared".to_owned()])
        );
    }

    #[test]
    fn search_without_namespace_options_is_unscoped() {
        let cli = Cli::try_parse_from(["needle", "search", "query"]).expect("parse");
        let namespaces = match cli.command {
            Command::Search { namespaces, .. } => Some(namespaces),
            _ => None,
        };
        assert_eq!(namespaces, Some(Vec::new()));
    }

    #[test]
    fn similar_accepts_repeated_namespace_options() {
        let cli = Cli::try_parse_from([
            "needle",
            "similar",
            "--namespace",
            "alpha",
            "--namespace",
            "shared",
        ])
        .expect("parse");
        let namespaces = match cli.command {
            Command::Similar { namespaces, .. } => Some(namespaces),
            _ => None,
        };
        assert_eq!(
            namespaces,
            Some(vec!["alpha".to_owned(), "shared".to_owned()])
        );
    }

    #[test]
    fn similar_without_namespace_options_is_unscoped() {
        let cli = Cli::try_parse_from(["needle", "similar"]).expect("parse");
        let namespaces = match cli.command {
            Command::Similar { namespaces, .. } => Some(namespaces),
            _ => None,
        };
        assert_eq!(namespaces, Some(Vec::new()));
    }

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
    fn service_role_parses_with_defaults() {
        let cli = Cli::try_parse_from(["needle", "service", "serve"]).expect("parse");
        let parsed = match cli.command {
            Command::Service(args) => Some((
                args.role,
                args.backend,
                args.exec_path,
                args.interval,
                args.log_level,
                args.host,
                args.port,
            )),
            _ => None,
        };
        assert_eq!(
            parsed,
            Some((
                service::Role::Serve,
                service::Backend::host(),
                None,
                "15m".parse().expect("interval"),
                "info".to_owned(),
                "127.0.0.1".parse::<IpAddr>().expect("IP"),
                8080,
            ))
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
