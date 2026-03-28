mod cli;
mod config;
mod db;
mod embed;
mod error;
mod fts;
mod hash;
mod index;
mod rank;
mod similar;
mod watch;

use std::io::{IsTerminal, Read};

use clap::Parser;

use crate::{
    cli::{Cli, Command},
    config::{CliEmbedArgs, CliWeights, Config},
    embed::Embedder,
    error::NeedleError,
};

fn is_dimension_mismatch(err: &anyhow::Error) -> bool {
    err.downcast_ref::<NeedleError>()
        .is_some_and(|e| matches!(e, NeedleError::DimensionMismatch { .. }))
}

/// Open the database, resetting it first when a dimension mismatch is
/// encountered and the caller intends to reindex (which rebuilds everything).
async fn open_db(
    db_path: &std::path::Path,
    dim: Option<usize>,
    allow_reset_on_mismatch: bool,
) -> anyhow::Result<(libsql::Database, libsql::Connection)> {
    match db::connect(db_path, dim).await {
        Ok(pair) => Ok(pair),
        Err(ref e) if allow_reset_on_mismatch && is_dimension_mismatch(e) => {
            tracing::warn!("dimension mismatch detected; resetting index for reindex");
            db::reset(db_path)?;
            db::connect(db_path, dim).await
        }
        Err(e) => Err(e),
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    run(Cli::parse()).await
}

fn search_needs_embedder(command: &Command, weights: &rank::RrfWeights) -> bool {
    match command {
        Command::Watch | Command::Reindex => true,
        Command::Search { .. } => weights.semantic > 0.0,
        _ => false,
    }
}

const fn extract_cli_weights(command: &Command) -> CliWeights {
    match command {
        Command::Search {
            w_semantic,
            w_fts,
            w_filename,
            ..
        } => CliWeights {
            semantic: *w_semantic,
            fts: *w_fts,
            filename: *w_filename,
        },
        _ => CliWeights {
            semantic: None,
            fts: None,
            filename: None,
        },
    }
}

async fn run(cli: Cli) -> anyhow::Result<()> {
    let cli_weights = extract_cli_weights(&cli.command);
    let cli_embed = CliEmbedArgs {
        provider: cli.provider,
        model: cli.model,
        api_base: cli.api_base,
    };
    let config = Config::resolve(cli.notes_dir, cli_weights, cli_embed)?;

    let embedder = if search_needs_embedder(&cli.command, &config.weights) {
        Some(Embedder::from_config(&config.embed)?)
    } else {
        None
    };
    let dim = embedder.as_ref().map(Embedder::dim);
    let is_reindex = matches!(cli.command, Command::Reindex);
    let (_db, conn) = open_db(&config.db_path, dim, is_reindex).await?;

    match cli.command {
        Command::Watch => {
            let fts = fts::FtsIndex::open_or_create(&config.tantivy_dir)?;
            let embedder = embedder.ok_or(NeedleError::NoEmbeddingProvider)?;
            watch::run_watcher(conn, fts, &embedder, config.notes_dir).await?;
        }
        Command::Search {
            query,
            limit,
            paths_only,
            ..
        } => {
            let query = resolve_query(
                query,
                &mut std::io::stdin().lock(),
                std::io::stdin().is_terminal(),
            )?;
            let fts = fts::FtsIndex::open_or_create(&config.tantivy_dir)?;
            let results = rank::search(
                &conn,
                &fts,
                embedder.as_ref(),
                &query,
                limit,
                &config.weights,
            )
            .await?;
            if paths_only {
                for result in &results {
                    println!("{}", result.path);
                }
            } else {
                for result in &results {
                    println!(
                        "{:.4}\t{}\t{}",
                        result.score,
                        result.path,
                        first_line(&result.snippet)
                    );
                }
            }
        }
        Command::Similar {
            threshold,
            limit,
            group,
            paths_only,
        } => {
            let pair_limit = if group { None } else { Some(limit) };
            let pairs = similar::find_similar(&conn, threshold, pair_limit).await?;
            print_similar(pairs, limit, group, paths_only);
        }
        Command::Related {
            path,
            limit,
            paths_only,
        } => {
            let results = similar::find_related(&conn, &path, limit).await?;
            if paths_only {
                for r in &results {
                    println!("{}", r.path);
                }
            } else {
                for r in &results {
                    println!("{:.4}\t{}", r.similarity, r.path);
                }
            }
        }
        Command::Reindex => {
            let fts = fts::FtsIndex::open_or_create(&config.tantivy_dir)?;
            let embedder = embedder.as_ref().ok_or(NeedleError::NoEmbeddingProvider)?;
            let stats = index::index_directory(&conn, &fts, embedder, &config.notes_dir).await?;
            tracing::info!(%stats, "reindex complete");
        }
    }

    Ok(())
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("")
}

fn print_similar(pairs: Vec<similar::SimilarPair>, limit: usize, group: bool, paths_only: bool) {
    if group {
        let mut groups = similar::group_pairs(pairs);
        groups.truncate(limit);
        if paths_only {
            for g in &groups {
                for path in &g.paths {
                    println!("{path}");
                }
            }
        } else {
            for (i, g) in groups.iter().enumerate() {
                if i > 0 {
                    println!();
                }
                println!("Group {} ({} documents):", i + 1, g.paths.len());
                for pair in &g.pairs {
                    println!(
                        "  {:.4}  {} <> {}",
                        pair.similarity, pair.path_a, pair.path_b
                    );
                }
            }
        }
    } else if paths_only {
        for pair in &pairs {
            println!("{}", pair.path_a);
            println!("{}", pair.path_b);
        }
    } else {
        for pair in &pairs {
            println!("{:.4}\t{}\t{}", pair.similarity, pair.path_a, pair.path_b);
        }
    }
}

const STDIN_QUERY_LIMIT_BYTES: u64 = 1024 * 1024; // 1 MiB

fn resolve_query(
    explicit: Option<String>,
    reader: &mut impl Read,
    is_terminal: bool,
) -> anyhow::Result<String> {
    if let Some(q) = explicit {
        return Ok(q);
    }
    if is_terminal {
        anyhow::bail!("no query provided; pass as argument or pipe to stdin");
    }
    let mut buf = String::new();
    reader
        .take(STDIN_QUERY_LIMIT_BYTES)
        .read_to_string(&mut buf)?;
    let trimmed = buf.trim().to_owned();
    if trimmed.is_empty() {
        anyhow::bail!("empty query from stdin");
    }
    Ok(trimmed)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn resolve_query_returns_explicit_argument() {
        let mut reader = Cursor::new(b"");
        let result = resolve_query(Some("hello".to_owned()), &mut reader, false);
        assert_eq!(result.expect("should succeed"), "hello");
    }

    #[test]
    fn resolve_query_reads_from_stdin_when_no_argument() {
        let mut reader = Cursor::new(b"from stdin");
        let result = resolve_query(None, &mut reader, false);
        assert_eq!(result.expect("should succeed"), "from stdin");
    }

    #[test]
    fn resolve_query_trims_whitespace_from_stdin() {
        let mut reader = Cursor::new(b"  hello world  \n");
        let result = resolve_query(None, &mut reader, false);
        assert_eq!(result.expect("should succeed"), "hello world");
    }

    #[test]
    fn resolve_query_errors_when_terminal_and_no_argument() {
        let mut reader = Cursor::new(b"");
        let result = resolve_query(None, &mut reader, true);
        assert!(result.is_err());
        let msg = result.expect_err("should fail").to_string();
        assert!(msg.contains("no query provided"), "got: {msg}");
    }

    #[test]
    fn resolve_query_errors_on_empty_stdin() {
        let mut reader = Cursor::new(b"");
        let result = resolve_query(None, &mut reader, false);
        assert!(result.is_err());
        let msg = result.expect_err("should fail").to_string();
        assert!(msg.contains("empty query"), "got: {msg}");
    }

    #[test]
    fn resolve_query_errors_on_whitespace_only_stdin() {
        let mut reader = Cursor::new(b"   \n  ");
        let result = resolve_query(None, &mut reader, false);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_query_truncates_stdin_at_limit_and_still_returns_content() {
        // Input larger than STDIN_QUERY_LIMIT_BYTES should not OOM or error;
        // the leading content within the limit is accepted as the query.
        let limit_bytes = usize::try_from(STDIN_QUERY_LIMIT_BYTES).expect("limit fits usize");
        let big_input = vec![b'x'; limit_bytes + 1];
        let mut reader = Cursor::new(big_input);
        let result = resolve_query(None, &mut reader, false);
        assert!(result.is_ok(), "oversized stdin should not error");
        let query = result.expect("should succeed");
        assert_eq!(query.len(), limit_bytes);
    }

    fn search_command(w_semantic: Option<f64>) -> Command {
        Command::Search {
            query: None,
            limit: 10,
            paths_only: false,
            w_semantic,
            w_fts: None,
            w_filename: None,
        }
    }

    #[test]
    fn search_with_positive_semantic_weight_needs_embedder() {
        let weights = rank::RrfWeights {
            semantic: 1.5,
            fts: 1.0,
            filename: 0.7,
        };
        assert!(search_needs_embedder(&search_command(None), &weights));
    }

    #[test]
    fn search_with_zero_semantic_weight_does_not_need_embedder() {
        let weights = rank::RrfWeights {
            semantic: 0.0,
            fts: 1.0,
            filename: 0.7,
        };
        assert!(!search_needs_embedder(&search_command(None), &weights));
    }

    #[test]
    fn watch_always_needs_embedder() {
        let weights = rank::RrfWeights {
            semantic: 0.0,
            fts: 0.0,
            filename: 0.0,
        };
        assert!(search_needs_embedder(&Command::Watch, &weights));
    }

    #[test]
    fn reindex_always_needs_embedder() {
        let weights = rank::RrfWeights {
            semantic: 0.0,
            fts: 0.0,
            filename: 0.0,
        };
        assert!(search_needs_embedder(&Command::Reindex, &weights));
    }

    #[test]
    fn similar_never_needs_embedder() {
        let weights = rank::RrfWeights {
            semantic: 1.5,
            fts: 1.0,
            filename: 0.7,
        };
        assert!(!search_needs_embedder(
            &Command::Similar {
                threshold: 0.85,
                limit: 50,
                group: false,
                paths_only: false,
            },
            &weights
        ));
    }

    #[tokio::test]
    async fn open_db_succeeds_when_dimensions_match() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.db");
        let result = open_db(&path, Some(384), false).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn open_db_propagates_mismatch_when_reset_not_allowed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.db");

        let (db, conn) = db::connect(&path, Some(384)).await.expect("first connect");
        drop(conn);
        drop(db);

        let result = open_db(&path, Some(1024), false).await;
        assert!(result.is_err());
        assert!(
            is_dimension_mismatch(&result.expect_err("should fail")),
            "error must be DimensionMismatch"
        );
    }

    #[tokio::test]
    async fn open_db_resets_and_reconnects_on_mismatch_when_allowed() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.db");

        let (db, conn) = db::connect(&path, Some(384)).await.expect("first connect");
        drop(conn);
        drop(db);

        let result = open_db(&path, Some(1024), true).await;
        assert!(
            result.is_ok(),
            "open_db must succeed after reset: {:?}",
            result.err()
        );

        let (_db, conn) = result.expect("open succeeded");
        assert!(
            db::chunks_table_exists(&conn).await.expect("check failed"),
            "schema must be initialised with the new dimension"
        );
    }
}
