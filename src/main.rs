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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    run(Cli::parse()).await
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

    let needs_embedder = matches!(
        cli.command,
        Command::Watch | Command::Reindex | Command::Search { .. }
    );
    let embedder = if needs_embedder {
        Some(Embedder::from_config(&config.embed)?)
    } else {
        None
    };
    let dim = embedder.as_ref().map(Embedder::dim);
    let (_db, conn) = db::connect(&config.db_path, dim).await?;

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
    reader.read_to_string(&mut buf)?;
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
}
