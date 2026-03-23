mod cli;
mod config;
mod db;
mod embed;
mod error;
mod fts;
mod hash;
mod index;
mod rank;
mod watch;

use clap::Parser;

use crate::{
    cli::{Cli, Command},
    config::{CliWeights, Config},
    embed::VoyageClient,
    rank::RrfWeights,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    let cli_weights = match &cli.command {
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
    };

    let config = Config::resolve(cli.notes_dir, cli_weights)?;
    let conn = db::connect(&config.db_path).await?;
    let fts = fts::FtsIndex::open_or_create(&config.tantivy_dir)?;
    let client = VoyageClient::new(&config.voyage_api_key);

    if fts.is_empty() {
        let chunks = db::all_chunks(&conn).await?;
        if !chunks.is_empty() {
            tracing::info!(
                chunks = chunks.len(),
                "rebuilding tantivy index from existing chunks"
            );
            fts.rebuild(chunks).await?;
        }
    }

    match cli.command {
        Command::Watch => {
            watch::run_watcher(conn, fts, client, config.notes_dir).await?;
        }
        Command::Search { query, limit, .. } => {
            let weights = RrfWeights {
                semantic: config.w_semantic,
                fts: config.w_fts,
                filename: config.w_filename,
            };
            let results = rank::search(&conn, &fts, &client, &query, limit, &weights).await?;
            for result in &results {
                println!(
                    "{:.4}\t{}\t{}",
                    result.score,
                    result.path,
                    first_line(&result.snippet)
                );
            }
        }
        Command::Reindex => {
            let stats = index::index_directory(&conn, &fts, &client, &config.notes_dir).await?;
            tracing::info!(%stats, "reindex complete");
        }
    }

    Ok(())
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("")
}
