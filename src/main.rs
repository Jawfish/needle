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

use clap::Parser;

use crate::{
    cli::{Cli, Command},
    config::{CliWeights, Config},
    embed::VoyageClient,
    error::NeedleError,
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
    let (_db, conn) = db::connect(&config.db_path).await?;
    let client = config
        .voyage_api_key
        .as_deref()
        .map(VoyageClient::new)
        .transpose()?;

    match cli.command {
        Command::Watch => {
            let fts = fts::FtsIndex::open_or_create(&config.tantivy_dir)?;
            let client = client.as_ref().ok_or(NeedleError::MissingApiKey)?;
            watch::run_watcher(conn, fts, client, config.notes_dir).await?;
        }
        Command::Search { query, limit, .. } => {
            let fts = fts::FtsIndex::open_or_create(&config.tantivy_dir)?;
            let results =
                rank::search(&conn, &fts, client.as_ref(), &query, limit, &config.weights).await?;
            for result in &results {
                println!(
                    "{:.4}\t{}\t{}",
                    result.score,
                    result.path,
                    first_line(&result.snippet)
                );
            }
        }
        Command::Similar {
            threshold,
            limit,
            group,
        } => {
            let pair_limit = if group { None } else { Some(limit) };
            let pairs = similar::find_similar(&conn, threshold, pair_limit).await?;
            if group {
                let mut groups = similar::group_pairs(pairs);
                groups.truncate(limit);
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
            } else {
                for pair in &pairs {
                    println!("{:.4}\t{}\t{}", pair.similarity, pair.path_a, pair.path_b);
                }
            }
        }
        Command::Reindex => {
            let fts = fts::FtsIndex::open_or_create(&config.tantivy_dir)?;
            let client = client.as_ref().ok_or(NeedleError::MissingApiKey)?;
            let stats = index::index_directory(&conn, &fts, client, &config.notes_dir).await?;
            tracing::info!(%stats, "reindex complete");
        }
    }

    Ok(())
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("")
}
