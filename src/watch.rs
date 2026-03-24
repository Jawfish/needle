use std::{collections::HashSet, path::PathBuf, time::Duration};

use libsql::Connection;
use notify::{EventKind, RecursiveMode, Watcher};
use tokio::sync::mpsc;

use crate::{db, embed::VoyageClient, fts::FtsIndex, index};

const DEBOUNCE_MS: u64 = 500;

pub async fn run_watcher(
    conn: Connection,
    fts: FtsIndex,
    client: &VoyageClient,
    notes_dir: PathBuf,
) -> anyhow::Result<()> {
    tracing::info!(dir = %notes_dir.display(), "initial indexing");
    let stats = index::index_directory(&conn, &fts, client, &notes_dir).await?;
    tracing::info!(%stats, "initial index complete");

    let (tx, mut rx) = mpsc::unbounded_channel::<PathBuf>();

    let watch_dir = notes_dir.clone();
    let mut watcher = {
        let notes_dir = watch_dir.clone();
        notify::recommended_watcher(move |event: Result<notify::Event, _>| {
            if let Ok(event) = event {
                match event.kind {
                    EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => {
                        for path in event.paths {
                            if path.extension().is_some_and(|e| e == "md")
                                && !index::is_in_hidden_dir(&path, &notes_dir)
                            {
                                let _ = tx.send(path);
                            }
                        }
                    }
                    _ => {}
                }
            }
        })?
    };

    watcher.watch(&watch_dir, RecursiveMode::Recursive)?;
    tracing::info!(dir = %notes_dir.display(), "watching for changes");

    let shutdown = tokio::signal::ctrl_c();
    tokio::pin!(shutdown);

    loop {
        let mut changed = HashSet::new();

        tokio::select! {
            Some(path) = rx.recv() => {
                changed.insert(path);
                // Debounce: drain additional events within the window
                tokio::time::sleep(Duration::from_millis(DEBOUNCE_MS)).await;
                while let Ok(path) = rx.try_recv() {
                    changed.insert(path);
                }
            }
            Ok(()) = &mut shutdown => {
                tracing::info!("shutting down");
                break;
            }
        }

        for path in &changed {
            if path.exists() {
                if let Err(e) = index::index_single_file(&conn, client, &notes_dir, path).await {
                    tracing::error!(path = %path.display(), error = %e, "failed to index");
                }
            } else {
                let rel = path.strip_prefix(&notes_dir).map_or_else(
                    |_| path.to_string_lossy().to_string(),
                    |p| p.to_string_lossy().to_string(),
                );
                match db::delete_note(&conn, &rel).await {
                    Ok(()) => tracing::info!(path = rel, "deleted from index"),
                    Err(e) => tracing::error!(path = rel, error = %e, "failed to delete from db"),
                }
            }
        }

        // Reconcile FTS from authoritative DB state after each batch
        if !changed.is_empty() {
            match db::all_chunks(&conn).await {
                Ok(chunks) => {
                    if let Err(e) = fts.rebuild(chunks).await {
                        tracing::error!(error = %e, "FTS reconciliation failed");
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to read chunks for FTS reconciliation");
                }
            }
        }
    }

    Ok(())
}
