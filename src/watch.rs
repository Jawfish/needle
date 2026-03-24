use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    time::Duration,
};

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

        if !changed.is_empty() {
            process_batch(&conn, &fts, client, &notes_dir, &changed).await;
        }
    }

    Ok(())
}

async fn process_batch(
    conn: &Connection,
    fts: &FtsIndex,
    client: &VoyageClient,
    notes_dir: &Path,
    changed: &HashSet<PathBuf>,
) {
    let mut needs_fts_reconcile = false;

    for path in changed {
        if path.exists() {
            match index::index_single_file(conn, fts, client, notes_dir, path).await {
                Ok(index::FtsStatus::Current) => {}
                Ok(index::FtsStatus::Stale) => needs_fts_reconcile = true,
                Err(e) => tracing::error!(path = %path.display(), error = %e, "failed to index"),
            }
        } else {
            let rel = path.strip_prefix(notes_dir).map_or_else(
                |_| path.to_string_lossy().to_string(),
                |p| p.to_string_lossy().to_string(),
            );
            match db::delete_note(conn, &rel).await {
                Ok(()) => {
                    if fts.delete(&rel).await.is_err() {
                        needs_fts_reconcile = true;
                    }
                    tracing::info!(path = rel, "deleted from index");
                }
                Err(e) => tracing::error!(path = rel, error = %e, "failed to delete from db"),
            }
        }
    }

    if needs_fts_reconcile {
        tracing::info!("reconciling FTS after partial failures");
        match db::all_chunks(conn).await {
            Ok(chunks) => {
                if let Err(e) = fts.rebuild(chunks).await {
                    tracing::error!(error = %e, "FTS reconciliation failed");
                }
            }
            Err(e) => tracing::error!(error = %e, "failed to read chunks for FTS"),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::{db, embed, fts::FtsIndex};

    fn create_file(dir: &Path, relative: &str, content: &str) {
        let path = dir.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("failed to create parent dirs");
        }
        std::fs::write(&path, content).expect("failed to write file");
    }

    async fn test_setup(
        notes_dir: &Path,
    ) -> (
        tempfile::TempDir,
        libsql::Database,
        Connection,
        tempfile::TempDir,
        FtsIndex,
        VoyageClient,
    ) {
        let db_dir = tempfile::tempdir().expect("tempdir");
        let (db, conn) = db::connect(&db_dir.path().join("test.db"))
            .await
            .expect("connect");
        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = FtsIndex::open_or_create(fts_dir.path()).expect("fts");
        let client = embed::VoyageClient::create_null();

        // Initial index to populate DB and FTS
        index::index_directory(&conn, &fts, &client, notes_dir)
            .await
            .expect("initial index");

        (db_dir, db, conn, fts_dir, fts, client)
    }

    #[tokio::test]
    async fn process_batch_indexes_new_file() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "note.md", "# Hello world");

        let (_db_dir, _db, conn, _fts_dir, fts, client) = test_setup(notes_dir.path()).await;

        // Add a new file and process the batch
        create_file(notes_dir.path(), "new.md", "# New note");
        let mut changed = HashSet::new();
        changed.insert(notes_dir.path().join("new.md"));

        process_batch(&conn, &fts, &client, notes_dir.path(), &changed).await;

        let hashes = db::all_note_hashes(&conn).await.expect("hashes");
        assert!(hashes.contains_key("new.md"));
        assert!(hashes.contains_key("note.md"));
    }

    #[tokio::test]
    async fn process_batch_handles_deleted_file() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "keep.md", "# Keep");
        create_file(notes_dir.path(), "remove.md", "# Remove");

        let (_db_dir, _db, conn, _fts_dir, fts, client) = test_setup(notes_dir.path()).await;

        // Delete the file on disk and process
        std::fs::remove_file(notes_dir.path().join("remove.md")).expect("remove");
        let mut changed = HashSet::new();
        changed.insert(notes_dir.path().join("remove.md"));

        process_batch(&conn, &fts, &client, notes_dir.path(), &changed).await;

        let hashes = db::all_note_hashes(&conn).await.expect("hashes");
        assert!(hashes.contains_key("keep.md"));
        assert!(!hashes.contains_key("remove.md"));
    }

    #[tokio::test]
    async fn process_batch_handles_mixed_changes() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "existing.md", "# Original");
        create_file(notes_dir.path(), "to_delete.md", "# Will be deleted");

        let (_db_dir, _db, conn, _fts_dir, fts, client) = test_setup(notes_dir.path()).await;

        // Modify, create, and delete in one batch
        create_file(notes_dir.path(), "existing.md", "# Modified content");
        create_file(notes_dir.path(), "brand_new.md", "# Brand new");
        std::fs::remove_file(notes_dir.path().join("to_delete.md")).expect("remove");

        let mut changed = HashSet::new();
        changed.insert(notes_dir.path().join("existing.md"));
        changed.insert(notes_dir.path().join("brand_new.md"));
        changed.insert(notes_dir.path().join("to_delete.md"));

        process_batch(&conn, &fts, &client, notes_dir.path(), &changed).await;

        let hashes = db::all_note_hashes(&conn).await.expect("hashes");
        assert!(hashes.contains_key("existing.md"));
        assert!(hashes.contains_key("brand_new.md"));
        assert!(!hashes.contains_key("to_delete.md"));
    }

    #[tokio::test]
    async fn process_batch_skips_unchanged_file() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "note.md", "# Unchanged");

        let (_db_dir, _db, conn, _fts_dir, fts, client) = test_setup(notes_dir.path()).await;

        let hash_before = db::note_hash(&conn, "note.md")
            .await
            .expect("hash")
            .expect("should exist");

        // Process the same file again without changes
        let mut changed = HashSet::new();
        changed.insert(notes_dir.path().join("note.md"));
        process_batch(&conn, &fts, &client, notes_dir.path(), &changed).await;

        let hash_after = db::note_hash(&conn, "note.md")
            .await
            .expect("hash")
            .expect("should exist");
        assert_eq!(hash_before, hash_after);
    }
}
