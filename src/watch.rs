use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use notify::{EventKind, RecursiveMode, Watcher};
use tokio::sync::mpsc;

#[cfg(test)]
use crate::document::DefaultPreparer;
#[cfg(test)]
use crate::document::PreparedChunk;
use crate::{db, document::DocumentPreparer, embed::Embedder, fts::FtsIndex, index};

const DEBOUNCE_MS: u64 = 500;

/// A store whose DB connection and FTS index are already open and ready.
///
/// Constructed by the composition root (`run_watch` in `main.rs`); passed into
/// the watcher so it contains no resource-construction logic itself.
pub struct OpenStore {
    pub notes_dir: PathBuf,
    pub conn: libsql::Connection,
    pub fts: FtsIndex,
}

/// Watch all `stores` for filesystem changes and keep their indices up to date.
///
/// Callers are responsible for opening each store (DB connection, FTS index,
/// initial indexing pass, lock acquisition) before calling this function.
pub async fn run_watcher(
    stores: Vec<OpenStore>,
    embedder: &Embedder,
    preparer: Arc<dyn DocumentPreparer>,
) -> anyhow::Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<PathBuf>();

    // Collect notes_dirs for the watcher closure. Config resolution guarantees
    // non-overlapping roots, so each path matches at most one entry here.
    let notes_dirs: Vec<PathBuf> = stores.iter().map(|s| s.notes_dir.clone()).collect();

    let mut watcher = {
        let notes_dirs_clone = notes_dirs.clone();
        let preparer = Arc::clone(&preparer);
        notify::recommended_watcher(move |event: Result<notify::Event, _>| {
            if let Ok(event) = event {
                match event.kind {
                    EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => {
                        for path in event.paths {
                            if should_index_path_with_preparer(
                                &path,
                                notes_dirs_clone.iter(),
                                preparer.as_ref(),
                            ) {
                                let _ = tx.send(path);
                            }
                        }
                    }
                    _ => {}
                }
            }
        })?
    };

    for notes_dir in &notes_dirs {
        watcher.watch(notes_dir, RecursiveMode::Recursive)?;
        tracing::info!(dir = %notes_dir.display(), "watching for changes");
    }

    let shutdown = tokio::signal::ctrl_c();
    tokio::pin!(shutdown);

    loop {
        let mut changed = HashSet::new();

        tokio::select! {
            Some(path) = rx.recv() => {
                changed.insert(path);
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
            dispatch_changes_with_preparer(
                &stores,
                &notes_dirs,
                embedder,
                &changed,
                preparer.as_ref(),
            )
            .await;
        }
    }

    Ok(())
}

/// Route each changed path to its owning store and process it.
///
/// Extracted so tests can drive dispatch directly without a real FS watcher.
#[cfg(test)]
pub async fn dispatch_changes(
    stores: &[OpenStore],
    notes_dirs: &[PathBuf],
    embedder: &Embedder,
    changed: &HashSet<PathBuf>,
) {
    dispatch_changes_with_preparer(
        stores,
        notes_dirs,
        embedder,
        changed,
        &DefaultPreparer::default(),
    )
    .await;
}

pub async fn dispatch_changes_with_preparer(
    stores: &[OpenStore],
    notes_dirs: &[PathBuf],
    embedder: &Embedder,
    changed: &HashSet<PathBuf>,
    preparer: &dyn DocumentPreparer,
) {
    for path in changed {
        // Non-overlapping roots (enforced at config time) guarantee at
        // most one store matches, so position is deterministic.
        let store_idx = notes_dirs
            .iter()
            .position(|dir| path.starts_with(dir.as_path()));

        if let Some(idx) = store_idx {
            let store = &stores[idx];
            process_single_file(
                &store.conn,
                &store.fts,
                embedder,
                &store.notes_dir,
                path,
                preparer,
            )
            .await;
        }
    }
}

#[cfg(test)]
pub fn should_index_path<'a>(
    path: &std::path::Path,
    roots: impl Iterator<Item = &'a PathBuf>,
) -> bool {
    should_index_path_with_preparer(path, roots, &DefaultPreparer::default())
}

pub fn should_index_path_with_preparer<'a>(
    path: &std::path::Path,
    roots: impl Iterator<Item = &'a PathBuf>,
    preparer: &dyn DocumentPreparer,
) -> bool {
    let owning_root = roots
        .into_iter()
        .find(|dir| path.starts_with(dir.as_path()));
    owning_root
        .is_some_and(|root| !index::is_in_hidden_dir(path, root) && preparer.supports_path(path))
}

async fn process_single_file(
    conn: &libsql::Connection,
    fts: &FtsIndex,
    embedder: &Embedder,
    notes_dir: &Path,
    path: &Path,
    preparer: &dyn DocumentPreparer,
) {
    if path.exists() {
        match index::index_single_file_with_preparer(conn, fts, embedder, notes_dir, path, preparer)
            .await
        {
            Ok(index::FtsStatus::Current) => {}
            Ok(index::FtsStatus::Stale) => reconcile_fts(conn, fts).await,
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
                    reconcile_fts(conn, fts).await;
                }
                tracing::info!(path = rel, "deleted from index");
            }
            Err(e) => tracing::error!(path = rel, error = %e, "failed to delete from db"),
        }
    }
}

async fn reconcile_fts(conn: &libsql::Connection, fts: &FtsIndex) {
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

#[cfg(test)]
async fn process_batch(
    conn: &libsql::Connection,
    fts: &FtsIndex,
    embedder: &Embedder,
    notes_dir: &Path,
    changed: &HashSet<PathBuf>,
) {
    for path in changed {
        process_single_file(
            conn,
            fts,
            embedder,
            notes_dir,
            path,
            &DefaultPreparer::default(),
        )
        .await;
    }
}

#[cfg(test)]
mod tests {
    use std::{
        path::Path,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use super::*;
    use crate::{db, embed, fts::FtsIndex};

    struct CountingMarkdownPreparer(Arc<AtomicUsize>);

    impl DocumentPreparer for CountingMarkdownPreparer {
        fn supports_path(&self, source_path: &Path) -> bool {
            source_path
                .extension()
                .is_some_and(|extension| extension == "md")
        }

        fn prepare(
            &self,
            _source_path: &Path,
            source: &[u8],
        ) -> anyhow::Result<Vec<PreparedChunk>> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(vec![PreparedChunk::from(
                std::str::from_utf8(source)?.to_owned(),
            )])
        }

        fn profile(&self) -> &'static str {
            "counting-markdown-v1"
        }
    }

    struct NotePreparer;

    impl DocumentPreparer for NotePreparer {
        fn supports_path(&self, source_path: &Path) -> bool {
            source_path
                .extension()
                .is_some_and(|extension| extension == "note")
        }

        fn prepare(
            &self,
            _source_path: &Path,
            source: &[u8],
        ) -> anyhow::Result<Vec<PreparedChunk>> {
            Ok(vec![PreparedChunk::from(
                std::str::from_utf8(source)?.to_owned(),
            )])
        }

        fn profile(&self) -> &'static str {
            "note-v1"
        }
    }

    fn create_file(dir: &Path, relative: &str, content: &str) {
        let path = dir.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("failed to create parent dirs");
        }
        std::fs::write(&path, content).expect("failed to write file");
    }

    /// Build a temp-backed `OpenStore` rooted at `notes_dir`.
    async fn open_store(notes_dir: &Path) -> (Vec<tempfile::TempDir>, OpenStore) {
        let db_dir = tempfile::tempdir().expect("db tempdir");
        let fts_dir = tempfile::tempdir().expect("fts tempdir");
        let (_, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");
        let fts = FtsIndex::open_or_create(fts_dir.path()).expect("fts");
        let store = OpenStore {
            notes_dir: notes_dir.to_path_buf(),
            conn,
            fts,
        };
        (vec![db_dir, fts_dir], store)
    }

    // ---------- routing tests -------------------------------------------------

    #[tokio::test]
    async fn dispatch_uses_preparer_for_supported_single_file_updates() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "entry.note", "prepared content");
        let (_temps, store) = open_store(notes_dir.path()).await;
        let notes_dirs = vec![notes_dir.path().to_path_buf()];
        let mut changed = HashSet::new();
        changed.insert(notes_dir.path().join("entry.note"));

        dispatch_changes_with_preparer(
            std::slice::from_ref(&store),
            &notes_dirs,
            &embed::Embedder::create_null(1024),
            &changed,
            &NotePreparer,
        )
        .await;

        assert!(
            db::all_note_hashes(&store.conn)
                .await
                .expect("hashes")
                .contains_key("entry.note")
        );
    }

    #[cfg(feature = "documents")]
    #[tokio::test]
    async fn dispatch_indexes_plain_text_with_default_preparer() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "watch.txt", "watch plain text");
        let (_temps, store) = open_store(notes_dir.path()).await;
        let notes_dirs = vec![notes_dir.path().to_path_buf()];
        let mut changed = HashSet::new();
        changed.insert(notes_dir.path().join("watch.txt"));

        dispatch_changes(
            std::slice::from_ref(&store),
            &notes_dirs,
            &embed::Embedder::create_null(1024),
            &changed,
        )
        .await;

        assert!(
            db::all_note_hashes(&store.conn)
                .await
                .expect("hashes")
                .contains_key("watch.txt")
        );
    }

    #[cfg(feature = "documents")]
    #[tokio::test]
    async fn default_watcher_indexes_document_fixtures_and_removes_deleted_pdf() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        let fixtures = [
            (
                "fixture.pdf",
                include_bytes!("../tests/fixtures/documents/fixture.pdf").as_slice(),
                "PDFFIXTURENEEDLE",
            ),
            (
                "fixture.epub",
                include_bytes!("../tests/fixtures/documents/fixture.epub").as_slice(),
                "EPUBFIXTURENEEDLE",
            ),
            (
                "fixture.html",
                include_bytes!("../tests/fixtures/documents/fixture.html").as_slice(),
                "HTMLFIXTURENEEDLE",
            ),
            (
                "fixture.docx",
                include_bytes!("../tests/fixtures/documents/fixture.docx").as_slice(),
                "DOCXFIXTURENEEDLE",
            ),
        ];
        for (path, bytes, _) in fixtures {
            std::fs::write(notes_dir.path().join(path), bytes).expect("write fixture");
        }
        let (_temps, store) = open_store(notes_dir.path()).await;
        let notes_dirs = vec![notes_dir.path().to_path_buf()];
        let changed: HashSet<PathBuf> = fixtures
            .iter()
            .map(|(path, _, _)| notes_dir.path().join(path))
            .collect();

        dispatch_changes(
            std::slice::from_ref(&store),
            &notes_dirs,
            &embed::Embedder::create_null(1024),
            &changed,
        )
        .await;

        let semantic_paths: Vec<String> = db::search_semantic(&store.conn, &[0.0; 1024], 10)
            .await
            .expect("semantic search")
            .into_iter()
            .map(|result| result.path)
            .collect();
        for (path, _, marker) in fixtures {
            assert!(semantic_paths.contains(&path.to_owned()), "{path}");
            assert_eq!(
                store
                    .fts
                    .search(marker, 10)
                    .await
                    .expect("full-text search")[0]
                    .path,
                path
            );
        }

        std::fs::remove_file(notes_dir.path().join("fixture.pdf")).expect("remove fixture");
        let changed = HashSet::from([notes_dir.path().join("fixture.pdf")]);
        dispatch_changes(
            std::slice::from_ref(&store),
            &notes_dirs,
            &embed::Embedder::create_null(1024),
            &changed,
        )
        .await;

        assert!(
            !db::search_semantic(&store.conn, &[0.0; 1024], 10)
                .await
                .expect("semantic search")
                .into_iter()
                .any(|result| result.path == "fixture.pdf")
        );
        assert!(
            store
                .fts
                .search("PDFFIXTURENEEDLE", 10)
                .await
                .expect("full-text search")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn dispatch_routes_file_to_correct_store() {
        let dir1 = tempfile::tempdir().expect("tempdir");
        let dir2 = tempfile::tempdir().expect("tempdir");
        create_file(dir1.path(), "alpha.md", "# Alpha");
        create_file(dir2.path(), "beta.md", "# Beta");

        let (_tmps1, store1) = open_store(dir1.path()).await;
        let (_tmps2, store2) = open_store(dir2.path()).await;
        let embedder = embed::Embedder::create_null(1024);

        let notes_dirs = vec![dir1.path().to_path_buf(), dir2.path().to_path_buf()];
        let open_stores = vec![store1, store2];

        // Index both directories first so hashes exist.
        index::index_directory(
            &open_stores[0].conn,
            &open_stores[0].fts,
            &embedder,
            dir1.path(),
        )
        .await
        .expect("index dir1");
        index::index_directory(
            &open_stores[1].conn,
            &open_stores[1].fts,
            &embedder,
            dir2.path(),
        )
        .await
        .expect("index dir2");

        // Add a new file to dir2 only.
        create_file(dir2.path(), "new_in_dir2.md", "# New");
        let mut changed = HashSet::new();
        changed.insert(dir2.path().join("new_in_dir2.md"));

        dispatch_changes(&open_stores, &notes_dirs, &embedder, &changed).await;

        // new_in_dir2.md must appear in store2, not store1.
        let hashes1 = db::all_note_hashes(&open_stores[0].conn)
            .await
            .expect("hashes1");
        let hashes2 = db::all_note_hashes(&open_stores[1].conn)
            .await
            .expect("hashes2");

        assert!(
            !hashes1.contains_key("new_in_dir2.md"),
            "file must not appear in the wrong store"
        );
        assert!(
            hashes2.contains_key("new_in_dir2.md"),
            "file must appear in its owning store"
        );
    }

    #[tokio::test]
    async fn dispatch_routes_delete_to_correct_store() {
        let dir1 = tempfile::tempdir().expect("tempdir");
        let dir2 = tempfile::tempdir().expect("tempdir");
        create_file(dir1.path(), "keep.md", "# Keep");
        create_file(dir2.path(), "remove.md", "# Remove");

        let (_tmps1, store1) = open_store(dir1.path()).await;
        let (_tmps2, store2) = open_store(dir2.path()).await;
        let embedder = embed::Embedder::create_null(1024);

        let notes_dirs = vec![dir1.path().to_path_buf(), dir2.path().to_path_buf()];
        let open_stores = vec![store1, store2];

        index::index_directory(
            &open_stores[0].conn,
            &open_stores[0].fts,
            &embedder,
            dir1.path(),
        )
        .await
        .expect("index dir1");
        index::index_directory(
            &open_stores[1].conn,
            &open_stores[1].fts,
            &embedder,
            dir2.path(),
        )
        .await
        .expect("index dir2");

        // Delete remove.md from dir2.
        std::fs::remove_file(dir2.path().join("remove.md")).expect("remove");
        let mut changed = HashSet::new();
        changed.insert(dir2.path().join("remove.md"));

        dispatch_changes(&open_stores, &notes_dirs, &embedder, &changed).await;

        let hashes1 = db::all_note_hashes(&open_stores[0].conn)
            .await
            .expect("hashes1");
        let hashes2 = db::all_note_hashes(&open_stores[1].conn)
            .await
            .expect("hashes2");

        assert!(
            hashes1.contains_key("keep.md"),
            "unrelated store must be untouched"
        );
        assert!(
            !hashes2.contains_key("remove.md"),
            "deleted file must be removed from its owning store"
        );
    }

    #[tokio::test]
    async fn dispatch_ignores_path_not_under_any_store() {
        let dir1 = tempfile::tempdir().expect("tempdir");
        let other = tempfile::tempdir().expect("tempdir");
        create_file(dir1.path(), "existing.md", "# Existing");

        let (_tmps1, store1) = open_store(dir1.path()).await;
        let embedder = embed::Embedder::create_null(1024);

        let notes_dirs = vec![dir1.path().to_path_buf()];
        let open_stores = vec![store1];

        index::index_directory(
            &open_stores[0].conn,
            &open_stores[0].fts,
            &embedder,
            dir1.path(),
        )
        .await
        .expect("index");

        // A path outside any configured store root.
        create_file(other.path(), "intruder.md", "# Intruder");
        let mut changed = HashSet::new();
        changed.insert(other.path().join("intruder.md"));

        dispatch_changes(&open_stores, &notes_dirs, &embedder, &changed).await;

        let hashes = db::all_note_hashes(&open_stores[0].conn)
            .await
            .expect("hashes");
        assert!(
            !hashes.contains_key("intruder.md"),
            "file outside all configured stores must not be indexed"
        );
    }

    // ---------- process_batch tests (single-store per-file processing) --------

    async fn test_setup(
        notes_dir: &Path,
    ) -> (
        tempfile::TempDir,
        libsql::Database,
        libsql::Connection,
        tempfile::TempDir,
        FtsIndex,
        Embedder,
    ) {
        let db_dir = tempfile::tempdir().expect("tempdir");
        let (db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");
        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = FtsIndex::open_or_create(fts_dir.path()).expect("fts");
        let client = embed::Embedder::create_null(1024);

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

        let mut changed = HashSet::new();
        changed.insert(notes_dir.path().join("note.md"));
        process_batch(&conn, &fts, &client, notes_dir.path(), &changed).await;

        let hash_after = db::note_hash(&conn, "note.md")
            .await
            .expect("hash")
            .expect("should exist");
        assert_eq!(hash_before, hash_after);
    }

    #[tokio::test]
    async fn watcher_unchanged_file_does_not_prepare_or_embed() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "note.md", "same");
        let (_db_dir, _db, conn, _fts_dir, fts, _) = test_setup(notes_dir.path()).await;
        let prepare_calls = Arc::new(AtomicUsize::new(0));
        let preparer = CountingMarkdownPreparer(Arc::clone(&prepare_calls));
        let initial = embed::Embedder::create_null(1024);
        let path = notes_dir.path().join("note.md");

        process_single_file(&conn, &fts, &initial, notes_dir.path(), &path, &preparer).await;
        prepare_calls.store(0, Ordering::SeqCst);
        let (embedder, embed_calls) = embed::Embedder::create_counting(1024, false);
        process_single_file(&conn, &fts, &embedder, notes_dir.path(), &path, &preparer).await;

        assert_eq!(prepare_calls.load(Ordering::SeqCst), 0);
        assert_eq!(embed_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn hidden_dir_file_is_not_indexed_when_second_unrelated_root_is_configured() {
        let configured_dirs: Vec<PathBuf> = vec![PathBuf::from("/dir1"), PathBuf::from("/dir2")];

        let visible = PathBuf::from("/dir1/notes/public.md");
        let hidden_in_dir1 = PathBuf::from("/dir1/.private/secret.md");
        let visible_in_dir2 = PathBuf::from("/dir2/public.md");
        let hidden_in_dir2 = PathBuf::from("/dir2/.hidden/secret.md");
        let unrelated = PathBuf::from("/other/note.md");

        assert!(
            should_index_path(&visible, configured_dirs.iter()),
            "visible file under root1 must be indexed"
        );
        assert!(
            !should_index_path(&hidden_in_dir1, configured_dirs.iter()),
            "hidden file under root1 must not be indexed even when root2 is configured"
        );
        assert!(
            should_index_path(&visible_in_dir2, configured_dirs.iter()),
            "visible file under root2 must be indexed"
        );
        assert!(
            !should_index_path(&hidden_in_dir2, configured_dirs.iter()),
            "hidden file under root2 must not be indexed"
        );
        assert!(
            !should_index_path(&unrelated, configured_dirs.iter()),
            "file not under any configured root must not be indexed"
        );
    }
}
