use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use notify::{EventKind, RecursiveMode, Watcher};
use tokio::sync::mpsc;

use std::time::Instant;

#[cfg(test)]
use crate::document::DefaultPreparer;
#[cfg(test)]
use crate::document::PreparedChunk;
use crate::{
    db, document::DocumentPreparer, embed::Embedder, fts::FtsIndex, index, vector::VectorIndex,
};

const DEBOUNCE_MS: u64 = 500;

/// A store whose DB connection and FTS index are already open and ready.
///
/// Constructed by the composition root (`run_watch` in `main.rs`); passed into
/// the watcher so it contains no resource-construction logic itself.
pub struct OpenStore {
    pub notes_dir: PathBuf,
    pub conn: libsql::Connection,
    pub fts: FtsIndex,
    #[cfg(unix)]
    pub vector_path: PathBuf,
    pub vector: Arc<VectorIndex>,
}

pub struct WatchedRoots {
    _watcher: notify::RecommendedWatcher,
    events: mpsc::UnboundedReceiver<PathBuf>,
}

pub fn arm(
    roots: Vec<PathBuf>,
    preparer: Arc<dyn DocumentPreparer>,
) -> anyhow::Result<WatchedRoots> {
    let (tx, events) = mpsc::unbounded_channel::<PathBuf>();
    let roots_for_callback = roots.clone();
    let mut watcher = notify::recommended_watcher(move |event: Result<notify::Event, _>| {
        if let Ok(event) = event {
            match event.kind {
                EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => {
                    for path in event.paths {
                        if should_index_path_with_preparer(
                            &path,
                            roots_for_callback.iter(),
                            preparer.as_ref(),
                        ) {
                            let _ = tx.send(path);
                        }
                    }
                }
                _ => {}
            }
        }
    })?;

    for root in roots {
        watcher.watch(&root, RecursiveMode::Recursive)?;
        tracing::debug!(dir = %root.display(), "watching for changes");
    }

    Ok(WatchedRoots {
        _watcher: watcher,
        events,
    })
}

/// Watch all `stores` for filesystem changes and keep their indices up to date.
///
/// Callers are responsible for opening each store (DB connection, FTS index,
/// initial indexing pass, lock acquisition) before calling this function.
pub async fn run_watcher(
    watched_roots: WatchedRoots,
    mut stores: Vec<OpenStore>,
    embedder: &Embedder,
    preparer: Arc<dyn DocumentPreparer>,
    profile: crate::types::IndexProfile,
) -> anyhow::Result<()> {
    let started_at = Instant::now();

    let WatchedRoots {
        _watcher,
        mut events,
    } = watched_roots;

    // Config resolution guarantees non-overlapping roots, so each path matches
    // at most one entry here.
    let notes_dirs: Vec<PathBuf> = stores.iter().map(|s| s.notes_dir.clone()).collect();
    let mut shutdown = crate::shutdown::Shutdown::install()?;
    let control = crate::control::ControlSocket::bind()?;

    index_pending(&stores, &notes_dirs, embedder, &preparer, &mut events).await;
    tracing::info!(roots = ?notes_dirs, "watching for changes");

    loop {
        let mut changed = HashSet::new();

        tokio::select! {
            Some(path) = events.recv() => {
                changed.insert(path);
                tokio::time::sleep(Duration::from_millis(DEBOUNCE_MS)).await;
                while let Ok(path) = events.try_recv() {
                    changed.insert(path);
                }
            },
            () = shutdown.requested() => {
                tracing::info!("shutting down");
                break;
            },
            result = control.serve_next(
                &mut stores,
                started_at,
                embedder,
                preparer.as_ref(),
                &profile,
            ) => {
                if let Err(error) = result {
                    tracing::warn!(error = %error, "failed to serve control request");
                }
            },
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

async fn index_pending(
    stores: &[OpenStore],
    notes_dirs: &[PathBuf],
    embedder: &Embedder,
    preparer: &Arc<dyn DocumentPreparer>,
    events: &mut mpsc::UnboundedReceiver<PathBuf>,
) {
    let mut changed = HashSet::new();
    while let Ok(path) = events.try_recv() {
        changed.insert(path);
    }
    if !changed.is_empty() {
        dispatch_changes_with_preparer(stores, notes_dirs, embedder, &changed, preparer.as_ref())
            .await;
    }
}

#[cfg(unix)]
pub async fn reindex_in_place(
    stores: &mut [OpenStore],
    embedder: &Embedder,
    preparer: &dyn DocumentPreparer,
    retry_failed: bool,
) -> anyhow::Result<Vec<(String, index::IndexStats)>> {
    let mut results = Vec::with_capacity(stores.len());
    for store in stores {
        if retry_failed {
            db::clear_failed_files(&store.conn).await?;
        }
        let indexed = index::index_directory_with_preparer(
            &store.conn,
            &store.fts,
            embedder,
            &store.notes_dir,
            preparer,
        )
        .await;
        let refreshed = VectorIndex::open_or_rebuild(&store.conn, store.vector_path.clone()).await;
        match (indexed, refreshed) {
            (Ok(stats), Ok(vector)) => {
                store.vector = Arc::new(vector);
                tracing::info!(%stats, dir = %store.notes_dir.display(), "reindex complete");
                results.push((store.notes_dir.to_string_lossy().into_owned(), stats));
            }
            (Err(index_error), Ok(vector)) => {
                store.vector = Arc::new(vector);
                return Err(index_error);
            }
            (Ok(_), Err(vector_error)) => return Err(vector_error),
            (Err(index_error), Err(vector_error)) => {
                return Err(anyhow::anyhow!(
                    "reindex failed: {index_error:#}; vector refresh failed: {vector_error:#}"
                ));
            }
        }
    }
    Ok(results)
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
    let mut mutated_stores = HashSet::new();
    for path in changed {
        let store_idx = notes_dirs
            .iter()
            .position(|dir| path.starts_with(dir.as_path()));

        if let Some(idx) = store_idx {
            let store = &stores[idx];
            if process_single_file(
                &store.conn,
                &store.fts,
                embedder,
                &store.notes_dir,
                path,
                preparer,
                Some(&store.vector),
            )
            .await
            {
                mutated_stores.insert(idx);
            }
        }
    }
    flush_vector_indices(stores, &mut mutated_stores).await;
}

async fn flush_vector_indices(stores: &[OpenStore], mutated_stores: &mut HashSet<usize>) {
    for index in mutated_stores.drain() {
        let store = &stores[index];
        let result = async {
            store
                .vector
                .save(&store.conn, db::chunk_index_state(&store.conn).await?)
                .await
        }
        .await;
        if result.is_err() {
            reconcile_vector(&store.conn, &store.vector).await;
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
    owning_root.is_some_and(|root| {
        !index::is_in_hidden_dir(path, root)
            && (preparer.supports_path(path) || crate::archive::is_archive(path))
    })
}

async fn process_single_file(
    conn: &libsql::Connection,
    fts: &FtsIndex,
    embedder: &Embedder,
    notes_dir: &Path,
    path: &Path,
    preparer: &dyn DocumentPreparer,
    vector: Option<&VectorIndex>,
) -> bool {
    if path.exists() {
        let indexed = if crate::archive::is_archive(path) {
            index::index_archive_with_preparer(
                conn, fts, embedder, notes_dir, path, preparer, vector,
            )
            .await
        } else {
            index::index_single_file_with_preparer(
                conn, fts, embedder, notes_dir, path, preparer, vector,
            )
            .await
        };
        match indexed {
            Ok(index::IndexStatus::Current { vector_mutated }) => vector_mutated,
            Ok(index::IndexStatus::FtsStale { vector_mutated }) => {
                reconcile_fts(conn, fts).await;
                vector_mutated
            }
            Ok(index::IndexStatus::VectorStale) => {
                if let Some(vector) = vector {
                    reconcile_vector(conn, vector).await;
                }
                false
            }
            Err(e) => {
                tracing::error!(path = %path.display(), error = %e, "failed to index");
                false
            }
        }
    } else if crate::archive::is_archive(path) {
        if let Err(e) =
            index::delete_archive_members(conn, fts, embedder, notes_dir, path, vector).await
        {
            tracing::error!(path = %path.display(), error = %e, "failed to delete archive members");
        }
        false
    } else {
        let rel = path.strip_prefix(notes_dir).map_or_else(
            |_| path.to_string_lossy().to_string(),
            |p| p.to_string_lossy().to_string(),
        );
        let old_embeddings =
            match db::chunk_embeddings_for_paths(conn, std::slice::from_ref(&rel)).await {
                Ok(embeddings) => embeddings,
                Err(e) => {
                    tracing::error!(path = rel, error = %e, "failed to read vectors before delete");
                    return false;
                }
            };
        match db::delete_note(conn, &rel).await {
            Ok(()) => {
                if fts.delete(&rel).await.is_err() {
                    reconcile_fts(conn, fts).await;
                }
                if let Some(vector) = vector {
                    let vector_result = old_embeddings
                        .iter()
                        .try_for_each(|(id, _)| vector.remove(*id));
                    if vector_result.is_err() {
                        reconcile_vector(conn, vector).await;
                        return false;
                    }
                    tracing::info!(path = rel, "deleted from index");
                    return !old_embeddings.is_empty();
                }
                tracing::info!(path = rel, "deleted from index");
                false
            }
            Err(e) => {
                tracing::error!(path = rel, error = %e, "failed to delete from db");
                false
            }
        }
    }
}

async fn reconcile_vector(conn: &libsql::Connection, vector: &VectorIndex) {
    tracing::info!("reconciling vector index after partial failures");
    if let Err(e) = VectorIndex::rebuild(conn, vector.path().to_owned()).await {
        tracing::error!(error = %e, "vector reconciliation failed");
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
            None,
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
    #[cfg(feature = "documents")]
    use crate::rank::SemanticSource;
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

    struct FailingContentPreparer;

    impl DocumentPreparer for FailingContentPreparer {
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
            anyhow::ensure!(source != b"broken", "test preparation failure");
            Ok(vec![PreparedChunk::from(
                std::str::from_utf8(source)?.to_owned(),
            )])
        }

        fn profile(&self) -> &'static str {
            "failing-content-v1"
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
        let vector_path = db_dir.path().join("test.usearch");
        let vector = Arc::new(
            VectorIndex::rebuild(&conn, vector_path.clone())
                .await
                .expect("vector"),
        );
        let store = OpenStore {
            notes_dir: notes_dir.to_path_buf(),
            conn,
            fts,
            #[cfg(unix)]
            vector_path,
            vector,
        };
        (vec![db_dir, fts_dir], store)
    }

    #[tokio::test]
    async fn a_change_buffered_before_startup_is_indexed() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        let (_temps, store) = open_store(notes_dir.path()).await;
        let stores = vec![store];
        let dirs = vec![notes_dir.path().to_path_buf()];
        let embedder = embed::Embedder::create_null(1024);
        let preparer: Arc<dyn DocumentPreparer> = Arc::new(DefaultPreparer::default());
        let path = notes_dir.path().join("buffered.md");
        create_file(notes_dir.path(), "buffered.md", "buffered content");
        let (sender, mut events) = mpsc::unbounded_channel();
        sender.send(path).expect("queue path");

        index_pending(&stores, &dirs, &embedder, &preparer, &mut events).await;

        assert!(
            db::all_note_hashes(&stores[0].conn)
                .await
                .expect("hashes")
                .contains_key("buffered.md")
        );
    }

    #[tokio::test]
    async fn a_deletion_buffered_before_startup_is_removed() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "removed.md", "removed content");
        let (_temps, store) = open_store(notes_dir.path()).await;
        let stores = vec![store];
        let dirs = vec![notes_dir.path().to_path_buf()];
        let embedder = embed::Embedder::create_null(1024);
        let preparer: Arc<dyn DocumentPreparer> = Arc::new(DefaultPreparer::default());
        let path = notes_dir.path().join("removed.md");
        let (sender, mut events) = mpsc::unbounded_channel();
        sender.send(path.clone()).expect("queue initial path");
        index_pending(&stores, &dirs, &embedder, &preparer, &mut events).await;
        assert!(
            db::all_note_hashes(&stores[0].conn)
                .await
                .expect("hashes")
                .contains_key("removed.md")
        );

        std::fs::remove_file(&path).expect("remove file");
        sender.send(path).expect("queue deleted path");
        index_pending(&stores, &dirs, &embedder, &preparer, &mut events).await;

        assert!(
            !db::all_note_hashes(&stores[0].conn)
                .await
                .expect("hashes")
                .contains_key("removed.md")
        );
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

        let vector = Arc::new(
            crate::vector::VectorReader::open(&store.conn, store.vector.path())
                .await
                .expect("reader"),
        );
        let semantic = db::DbSemanticSource::new(store.conn.clone(), vector);
        let semantic_paths: Vec<String> = semantic
            .search_semantic(&[0.0; 1024], 10)
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

        let vector = Arc::new(
            crate::vector::VectorReader::open(&store.conn, store.vector.path())
                .await
                .expect("reader"),
        );
        let semantic = db::DbSemanticSource::new(store.conn.clone(), vector);
        assert!(
            !semantic
                .search_semantic(&[0.0; 1024], 10)
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

        process_single_file(
            &conn,
            &fts,
            &initial,
            notes_dir.path(),
            &path,
            &preparer,
            None,
        )
        .await;
        prepare_calls.store(0, Ordering::SeqCst);
        let (embedder, embed_calls) = embed::Embedder::create_counting(1024, false);
        process_single_file(
            &conn,
            &fts,
            &embedder,
            notes_dir.path(),
            &path,
            &preparer,
            None,
        )
        .await;

        assert_eq!(prepare_calls.load(Ordering::SeqCst), 0);
        assert_eq!(embed_calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn watcher_records_preparation_failure_and_clears_it_after_success() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "broken.md", "broken");
        let (_temps, store) = open_store(notes_dir.path()).await;
        let embedder = embed::Embedder::create_null(1024);
        let path = notes_dir.path().join("broken.md");

        process_single_file(
            &store.conn,
            &store.fts,
            &embedder,
            notes_dir.path(),
            &path,
            &FailingContentPreparer,
            None,
        )
        .await;
        assert!(
            db::failed_file_hash(&store.conn, "broken.md")
                .await
                .expect("failure hash")
                .is_some()
        );

        create_file(notes_dir.path(), "broken.md", "fixed");
        process_single_file(
            &store.conn,
            &store.fts,
            &embedder,
            notes_dir.path(),
            &path,
            &FailingContentPreparer,
            None,
        )
        .await;
        assert!(
            db::failed_file_hash(&store.conn, "broken.md")
                .await
                .expect("failure hash")
                .is_none()
        );
        assert!(
            db::note_hash(&store.conn, "broken.md")
                .await
                .expect("note hash")
                .is_some()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn retrying_failed_reindex_indexes_the_fixed_file() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "fixed.md", "fixed");
        let (_temps, store) = open_store(notes_dir.path()).await;
        db::record_failed_file(
            &store.conn,
            "fixed.md",
            &crate::hash::content_hash("fixed"),
            "embedder outage",
        )
        .await
        .expect("record failure");
        let mut stores = vec![store];

        let stats = reindex_in_place(
            &mut stores,
            &embed::Embedder::create_null(1024),
            &DefaultPreparer::default(),
            true,
        )
        .await
        .expect("retry reindex");

        assert_eq!(stats[0].1.added, 1);
        assert!(
            db::failed_files(&stores[0].conn)
                .await
                .expect("failures")
                .is_empty()
        );
        assert!(
            db::note_hash(&stores[0].conn, "fixed.md")
                .await
                .expect("note hash")
                .is_some()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn reindexing_in_place_refreshes_the_vector_for_new_files() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        let (_temps, store) = open_store(notes_dir.path()).await;
        let mut stores = vec![store];
        create_file(notes_dir.path(), "new.md", "new content");

        let stats = reindex_in_place(
            &mut stores,
            &embed::Embedder::create_null(1024),
            &DefaultPreparer::default(),
            false,
        )
        .await
        .expect("reindex");

        assert_eq!(stats[0].1.added, 1);
        assert!(
            db::note_hash(&stores[0].conn, "new.md")
                .await
                .expect("note hash")
                .is_some()
        );
        assert_vector_matches_chunks(&stores[0]).await;
    }

    async fn assert_vector_matches_chunks(store: &OpenStore) {
        let ids: HashSet<i64> = db::all_chunk_embeddings_with_ids(&store.conn)
            .await
            .expect("embeddings")
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(store.vector.len(), ids.len());
        assert!(ids.iter().all(|id| store.vector.contains(*id)));
    }

    #[tokio::test]
    async fn a_debounce_batch_saves_each_touched_store_once() {
        let first_dir = tempfile::tempdir().expect("tempdir");
        let second_dir = tempfile::tempdir().expect("tempdir");
        for index in 0..3 {
            create_file(
                first_dir.path(),
                &format!("note{index}.md"),
                &format!("content {index}"),
            );
        }
        let (_first_temps, first) = open_store(first_dir.path()).await;
        let (_second_temps, second) = open_store(second_dir.path()).await;
        first.vector.reset_save_count();
        second.vector.reset_save_count();
        let stores = vec![first, second];
        let dirs = vec![
            first_dir.path().to_path_buf(),
            second_dir.path().to_path_buf(),
        ];
        let changed = (0..3)
            .map(|index| first_dir.path().join(format!("note{index}.md")))
            .collect();

        dispatch_changes(
            &stores,
            &dirs,
            &embed::Embedder::create_null(1024),
            &changed,
        )
        .await;

        assert_eq!(stores[0].vector.save_count(), 1);
        assert_eq!(stores[1].vector.save_count(), 0);
        assert_vector_matches_chunks(&stores[0]).await;
    }

    #[tokio::test]
    async fn a_batch_keeps_the_vector_index_aligned_with_edits_and_deletes() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "edited.md", "before");
        create_file(notes_dir.path(), "deleted.md", "gone");
        let (_temps, store) = open_store(notes_dir.path()).await;
        let dirs = vec![notes_dir.path().to_path_buf()];
        let initial = HashSet::from([
            notes_dir.path().join("edited.md"),
            notes_dir.path().join("deleted.md"),
        ]);
        let embedder = embed::Embedder::create_null(1024);
        dispatch_changes(std::slice::from_ref(&store), &dirs, &embedder, &initial).await;
        let deleted_ids: HashSet<i64> =
            db::chunk_embeddings_for_paths(&store.conn, &["deleted.md".to_owned()])
                .await
                .expect("deleted embeddings")
                .into_iter()
                .map(|(id, _)| id)
                .collect();
        create_file(notes_dir.path(), "edited.md", "after");
        std::fs::remove_file(notes_dir.path().join("deleted.md")).expect("remove");
        create_file(notes_dir.path(), "added.md", "added");
        let changed = HashSet::from([
            notes_dir.path().join("edited.md"),
            notes_dir.path().join("deleted.md"),
            notes_dir.path().join("added.md"),
        ]);

        dispatch_changes(std::slice::from_ref(&store), &dirs, &embedder, &changed).await;

        assert_vector_matches_chunks(&store).await;
        assert!(deleted_ids.iter().all(|id| !store.vector.contains(*id)));
    }

    #[tokio::test]
    async fn a_batch_leaves_nothing_pending_for_shutdown() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        for index in 0..3 {
            create_file(notes_dir.path(), &format!("note{index}.md"), "content");
        }
        let (_temps, store) = open_store(notes_dir.path()).await;
        let dirs = vec![notes_dir.path().to_path_buf()];
        let changed: HashSet<PathBuf> = (0..3)
            .map(|index| notes_dir.path().join(format!("note{index}.md")))
            .collect();

        dispatch_changes(
            std::slice::from_ref(&store),
            &dirs,
            &embed::Embedder::create_null(1024),
            &changed,
        )
        .await;

        assert_eq!(
            db::vector_index_state(&store.conn).await.expect("state"),
            Some(
                db::chunk_index_state(&store.conn)
                    .await
                    .expect("chunk state")
            ),
            "a completed batch must persist its state, leaving shutdown nothing to flush"
        );
        let persisted = std::fs::read(store.vector.path()).expect("read index");

        crate::vector::VectorIndex::open_or_rebuild(&store.conn, store.vector.path().to_owned())
            .await
            .expect("reopen");

        assert_eq!(
            std::fs::read(store.vector.path()).expect("read index"),
            persisted,
            "reopening after a batch must find the index current rather than rebuild it"
        );
    }

    #[tokio::test]
    async fn flushing_pending_changes_persists_the_vector_index_state() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "note.md", "content");
        let (_temps, store) = open_store(notes_dir.path()).await;
        let changed = process_single_file(
            &store.conn,
            &store.fts,
            &embed::Embedder::create_null(1024),
            notes_dir.path(),
            &notes_dir.path().join("note.md"),
            &DefaultPreparer::default(),
            Some(&store.vector),
        )
        .await;
        assert!(changed);
        assert_ne!(
            db::vector_index_state(&store.conn).await.expect("state"),
            Some(
                db::chunk_index_state(&store.conn)
                    .await
                    .expect("chunk state")
            )
        );
        let mut pending = HashSet::from([0]);

        flush_vector_indices(std::slice::from_ref(&store), &mut pending).await;

        assert!(store.vector.path().exists());
        assert_eq!(
            db::vector_index_state(&store.conn).await.expect("state"),
            Some(
                db::chunk_index_state(&store.conn)
                    .await
                    .expect("chunk state")
            )
        );
    }

    #[tokio::test]
    async fn an_unflushed_batch_rebuilds_from_stored_embeddings() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "note.md", "content");
        let (_temps, store) = open_store(notes_dir.path()).await;
        let changed = process_single_file(
            &store.conn,
            &store.fts,
            &embed::Embedder::create_null(1024),
            notes_dir.path(),
            &notes_dir.path().join("note.md"),
            &DefaultPreparer::default(),
            Some(&store.vector),
        )
        .await;
        assert!(changed);
        let before = db::all_chunk_embeddings_with_ids(&store.conn)
            .await
            .expect("embeddings");

        let rebuilt = VectorIndex::open_or_rebuild(&store.conn, store.vector.path().to_owned())
            .await
            .expect("rebuild");

        assert_eq!(
            db::all_chunk_embeddings_with_ids(&store.conn)
                .await
                .expect("embeddings"),
            before
        );
        assert_eq!(rebuilt.len(), before.len());
        assert_eq!(
            db::vector_index_state(&store.conn).await.expect("state"),
            Some(
                db::chunk_index_state(&store.conn)
                    .await
                    .expect("chunk state")
            )
        );
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

    fn write_archive(path: &Path, entries: &[(&str, &str)]) {
        use std::io::Write as _;

        let file = std::fs::File::create(path).expect("create archive");
        let mut writer = zip::write::ZipWriter::new(file);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for (name, content) in entries {
            writer.start_file(*name, options).expect("start entry");
            writer.write_all(content.as_bytes()).expect("write entry");
        }
        writer.finish().expect("finish archive");
    }

    async fn dispatch_archive_change(
        store: &OpenStore,
        archive_path: &Path,
        calls: &Arc<AtomicUsize>,
    ) {
        let notes_dirs = vec![store.notes_dir.clone()];
        let mut changed = HashSet::new();
        changed.insert(archive_path.to_path_buf());
        dispatch_changes_with_preparer(
            std::slice::from_ref(store),
            &notes_dirs,
            &embed::Embedder::create_null(1024),
            &changed,
            &CountingMarkdownPreparer(Arc::clone(calls)),
        )
        .await;
    }

    #[tokio::test]
    async fn editing_an_archive_reindexes_only_the_members_that_changed() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        let archive_path = notes_dir.path().join("docs.zip");
        write_archive(&archive_path, &[("one.md", "one"), ("two.md", "two")]);
        let (_temps, store) = open_store(notes_dir.path()).await;
        let calls = Arc::new(AtomicUsize::new(0));
        dispatch_archive_change(&store, &archive_path, &calls).await;

        write_archive(
            &archive_path,
            &[("one.md", "one"), ("two.md", "two edited")],
        );
        dispatch_archive_change(&store, &archive_path, &calls).await;

        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "only the edited member is prepared on the second pass"
        );
        assert!(
            db::all_chunks(&store.conn)
                .await
                .expect("chunks")
                .iter()
                .any(|(path, chunk)| path == "docs.zip!two.md" && chunk.content == "two edited")
        );
    }

    #[tokio::test]
    async fn deleting_an_archive_removes_every_member_it_contributed() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        let archive_path = notes_dir.path().join("docs.zip");
        write_archive(&archive_path, &[("one.md", "one"), ("two.md", "two")]);
        create_file(notes_dir.path(), "loose.md", "loose");
        let (_temps, store) = open_store(notes_dir.path()).await;
        let calls = Arc::new(AtomicUsize::new(0));
        dispatch_archive_change(&store, &archive_path, &calls).await;
        dispatch_archive_change(&store, &notes_dir.path().join("loose.md"), &calls).await;

        std::fs::remove_file(&archive_path).expect("remove archive");
        dispatch_archive_change(&store, &archive_path, &calls).await;

        assert_eq!(
            db::all_note_hashes(&store.conn)
                .await
                .expect("hashes")
                .into_keys()
                .collect::<Vec<String>>(),
            vec!["loose.md"],
            "archive members go, unrelated documents stay"
        );
    }
}
