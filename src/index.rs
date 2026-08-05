use std::{collections::HashMap, path::Path};

use anyhow::Context;
use libsql::Connection;

use crate::{db, document::DocumentPreparer, embed::Embedder, fts::FtsIndex, hash};

#[derive(Debug)]
pub struct IndexStats {
    pub added: usize,
    pub updated: usize,
    pub deleted: usize,
    pub unchanged: usize,
}

impl std::fmt::Display for IndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "added={}, updated={}, deleted={}, unchanged={}",
            self.added, self.updated, self.deleted, self.unchanged
        )
    }
}

pub enum FtsStatus {
    Current,
    Stale,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DiskFile {
    pub content_hash: String,
    pub chunks: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct FileToIndex {
    pub rel_path: String,
    pub content_hash: String,
    pub chunks: Vec<String>,
    pub is_new: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DirectoryIndexPlan {
    pub to_add: Vec<FileToIndex>,
    pub to_update: Vec<FileToIndex>,
    pub to_delete: Vec<String>,
    pub unchanged_count: usize,
}

struct SourceFile {
    abs_path: std::path::PathBuf,
    content_hash: String,
    source: Vec<u8>,
}

pub enum SingleFilePlan {
    Unchanged,
    NeedsIndex {
        rel_path: String,
        content_hash: String,
        chunks: Vec<String>,
    },
}

const FILE_BATCH_SIZE: usize = 50;

pub fn plan_directory_index(
    existing_hashes: &HashMap<String, String>,
    disk_files: &HashMap<String, DiskFile>,
) -> DirectoryIndexPlan {
    let mut to_add = Vec::new();
    let mut to_update = Vec::new();
    let mut unchanged_count = 0usize;

    for (rel_path, disk_file) in disk_files {
        match existing_hashes.get(rel_path) {
            None => to_add.push(FileToIndex {
                rel_path: rel_path.clone(),
                content_hash: disk_file.content_hash.clone(),
                chunks: disk_file.chunks.clone(),
                is_new: true,
            }),
            Some(stored) if stored != &disk_file.content_hash => to_update.push(FileToIndex {
                rel_path: rel_path.clone(),
                content_hash: disk_file.content_hash.clone(),
                chunks: disk_file.chunks.clone(),
                is_new: false,
            }),
            _ => unchanged_count += 1,
        }
    }

    let to_delete = existing_hashes
        .keys()
        .filter(|p| !disk_files.contains_key(p.as_str()))
        .cloned()
        .collect();

    DirectoryIndexPlan {
        to_add,
        to_update,
        to_delete,
        unchanged_count,
    }
}

pub fn plan_single_file(
    rel_path: String,
    stored_hash: Option<&str>,
    content_hash: String,
    chunks: Vec<String>,
) -> SingleFilePlan {
    let file_hash = content_hash;
    if stored_hash.is_some_and(|h| h == file_hash) {
        return SingleFilePlan::Unchanged;
    }
    SingleFilePlan::NeedsIndex {
        rel_path,
        content_hash: file_hash,
        chunks,
    }
}

pub async fn execute_directory_plan(
    conn: &Connection,
    fts: &FtsIndex,
    embedder: &Embedder,
    plan: DirectoryIndexPlan,
) -> anyhow::Result<IndexStats> {
    let added_count = plan.to_add.len();
    let updated_count = plan.to_update.len();
    let to_embed: Vec<&FileToIndex> = plan.to_add.iter().chain(plan.to_update.iter()).collect();
    let upserts = embed_files(embedder, &to_embed).await?;

    db::apply_directory_changes(conn, &plan.to_delete, &upserts).await?;

    for path in &plan.to_delete {
        tracing::info!(path, "deleted from index");
    }
    for file in &to_embed {
        let action = if file.is_new { "indexed" } else { "updated" };
        tracing::info!(path = file.rel_path, action);
    }

    let all_chunks = db::all_chunks(conn).await?;
    fts.rebuild(all_chunks).await?;

    Ok(IndexStats {
        added: added_count,
        updated: updated_count,
        deleted: plan.to_delete.len(),
        unchanged: plan.unchanged_count,
    })
}

pub async fn execute_single_file_plan(
    conn: &Connection,
    fts: &FtsIndex,
    embedder: &Embedder,
    plan: SingleFilePlan,
) -> anyhow::Result<FtsStatus> {
    match plan {
        SingleFilePlan::Unchanged => Ok(FtsStatus::Current),
        SingleFilePlan::NeedsIndex {
            rel_path,
            content_hash,
            chunks,
        } => {
            let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
            let embeddings = embedder.embed_documents(&chunk_refs).await?;
            anyhow::ensure!(
                embeddings.len() == chunks.len(),
                "embedder returned {} embeddings for {} chunks",
                embeddings.len(),
                chunks.len()
            );
            let paired: Vec<(String, Vec<f32>)> = chunks.into_iter().zip(embeddings).collect();
            db::upsert_note(conn, &rel_path, &content_hash, &paired).await?;

            let fts_chunks: Vec<String> = paired.into_iter().map(|(text, _)| text).collect();
            let status = if fts.upsert(&rel_path, &fts_chunks).await.is_ok() {
                FtsStatus::Current
            } else {
                tracing::warn!(path = rel_path, "FTS upsert failed");
                FtsStatus::Stale
            };

            tracing::info!(path = rel_path, "indexed");
            Ok(status)
        }
    }
}

pub async fn index_directory(
    conn: &Connection,
    fts: &FtsIndex,
    embedder: &Embedder,
    notes_dir: &Path,
) -> anyhow::Result<IndexStats> {
    index_directory_with_preparer(
        conn,
        fts,
        embedder,
        notes_dir,
        &crate::document::MarkdownPreparer,
    )
    .await
}

pub async fn index_directory_with_preparer(
    conn: &Connection,
    fts: &FtsIndex,
    embedder: &Embedder,
    notes_dir: &Path,
    preparer: &dyn DocumentPreparer,
) -> anyhow::Result<IndexStats> {
    let existing_hashes = db::all_note_hashes(conn).await?;
    let source_files = read_disk_sources(notes_dir, preparer).await?;
    let mut disk_files = HashMap::with_capacity(source_files.len());
    for (rel_path, source_file) in source_files {
        if existing_hashes.get(&rel_path) == Some(&source_file.content_hash) {
            disk_files.insert(
                rel_path,
                DiskFile {
                    content_hash: source_file.content_hash,
                    chunks: Vec::new(),
                },
            );
            continue;
        }
        disk_files.insert(
            rel_path,
            DiskFile {
                content_hash: source_file.content_hash,
                chunks: preparer.prepare(&source_file.abs_path, &source_file.source)?,
            },
        );
    }
    let plan = plan_directory_index(&existing_hashes, &disk_files);
    execute_directory_plan(conn, fts, embedder, plan).await
}

pub async fn index_single_file_with_preparer(
    conn: &Connection,
    fts: &FtsIndex,
    embedder: &Embedder,
    notes_dir: &Path,
    abs_path: &Path,
    preparer: &dyn DocumentPreparer,
) -> anyhow::Result<FtsStatus> {
    if !preparer.supports_path(abs_path) {
        return Ok(FtsStatus::Current);
    }

    let rel_path = abs_path.strip_prefix(notes_dir).map_or_else(
        |_| abs_path.to_string_lossy().to_string(),
        |p| p.to_string_lossy().to_string(),
    );
    let source = tokio::fs::read(abs_path).await?;
    let stored_hash = db::note_hash(conn, &rel_path).await?;
    let content_hash = hash::content_hash_bytes(&source);
    if stored_hash.as_deref() == Some(content_hash.as_str()) {
        return execute_single_file_plan(conn, fts, embedder, SingleFilePlan::Unchanged).await;
    }
    let plan = plan_single_file(
        rel_path,
        stored_hash.as_deref(),
        content_hash,
        preparer.prepare(abs_path, &source)?,
    );
    execute_single_file_plan(conn, fts, embedder, plan).await
}

async fn read_disk_sources(
    dir: &Path,
    preparer: &dyn DocumentPreparer,
) -> anyhow::Result<HashMap<String, SourceFile>> {
    let files = collect_supported_files(dir, preparer)?;
    let mut out = HashMap::with_capacity(files.len());
    for (rel_path, abs_path) in files {
        let source = tokio::fs::read(&abs_path).await?;
        out.insert(
            rel_path,
            SourceFile {
                abs_path,
                content_hash: hash::content_hash_bytes(&source),
                source,
            },
        );
    }
    Ok(out)
}

async fn embed_files(
    embedder: &Embedder,
    files: &[&FileToIndex],
) -> anyhow::Result<Vec<(String, String, Vec<(String, Vec<f32>)>)>> {
    let total_chunks: usize = files.iter().map(|f| f.chunks.len()).sum();
    tracing::info!(files = files.len(), chunks = total_chunks, "embedding");

    let mut upserts = Vec::with_capacity(files.len());
    for batch in files.chunks(FILE_BATCH_SIZE) {
        let batch_texts: Vec<&str> = batch
            .iter()
            .flat_map(|f| f.chunks.iter().map(String::as_str))
            .collect();

        let batch_embeddings = embedder.embed_documents(&batch_texts).await?;
        anyhow::ensure!(
            batch_embeddings.len() == batch_texts.len(),
            "embedder returned {} embeddings for {} chunks",
            batch_embeddings.len(),
            batch_texts.len()
        );

        let mut offset = 0;
        for file in batch {
            let paired: Vec<(String, Vec<f32>)> = file
                .chunks
                .iter()
                .enumerate()
                .map(|(i, text)| {
                    let emb = batch_embeddings
                        .get(offset + i)
                        .context("embedding index out of bounds")?
                        .clone();
                    Ok((text.clone(), emb))
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            offset += file.chunks.len();

            upserts.push((file.rel_path.clone(), file.content_hash.clone(), paired));
        }
    }

    Ok(upserts)
}

pub fn is_in_hidden_dir(path: &Path, root: &Path) -> bool {
    let Ok(rel) = path.strip_prefix(root) else {
        return false;
    };
    let Some(parent) = rel.parent() else {
        return false;
    };
    parent
        .components()
        .any(|c| c.as_os_str().to_string_lossy().starts_with('.'))
}

fn collect_supported_files(
    dir: &Path,
    preparer: &dyn DocumentPreparer,
) -> anyhow::Result<HashMap<String, std::path::PathBuf>> {
    let mut files = HashMap::new();
    collect_recursive(dir, dir, preparer, &mut files)?;
    Ok(files)
}

#[cfg(test)]
fn collect_markdown_files(dir: &Path) -> anyhow::Result<HashMap<String, std::path::PathBuf>> {
    collect_supported_files(dir, &crate::document::MarkdownPreparer)
}

fn collect_recursive(
    root: &Path,
    dir: &Path,
    preparer: &dyn DocumentPreparer,
    files: &mut HashMap<String, std::path::PathBuf>,
) -> anyhow::Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            if path
                .file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with('.'))
            {
                continue;
            }
            collect_recursive(root, &path, preparer, files)?;
        } else if preparer.supports_path(&path) {
            let rel = path.strip_prefix(root).map_or_else(
                |_| path.to_string_lossy().to_string(),
                |p| p.to_string_lossy().to_string(),
            );
            files.insert(rel, path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use crate::embed;

    fn create_file(dir: &Path, relative: &str, content: &str) {
        let path = dir.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("failed to create parent dirs");
        }
        std::fs::write(&path, content).expect("failed to write file");
    }

    async fn assert_old_note_content(conn: &Connection) {
        assert_eq!(
            db::all_chunks(conn)
                .await
                .expect("chunks")
                .into_iter()
                .find(|(path, _)| path == "note.md")
                .expect("note chunk")
                .1,
            "before"
        );
    }

    fn disk_files_from(entries: &[(&str, &str)]) -> HashMap<String, DiskFile> {
        entries
            .iter()
            .map(|(rel, h)| {
                (
                    rel.to_string(),
                    DiskFile {
                        content_hash: h.to_string(),
                        chunks: vec![rel.to_string()],
                    },
                )
            })
            .collect()
    }

    fn assert_file_to_index(f: &FileToIndex, rel_path: &str, is_new: bool) {
        assert_eq!(f.rel_path, rel_path);
        assert_eq!(f.is_new, is_new);
    }

    struct CountingPreparer {
        calls: Arc<AtomicUsize>,
        fails: bool,
    }

    impl DocumentPreparer for CountingPreparer {
        fn supports_path(&self, source_path: &Path) -> bool {
            source_path
                .extension()
                .is_some_and(|extension| extension == "md")
        }

        fn prepare(&self, _source_path: &Path, source: &[u8]) -> anyhow::Result<Vec<String>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            anyhow::ensure!(!self.fails, "test preparation failure");
            Ok(vec![std::str::from_utf8(source)?.to_owned()])
        }

        fn profile(&self) -> &'static str {
            "counting-v1"
        }
    }

    struct NotePreparer;

    impl DocumentPreparer for NotePreparer {
        fn supports_path(&self, source_path: &Path) -> bool {
            source_path
                .extension()
                .is_some_and(|extension| extension == "note")
        }

        fn prepare(&self, _source_path: &Path, source: &[u8]) -> anyhow::Result<Vec<String>> {
            Ok(vec![std::str::from_utf8(source)?.to_owned()])
        }

        fn profile(&self) -> &'static str {
            "note-v1"
        }
    }

    #[test]
    fn plan_marks_all_disk_files_as_new_when_db_is_empty() {
        let disk = disk_files_from(&[("a.md", "h1"), ("b.md", "h2")]);
        let plan = plan_directory_index(&HashMap::new(), &disk);

        assert_eq!(plan.to_add.len(), 2);
        assert!(plan.to_add.iter().all(|f| f.is_new));
        assert!(plan.to_update.is_empty());
        assert!(plan.to_delete.is_empty());
        assert_eq!(plan.unchanged_count, 0);
    }

    #[test]
    fn plan_marks_unchanged_files_correctly() {
        let disk = disk_files_from(&[("a.md", "hash1")]);
        let existing = HashMap::from([("a.md".to_string(), "hash1".to_string())]);

        let plan = plan_directory_index(&existing, &disk);

        assert!(plan.to_add.is_empty());
        assert!(plan.to_update.is_empty());
        assert!(plan.to_delete.is_empty());
        assert_eq!(plan.unchanged_count, 1);
    }

    #[test]
    fn plan_marks_changed_files_for_update() {
        let disk = disk_files_from(&[("a.md", "newhash")]);
        let existing = HashMap::from([("a.md".to_string(), "oldhash".to_string())]);

        let plan = plan_directory_index(&existing, &disk);

        assert!(plan.to_add.is_empty());
        assert_eq!(plan.to_update.len(), 1);
        assert_file_to_index(&plan.to_update[0], "a.md", false);
        assert!(plan.to_delete.is_empty());
        assert_eq!(plan.unchanged_count, 0);
    }

    #[test]
    fn plan_marks_removed_paths_for_deletion() {
        let disk = disk_files_from(&[("kept.md", "h1")]);
        let existing = HashMap::from([
            ("kept.md".to_string(), "h1".to_string()),
            ("gone.md".to_string(), "h2".to_string()),
        ]);

        let plan = plan_directory_index(&existing, &disk);

        assert_eq!(plan.to_delete, vec!["gone.md".to_string()]);
        assert_eq!(plan.unchanged_count, 1);
    }

    #[test]
    fn plan_is_empty_when_disk_and_db_are_both_empty() {
        let plan = plan_directory_index(&HashMap::new(), &HashMap::new());

        assert!(plan.to_add.is_empty());
        assert!(plan.to_update.is_empty());
        assert!(plan.to_delete.is_empty());
        assert_eq!(plan.unchanged_count, 0);
    }

    #[test]
    fn single_file_plan_is_unchanged_when_hash_matches() {
        let content = "hello world";
        let stored = hash::content_hash(content);
        let plan = plan_single_file(
            "note.md".to_string(),
            Some(&stored),
            hash::content_hash(content),
            vec![content.to_string()],
        );
        assert!(matches!(plan, SingleFilePlan::Unchanged));
    }

    #[test]
    fn single_file_plan_needs_index_when_no_stored_hash() {
        let content = "hello world";
        let plan = plan_single_file(
            "note.md".to_string(),
            None,
            hash::content_hash(content),
            vec![content.to_string()],
        );
        match plan {
            SingleFilePlan::NeedsIndex {
                rel_path,
                content_hash,
                chunks,
            } => {
                assert_eq!(rel_path, "note.md");
                assert_eq!(content_hash, hash::content_hash(content));
                assert!(!chunks.is_empty());
            }
            SingleFilePlan::Unchanged => unreachable!("expected NeedsIndex"),
        }
    }

    #[test]
    fn single_file_plan_needs_index_when_hash_differs() {
        let content = "updated content";
        let plan = plan_single_file(
            "note.md".to_string(),
            Some("oldhash"),
            hash::content_hash(content),
            vec![content.to_string()],
        );
        match plan {
            SingleFilePlan::NeedsIndex { content_hash, .. } => {
                assert_eq!(content_hash, hash::content_hash(content));
            }
            SingleFilePlan::Unchanged => unreachable!("expected NeedsIndex"),
        }
    }

    #[test]
    fn collects_markdown_files_from_root() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        create_file(dir.path(), "note1.md", "");
        create_file(dir.path(), "note2.md", "");

        let files = collect_markdown_files(dir.path()).expect("collect failed");
        assert_eq!(files.len(), 2);
        assert!(files.contains_key("note1.md"));
        assert!(files.contains_key("note2.md"));
    }

    #[test]
    fn collects_markdown_from_subdirectories() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        create_file(dir.path(), "sub/deep/note.md", "");

        let files = collect_markdown_files(dir.path()).expect("collect failed");
        assert_eq!(files.len(), 1);
        assert!(files.contains_key("sub/deep/note.md"));
    }

    #[test]
    fn ignores_non_markdown_files() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        create_file(dir.path(), "note.md", "");
        create_file(dir.path(), "readme.txt", "");
        create_file(dir.path(), "image.png", "");
        create_file(dir.path(), "data.json", "");

        let files = collect_markdown_files(dir.path()).expect("collect failed");
        assert_eq!(files.len(), 1);
        assert!(files.contains_key("note.md"));
    }

    #[test]
    fn skips_dot_directories() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        create_file(dir.path(), "visible/note.md", "");
        create_file(dir.path(), ".hidden/secret.md", "");
        create_file(dir.path(), ".git/objects/ab.md", "");

        let files = collect_markdown_files(dir.path()).expect("collect failed");
        assert_eq!(files.len(), 1);
        assert!(files.contains_key("visible/note.md"));
    }

    #[test]
    fn empty_directory_returns_empty_map() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let files = collect_markdown_files(dir.path()).expect("collect failed");
        assert!(files.is_empty());
    }

    #[test]
    fn relative_paths_do_not_have_leading_slash() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        create_file(dir.path(), "notes/topic.md", "");

        let files = collect_markdown_files(dir.path()).expect("collect failed");
        for key in files.keys() {
            assert!(
                !key.starts_with('/'),
                "relative path should not start with /: {key}"
            );
        }
    }

    #[tokio::test]
    async fn directory_index_uses_preparer_for_discovery_and_chunks() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "entry.note", "prepared content");

        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");
        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");

        let stats = index_directory_with_preparer(
            &conn,
            &fts,
            &Embedder::create_null(1024),
            notes_dir.path(),
            &NotePreparer,
        )
        .await
        .expect("index");

        assert_eq!(stats.added, 1);
        assert_eq!(
            db::all_chunks(&conn).await.expect("chunks"),
            vec![("entry.note".to_string(), "prepared content".to_string(),)]
        );
    }

    #[tokio::test]
    async fn index_directory_indexes_new_files() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "alpha.md", "# Alpha");
        create_file(notes_dir.path(), "sub/beta.md", "# Beta");
        create_file(notes_dir.path(), ".hidden/secret.md", "# Secret");

        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");

        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");

        let client = embed::Embedder::create_null(1024);

        let stats = index_directory(&conn, &fts, &client, notes_dir.path())
            .await
            .expect("index");

        assert_eq!(stats.added, 2);
        assert_eq!(stats.deleted, 0);
        assert_eq!(stats.unchanged, 0);

        let hashes = db::all_note_hashes(&conn).await.expect("hashes");
        assert!(hashes.contains_key("alpha.md"));
        assert!(hashes.contains_key("sub/beta.md"));
        assert!(!hashes.contains_key(".hidden/secret.md"));
    }

    #[tokio::test]
    async fn index_directory_detects_unchanged_files() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "note.md", "# Note");

        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");

        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");

        let client = embed::Embedder::create_null(1024);

        let first = index_directory(&conn, &fts, &client, notes_dir.path())
            .await
            .expect("first index");
        assert_eq!(first.added, 1);

        let second = index_directory(&conn, &fts, &client, notes_dir.path())
            .await
            .expect("second index");
        assert_eq!(second.added, 0);
        assert_eq!(second.unchanged, 1);
    }

    #[tokio::test]
    async fn index_directory_detects_deleted_files() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "keep.md", "# Keep");
        create_file(notes_dir.path(), "remove.md", "# Remove");

        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");

        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");

        let client = embed::Embedder::create_null(1024);

        index_directory(&conn, &fts, &client, notes_dir.path())
            .await
            .expect("first index");

        std::fs::remove_file(notes_dir.path().join("remove.md")).expect("remove");

        let stats = index_directory(&conn, &fts, &client, notes_dir.path())
            .await
            .expect("second index");
        assert_eq!(stats.deleted, 1);
        assert_eq!(stats.unchanged, 1);

        let hashes = db::all_note_hashes(&conn).await.expect("hashes");
        assert!(hashes.contains_key("keep.md"));
        assert!(!hashes.contains_key("remove.md"));
    }

    #[test]
    fn is_in_hidden_dir_checks_directory_ancestors() {
        let root = Path::new("/notes");
        assert!(!is_in_hidden_dir(Path::new("/notes/note.md"), root));
        assert!(!is_in_hidden_dir(Path::new("/notes/sub/note.md"), root));
        assert!(!is_in_hidden_dir(Path::new("/notes/.dotfile.md"), root));
        assert!(is_in_hidden_dir(Path::new("/notes/.hidden/note.md"), root));
        assert!(is_in_hidden_dir(Path::new("/notes/sub/.git/note.md"), root));
    }

    #[tokio::test]
    async fn directory_reindex_prepares_and_embeds_only_changed_files() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "keep.md", "keep");
        create_file(notes_dir.path(), "change.md", "before");
        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");
        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
        let prepare_calls = Arc::new(AtomicUsize::new(0));
        let preparer = CountingPreparer {
            calls: Arc::clone(&prepare_calls),
            fails: false,
        };
        let (embedder, embed_calls) = Embedder::create_counting(1024, false);

        index_directory_with_preparer(&conn, &fts, &embedder, notes_dir.path(), &preparer)
            .await
            .expect("first index");
        index_directory_with_preparer(&conn, &fts, &embedder, notes_dir.path(), &preparer)
            .await
            .expect("unchanged index");
        assert_eq!(prepare_calls.load(Ordering::SeqCst), 2);
        assert_eq!(embed_calls.load(Ordering::SeqCst), 1);

        create_file(notes_dir.path(), "change.md", "after");
        index_directory_with_preparer(&conn, &fts, &embedder, notes_dir.path(), &preparer)
            .await
            .expect("changed index");
        assert_eq!(prepare_calls.load(Ordering::SeqCst), 3);
        assert_eq!(embed_calls.load(Ordering::SeqCst), 2);

        std::fs::remove_file(notes_dir.path().join("change.md")).expect("remove");
        index_directory_with_preparer(&conn, &fts, &embedder, notes_dir.path(), &preparer)
            .await
            .expect("delete index");
        assert_eq!(prepare_calls.load(Ordering::SeqCst), 3);
        assert_eq!(embed_calls.load(Ordering::SeqCst), 2);
        assert!(
            !db::all_note_hashes(&conn)
                .await
                .expect("hashes")
                .contains_key("change.md")
        );
    }

    #[tokio::test]
    async fn failed_preparation_or_embedding_preserves_existing_note() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "note.md", "before");
        create_file(notes_dir.path(), "gone.md", "gone");
        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");
        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
        let initial_calls = Arc::new(AtomicUsize::new(0));
        let initial = CountingPreparer {
            calls: initial_calls,
            fails: false,
        };
        index_directory_with_preparer(
            &conn,
            &fts,
            &Embedder::create_null(1024),
            notes_dir.path(),
            &initial,
        )
        .await
        .expect("initial index");
        let old_hash = db::note_hash(&conn, "note.md").await.expect("hash");

        create_file(notes_dir.path(), "note.md", "after");
        std::fs::remove_file(notes_dir.path().join("gone.md")).expect("remove");
        let failed_preparer = CountingPreparer {
            calls: Arc::new(AtomicUsize::new(0)),
            fails: true,
        };
        assert!(
            index_directory_with_preparer(
                &conn,
                &fts,
                &Embedder::create_null(1024),
                notes_dir.path(),
                &failed_preparer,
            )
            .await
            .is_err()
        );
        assert_eq!(
            db::note_hash(&conn, "note.md").await.expect("hash"),
            old_hash
        );
        assert_old_note_content(&conn).await;
        assert!(
            db::note_hash(&conn, "gone.md")
                .await
                .expect("hash")
                .is_some()
        );

        let working_preparer = CountingPreparer {
            calls: Arc::new(AtomicUsize::new(0)),
            fails: false,
        };
        let (failing_embedder, _) = Embedder::create_counting(1024, true);
        assert!(
            index_directory_with_preparer(
                &conn,
                &fts,
                &failing_embedder,
                notes_dir.path(),
                &working_preparer,
            )
            .await
            .is_err()
        );
        assert_eq!(
            db::note_hash(&conn, "note.md").await.expect("hash"),
            old_hash
        );
        assert_old_note_content(&conn).await;
        assert!(
            db::note_hash(&conn, "gone.md")
                .await
                .expect("hash")
                .is_some()
        );
    }

    #[test]
    fn index_stats_display_format() {
        let stats = IndexStats {
            added: 5,
            updated: 3,
            deleted: 1,
            unchanged: 10,
        };
        assert_eq!(
            stats.to_string(),
            "added=5, updated=3, deleted=1, unchanged=10"
        );
    }
}
