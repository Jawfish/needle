use std::{collections::HashMap, path::Path};

use anyhow::Context;
use libsql::Connection;

use crate::{
    db,
    embed::{self, Embedder},
    fts::FtsIndex,
    hash,
};

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

pub async fn plan_single_file_index(
    conn: &Connection,
    notes_dir: &Path,
    abs_path: &Path,
) -> anyhow::Result<SingleFilePlan> {
    let rel_path = abs_path.strip_prefix(notes_dir).map_or_else(
        |_| abs_path.to_string_lossy().to_string(),
        |p| p.to_string_lossy().to_string(),
    );

    let content = tokio::fs::read_to_string(abs_path).await?;
    let file_hash = hash::content_hash(&content);

    if db::note_hash(conn, &rel_path)
        .await?
        .is_some_and(|h| h == file_hash)
    {
        return Ok(SingleFilePlan::Unchanged);
    }

    let chunks = embed::chunk_text(&content);
    Ok(SingleFilePlan::NeedsIndex {
        rel_path,
        content_hash: file_hash,
        chunks,
    })
}

pub async fn execute_directory_plan(
    conn: &Connection,
    fts: &FtsIndex,
    embedder: &Embedder,
    plan: DirectoryIndexPlan,
) -> anyhow::Result<IndexStats> {
    let added_count = plan.to_add.len();
    let updated_count = plan.to_update.len();

    for path in &plan.to_delete {
        db::delete_note(conn, path).await?;
        tracing::info!(path, "deleted from index");
    }

    let to_embed: Vec<&FileToIndex> = plan.to_add.iter().chain(plan.to_update.iter()).collect();

    let embed_result = if to_embed.is_empty() {
        Ok(())
    } else {
        embed_and_upsert(conn, embedder, &to_embed).await
    };

    let all_chunks = db::all_chunks(conn).await?;
    fts.rebuild(all_chunks).await?;

    embed_result?;

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
    let existing_hashes = db::all_note_hashes(conn).await?;
    let disk_files = read_disk_hashes(notes_dir).await?;
    let plan = plan_directory_index(&existing_hashes, &disk_files);
    execute_directory_plan(conn, fts, embedder, plan).await
}

pub async fn index_single_file(
    conn: &Connection,
    fts: &FtsIndex,
    embedder: &Embedder,
    notes_dir: &Path,
    abs_path: &Path,
) -> anyhow::Result<FtsStatus> {
    let plan = plan_single_file_index(conn, notes_dir, abs_path).await?;
    execute_single_file_plan(conn, fts, embedder, plan).await
}

async fn read_disk_hashes(dir: &Path) -> anyhow::Result<HashMap<String, DiskFile>> {
    let files = collect_markdown_files(dir)?;
    let mut out = HashMap::with_capacity(files.len());
    for (rel_path, abs_path) in files {
        let content = tokio::fs::read_to_string(&abs_path).await?;
        out.insert(
            rel_path,
            DiskFile {
                content_hash: hash::content_hash(&content),
                chunks: embed::chunk_text(&content),
            },
        );
    }
    Ok(out)
}

async fn embed_and_upsert(
    conn: &Connection,
    embedder: &Embedder,
    files: &[&FileToIndex],
) -> anyhow::Result<()> {
    let total_chunks: usize = files.iter().map(|f| f.chunks.len()).sum();
    tracing::info!(files = files.len(), chunks = total_chunks, "embedding");

    for batch in files.chunks(FILE_BATCH_SIZE) {
        let batch_texts: Vec<&str> = batch
            .iter()
            .flat_map(|f| f.chunks.iter().map(String::as_str))
            .collect();

        let batch_embeddings = embedder.embed_documents(&batch_texts).await?;

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

            db::upsert_note(conn, &file.rel_path, &file.content_hash, &paired).await?;

            let action = if file.is_new { "indexed" } else { "updated" };
            tracing::info!(path = file.rel_path, action);
        }
    }

    Ok(())
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

fn collect_markdown_files(dir: &Path) -> anyhow::Result<HashMap<String, std::path::PathBuf>> {
    let mut files = HashMap::new();
    collect_recursive(dir, dir, &mut files)?;
    Ok(files)
}

fn collect_recursive(
    root: &Path,
    dir: &Path,
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
            collect_recursive(root, &path, files)?;
        } else if path.extension().is_some_and(|e| e == "md") {
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
    use super::*;

    fn create_file(dir: &Path, relative: &str, content: &str) {
        let path = dir.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("failed to create parent dirs");
        }
        std::fs::write(&path, content).expect("failed to write file");
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
