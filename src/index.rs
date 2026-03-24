use std::{collections::HashMap, path::Path};

use anyhow::Context;
use libsql::Connection;

use crate::{
    db,
    embed::{self, VoyageClient},
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

struct PendingFile {
    path: String,
    content_hash: String,
    chunks: Vec<String>,
    is_new: bool,
}

pub async fn index_directory(
    conn: &Connection,
    fts: &FtsIndex,
    client: &VoyageClient,
    notes_dir: &Path,
) -> anyhow::Result<IndexStats> {
    let existing_hashes = db::all_note_hashes(conn).await?;
    let disk_files = collect_markdown_files(notes_dir)?;

    let mut pending = Vec::new();
    let mut unchanged = 0usize;

    for (rel_path, abs_path) in &disk_files {
        let content = tokio::fs::read_to_string(abs_path).await?;
        let file_hash = hash::content_hash(&content);

        if existing_hashes
            .get(rel_path)
            .is_some_and(|h| *h == file_hash)
        {
            unchanged += 1;
            continue;
        }

        let is_new = !existing_hashes.contains_key(rel_path);
        let chunks = embed::chunk_text(&content);

        pending.push(PendingFile {
            path: rel_path.clone(),
            content_hash: file_hash,
            chunks,
            is_new,
        });
    }

    let deleted_paths: Vec<String> = existing_hashes
        .keys()
        .filter(|p| !disk_files.contains_key(p.as_str()))
        .cloned()
        .collect();

    for path in &deleted_paths {
        db::delete_note(conn, path).await?;
        tracing::info!(path, "deleted from index");
    }

    // Embed and write pending files (may fail partway through)
    let embed_result = if pending.is_empty() {
        Ok(())
    } else {
        embed_and_write_pending(conn, client, &pending).await
    };

    // Always rebuild FTS from authoritative DB state
    let all_chunks = db::all_chunks(conn).await?;
    fts.rebuild(all_chunks).await?;

    // Propagate embedding error after FTS is reconciled
    embed_result?;

    let added = pending.iter().filter(|f| f.is_new).count();
    let updated = pending.len() - added;

    Ok(IndexStats {
        added,
        updated,
        deleted: deleted_paths.len(),
        unchanged,
    })
}

pub async fn index_single_file(
    conn: &Connection,
    client: &VoyageClient,
    notes_dir: &Path,
    abs_path: &Path,
) -> anyhow::Result<()> {
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
        return Ok(());
    }

    let chunks = embed::chunk_text(&content);
    let chunk_refs: Vec<&str> = chunks.iter().map(String::as_str).collect();
    let embeddings = client.embed_documents(&chunk_refs).await?;

    let paired: Vec<(String, Vec<f32>)> = chunks.into_iter().zip(embeddings).collect();
    db::upsert_note(conn, &rel_path, &file_hash, &paired).await?;

    tracing::info!(path = rel_path, "indexed");
    Ok(())
}

async fn embed_and_write_pending(
    conn: &Connection,
    client: &VoyageClient,
    pending: &[PendingFile],
) -> anyhow::Result<()> {
    let all_chunk_texts: Vec<&str> = pending
        .iter()
        .flat_map(|f| f.chunks.iter().map(String::as_str))
        .collect();

    tracing::info!(
        files = pending.len(),
        chunks = all_chunk_texts.len(),
        "embedding"
    );

    let all_embeddings = client.embed_documents(&all_chunk_texts).await?;

    let mut embedding_offset = 0;
    for file in pending {
        let file_chunks: Vec<(String, Vec<f32>)> = file
            .chunks
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let emb = all_embeddings
                    .get(embedding_offset + i)
                    .context("embedding index out of bounds")?
                    .clone();
                Ok((text.clone(), emb))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        embedding_offset += file.chunks.len();

        db::upsert_note(conn, &file.path, &file.content_hash, &file_chunks).await?;

        let action = if file.is_new { "indexed" } else { "updated" };
        tracing::info!(path = file.path, action);
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

    fn create_file(dir: &Path, relative: &str) {
        let path = dir.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("failed to create parent dirs");
        }
        std::fs::write(&path, "# Test note").expect("failed to write file");
    }

    #[test]
    fn collects_markdown_files_from_root() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        create_file(dir.path(), "note1.md");
        create_file(dir.path(), "note2.md");

        let files = collect_markdown_files(dir.path()).expect("collect failed");
        assert_eq!(files.len(), 2);
        assert!(files.contains_key("note1.md"));
        assert!(files.contains_key("note2.md"));
    }

    #[test]
    fn collects_markdown_from_subdirectories() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        create_file(dir.path(), "sub/deep/note.md");

        let files = collect_markdown_files(dir.path()).expect("collect failed");
        assert_eq!(files.len(), 1);
        assert!(files.contains_key("sub/deep/note.md"));
    }

    #[test]
    fn ignores_non_markdown_files() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        create_file(dir.path(), "note.md");
        create_file(dir.path(), "readme.txt");
        create_file(dir.path(), "image.png");
        create_file(dir.path(), "data.json");

        let files = collect_markdown_files(dir.path()).expect("collect failed");
        assert_eq!(files.len(), 1);
        assert!(files.contains_key("note.md"));
    }

    #[test]
    fn skips_dot_directories() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        create_file(dir.path(), "visible/note.md");
        create_file(dir.path(), ".hidden/secret.md");
        create_file(dir.path(), ".git/objects/ab.md");

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
        create_file(dir.path(), "notes/topic.md");

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
        create_file(notes_dir.path(), "alpha.md");
        create_file(notes_dir.path(), "sub/beta.md");
        create_file(notes_dir.path(), ".hidden/secret.md");

        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"))
            .await
            .expect("connect");

        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");

        let client = embed::VoyageClient::create_null();

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
        create_file(notes_dir.path(), "note.md");

        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"))
            .await
            .expect("connect");

        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");

        let client = embed::VoyageClient::create_null();

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
        create_file(notes_dir.path(), "keep.md");
        create_file(notes_dir.path(), "remove.md");

        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"))
            .await
            .expect("connect");

        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");

        let client = embed::VoyageClient::create_null();

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
        assert!(!is_in_hidden_dir(Path::new("/notes/.dotfile.md"), root,));
        assert!(is_in_hidden_dir(Path::new("/notes/.hidden/note.md"), root,));
        assert!(is_in_hidden_dir(Path::new("/notes/sub/.git/note.md"), root,));
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
