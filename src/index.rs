use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use anyhow::Context;
use libsql::Connection;

use crate::{
    archive, db,
    document::{DocumentPreparer, PreparedChunk},
    embed::Embedder,
    fts::FtsIndex,
    hash,
    vector::VectorIndex,
};

#[derive(Debug)]
pub struct IndexStats {
    pub added: usize,
    pub updated: usize,
    pub deleted: usize,
    pub unchanged: usize,
    pub failed: usize,
}

impl std::fmt::Display for IndexStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "added={}, updated={}, deleted={}, unchanged={}, failed={}",
            self.added, self.updated, self.deleted, self.unchanged, self.failed
        )
    }
}

pub enum IndexStatus {
    Current { vector_mutated: bool },
    FtsStale { vector_mutated: bool },
    VectorStale,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DiskFile {
    pub content_hash: String,
    pub chunks: Vec<PreparedChunk>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct FileToIndex {
    pub rel_path: String,
    pub content_hash: String,
    pub chunks: Vec<PreparedChunk>,
    pub is_new: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DirectoryIndexPlan {
    pub to_add: Vec<FileToIndex>,
    pub to_update: Vec<FileToIndex>,
    pub to_delete: Vec<String>,
    pub unchanged_count: usize,
    pub failed_count: usize,
}

enum SourceOrigin {
    File(std::path::PathBuf),
    ArchiveMember {
        archive: std::path::PathBuf,
        index: usize,
        name: String,
    },
}

struct SourceRef {
    origin: SourceOrigin,
    prepare_path: std::path::PathBuf,
    content_hash: String,
}

impl SourceRef {
    fn load(&self) -> anyhow::Result<Vec<u8>> {
        match &self.origin {
            SourceOrigin::File(abs_path) => std::fs::read(abs_path)
                .with_context(|| format!("failed to read {}", abs_path.display())),
            SourceOrigin::ArchiveMember {
                archive,
                index,
                name,
            } => archive::read_member(archive, *index, name),
        }
    }
}

struct RejectedSource {
    rel_path: String,
    content_hash: String,
    reason: String,
}

#[derive(Default)]
struct DiskScan {
    sources: HashMap<String, SourceRef>,
    unreadable: HashSet<String>,
    unreadable_archives: Vec<String>,
    rejected: Vec<RejectedSource>,
}

#[derive(Default)]
struct DiscoveredPaths {
    files: HashMap<String, std::path::PathBuf>,
    archives: Vec<(String, std::path::PathBuf)>,
}

pub enum SingleFilePlan {
    Unchanged,
    NeedsIndex {
        rel_path: String,
        content_hash: String,
        chunks: Vec<PreparedChunk>,
    },
}

const FILE_BATCH_SIZE: usize = 50;

pub fn plan_directory_index(
    existing_hashes: &HashMap<String, String>,
    failed_hashes: &HashMap<String, String>,
    disk_files: &HashMap<String, DiskFile>,
    failed_paths: &HashSet<String>,
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
        .chain(failed_hashes.keys())
        .filter(|p| !disk_files.contains_key(p.as_str()) && !failed_paths.contains(p.as_str()))
        .cloned()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();

    DirectoryIndexPlan {
        to_add,
        to_update,
        to_delete,
        unchanged_count,
        failed_count: failed_paths.len(),
    }
}

pub fn plan_single_file(
    rel_path: String,
    stored_hash: Option<&str>,
    content_hash: String,
    chunks: Vec<PreparedChunk>,
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
    execute_directory_plan_with_vector(conn, fts, embedder, plan, None).await
}

pub async fn execute_directory_plan_with_vector(
    conn: &Connection,
    fts: &FtsIndex,
    embedder: &Embedder,
    plan: DirectoryIndexPlan,
    vector: Option<&VectorIndex>,
) -> anyhow::Result<IndexStats> {
    let added_count = plan.to_add.len();
    let updated_count = plan.to_update.len();
    let to_embed: Vec<&FileToIndex> = plan.to_add.iter().chain(plan.to_update.iter()).collect();
    let upserts = embed_files(embedder, &to_embed).await?;

    let changed_paths: Vec<String> = plan
        .to_delete
        .iter()
        .chain(upserts.iter().map(|(path, _, _)| path))
        .cloned()
        .collect();
    let old_embeddings = db::chunk_embeddings_for_paths(conn, &changed_paths).await?;
    db::apply_directory_changes(conn, &plan.to_delete, &upserts).await?;

    if let Some(vector) = vector
        && !changed_paths.is_empty()
        && async {
            update_vector(conn, vector, &old_embeddings, &changed_paths).await?;
            vector.save(conn, db::chunk_index_state(conn).await?).await
        }
        .await
        .is_err()
    {
        tracing::warn!("incremental vector update failed");
    }

    if !plan.to_delete.is_empty() || !upserts.is_empty() {
        let incremental_result = async {
            for path in &plan.to_delete {
                fts.delete(path)
                    .await
                    .with_context(|| format!("failed to delete {path} from FTS"))?;
            }
            for (path, _, chunks) in &upserts {
                let prepared: Vec<PreparedChunk> =
                    chunks.iter().map(|(chunk, _)| chunk.clone()).collect();
                fts.upsert(path, &prepared)
                    .await
                    .with_context(|| format!("failed to upsert {path} into FTS"))?;
            }
            anyhow::Ok(())
        }
        .await;

        if let Err(incremental_error) = incremental_result {
            tracing::warn!(error = %incremental_error, "incremental FTS update failed; rebuilding");
            let recovery_result = async {
                let chunks = db::all_chunks(conn)
                    .await
                    .context("failed to read committed DB chunks for FTS recovery")?;
                fts.rebuild(chunks)
                    .await
                    .context("failed to rebuild FTS from committed DB chunks")
            }
            .await;
            if let Err(recovery_error) = recovery_result {
                return Err(anyhow::anyhow!(
                    "incremental FTS update failed: {incremental_error:#}; FTS recovery failed: {recovery_error:#}"
                ));
            }
        }
    }

    for path in &plan.to_delete {
        tracing::info!(path, "deleted from index");
    }
    for file in &to_embed {
        let action = if file.is_new { "indexed" } else { "updated" };
        tracing::info!(path = file.rel_path, action);
    }

    Ok(IndexStats {
        added: added_count,
        updated: updated_count,
        deleted: plan.to_delete.len(),
        unchanged: plan.unchanged_count,
        failed: plan.failed_count,
    })
}

pub async fn execute_single_file_plan(
    conn: &Connection,
    fts: &FtsIndex,
    embedder: &Embedder,
    plan: SingleFilePlan,
) -> anyhow::Result<IndexStatus> {
    execute_single_file_plan_with_vector(conn, fts, embedder, plan, None).await
}

pub async fn execute_single_file_plan_with_vector(
    conn: &Connection,
    fts: &FtsIndex,
    embedder: &Embedder,
    plan: SingleFilePlan,
    vector: Option<&VectorIndex>,
) -> anyhow::Result<IndexStatus> {
    match plan {
        SingleFilePlan::Unchanged => Ok(IndexStatus::Current {
            vector_mutated: false,
        }),
        SingleFilePlan::NeedsIndex {
            rel_path,
            content_hash,
            chunks,
        } => {
            let chunk_refs: Vec<&str> = chunks.iter().map(|chunk| chunk.content.as_str()).collect();
            let embeddings = embedder.embed_documents(&chunk_refs).await?;
            anyhow::ensure!(
                embeddings.len() == chunks.len(),
                "embedder returned {} embeddings for {} chunks",
                embeddings.len(),
                chunks.len()
            );
            let paired: Vec<(PreparedChunk, Vec<f32>)> =
                chunks.into_iter().zip(embeddings).collect();
            let old_embeddings =
                db::chunk_embeddings_for_paths(conn, std::slice::from_ref(&rel_path)).await?;
            db::upsert_note(conn, &rel_path, &content_hash, &paired).await?;
            let vector_update = match vector {
                Some(vector) => {
                    update_vector(
                        conn,
                        vector,
                        &old_embeddings,
                        std::slice::from_ref(&rel_path),
                    )
                    .await
                }
                None => Ok(false),
            };
            let vector_mutated = vector_update.as_ref().is_ok_and(|mutated| *mutated);
            let vector_is_stale = vector_update.is_err();

            let fts_chunks: Vec<PreparedChunk> =
                paired.iter().map(|(chunk, _)| chunk.clone()).collect();
            let status = if fts.upsert(&rel_path, &fts_chunks).await.is_ok() {
                IndexStatus::Current { vector_mutated }
            } else {
                tracing::warn!(path = rel_path, "FTS upsert failed");
                IndexStatus::FtsStale { vector_mutated }
            };

            tracing::info!(path = rel_path, "indexed");
            if vector_is_stale {
                tracing::warn!(path = rel_path, "incremental vector update failed");
                Ok(IndexStatus::VectorStale)
            } else {
                Ok(status)
            }
        }
    }
}

#[cfg(test)]
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
        &crate::document::DefaultPreparer::default(),
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
    let failed_hashes = db::all_failed_file_hashes(conn).await?;
    let scan = scan_sources(notes_dir, preparer)?;
    let plan = plan_from_scan(conn, preparer, &existing_hashes, &failed_hashes, scan).await?;
    execute_directory_plan(conn, fts, embedder, plan).await
}

pub async fn index_archive_with_preparer(
    conn: &Connection,
    fts: &FtsIndex,
    embedder: &Embedder,
    notes_dir: &Path,
    archive_path: &Path,
    preparer: &dyn DocumentPreparer,
    vector: Option<&VectorIndex>,
) -> anyhow::Result<IndexStatus> {
    let archive_rel = relative_path(notes_dir, archive_path);
    let prefix = archive::archive_prefix(&archive_rel);
    let existing_hashes = hashes_under_prefix(db::all_note_hashes(conn).await?, &prefix);
    let failed_hashes = hashes_under_prefix(db::all_failed_file_hashes(conn).await?, &prefix);

    let mut scan = DiskScan::default();
    match archive::scan(archive_path, &|path| preparer.supports_path(path)) {
        Ok(members) => merge_archive_scan(&mut scan, &archive_rel, archive_path, members),
        Err(error) => {
            tracing::warn!(path = archive_rel, error = %error, "failed to read archive");
            return Ok(IndexStatus::Current {
                vector_mutated: false,
            });
        }
    }

    let plan = plan_from_scan(conn, preparer, &existing_hashes, &failed_hashes, scan).await?;
    if plan.to_add.is_empty() && plan.to_update.is_empty() && plan.to_delete.is_empty() {
        return Ok(IndexStatus::Current {
            vector_mutated: false,
        });
    }
    execute_directory_plan_with_vector(conn, fts, embedder, plan, vector).await?;
    Ok(IndexStatus::Current {
        vector_mutated: false,
    })
}

pub async fn delete_archive_members(
    conn: &Connection,
    fts: &FtsIndex,
    embedder: &Embedder,
    notes_dir: &Path,
    archive_path: &Path,
    vector: Option<&VectorIndex>,
) -> anyhow::Result<()> {
    let prefix = archive::archive_prefix(&relative_path(notes_dir, archive_path));
    let to_delete: Vec<String> = hashes_under_prefix(db::all_note_hashes(conn).await?, &prefix)
        .into_keys()
        .collect();
    if to_delete.is_empty() {
        return Ok(());
    }
    let plan = DirectoryIndexPlan {
        to_add: Vec::new(),
        to_update: Vec::new(),
        to_delete,
        unchanged_count: 0,
        failed_count: 0,
    };
    execute_directory_plan_with_vector(conn, fts, embedder, plan, vector).await?;
    Ok(())
}

fn hashes_under_prefix(hashes: HashMap<String, String>, prefix: &str) -> HashMap<String, String> {
    hashes
        .into_iter()
        .filter(|(path, _)| path.starts_with(prefix))
        .collect()
}

async fn plan_from_scan(
    conn: &Connection,
    preparer: &dyn DocumentPreparer,
    existing_hashes: &HashMap<String, String>,
    failed_hashes: &HashMap<String, String>,
    scan: DiskScan,
) -> anyhow::Result<DirectoryIndexPlan> {
    let mut failed_paths = scan.unreadable;
    for archive_rel in &scan.unreadable_archives {
        let prefix = archive::archive_prefix(archive_rel);
        failed_paths.extend(
            existing_hashes
                .keys()
                .filter(|path| path.starts_with(&prefix))
                .cloned(),
        );
    }
    for rejected in scan.rejected {
        if failed_hashes.get(&rejected.rel_path) != Some(&rejected.content_hash) {
            db::record_failed_file(
                conn,
                &rejected.rel_path,
                &rejected.content_hash,
                &rejected.reason,
            )
            .await?;
            tracing::warn!(
                path = rejected.rel_path,
                reason = rejected.reason,
                "skipping archive member"
            );
        }
        failed_paths.insert(rejected.rel_path);
    }

    let mut disk_files = HashMap::with_capacity(scan.sources.len());
    for (rel_path, source) in scan.sources {
        if existing_hashes.get(&rel_path) == Some(&source.content_hash) {
            db::clear_failed_file(conn, &rel_path).await?;
            disk_files.insert(
                rel_path,
                DiskFile {
                    content_hash: source.content_hash,
                    chunks: Vec::new(),
                },
            );
            continue;
        }
        if failed_hashes.get(&rel_path) == Some(&source.content_hash) {
            tracing::warn!(
                path = rel_path,
                "skipping file with unchanged preparation failure"
            );
            failed_paths.insert(rel_path);
            continue;
        }
        let source_bytes = match source.load() {
            Ok(source_bytes) => source_bytes,
            Err(error) => {
                tracing::warn!(path = rel_path, error = %error, "failed to read file");
                failed_paths.insert(rel_path);
                continue;
            }
        };
        match preparer.prepare(&source.prepare_path, &source_bytes) {
            Ok(chunks) => {
                disk_files.insert(
                    rel_path,
                    DiskFile {
                        content_hash: source.content_hash,
                        chunks,
                    },
                );
            }
            Err(error) => {
                db::record_failed_file(
                    conn,
                    &rel_path,
                    &source.content_hash,
                    &format!("{error:#}"),
                )
                .await?;
                tracing::warn!(path = rel_path, error = %error, "failed to prepare file");
                failed_paths.insert(rel_path);
            }
        }
    }
    Ok(plan_directory_index(
        existing_hashes,
        failed_hashes,
        &disk_files,
        &failed_paths,
    ))
}

pub async fn index_single_file_with_preparer(
    conn: &Connection,
    fts: &FtsIndex,
    embedder: &Embedder,
    notes_dir: &Path,
    abs_path: &Path,
    preparer: &dyn DocumentPreparer,
    vector: Option<&VectorIndex>,
) -> anyhow::Result<IndexStatus> {
    if !preparer.supports_path(abs_path) {
        return Ok(IndexStatus::Current {
            vector_mutated: false,
        });
    }

    let rel_path = relative_path(notes_dir, abs_path);
    let source = match tokio::fs::read(abs_path).await {
        Ok(source) => source,
        Err(error) => {
            tracing::warn!(path = %abs_path.display(), error = %error, "failed to read file");
            return Ok(IndexStatus::Current {
                vector_mutated: false,
            });
        }
    };
    let content_hash = hash::content_hash_bytes(&source);
    let stored_hash = db::note_hash(conn, &rel_path).await?;
    if stored_hash.as_deref() == Some(content_hash.as_str()) {
        db::clear_failed_file(conn, &rel_path).await?;
        return execute_single_file_plan(conn, fts, embedder, SingleFilePlan::Unchanged).await;
    }
    if db::failed_file_hash(conn, &rel_path).await?.as_deref() == Some(content_hash.as_str()) {
        tracing::warn!(
            path = rel_path,
            "skipping file with unchanged preparation failure"
        );
        return Ok(IndexStatus::Current {
            vector_mutated: false,
        });
    }
    let chunks = match preparer.prepare(abs_path, &source) {
        Ok(chunks) => chunks,
        Err(error) => {
            db::record_failed_file(conn, &rel_path, &content_hash, &format!("{error:#}")).await?;
            tracing::warn!(path = rel_path, error = %error, "failed to prepare file");
            return Ok(IndexStatus::Current {
                vector_mutated: false,
            });
        }
    };
    let plan = plan_single_file(rel_path, stored_hash.as_deref(), content_hash, chunks);
    execute_single_file_plan_with_vector(conn, fts, embedder, plan, vector).await
}

fn scan_sources(dir: &Path, preparer: &dyn DocumentPreparer) -> anyhow::Result<DiskScan> {
    let discovered = collect_paths(dir, preparer)?;
    let mut scan = DiskScan::default();
    for (rel_path, abs_path) in discovered.files {
        match std::fs::read(&abs_path) {
            Ok(source) => {
                scan.sources.insert(
                    rel_path,
                    SourceRef {
                        content_hash: hash::content_hash_bytes(&source),
                        prepare_path: abs_path.clone(),
                        origin: SourceOrigin::File(abs_path),
                    },
                );
            }
            Err(error) => {
                tracing::warn!(path = rel_path, error = %error, "failed to read file");
                scan.unreadable.insert(rel_path);
            }
        }
    }
    for (rel_path, abs_path) in discovered.archives {
        match archive::scan(&abs_path, &|path| preparer.supports_path(path)) {
            Ok(members) => merge_archive_scan(&mut scan, &rel_path, &abs_path, members),
            Err(error) => {
                tracing::warn!(path = rel_path, error = %error, "failed to read archive");
                scan.unreadable_archives.push(rel_path);
            }
        }
    }
    Ok(scan)
}

fn merge_archive_scan(
    scan: &mut DiskScan,
    archive_rel: &str,
    archive_abs: &Path,
    members: archive::Scan,
) {
    for member in members.members {
        let name = member.name;
        scan.sources.insert(
            archive::member_path(archive_rel, &name),
            SourceRef {
                prepare_path: std::path::PathBuf::from(&name),
                origin: SourceOrigin::ArchiveMember {
                    archive: archive_abs.to_path_buf(),
                    index: member.index,
                    name,
                },
                content_hash: member.content_hash,
            },
        );
    }
    for rejected in members.rejected {
        scan.rejected.push(RejectedSource {
            rel_path: archive::member_path(archive_rel, &rejected.name),
            content_hash: rejected.content_hash,
            reason: rejected.reason,
        });
    }
}

async fn update_vector(
    conn: &Connection,
    vector: &VectorIndex,
    old_embeddings: &[(i64, Vec<f32>)],
    changed_paths: &[String],
) -> anyhow::Result<bool> {
    let new_embeddings = db::chunk_embeddings_for_paths(conn, changed_paths).await?;
    for (id, _) in old_embeddings {
        vector.remove(*id)?;
    }
    for (id, embedding) in &new_embeddings {
        vector.add(*id, embedding)?;
    }
    Ok(!old_embeddings.is_empty() || !new_embeddings.is_empty())
}

async fn embed_files(
    embedder: &Embedder,
    files: &[&FileToIndex],
) -> anyhow::Result<Vec<db::NoteUpsert>> {
    let total_chunks: usize = files.iter().map(|f| f.chunks.len()).sum();
    tracing::info!(files = files.len(), chunks = total_chunks, "embedding");

    let mut upserts = Vec::with_capacity(files.len());
    for batch in files.chunks(FILE_BATCH_SIZE) {
        let batch_texts: Vec<&str> = batch
            .iter()
            .flat_map(|f| f.chunks.iter().map(|chunk| chunk.content.as_str()))
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
            let paired: Vec<(PreparedChunk, Vec<f32>)> = file
                .chunks
                .iter()
                .enumerate()
                .map(|(i, chunk)| {
                    let emb = batch_embeddings
                        .get(offset + i)
                        .context("embedding index out of bounds")?
                        .clone();
                    Ok((chunk.clone(), emb))
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

fn relative_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root).map_or_else(
        |_| path.to_string_lossy().to_string(),
        |rel| rel.to_string_lossy().to_string(),
    )
}

fn collect_paths(dir: &Path, preparer: &dyn DocumentPreparer) -> anyhow::Result<DiscoveredPaths> {
    let mut discovered = DiscoveredPaths::default();
    collect_recursive(dir, dir, preparer, &mut discovered)?;
    Ok(discovered)
}

#[cfg(test)]
fn collect_markdown_files(dir: &Path) -> anyhow::Result<HashMap<String, std::path::PathBuf>> {
    Ok(collect_paths(dir, &crate::document::DefaultPreparer::default())?.files)
}

fn collect_recursive(
    root: &Path,
    dir: &Path,
    preparer: &dyn DocumentPreparer,
    discovered: &mut DiscoveredPaths,
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
            collect_recursive(root, &path, preparer, discovered)?;
        } else if preparer.supports_path(&path) {
            discovered.files.insert(relative_path(root, &path), path);
        } else if archive::is_archive(&path) {
            discovered.archives.push((relative_path(root, &path), path));
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
    #[cfg(feature = "documents")]
    use crate::rank::SemanticSource;

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
            PreparedChunk::from("before".to_string())
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
                        chunks: vec![PreparedChunk::from(rel.to_string())],
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

        fn prepare(
            &self,
            _source_path: &Path,
            source: &[u8],
        ) -> anyhow::Result<Vec<PreparedChunk>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            anyhow::ensure!(!self.fails, "test preparation failure");
            Ok(vec![PreparedChunk::from(
                std::str::from_utf8(source)?.to_owned(),
            )])
        }

        fn profile(&self) -> &'static str {
            "counting-v1"
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
            anyhow::ensure!(source != b"corrupt", "test preparation failure");
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

    #[test]
    fn plan_marks_all_disk_files_as_new_when_db_is_empty() {
        let disk = disk_files_from(&[("a.md", "h1"), ("b.md", "h2")]);
        let plan = plan_directory_index(&HashMap::new(), &HashMap::new(), &disk, &HashSet::new());

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

        let plan = plan_directory_index(&existing, &HashMap::new(), &disk, &HashSet::new());

        assert!(plan.to_add.is_empty());
        assert!(plan.to_update.is_empty());
        assert!(plan.to_delete.is_empty());
        assert_eq!(plan.unchanged_count, 1);
    }

    #[test]
    fn plan_marks_changed_files_for_update() {
        let disk = disk_files_from(&[("a.md", "newhash")]);
        let existing = HashMap::from([("a.md".to_string(), "oldhash".to_string())]);

        let plan = plan_directory_index(&existing, &HashMap::new(), &disk, &HashSet::new());

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

        let plan = plan_directory_index(&existing, &HashMap::new(), &disk, &HashSet::new());

        assert_eq!(plan.to_delete, vec!["gone.md".to_string()]);
        assert_eq!(plan.unchanged_count, 1);
    }

    #[test]
    fn plan_is_empty_when_disk_and_db_are_both_empty() {
        let plan = plan_directory_index(
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &HashSet::new(),
        );

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
            vec![PreparedChunk::from(content.to_string())],
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
            vec![PreparedChunk::from(content.to_string())],
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
            vec![PreparedChunk::from(content.to_string())],
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
    fn collects_markdown_and_text_but_ignores_unsupported_files() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        create_file(dir.path(), "note.md", "");
        create_file(dir.path(), "readme.txt", "");
        create_file(dir.path(), "image.png", "");
        create_file(dir.path(), "data.json", "");

        let files = collect_markdown_files(dir.path()).expect("collect failed");
        assert!(files.contains_key("note.md"));
        #[cfg(feature = "documents")]
        {
            assert_eq!(files.len(), 2);
            assert!(files.contains_key("readme.txt"));
        }
        #[cfg(not(feature = "documents"))]
        assert_eq!(files.len(), 1);
    }

    #[cfg(feature = "documents")]
    const DOCUMENT_FIXTURES: [(&str, &[u8], &str); 4] = [
        (
            "fixture.pdf",
            include_bytes!("../tests/fixtures/documents/fixture.pdf"),
            "PDFFIXTURENEEDLE",
        ),
        (
            "fixture.epub",
            include_bytes!("../tests/fixtures/documents/fixture.epub"),
            "EPUBFIXTURENEEDLE",
        ),
        (
            "fixture.html",
            include_bytes!("../tests/fixtures/documents/fixture.html"),
            "HTMLFIXTURENEEDLE",
        ),
        (
            "fixture.docx",
            include_bytes!("../tests/fixtures/documents/fixture.docx"),
            "DOCXFIXTURENEEDLE",
        ),
    ];

    #[cfg(feature = "documents")]
    #[tokio::test]
    async fn default_preparer_indexes_plain_text_and_empty_documents() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "guide.txt", "needle text search phrase");
        create_file(notes_dir.path(), "empty.txt", "");

        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");
        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");

        let stats = index_directory(&conn, &fts, &Embedder::create_null(1024), notes_dir.path())
            .await
            .expect("index");

        assert_eq!(stats.added, 2);
        assert!(
            db::all_chunks(&conn)
                .await
                .expect("chunks")
                .iter()
                .any(|(path, chunk)| path == "guide.txt" && chunk.content.contains("search phrase"))
        );
        assert!(
            db::all_note_hashes(&conn)
                .await
                .expect("hashes")
                .contains_key("empty.txt")
        );
        assert!(
            !db::all_chunks(&conn)
                .await
                .expect("chunks")
                .iter()
                .any(|(path, _)| path == "empty.txt")
        );

        let vector_path = db_dir.path().join("test.usearch");
        VectorIndex::open_or_rebuild(&conn, vector_path.clone())
            .await
            .expect("vector");
        let vector = Arc::new(
            crate::vector::VectorReader::open(&conn, &vector_path)
                .await
                .expect("reader"),
        );
        let semantic = db::DbSemanticSource::new(conn.clone(), vector);
        let fts_source = crate::fts::FtsFtsSource::new(fts);
        let paths = db::DbPathSource::new(conn);
        let results = crate::rank::search(
            &semantic,
            &fts_source,
            &paths,
            None,
            "empty",
            10,
            &crate::rank::RrfWeights {
                semantic: 0.0,
                fts: 0.0,
                filename: 1.0,
            },
        )
        .await
        .expect("search");
        assert_eq!(
            results.first().map(|result| result.path.as_str()),
            Some("empty.txt")
        );
    }

    #[cfg(feature = "documents")]
    #[tokio::test]
    async fn default_reindex_indexes_document_fixtures_for_semantic_and_full_text_search() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        for (path, bytes, _) in DOCUMENT_FIXTURES {
            std::fs::write(notes_dir.path().join(path), bytes).expect("write fixture");
        }

        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");
        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");

        let stats = index_directory(&conn, &fts, &Embedder::create_null(1024), notes_dir.path())
            .await
            .expect("index");
        assert_eq!(stats.added, DOCUMENT_FIXTURES.len());

        let vector_path = db_dir.path().join("test.usearch");
        VectorIndex::open_or_rebuild(&conn, vector_path.clone())
            .await
            .expect("vector");
        let vector = Arc::new(
            crate::vector::VectorReader::open(&conn, &vector_path)
                .await
                .expect("reader"),
        );
        let semantic = db::DbSemanticSource::new(conn.clone(), vector);
        let semantic_paths: Vec<String> = semantic
            .search_semantic(&[0.0; 1024], 10)
            .await
            .expect("semantic search")
            .into_iter()
            .map(|result| result.path)
            .collect();
        for (path, _, marker) in DOCUMENT_FIXTURES {
            assert!(semantic_paths.contains(&path.to_owned()), "{path}");
            assert_eq!(
                fts.search(marker, 10).await.expect("full-text search")[0].path,
                path
            );
        }
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
            vec![(
                "entry.note".to_string(),
                PreparedChunk::from("prepared content".to_string()),
            )]
        );
    }

    #[tokio::test]
    async fn directory_index_continues_after_a_file_fails_preparation() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "healthy.md", "healthy content");
        create_file(notes_dir.path(), "corrupt.md", "corrupt");
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
            &FailingContentPreparer,
        )
        .await
        .expect("index");

        assert_eq!(stats.added, 1);
        assert_eq!(stats.failed, 1);
        assert!(
            db::note_hash(&conn, "healthy.md")
                .await
                .expect("hash")
                .is_some()
        );
        assert!(
            db::note_hash(&conn, "corrupt.md")
                .await
                .expect("hash")
                .is_none()
        );
    }

    #[tokio::test]
    async fn directory_index_skips_unchanged_preparation_failures_and_clears_after_success() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "broken.md", "broken");
        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");
        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
        let calls = Arc::new(AtomicUsize::new(0));
        let failing = CountingPreparer {
            calls: Arc::clone(&calls),
            fails: true,
        };
        let embedder = Embedder::create_null(1024);

        let first =
            index_directory_with_preparer(&conn, &fts, &embedder, notes_dir.path(), &failing)
                .await
                .expect("first index");
        let second =
            index_directory_with_preparer(&conn, &fts, &embedder, notes_dir.path(), &failing)
                .await
                .expect("second index");
        assert_eq!(first.failed, 1);
        assert_eq!(second.failed, 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        create_file(notes_dir.path(), "broken.md", "fixed");
        let working_calls = Arc::new(AtomicUsize::new(0));
        let working = CountingPreparer {
            calls: Arc::clone(&working_calls),
            fails: false,
        };
        let stats =
            index_directory_with_preparer(&conn, &fts, &embedder, notes_dir.path(), &working)
                .await
                .expect("retry index");
        assert_eq!(working_calls.load(Ordering::SeqCst), 1);
        assert_eq!(stats.added, 1);
        assert!(
            db::failed_file_hash(&conn, "broken.md")
                .await
                .expect("failure hash")
                .is_none()
        );
    }

    #[tokio::test]
    async fn clearing_remembered_failures_reprepares_unchanged_content() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "broken.md", "broken");
        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");
        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
        let embedder = Embedder::create_null(1024);
        let failing = CountingPreparer {
            calls: Arc::new(AtomicUsize::new(0)),
            fails: true,
        };

        index_directory_with_preparer(&conn, &fts, &embedder, notes_dir.path(), &failing)
            .await
            .expect("first index");
        db::clear_failed_files(&conn).await.expect("clear failures");

        let retry_calls = Arc::new(AtomicUsize::new(0));
        let working = CountingPreparer {
            calls: Arc::clone(&retry_calls),
            fails: false,
        };
        let stats =
            index_directory_with_preparer(&conn, &fts, &embedder, notes_dir.path(), &working)
                .await
                .expect("retry index");

        assert_eq!(retry_calls.load(Ordering::SeqCst), 1);
        assert_eq!(stats.added, 1);
        assert_eq!(stats.failed, 0);
    }

    #[tokio::test]
    async fn recorded_failure_keeps_the_underlying_cause() {
        struct ContextualFailurePreparer;

        impl DocumentPreparer for ContextualFailurePreparer {
            fn supports_path(&self, source_path: &Path) -> bool {
                source_path
                    .extension()
                    .is_some_and(|extension| extension == "md")
            }

            fn prepare(
                &self,
                _source_path: &Path,
                _source: &[u8],
            ) -> anyhow::Result<Vec<PreparedChunk>> {
                Err(anyhow::anyhow!("not a char boundary at byte 71")
                    .context("extracting book.md with Xberg"))
            }

            fn profile(&self) -> &'static str {
                "contextual-failure-v1"
            }
        }

        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "book.md", "content");
        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");
        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");

        index_directory_with_preparer(
            &conn,
            &fts,
            &Embedder::create_null(1024),
            notes_dir.path(),
            &ContextualFailurePreparer,
        )
        .await
        .expect("index");

        let failures = db::failed_files(&conn).await.expect("failures");
        let (_, error) = failures.first().expect("one failure");
        assert!(error.contains("extracting book.md with Xberg"), "{error}");
        assert!(error.contains("not a char boundary at byte 71"), "{error}");
    }

    #[tokio::test]
    async fn directory_index_removes_failed_record_for_deleted_unindexed_file() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "broken.md", "broken");
        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");
        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
        let failing = CountingPreparer {
            calls: Arc::new(AtomicUsize::new(0)),
            fails: true,
        };
        let embedder = Embedder::create_null(1024);

        index_directory_with_preparer(&conn, &fts, &embedder, notes_dir.path(), &failing)
            .await
            .expect("failed index");
        std::fs::remove_file(notes_dir.path().join("broken.md")).expect("remove");
        let stats =
            index_directory_with_preparer(&conn, &fts, &embedder, notes_dir.path(), &failing)
                .await
                .expect("delete index");

        assert_eq!(stats.deleted, 1);
        assert!(
            db::failed_file_hash(&conn, "broken.md")
                .await
                .expect("failure hash")
                .is_none()
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
    async fn failed_preparation_and_embedding_preserve_existing_note() {
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
        let stats = index_directory_with_preparer(
            &conn,
            &fts,
            &Embedder::create_null(1024),
            notes_dir.path(),
            &failed_preparer,
        )
        .await
        .expect("tolerant index");
        assert_eq!(stats.failed, 1);
        assert_eq!(stats.deleted, 1);
        assert_eq!(
            db::note_hash(&conn, "note.md").await.expect("hash"),
            old_hash
        );
        assert_old_note_content(&conn).await;
        assert_eq!(
            fts.search("before", 10).await.expect("search")[0].path,
            "note.md"
        );
        assert!(
            db::note_hash(&conn, "gone.md")
                .await
                .expect("hash")
                .is_none()
        );

        let working_preparer = CountingPreparer {
            calls: Arc::new(AtomicUsize::new(0)),
            fails: false,
        };
        create_file(notes_dir.path(), "note.md", "after retry");
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
                .is_none()
        );
    }

    #[tokio::test]
    async fn unchanged_directory_scan_makes_no_fts_mutations() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "note.md", "unchanged content");
        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");
        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
        let embedder = Embedder::create_null(1024);

        index_directory(&conn, &fts, &embedder, notes_dir.path())
            .await
            .expect("initial index");
        fts.reset_mutation_count();

        index_directory(&conn, &fts, &embedder, notes_dir.path())
            .await
            .expect("unchanged index");

        assert_eq!(fts.mutation_count(), 0);
    }

    #[tokio::test]
    async fn directory_scan_incrementally_applies_add_update_and_delete() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "keep.md", "keep token");
        create_file(notes_dir.path(), "update.md", "before token");
        create_file(notes_dir.path(), "remove.md", "remove token");
        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");
        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
        let embedder = Embedder::create_null(1024);

        index_directory(&conn, &fts, &embedder, notes_dir.path())
            .await
            .expect("initial index");
        create_file(notes_dir.path(), "update.md", "after token");
        create_file(notes_dir.path(), "add.md", "add token");
        std::fs::remove_file(notes_dir.path().join("remove.md")).expect("remove");
        fts.reset_mutation_count();

        let stats = index_directory(&conn, &fts, &embedder, notes_dir.path())
            .await
            .expect("mixed index");

        assert_eq!(stats.added, 1);
        assert_eq!(stats.updated, 1);
        assert_eq!(stats.deleted, 1);
        assert_eq!(fts.mutation_count(), 3);
        assert_eq!(
            fts.search("after", 10).await.expect("search")[0].path,
            "update.md"
        );
        assert_eq!(
            fts.search("add", 10).await.expect("search")[0].path,
            "add.md"
        );
        assert!(fts.search("remove", 10).await.expect("search").is_empty());
    }

    #[tokio::test]
    async fn incremental_fts_failure_rebuilds_from_committed_db_chunks() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "note.md", "before token");
        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");
        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
        let embedder = Embedder::create_null(1024);

        index_directory(&conn, &fts, &embedder, notes_dir.path())
            .await
            .expect("initial index");
        create_file(notes_dir.path(), "note.md", "after token");
        fts.reset_mutation_count();
        fts.fail_next_mutations(1);

        index_directory(&conn, &fts, &embedder, notes_dir.path())
            .await
            .expect("recovery index");

        assert_eq!(fts.mutation_count(), 2);
        assert_eq!(
            db::all_chunks(&conn).await.expect("chunks")[0].1,
            PreparedChunk::from("after token".to_string())
        );
        let db_chunk = db::all_chunks(&conn).await.expect("chunks").remove(0).1;
        let fts_result = fts.search("after", 10).await.expect("search").remove(0);
        assert_eq!(fts_result.path, "note.md");
        assert_eq!(fts_result.locator, db_chunk.locator);
        assert!(fts.search("before", 10).await.expect("search").is_empty());
    }

    #[tokio::test]
    async fn failed_fts_recovery_returns_incremental_and_recovery_context() {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        create_file(notes_dir.path(), "note.md", "before token");
        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");
        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
        let embedder = Embedder::create_null(1024);

        index_directory(&conn, &fts, &embedder, notes_dir.path())
            .await
            .expect("initial index");
        create_file(notes_dir.path(), "note.md", "after token");
        fts.fail_next_mutations(2);

        let error = index_directory(&conn, &fts, &embedder, notes_dir.path())
            .await
            .expect_err("recovery failure should fail indexing")
            .to_string();

        assert!(error.contains("incremental FTS update failed"));
        assert!(error.contains("FTS recovery failed"));
        assert_eq!(
            db::all_chunks(&conn).await.expect("chunks")[0].1,
            PreparedChunk::from("after token".to_string())
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

    struct TestStore {
        notes_dir: tempfile::TempDir,
        conn: Connection,
        fts: crate::fts::FtsIndex,
        _db: libsql::Database,
        _db_dir: tempfile::TempDir,
        _fts_dir: tempfile::TempDir,
    }

    async fn archive_test_store() -> TestStore {
        let notes_dir = tempfile::tempdir().expect("tempdir");
        let db_dir = tempfile::tempdir().expect("tempdir");
        let (db, conn) = db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");
        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
        TestStore {
            notes_dir,
            conn,
            fts,
            _db: db,
            _db_dir: db_dir,
            _fts_dir: fts_dir,
        }
    }

    async fn index_with_counting(
        conn: &Connection,
        fts: &crate::fts::FtsIndex,
        notes_dir: &Path,
        calls: &Arc<AtomicUsize>,
    ) -> IndexStats {
        let preparer = CountingPreparer {
            calls: Arc::clone(calls),
            fails: false,
        };
        index_directory_with_preparer(
            conn,
            fts,
            &Embedder::create_null(1024),
            notes_dir,
            &preparer,
        )
        .await
        .expect("index")
    }

    #[tokio::test]
    async fn each_document_inside_an_archive_becomes_its_own_note() {
        let store = archive_test_store().await;
        let (notes_dir, conn, fts) = (&store.notes_dir, &store.conn, &store.fts);
        write_archive(
            &notes_dir.path().join("docs.zip"),
            &[
                ("guide.md", "archive guide body"),
                ("nested/reference.md", "archive reference body"),
                ("picture.png", "not a document"),
            ],
        );

        let stats =
            index_with_counting(conn, fts, notes_dir.path(), &Arc::new(AtomicUsize::new(0))).await;

        assert_eq!(stats.added, 2, "only supported members are indexed");
        let mut paths: Vec<String> = db::all_note_hashes(conn)
            .await
            .expect("hashes")
            .into_keys()
            .collect();
        paths.sort();
        assert_eq!(
            paths,
            vec!["docs.zip!guide.md", "docs.zip!nested/reference.md"]
        );
        assert!(
            db::all_chunks(conn)
                .await
                .expect("chunks")
                .iter()
                .any(|(path, chunk)| path == "docs.zip!guide.md"
                    && chunk.content == "archive guide body")
        );
    }

    #[tokio::test]
    async fn unchanged_archive_members_are_never_extracted_again() {
        let store = archive_test_store().await;
        let (notes_dir, conn, fts) = (&store.notes_dir, &store.conn, &store.fts);
        write_archive(
            &notes_dir.path().join("docs.zip"),
            &[("one.md", "one"), ("two.md", "two")],
        );
        let calls = Arc::new(AtomicUsize::new(0));

        index_with_counting(conn, fts, notes_dir.path(), &calls).await;
        let after_first = calls.load(Ordering::SeqCst);
        let stats = index_with_counting(conn, fts, notes_dir.path(), &calls).await;

        assert_eq!(after_first, 2);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            after_first,
            "a second pass must not extract or prepare unchanged members"
        );
        assert_eq!(stats.unchanged, 2);
    }

    #[tokio::test]
    async fn editing_one_member_reindexes_only_that_member() {
        let store = archive_test_store().await;
        let (notes_dir, conn, fts) = (&store.notes_dir, &store.conn, &store.fts);
        let archive_path = notes_dir.path().join("docs.zip");
        write_archive(&archive_path, &[("one.md", "one"), ("two.md", "two")]);
        let calls = Arc::new(AtomicUsize::new(0));
        index_with_counting(conn, fts, notes_dir.path(), &calls).await;

        write_archive(
            &archive_path,
            &[("one.md", "one"), ("two.md", "two edited")],
        );
        let stats = index_with_counting(conn, fts, notes_dir.path(), &calls).await;

        assert_eq!(stats.updated, 1);
        assert_eq!(stats.unchanged, 1);
        assert!(
            db::all_chunks(conn)
                .await
                .expect("chunks")
                .iter()
                .any(|(path, chunk)| path == "docs.zip!two.md" && chunk.content == "two edited")
        );
    }

    #[tokio::test]
    async fn a_member_removed_from_the_archive_is_removed_from_the_index() {
        let store = archive_test_store().await;
        let (notes_dir, conn, fts) = (&store.notes_dir, &store.conn, &store.fts);
        let archive_path = notes_dir.path().join("docs.zip");
        write_archive(&archive_path, &[("one.md", "one"), ("two.md", "two")]);
        let calls = Arc::new(AtomicUsize::new(0));
        index_with_counting(conn, fts, notes_dir.path(), &calls).await;

        write_archive(&archive_path, &[("one.md", "one")]);
        let stats = index_with_counting(conn, fts, notes_dir.path(), &calls).await;

        assert_eq!(stats.deleted, 1);
        assert_eq!(
            db::all_note_hashes(conn)
                .await
                .expect("hashes")
                .into_keys()
                .collect::<Vec<String>>(),
            vec!["docs.zip!one.md"]
        );
    }

    #[tokio::test]
    async fn an_archive_that_cannot_be_read_keeps_its_indexed_members() {
        let store = archive_test_store().await;
        let (notes_dir, conn, fts) = (&store.notes_dir, &store.conn, &store.fts);
        let archive_path = notes_dir.path().join("docs.zip");
        write_archive(&archive_path, &[("one.md", "one"), ("two.md", "two")]);
        let calls = Arc::new(AtomicUsize::new(0));
        index_with_counting(conn, fts, notes_dir.path(), &calls).await;

        std::fs::write(&archive_path, b"truncated garbage").expect("corrupt archive");
        let stats = index_with_counting(conn, fts, notes_dir.path(), &calls).await;

        assert_eq!(
            stats.deleted, 0,
            "a broken archive must not wipe its members"
        );
        assert_eq!(db::all_note_hashes(conn).await.expect("hashes").len(), 2);
    }

    #[tokio::test]
    async fn a_member_that_expands_too_far_is_reported_as_a_failure() {
        let store = archive_test_store().await;
        let (notes_dir, conn, fts) = (&store.notes_dir, &store.conn, &store.fts);
        let bomb = "a".repeat(2 * 1024 * 1024);
        write_archive(
            &notes_dir.path().join("docs.zip"),
            &[("one.md", "one"), ("bomb.md", &bomb)],
        );

        let stats =
            index_with_counting(conn, fts, notes_dir.path(), &Arc::new(AtomicUsize::new(0))).await;

        assert_eq!(stats.added, 1);
        assert_eq!(stats.failed, 1);
        let failures = db::failed_files(conn).await.expect("failures");
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, "docs.zip!bomb.md");
        assert!(
            failures[0].1.contains("expands"),
            "the failure should explain itself: {}",
            failures[0].1
        );
    }

    #[test]
    fn index_stats_display_format() {
        let stats = IndexStats {
            added: 5,
            updated: 3,
            deleted: 1,
            unchanged: 10,
            failed: 2,
        };
        assert_eq!(
            stats.to_string(),
            "added=5, updated=3, deleted=1, unchanged=10, failed=2"
        );
    }
}
