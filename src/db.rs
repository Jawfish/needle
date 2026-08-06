use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use anyhow::{Context, bail};
use libsql::Connection;

use crate::{
    document::PreparedChunk,
    error::NeedleError,
    rank::{Candidate, PathSource, SemanticSource},
    similar::{AllChunkEmbeddingsSource, NoteEmbeddingsSource, RelatedResult, RelatedSearchSource},
    types::IndexProfile,
};

const BYTES_PER_F32: usize = 4;

pub type NoteUpsert = (String, String, Vec<(PreparedChunk, Vec<f32>)>);

pub fn decode_embedding(blob: &[u8]) -> anyhow::Result<Vec<f32>> {
    if !blob.len().is_multiple_of(BYTES_PER_F32) {
        bail!(
            "embedding blob size {} is not a multiple of {BYTES_PER_F32}",
            blob.len()
        );
    }
    Ok(blob
        .chunks_exact(BYTES_PER_F32)
        .map(|chunk| {
            let bytes: [u8; BYTES_PER_F32] = chunk
                .try_into()
                .expect("chunks_exact guarantees exactly 4 bytes");
            f32::from_le_bytes(bytes)
        })
        .collect())
}

pub struct SearchResult {
    pub path: String,
    pub snippet: String,
    pub locator: Option<String>,
}

pub async fn connect(
    db_path: &Path,
    expected_dim: Option<usize>,
) -> anyhow::Result<(libsql::Database, Connection)> {
    connect_inner(db_path, expected_dim, None).await
}

pub async fn connect_with_profile(
    db_path: &Path,
    profile: &IndexProfile,
) -> anyhow::Result<(libsql::Database, Connection)> {
    connect_inner(db_path, Some(profile.embedder.dimension), Some(profile)).await
}

async fn connect_inner(
    db_path: &Path,
    expected_dim: Option<usize>,
    profile: Option<&IndexProfile>,
) -> anyhow::Result<(libsql::Database, Connection)> {
    let db = libsql::Builder::new_local(db_path).build().await?;
    let conn = db.connect()?;
    init_schema(&conn, expected_dim, profile).await?;
    Ok((db, conn))
}

async fn init_schema(
    conn: &Connection,
    expected_dim: Option<usize>,
    profile: Option<&IndexProfile>,
) -> anyhow::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS notes (
            path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS failed_files (
            path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            error TEXT NOT NULL
        );",
    )
    .await?;

    if let Some(profile) = profile {
        if chunks_table_exists(conn).await? && !chunks_have_locator(conn).await? {
            return Err(NeedleError::IndexProfileMismatch {
                reason: "index has no chunk locator column".to_owned(),
            }
            .into());
        }
        resolve_profile(conn, profile).await?;
    }

    // When neither the caller nor stored metadata can supply a dimension (fresh DB
    // opened by a read-only command like `similar`) skip chunks infrastructure
    // entirely. Writing a guessed dimension here would permanently corrupt the
    // schema for the first real indexing run.
    let Some(dim) = resolve_schema_dim(conn, expected_dim).await? else {
        return Ok(());
    };

    let create_chunks = format!(
        "CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL REFERENCES notes(path) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            locator TEXT,
            embedding F32_BLOB({dim})
        )"
    );
    conn.execute(&create_chunks, ()).await?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)",
        (),
    )
    .await?;

    Ok(())
}

// Returns the concrete dimension to use for the chunks schema, or None when
// this is a fresh DB opened by a command that does not produce embeddings.
async fn resolve_schema_dim(
    conn: &Connection,
    expected_dim: Option<usize>,
) -> anyhow::Result<Option<usize>> {
    let stored = stored_dim(conn).await?;

    // Legacy/migration path: chunks table already exists but the metadata row
    // was never written (older DB). Recover the dimension from the column
    // definition and persist it so future opens do not need to re-infer.
    let effective_stored = if stored.is_none() && chunks_table_exists(conn).await? {
        let inferred = infer_dim_from_chunks_schema(conn).await?;
        store_dim(conn, inferred).await?;
        Some(inferred)
    } else {
        stored
    };

    match (expected_dim, effective_stored) {
        (Some(expected), Some(stored)) => {
            if expected != stored {
                return Err(crate::error::NeedleError::DimensionMismatch {
                    db: stored,
                    provider: expected,
                }
                .into());
            }
            Ok(Some(expected))
        }
        (Some(expected), None) => {
            store_dim(conn, expected).await?;
            Ok(Some(expected))
        }
        (None, Some(stored)) => Ok(Some(stored)),
        (None, None) => Ok(None),
    }
}

async fn resolve_profile(conn: &Connection, expected: &IndexProfile) -> anyhow::Result<()> {
    let mut rows = conn
        .query("SELECT value FROM metadata WHERE key = 'index_profile'", ())
        .await?;
    let Some(row) = rows.next().await? else {
        if chunks_table_exists(conn).await? {
            return Err(NeedleError::IndexProfileMismatch {
                reason: "legacy index has no profile metadata".to_owned(),
            }
            .into());
        }
        let value = serde_json::to_string(expected)?;
        conn.execute(
            "INSERT INTO metadata (key, value) VALUES ('index_profile', ?1)",
            [value],
        )
        .await?;
        return Ok(());
    };
    let value: String = row.get(0)?;
    let stored: IndexProfile =
        serde_json::from_str(&value).map_err(|e| NeedleError::IndexProfileMismatch {
            reason: format!("stored profile metadata is invalid: {e}"),
        })?;
    if stored != *expected {
        return Err(NeedleError::IndexProfileMismatch {
            reason: "stored profile differs from the configured embedder or document preparer"
                .to_owned(),
        }
        .into());
    }
    Ok(())
}

pub async fn embedding_dimension(conn: &Connection) -> anyhow::Result<Option<usize>> {
    stored_dim(conn).await
}

async fn stored_dim(conn: &Connection) -> anyhow::Result<Option<usize>> {
    let mut rows = conn
        .query("SELECT value FROM metadata WHERE key = 'embedding_dim'", ())
        .await?;
    match rows.next().await? {
        Some(row) => {
            let val: String = row.get(0)?;
            Ok(Some(val.parse().context("invalid stored embedding_dim")?))
        }
        None => Ok(None),
    }
}

async fn store_dim(conn: &Connection, dim: usize) -> anyhow::Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('embedding_dim', ?1)",
        [dim.to_string()],
    )
    .await?;
    Ok(())
}

pub async fn chunks_table_exists(conn: &Connection) -> anyhow::Result<bool> {
    Ok(conn
        .query(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks'",
            (),
        )
        .await?
        .next()
        .await?
        .is_some())
}

async fn chunks_have_locator(conn: &Connection) -> anyhow::Result<bool> {
    let mut rows = conn.query("PRAGMA table_info(chunks)", ()).await?;
    while let Some(row) = rows.next().await? {
        let name: String = row.get(1)?;
        if name == "locator" {
            return Ok(true);
        }
    }
    Ok(false)
}

async fn infer_dim_from_chunks_schema(conn: &Connection) -> anyhow::Result<usize> {
    let mut rows = conn
        .query(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks'",
            (),
        )
        .await?;
    let row = rows
        .next()
        .await?
        .ok_or_else(|| anyhow::anyhow!("chunks table not found in schema"))?;
    let sql: String = row.get(0)?;
    parse_dim_from_schema_sql(&sql)
}

fn parse_dim_from_schema_sql(sql: &str) -> anyhow::Result<usize> {
    let upper = sql.to_ascii_uppercase();
    let marker = "F32_BLOB(";
    let start = upper
        .find(marker)
        .ok_or_else(|| anyhow::anyhow!("embedding column not found in chunks schema"))?;
    let after_paren = start + marker.len();
    let close = upper[after_paren..]
        .find(')')
        .ok_or_else(|| anyhow::anyhow!("malformed F32_BLOB in chunks schema"))?;
    upper[after_paren..after_paren + close]
        .trim()
        .parse::<usize>()
        .context("invalid dimension in chunks schema")
}

pub async fn all_note_hashes(conn: &Connection) -> anyhow::Result<HashMap<String, String>> {
    let mut rows = conn
        .query("SELECT path, content_hash FROM notes", ())
        .await?;

    let mut hashes = HashMap::new();
    while let Some(row) = rows.next().await? {
        let path: String = row.get(0)?;
        let hash: String = row.get(1)?;
        hashes.insert(path, hash);
    }
    Ok(hashes)
}

pub async fn note_hash(conn: &Connection, path: &str) -> anyhow::Result<Option<String>> {
    let mut rows = conn
        .query("SELECT content_hash FROM notes WHERE path = ?1", [path])
        .await?;
    match rows.next().await? {
        Some(row) => Ok(Some(row.get(0)?)),
        None => Ok(None),
    }
}

pub async fn all_failed_file_hashes(conn: &Connection) -> anyhow::Result<HashMap<String, String>> {
    let mut rows = conn
        .query("SELECT path, content_hash FROM failed_files", ())
        .await?;
    let mut hashes = HashMap::new();
    while let Some(row) = rows.next().await? {
        hashes.insert(row.get(0)?, row.get(1)?);
    }
    Ok(hashes)
}

pub async fn failed_files(conn: &Connection) -> anyhow::Result<Vec<(String, String)>> {
    let mut rows = conn
        .query("SELECT path, error FROM failed_files ORDER BY path", ())
        .await?;
    let mut failures = Vec::new();
    while let Some(row) = rows.next().await? {
        failures.push((row.get(0)?, row.get(1)?));
    }
    Ok(failures)
}

pub async fn failed_file_hash(conn: &Connection, path: &str) -> anyhow::Result<Option<String>> {
    let mut rows = conn
        .query(
            "SELECT content_hash FROM failed_files WHERE path = ?1",
            [path],
        )
        .await?;
    match rows.next().await? {
        Some(row) => Ok(Some(row.get(0)?)),
        None => Ok(None),
    }
}

pub async fn record_failed_file(
    conn: &Connection,
    path: &str,
    content_hash: &str,
    error: &str,
) -> anyhow::Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO failed_files (path, content_hash, error) VALUES (?1, ?2, ?3)",
        [path, content_hash, error],
    )
    .await?;
    Ok(())
}

pub async fn clear_failed_file(conn: &Connection, path: &str) -> anyhow::Result<()> {
    conn.execute("DELETE FROM failed_files WHERE path = ?1", [path])
        .await?;
    Ok(())
}

pub async fn clear_failed_files(conn: &Connection) -> anyhow::Result<()> {
    conn.execute("DELETE FROM failed_files", ()).await?;
    Ok(())
}

pub async fn upsert_note<T>(
    conn: &Connection,
    path: &str,
    hash: &str,
    chunks: &[(T, Vec<f32>)],
) -> anyhow::Result<()>
where
    T: Clone + Into<PreparedChunk> + Sync,
{
    let tx = conn.transaction().await?;

    tx.execute("DELETE FROM chunks WHERE path = ?1", [path])
        .await?;

    tx.execute(
        "INSERT OR REPLACE INTO notes (path, content_hash, updated_at) VALUES (?1, ?2, unixepoch())",
        [path, hash],
    )
    .await?;
    tx.execute("DELETE FROM failed_files WHERE path = ?1", [path])
        .await?;

    for (i, (chunk, embedding)) in chunks.iter().enumerate() {
        let chunk: PreparedChunk = chunk.clone().into();
        let embedding_json = serde_json::to_string(embedding)?;
        let chunk_index = i64::try_from(i).context("chunk index exceeds i64 range")?;

        tx.execute(
            "INSERT INTO chunks (path, chunk_index, content, locator, embedding) VALUES (?1, ?2, ?3, ?4, vector32(?5))",
            libsql::params![path, chunk_index, chunk.content.as_str(), chunk.locator.as_deref(), embedding_json],
        )
        .await?;
    }

    tx.commit().await?;
    Ok(())
}

pub async fn delete_note(conn: &Connection, path: &str) -> anyhow::Result<()> {
    apply_directory_changes(conn, &[path.to_owned()], &[]).await
}

pub async fn apply_directory_changes(
    conn: &Connection,
    deleted_paths: &[String],
    upserts: &[NoteUpsert],
) -> anyhow::Result<()> {
    let tx = conn.transaction().await?;

    for path in deleted_paths {
        tx.execute("DELETE FROM chunks WHERE path = ?1", [path.as_str()])
            .await?;
        tx.execute("DELETE FROM notes WHERE path = ?1", [path.as_str()])
            .await?;
        tx.execute("DELETE FROM failed_files WHERE path = ?1", [path.as_str()])
            .await?;
    }

    for (path, hash, chunks) in upserts {
        tx.execute("DELETE FROM chunks WHERE path = ?1", [path.as_str()])
            .await?;
        tx.execute(
            "INSERT OR REPLACE INTO notes (path, content_hash, updated_at) VALUES (?1, ?2, unixepoch())",
            [path.as_str(), hash.as_str()],
        )
        .await?;
        tx.execute("DELETE FROM failed_files WHERE path = ?1", [path.as_str()])
            .await?;

        for (i, (chunk, embedding)) in chunks.iter().enumerate() {
            let embedding_json = serde_json::to_string(embedding)?;
            let chunk_index = i64::try_from(i).context("chunk index exceeds i64 range")?;
            tx.execute(
                "INSERT INTO chunks (path, chunk_index, content, locator, embedding) VALUES (?1, ?2, ?3, ?4, vector32(?5))",
                libsql::params![path.as_str(), chunk_index, chunk.content.as_str(), chunk.locator.as_deref(), embedding_json],
            )
            .await?;
        }
    }

    tx.commit().await?;
    Ok(())
}

pub async fn document_chunks(
    conn: &Connection,
    path: &str,
) -> anyhow::Result<Option<Vec<PreparedChunk>>> {
    let mut rows = conn
        .query(
            "SELECT n.path, c.content, c.locator
             FROM notes n
             LEFT JOIN chunks c ON c.path = n.path
             WHERE n.path = ?1
             ORDER BY c.chunk_index",
            [path],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(None);
    };
    let mut chunks = Vec::new();
    if let Some(content) = row.get::<Option<String>>(1)? {
        chunks.push(PreparedChunk {
            content,
            locator: row.get(2)?,
        });
    }
    while let Some(row) = rows.next().await? {
        chunks.push(PreparedChunk {
            content: row.get(1)?,
            locator: row.get(2)?,
        });
    }
    Ok(Some(chunks))
}

pub async fn all_chunks(conn: &Connection) -> anyhow::Result<Vec<(String, PreparedChunk)>> {
    let mut rows = conn
        .query(
            "SELECT path, content, locator FROM chunks ORDER BY path, chunk_index",
            (),
        )
        .await?;
    let mut results = Vec::new();
    while let Some(row) = rows.next().await? {
        results.push((
            row.get(0)?,
            PreparedChunk {
                content: row.get(1)?,
                locator: row.get(2)?,
            },
        ));
    }
    Ok(results)
}

pub async fn all_chunk_embeddings_with_ids(
    conn: &Connection,
) -> anyhow::Result<Vec<(i64, Vec<f32>)>> {
    if !chunks_table_exists(conn).await? {
        return Ok(Vec::new());
    }
    let mut rows = conn
        .query(
            "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id",
            (),
        )
        .await?;
    let mut embeddings = Vec::new();
    while let Some(row) = rows.next().await? {
        embeddings.push((row.get(0)?, decode_embedding(&row.get::<Vec<u8>>(1)?)?));
    }
    Ok(embeddings)
}

pub async fn chunk_embeddings_for_paths(
    conn: &Connection,
    paths: &[String],
) -> anyhow::Result<Vec<(i64, Vec<f32>)>> {
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = (1..=paths.len())
        .map(|position| format!("?{position}"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT id, embedding FROM chunks WHERE path IN ({placeholders}) AND embedding IS NOT NULL ORDER BY id"
    );
    let params = paths.iter().cloned().map(libsql::Value::Text);
    let mut rows = conn.query(&sql, libsql::params_from_iter(params)).await?;
    let mut embeddings = Vec::new();
    while let Some(row) = rows.next().await? {
        embeddings.push((row.get(0)?, decode_embedding(&row.get::<Vec<u8>>(1)?)?));
    }
    Ok(embeddings)
}

pub async fn chunk_index_state(conn: &Connection) -> anyhow::Result<(usize, i64)> {
    if !chunks_table_exists(conn).await? {
        return Ok((0, 0));
    }
    let row = conn
        .query("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks", ())
        .await?
        .next()
        .await?
        .context("missing chunk state")?;
    Ok((
        usize::try_from(row.get::<i64>(0)?).context("chunk count exceeds usize")?,
        row.get(1)?,
    ))
}

pub async fn vector_index_state(conn: &Connection) -> anyhow::Result<Option<(usize, i64)>> {
    let mut rows = conn
        .query(
            "SELECT value FROM metadata WHERE key = 'vector_index_state'",
            (),
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(None);
    };
    let value: String = row.get(0)?;
    let (count, max_id) = value
        .split_once(':')
        .context("invalid vector index state")?;
    Ok(Some((
        count.parse().context("invalid vector index count")?,
        max_id.parse().context("invalid vector index max id")?,
    )))
}

pub async fn store_vector_index_state(
    conn: &Connection,
    state: (usize, i64),
) -> anyhow::Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO metadata (key, value) VALUES ('vector_index_state', ?1)",
        [format!("{}:{}", state.0, state.1)],
    )
    .await?;
    Ok(())
}

pub async fn search_semantic_candidates(
    conn: &Connection,
    query_embedding: &[f32],
    candidates: Vec<i64>,
) -> anyhow::Result<Vec<SearchResult>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = (2..=candidates.len() + 1)
        .map(|position| format!("?{position}"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT path, content, locator, vector_distance_cos(embedding, vector32(?1)) FROM chunks WHERE id IN ({placeholders})"
    );
    let mut params = Vec::with_capacity(candidates.len() + 1);
    params.push(libsql::Value::Text(serde_json::to_string(query_embedding)?));
    params.extend(candidates.into_iter().map(libsql::Value::Integer));
    let mut rows = conn.query(&sql, libsql::params_from_iter(params)).await?;
    let mut scored = Vec::new();
    while let Some(row) = rows.next().await? {
        scored.push((
            row.get::<Option<f64>>(3)?.unwrap_or(1.0),
            SearchResult {
                path: row.get(0)?,
                snippet: row.get(1)?,
                locator: row.get(2)?,
            },
        ));
    }
    scored.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut paths = HashSet::new();
    Ok(scored
        .into_iter()
        .filter_map(|(_, result)| paths.insert(result.path.clone()).then_some(result))
        .collect())
}

pub async fn chunk_embeddings_for_path(
    conn: &Connection,
    path: &str,
) -> anyhow::Result<Vec<Vec<f32>>> {
    if !chunks_table_exists(conn).await? {
        return Ok(vec![]);
    }

    let mut rows = conn
        .query(
            "SELECT embedding FROM chunks WHERE path = ?1 AND embedding IS NOT NULL ORDER BY chunk_index",
            [path],
        )
        .await?;

    let mut embeddings = Vec::new();
    while let Some(row) = rows.next().await? {
        let blob: Vec<u8> = row.get(0)?;
        match decode_embedding(&blob) {
            Ok(emb) => embeddings.push(emb),
            Err(err) => tracing::warn!(path, %err, "skipping chunk with corrupt embedding"),
        }
    }
    Ok(embeddings)
}

// Upper bound on how many index candidates search_related will ever request.
// Prevents unbounded growth if the index is very large and results are scarce.
const RELATED_MAX_K: usize = 100_000;

async fn search_related_candidates(
    conn: &Connection,
    query_embedding: &[f32],
    exclude_path: &str,
    limit: usize,
    vector: &crate::vector::VectorIndex,
) -> anyhow::Result<Vec<RelatedResult>> {
    let embedding_json = serde_json::to_string(query_embedding)?;
    let mut k = limit.saturating_mul(5).max(20);
    loop {
        let candidates = vector.search(query_embedding, k)?;
        let candidate_count = candidates.len();
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = (2..=candidates.len() + 1)
            .map(|position| format!("?{position}"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT path, vector_distance_cos(embedding, vector32(?1)) FROM chunks WHERE id IN ({placeholders})"
        );
        let mut params = Vec::with_capacity(candidates.len() + 1);
        params.push(libsql::Value::Text(embedding_json.clone()));
        params.extend(candidates.into_iter().map(libsql::Value::Integer));
        let mut rows = conn.query(&sql, libsql::params_from_iter(params)).await?;
        let mut scored = Vec::new();
        while let Some(row) = rows.next().await? {
            scored.push((
                row.get::<Option<f64>>(1)?.unwrap_or(1.0),
                row.get::<String>(0)?,
            ));
        }
        scored.sort_by(|left, right| left.0.total_cmp(&right.0));
        let mut paths = HashSet::new();
        let results: Vec<_> = scored
            .into_iter()
            .filter_map(|(distance, path)| {
                (path != exclude_path && paths.insert(path.clone())).then_some(RelatedResult {
                    path,
                    similarity: 1.0 - distance,
                })
            })
            .take(limit)
            .collect();
        if results.len() >= limit || k >= RELATED_MAX_K || candidate_count < k {
            return Ok(results);
        }
        k = k.saturating_mul(2).min(RELATED_MAX_K);
    }
}

pub async fn all_note_paths(conn: &Connection) -> anyhow::Result<Vec<String>> {
    let mut rows = conn.query("SELECT path FROM notes", ()).await?;
    let mut paths = Vec::new();
    while let Some(row) = rows.next().await? {
        let path: String = row.get(0)?;
        paths.push(path);
    }
    Ok(paths)
}

/// Adapter: implements `SemanticSource` against a live libsql connection.
pub struct DbSemanticSource {
    conn: Connection,
    vector: std::sync::Arc<crate::vector::VectorIndex>,
}

impl DbSemanticSource {
    pub const fn new(conn: Connection, vector: std::sync::Arc<crate::vector::VectorIndex>) -> Self {
        Self { conn, vector }
    }
}

impl SemanticSource for DbSemanticSource {
    fn search_semantic<'a>(
        &'a self,
        query_embedding: &'a [f32],
        limit: usize,
    ) -> crate::rank::SearchFuture<'a, Vec<Candidate>> {
        Box::pin(async move {
            Ok(search_semantic_candidates(
                &self.conn,
                query_embedding,
                self.vector
                    .search(query_embedding, limit.saturating_mul(3))?,
            )
            .await?
            .into_iter()
            .take(limit)
            .map(|r| Candidate {
                path: r.path,
                snippet: r.snippet,
                locator: r.locator,
            })
            .collect())
        })
    }
}

/// Adapter: implements `PathSource` against a live libsql connection.
pub struct DbPathSource {
    conn: Connection,
}

impl DbPathSource {
    pub const fn new(conn: Connection) -> Self {
        Self { conn }
    }
}

impl PathSource for DbPathSource {
    fn all_paths(&self) -> crate::rank::SearchFuture<'_, Vec<String>> {
        Box::pin(all_note_paths(&self.conn))
    }
}

pub trait DocumentChunksSource: Send + Sync {
    fn document_chunks<'a>(
        &'a self,
        path: &'a str,
    ) -> crate::rank::SearchFuture<'a, Option<Vec<PreparedChunk>>>;
}

pub struct DbDocumentChunksSource {
    conn: Connection,
}

impl DbDocumentChunksSource {
    pub const fn new(conn: Connection) -> Self {
        Self { conn }
    }
}

impl DocumentChunksSource for DbDocumentChunksSource {
    fn document_chunks<'a>(
        &'a self,
        path: &'a str,
    ) -> crate::rank::SearchFuture<'a, Option<Vec<PreparedChunk>>> {
        Box::pin(document_chunks(&self.conn, path))
    }
}

pub struct DbAllChunkEmbeddingsSource {
    conn: Connection,
}

impl DbAllChunkEmbeddingsSource {
    pub const fn new(conn: Connection) -> Self {
        Self { conn }
    }
}

impl AllChunkEmbeddingsSource for DbAllChunkEmbeddingsSource {
    fn has_embeddings(&self) -> crate::similar::SimilarFuture<'_, bool> {
        Box::pin(chunks_table_exists(&self.conn))
    }

    fn all_chunk_embeddings(&self) -> crate::similar::SimilarFuture<'_, Vec<(String, Vec<f32>)>> {
        Box::pin(async move {
            let mut rows = self
                .conn
                .query(
                    "SELECT path, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY path",
                    (),
                )
                .await?;
            let mut out = Vec::new();
            while let Some(row) = rows.next().await? {
                let path: String = row.get(0)?;
                let blob: Vec<u8> = row.get(1)?;
                match decode_embedding(&blob) {
                    Ok(emb) => out.push((path, emb)),
                    Err(err) => tracing::warn!(path, %err, "skipping chunk with corrupt embedding"),
                }
            }
            Ok(out)
        })
    }
}

pub struct DbNoteEmbeddingsSource {
    conn: Connection,
}

impl DbNoteEmbeddingsSource {
    pub const fn new(conn: Connection) -> Self {
        Self { conn }
    }
}

impl NoteEmbeddingsSource for DbNoteEmbeddingsSource {
    fn chunk_embeddings_for_path<'a>(
        &'a self,
        path: &'a str,
    ) -> crate::similar::SimilarFuture<'a, Vec<Vec<f32>>> {
        Box::pin(chunk_embeddings_for_path(&self.conn, path))
    }
}

pub struct DbRelatedSearchSource {
    conn: Connection,
    vector: std::sync::Arc<crate::vector::VectorIndex>,
}

impl DbRelatedSearchSource {
    pub const fn new(conn: Connection, vector: std::sync::Arc<crate::vector::VectorIndex>) -> Self {
        Self { conn, vector }
    }
}

impl RelatedSearchSource for DbRelatedSearchSource {
    fn search_related<'a>(
        &'a self,
        embedding: &'a [f32],
        exclude_path: &'a str,
        limit: usize,
    ) -> crate::similar::SimilarFuture<'a, Vec<RelatedResult>> {
        Box::pin(search_related_candidates(
            &self.conn,
            embedding,
            exclude_path,
            limit,
            &self.vector,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_DIM: usize = 1024;

    async fn test_db() -> (tempfile::TempDir, libsql::Database, Connection) {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let (db, conn) = connect(&db_path, Some(TEST_DIM))
            .await
            .expect("connect failed");
        (dir, db, conn)
    }

    async fn test_vector(
        conn: &Connection,
        dir: &tempfile::TempDir,
    ) -> std::sync::Arc<crate::vector::VectorIndex> {
        std::sync::Arc::new(
            crate::vector::VectorIndex::open_or_rebuild(conn, dir.path().join("test.usearch"))
                .await
                .expect("vector"),
        )
    }

    fn dummy_embedding() -> Vec<f32> {
        vec![0.0; TEST_DIM]
    }

    fn test_profile() -> IndexProfile {
        IndexProfile {
            embedder: crate::types::EmbedderProfile {
                provider: "openai".to_owned(),
                endpoint: Some("https://example.test/v1".to_owned()),
                model: "model-a".to_owned(),
                dimension: TEST_DIM,
            },
            preparer: "markdown-v1".to_owned(),
        }
    }

    #[tokio::test]
    async fn profile_open_persists_and_requires_an_exact_match() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.db");
        let profile = test_profile();
        let (_db, conn) = connect_with_profile(&path, &profile).await.expect("create");
        let mut rows = conn
            .query("SELECT value FROM metadata WHERE key = 'index_profile'", ())
            .await
            .expect("query");
        let row = rows.next().await.expect("row").expect("profile row");
        let stored: String = row.get(0).expect("value");
        assert_eq!(stored, serde_json::to_string(&profile).expect("serialize"));
        drop(conn);
        connect_with_profile(&path, &profile)
            .await
            .expect("matching profile");
    }

    #[tokio::test]
    async fn profile_open_rejects_pre_locator_chunks_schema() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.db");
        let profile = test_profile();
        let (_db, conn) = connect_with_profile(&path, &profile).await.expect("create");
        conn.execute_batch(
            "DROP TABLE chunks;
             CREATE TABLE chunks (
                 id INTEGER PRIMARY KEY,
                 path TEXT NOT NULL,
                 chunk_index INTEGER NOT NULL,
                 content TEXT NOT NULL,
                 embedding F32_BLOB(1024)
             );",
        )
        .await
        .expect("replace chunks schema");
        drop(conn);

        let err = connect_with_profile(&path, &profile)
            .await
            .expect_err("pre-locator schema rejected");
        assert!(matches!(
            err.downcast_ref::<NeedleError>(),
            Some(NeedleError::IndexProfileMismatch { .. })
        ));
    }

    #[tokio::test]
    async fn profile_open_rejects_legacy_and_all_identity_mismatches() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.db");
        let profile = test_profile();
        let (_db, _conn) = connect(&path, Some(TEST_DIM)).await.expect("legacy index");
        let err = connect_with_profile(&path, &profile)
            .await
            .expect_err("legacy rejected");
        assert!(matches!(
            err.downcast_ref::<NeedleError>(),
            Some(NeedleError::IndexProfileMismatch { .. })
        ));

        std::fs::remove_file(&path).expect("remove legacy");
        let (_db, _conn) = connect_with_profile(&path, &profile).await.expect("create");
        for changed in [
            IndexProfile {
                embedder: crate::types::EmbedderProfile {
                    provider: "voyage".to_owned(),
                    ..profile.embedder.clone()
                },
                preparer: profile.preparer.clone(),
            },
            IndexProfile {
                embedder: crate::types::EmbedderProfile {
                    model: "model-b".to_owned(),
                    ..profile.embedder.clone()
                },
                preparer: profile.preparer.clone(),
            },
            IndexProfile {
                embedder: crate::types::EmbedderProfile {
                    endpoint: Some("https://other.test/v1".to_owned()),
                    ..profile.embedder.clone()
                },
                preparer: profile.preparer.clone(),
            },
            IndexProfile {
                embedder: crate::types::EmbedderProfile {
                    dimension: 384,
                    ..profile.embedder.clone()
                },
                preparer: profile.preparer.clone(),
            },
            IndexProfile {
                embedder: profile.embedder.clone(),
                preparer: "other-v1".to_owned(),
            },
        ] {
            let err = connect_with_profile(&path, &changed)
                .await
                .expect_err("mismatch rejected");
            assert!(matches!(
                err.downcast_ref::<NeedleError>(),
                Some(
                    NeedleError::IndexProfileMismatch { .. }
                        | NeedleError::DimensionMismatch { .. }
                )
            ));
        }
    }

    fn make_chunks(texts: &[&str]) -> Vec<(String, Vec<f32>)> {
        texts
            .iter()
            .map(|t| ((*t).to_owned(), dummy_embedding()))
            .collect()
    }

    #[test]
    fn parse_dim_from_schema_sql_extracts_dimension() {
        let sql =
            "CREATE TABLE chunks (\n    id INTEGER PRIMARY KEY,\n    embedding F32_BLOB(384)\n)";
        assert_eq!(parse_dim_from_schema_sql(sql).expect("should parse"), 384);
    }

    #[test]
    fn parse_dim_from_schema_sql_is_case_insensitive() {
        let sql = "CREATE TABLE chunks (embedding f32_blob(1024))";
        assert_eq!(parse_dim_from_schema_sql(sql).expect("should parse"), 1024);
    }

    #[test]
    fn parse_dim_from_schema_sql_errors_when_column_absent() {
        let sql = "CREATE TABLE chunks (id INTEGER PRIMARY KEY, content TEXT)";
        assert!(parse_dim_from_schema_sql(sql).is_err());
    }

    #[tokio::test]
    async fn connect_without_dim_on_fresh_db_skips_chunks_table() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let (_db, conn) = connect(&db_path, None)
            .await
            .expect("connect without dim should succeed");
        assert!(
            !chunks_table_exists(&conn).await.expect("check failed"),
            "chunks table must not be created when dim is unknown"
        );
    }

    #[tokio::test]
    async fn connect_without_dim_does_not_write_metadata() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let (_db, conn) = connect(&db_path, None)
            .await
            .expect("connect without dim should succeed");
        assert!(
            stored_dim(&conn).await.expect("query failed").is_none(),
            "no dim should be written to metadata when dim is unknown"
        );
    }

    #[tokio::test]
    async fn connect_without_dim_then_with_dim_succeeds() {
        // Regression: similar (no embedder) on a fresh DB must not wedge the schema
        // so that a subsequent reindex with a real provider can succeed.
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");

        let (db1, conn1) = connect(&db_path, None)
            .await
            .expect("first connect (no dim) should succeed");
        drop(conn1);
        drop(db1);

        let (_db2, conn2) = connect(&db_path, Some(384))
            .await
            .expect("second connect with dim should succeed after no-dim connect");

        let embedding = vec![0.0_f32; 384];
        upsert_note(
            &conn2,
            "note.md",
            "abc",
            &[("content".to_owned(), embedding)],
        )
        .await
        .expect("upsert with correct dim must succeed");
    }

    #[tokio::test]
    async fn connect_with_dim_validates_against_stored_dim() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let (db, conn) = connect(&db_path, Some(1024))
            .await
            .expect("initial connect");
        drop(conn);
        drop(db);

        let result = connect(&db_path, Some(384)).await;
        assert!(result.is_err());
        let msg = result.expect_err("should fail").to_string();
        assert!(
            msg.contains("dimension mismatch"),
            "expected dimension mismatch error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn connect_recovers_dim_from_schema_when_metadata_missing() {
        // Simulate a legacy DB: chunks table exists with F32_BLOB(1024) but the
        // embedding_dim metadata row was never written.
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let (db, conn) = connect(&db_path, Some(TEST_DIM))
            .await
            .expect("initial connect");
        conn.execute("DELETE FROM metadata WHERE key = 'embedding_dim'", ())
            .await
            .expect("delete metadata");
        drop(conn);
        drop(db);

        // Reconnect with the matching dim -- should recover and succeed.
        let (_db2, conn2) = connect(&db_path, Some(TEST_DIM))
            .await
            .expect("connect should recover dim from schema");
        assert_eq!(
            stored_dim(&conn2).await.expect("query failed"),
            Some(TEST_DIM),
            "metadata should be repopulated after recovery"
        );
    }

    #[tokio::test]
    async fn connect_fails_on_dim_mismatch_with_legacy_db() {
        // Legacy DB: chunks F32_BLOB(1024), no metadata. Connecting with a different
        // provider dim must fail rather than silently corrupting the schema.
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let (db, conn) = connect(&db_path, Some(TEST_DIM))
            .await
            .expect("initial connect");
        conn.execute("DELETE FROM metadata WHERE key = 'embedding_dim'", ())
            .await
            .expect("delete metadata");
        drop(conn);
        drop(db);

        let result = connect(&db_path, Some(384)).await;
        assert!(result.is_err());
        let msg = result.expect_err("should fail").to_string();
        assert!(
            msg.contains("dimension mismatch"),
            "expected dimension mismatch error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn connect_creates_schema() {
        let (_dir, _db, conn) = test_db().await;
        let mut rows = conn
            .query(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
                (),
            )
            .await
            .expect("query failed");

        let mut tables = Vec::new();
        while let Some(row) = rows.next().await.expect("row failed") {
            let name: String = row.get(0).expect("get failed");
            tables.push(name);
        }

        assert!(tables.contains(&"notes".to_owned()));
        assert!(tables.contains(&"chunks".to_owned()));
        assert!(tables.contains(&"failed_files".to_owned()));
    }

    #[tokio::test]
    async fn all_note_hashes_empty_on_fresh_db() {
        let (_dir, _db, conn) = test_db().await;
        let hashes = all_note_hashes(&conn).await.expect("query failed");
        assert!(hashes.is_empty());
    }

    #[tokio::test]
    async fn note_hash_returns_none_for_missing_note() {
        let (_dir, _db, conn) = test_db().await;
        let hash = note_hash(&conn, "nonexistent.md")
            .await
            .expect("query failed");
        assert!(hash.is_none());
    }

    #[tokio::test]
    async fn note_hash_returns_hash_for_existing_note() {
        let (_dir, _db, conn) = test_db().await;
        let chunks = make_chunks(&["content"]);
        upsert_note(&conn, "note.md", "abc123", &chunks)
            .await
            .expect("upsert failed");
        let hash = note_hash(&conn, "note.md").await.expect("query failed");
        assert_eq!(hash, Some("abc123".to_owned()));
    }

    #[tokio::test]
    async fn failed_file_helpers_list_sorted_failures_and_clear_all() {
        let (_dir, _db, conn) = test_db().await;
        record_failed_file(&conn, "zebra.md", "hash-z", "zebra error")
            .await
            .expect("record");
        record_failed_file(&conn, "alpha.md", "hash-a", "alpha error")
            .await
            .expect("record");
        assert_eq!(
            failed_files(&conn).await.expect("failures"),
            vec![
                ("alpha.md".to_owned(), "alpha error".to_owned()),
                ("zebra.md".to_owned(), "zebra error".to_owned()),
            ]
        );
        assert_eq!(
            failed_file_hash(&conn, "zebra.md").await.expect("hash"),
            Some("hash-z".to_owned())
        );
        assert_eq!(
            all_failed_file_hashes(&conn)
                .await
                .expect("hashes")
                .get("zebra.md"),
            Some(&"hash-z".to_owned())
        );
        clear_failed_files(&conn).await.expect("clear all");
        assert!(failed_files(&conn).await.expect("failures").is_empty());
    }

    #[tokio::test]
    async fn upsert_and_retrieve_hashes() {
        let (_dir, _db, conn) = test_db().await;
        let chunks = make_chunks(&["hello world"]);
        upsert_note(&conn, "note.md", "abc123", &chunks)
            .await
            .expect("upsert failed");

        let hashes = all_note_hashes(&conn).await.expect("query failed");
        assert_eq!(hashes.len(), 1);
        assert_eq!(hashes.get("note.md"), Some(&"abc123".to_owned()));
    }

    #[tokio::test]
    async fn upsert_replaces_existing_note() {
        let (_dir, _db, conn) = test_db().await;
        let chunks_v1 = make_chunks(&["version one"]);
        upsert_note(&conn, "note.md", "hash_v1", &chunks_v1)
            .await
            .expect("upsert failed");

        let chunks_v2 = make_chunks(&["version two"]);
        upsert_note(&conn, "note.md", "hash_v2", &chunks_v2)
            .await
            .expect("upsert failed");

        let hashes = all_note_hashes(&conn).await.expect("query failed");
        assert_eq!(hashes.len(), 1);
        assert_eq!(hashes.get("note.md"), Some(&"hash_v2".to_owned()));

        let chunks = all_chunks(&conn).await.expect("query failed");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].1.content, "version two");
    }

    #[tokio::test]
    async fn upsert_stores_multiple_chunks() {
        let (_dir, _db, conn) = test_db().await;
        let chunks = make_chunks(&["chunk one", "chunk two", "chunk three"]);
        upsert_note(&conn, "note.md", "abc", &chunks)
            .await
            .expect("upsert failed");

        let all = all_chunks(&conn).await.expect("query failed");
        assert_eq!(all.len(), 3);
        assert!(all.iter().all(|(path, _)| path == "note.md"));
    }

    #[tokio::test]
    async fn delete_note_removes_note_and_chunks() {
        let (_dir, _db, conn) = test_db().await;
        let chunks = make_chunks(&["some content"]);
        upsert_note(&conn, "note.md", "abc", &chunks)
            .await
            .expect("upsert failed");

        delete_note(&conn, "note.md").await.expect("delete failed");

        let hashes = all_note_hashes(&conn).await.expect("query failed");
        assert!(hashes.is_empty());

        let chunks = all_chunks(&conn).await.expect("query failed");
        assert!(chunks.is_empty());
    }

    #[tokio::test]
    async fn delete_nonexistent_note_is_not_an_error() {
        let (_dir, _db, conn) = test_db().await;
        let result = delete_note(&conn, "does_not_exist.md").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn document_chunks_returns_only_the_exact_path_in_chunk_order() {
        let (_dir, _db, conn) = test_db().await;
        upsert_note(
            &conn,
            "nested/note.md",
            "h1",
            &[
                (
                    PreparedChunk {
                        content: "second".to_owned(),
                        locator: None,
                    },
                    dummy_embedding(),
                ),
                (
                    PreparedChunk {
                        content: "third".to_owned(),
                        locator: Some("p. 3".to_owned()),
                    },
                    dummy_embedding(),
                ),
            ],
        )
        .await
        .expect("upsert");
        upsert_note(&conn, "other.md", "h2", &make_chunks(&["other"]))
            .await
            .expect("upsert");

        let chunks = document_chunks(&conn, "nested/note.md")
            .await
            .expect("read")
            .expect("present");
        assert_eq!(chunks[0].content, "second");
        assert_eq!(chunks[0].locator, None);
        assert_eq!(chunks[1].content, "third");
        assert_eq!(chunks[1].locator.as_deref(), Some("p. 3"));
        assert!(
            document_chunks(&conn, "missing.md")
                .await
                .expect("read")
                .is_none()
        );

        conn.execute(
            "INSERT INTO notes (path, content_hash, updated_at) VALUES (?1, 'empty', unixepoch())",
            ["empty.md"],
        )
        .await
        .expect("empty note");
        assert_eq!(
            document_chunks(&conn, "empty.md").await.expect("read"),
            Some(vec![])
        );
    }

    #[tokio::test]
    async fn all_chunks_returns_all_paths_and_content() {
        let (_dir, _db, conn) = test_db().await;
        upsert_note(&conn, "a.md", "h1", &make_chunks(&["alpha"]))
            .await
            .expect("upsert failed");
        upsert_note(&conn, "b.md", "h2", &make_chunks(&["beta"]))
            .await
            .expect("upsert failed");

        let chunks = all_chunks(&conn).await.expect("query failed");
        assert_eq!(chunks.len(), 2);

        let paths: Vec<&str> = chunks.iter().map(|(p, _)| p.as_str()).collect();
        assert!(paths.contains(&"a.md"));
        assert!(paths.contains(&"b.md"));
    }

    #[test]
    fn decode_embedding_converts_known_bytes() {
        #[allow(clippy::cast_precision_loss)]
        let values: Vec<f32> = (0..TEST_DIM).map(|i| i as f32 * 0.1).collect();
        let blob: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let decoded = decode_embedding(&blob).expect("decode failed");
        assert_eq!(decoded.len(), TEST_DIM);
        for (a, b) in decoded.iter().zip(values.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn decode_embedding_rejects_non_aligned_size() {
        let blob = vec![0u8; 5];
        assert!(decode_embedding(&blob).is_err());
    }

    #[test]
    fn decode_embedding_accepts_any_aligned_size() {
        let blob = vec![0u8; 12]; // 3 floats
        let decoded = decode_embedding(&blob).expect("decode failed");
        assert_eq!(decoded.len(), 3);
    }

    #[tokio::test]
    async fn search_semantic_returns_stored_locator() {
        let (dir, _db, conn) = test_db().await;
        let chunk = PreparedChunk {
            content: "test content".to_owned(),
            locator: Some("Heading > Details".to_owned()),
        };
        upsert_note(&conn, "note.md", "abc", &[(chunk, vec![1.0; TEST_DIM])])
            .await
            .expect("upsert failed");

        let source = DbSemanticSource::new(conn.clone(), test_vector(&conn, &dir).await);
        let results = source
            .search_semantic(&vec![1.0; TEST_DIM], 10)
            .await
            .expect("search failed");
        assert_eq!(results[0].locator.as_deref(), Some("Heading > Details"));
    }

    #[tokio::test]
    async fn search_semantic_returns_results() {
        let (dir, _db, conn) = test_db().await;
        let embedding = vec![1.0; TEST_DIM];
        let chunks = vec![("test content".to_owned(), embedding)];
        upsert_note(&conn, "note.md", "abc", &chunks)
            .await
            .expect("upsert failed");

        let query_embedding = vec![1.0; TEST_DIM];
        let source = DbSemanticSource::new(conn.clone(), test_vector(&conn, &dir).await);
        let results = source
            .search_semantic(&query_embedding, 10)
            .await
            .expect("search failed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "note.md");
        assert_eq!(results[0].snippet, "test content");
    }

    #[tokio::test]
    async fn search_semantic_deduplicates_by_path() {
        let (dir, _db, conn) = test_db().await;
        let embedding = vec![1.0; TEST_DIM];
        let chunks = vec![
            ("chunk one".to_owned(), embedding.clone()),
            ("chunk two".to_owned(), embedding),
        ];
        upsert_note(&conn, "note.md", "abc", &chunks)
            .await
            .expect("upsert failed");

        let query_embedding = vec![1.0; TEST_DIM];
        let source = DbSemanticSource::new(conn.clone(), test_vector(&conn, &dir).await);
        let results = source
            .search_semantic(&query_embedding, 10)
            .await
            .expect("search failed");

        assert_eq!(results.len(), 1, "should deduplicate chunks from same path");
    }

    #[tokio::test]
    async fn search_semantic_respects_limit() {
        let (dir, _db, conn) = test_db().await;
        for i in 0..5 {
            let embedding = vec![1.0; TEST_DIM];
            let chunks = vec![(format!("content {i}"), embedding)];
            upsert_note(&conn, &format!("note{i}.md"), &format!("h{i}"), &chunks)
                .await
                .expect("upsert failed");
        }

        let query_embedding = vec![1.0; TEST_DIM];
        let source = DbSemanticSource::new(conn.clone(), test_vector(&conn, &dir).await);
        let results = source
            .search_semantic(&query_embedding, 2)
            .await
            .expect("search failed");

        assert!(results.len() <= 2, "should respect limit");
    }

    #[tokio::test]
    async fn chunk_embeddings_for_path_returns_embeddings_in_order() {
        let (_dir, _db, conn) = test_db().await;
        let mut emb_a = vec![1.0_f32; TEST_DIM];
        let mut emb_b = vec![2.0_f32; 1024];
        emb_a[0] = 0.1;
        emb_b[0] = 0.2;
        let chunks = vec![
            ("chunk one".to_owned(), emb_a),
            ("chunk two".to_owned(), emb_b),
        ];
        upsert_note(&conn, "note.md", "abc", &chunks)
            .await
            .expect("upsert failed");

        let embeddings = chunk_embeddings_for_path(&conn, "note.md")
            .await
            .expect("query failed");
        assert_eq!(embeddings.len(), 2);
        assert!((embeddings[0][0] - 0.1).abs() < f32::EPSILON);
        assert!((embeddings[1][0] - 0.2).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn chunk_embeddings_for_path_returns_empty_for_missing_path() {
        let (_dir, _db, conn) = test_db().await;
        let embeddings = chunk_embeddings_for_path(&conn, "nonexistent.md")
            .await
            .expect("query failed");
        assert!(embeddings.is_empty());
    }

    #[tokio::test]
    async fn search_related_excludes_specified_path() {
        let (dir, _db, conn) = test_db().await;
        let embedding = vec![1.0; TEST_DIM];
        upsert_note(
            &conn,
            "a.md",
            "h1",
            &[("content a".to_owned(), embedding.clone())],
        )
        .await
        .expect("upsert failed");
        upsert_note(
            &conn,
            "b.md",
            "h2",
            &[("content b".to_owned(), embedding.clone())],
        )
        .await
        .expect("upsert failed");

        let source = DbRelatedSearchSource::new(conn.clone(), test_vector(&conn, &dir).await);
        let results = source
            .search_related(&embedding, "a.md", 10)
            .await
            .expect("search failed");

        assert!(
            results.iter().all(|r| r.path != "a.md"),
            "should exclude the queried path"
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "b.md");
    }

    #[tokio::test]
    async fn search_related_finds_results_when_excluded_note_dominates_candidates() {
        // Reproduce: excluded note has many chunks that fill the initial candidate
        // pool, leaving no room for other documents. The adaptive loop must expand K
        // until the other document appears.
        let (dir, _db, conn) = test_db().await;
        let embedding = vec![1.0; TEST_DIM];

        // "source.md" has 12 chunks -- more than limit(1) * 5 = 5 initial candidates.
        let source_chunks: Vec<(String, Vec<f32>)> = (0..12)
            .map(|i| (format!("source chunk {i}"), embedding.clone()))
            .collect();
        upsert_note(&conn, "source.md", "h_src", &source_chunks)
            .await
            .expect("upsert failed");

        upsert_note(
            &conn,
            "other.md",
            "h_other",
            &[("other content".to_owned(), embedding.clone())],
        )
        .await
        .expect("upsert failed");

        let source = DbRelatedSearchSource::new(conn.clone(), test_vector(&conn, &dir).await);
        let results = source
            .search_related(&embedding, "source.md", 1)
            .await
            .expect("search failed");

        assert_eq!(
            results.len(),
            1,
            "should find the other document despite excluded note dominating candidates"
        );
        assert_eq!(results[0].path, "other.md");
    }

    #[tokio::test]
    async fn search_related_deduplicates_by_path() {
        let (dir, _db, conn) = test_db().await;
        let embedding = vec![1.0; TEST_DIM];
        upsert_note(
            &conn,
            "a.md",
            "h1",
            &[("content a".to_owned(), embedding.clone())],
        )
        .await
        .expect("upsert failed");
        upsert_note(
            &conn,
            "b.md",
            "h2",
            &[
                ("chunk 1".to_owned(), embedding.clone()),
                ("chunk 2".to_owned(), embedding.clone()),
            ],
        )
        .await
        .expect("upsert failed");

        let source = DbRelatedSearchSource::new(conn.clone(), test_vector(&conn, &dir).await);
        let results = source
            .search_related(&embedding, "a.md", 10)
            .await
            .expect("search failed");

        assert_eq!(results.len(), 1, "should deduplicate chunks from same path");
    }
}
