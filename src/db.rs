use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use anyhow::{Context, bail};
use libsql::Connection;

const BYTES_PER_F32: usize = 4;

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
}

pub struct RelatedResult {
    pub path: String,
    pub similarity: f64,
}

pub async fn connect(
    db_path: &Path,
    expected_dim: Option<usize>,
) -> anyhow::Result<(libsql::Database, Connection)> {
    let db = libsql::Builder::new_local(db_path).build().await?;
    let conn = db.connect()?;
    init_schema(&conn, expected_dim).await?;
    Ok((db, conn))
}

/// Delete the database file so the next `connect` starts with a clean slate.
///
/// Used by `reindex` when a dimension mismatch would otherwise prevent it from
/// running: the command's purpose is to rebuild the index, so resetting is
/// safe and expected.
pub fn reset(db_path: &Path) -> anyhow::Result<()> {
    if db_path.exists() {
        std::fs::remove_file(db_path)
            .with_context(|| format!("removing database at {}", db_path.display()))?;
    }
    Ok(())
}

async fn init_schema(conn: &Connection, expected_dim: Option<usize>) -> anyhow::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS notes (
            path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        );",
    )
    .await?;

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
            embedding F32_BLOB({dim})
        )"
    );
    conn.execute(&create_chunks, ()).await?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)",
        (),
    )
    .await?;

    let has_vector_idx = conn
        .query("SELECT 1 FROM sqlite_master WHERE name='chunks_idx'", ())
        .await?
        .next()
        .await?
        .is_some();

    if !has_vector_idx {
        conn.execute(
            "CREATE INDEX chunks_idx ON chunks(libsql_vector_idx(embedding, 'metric=cosine'))",
            (),
        )
        .await?;
    }

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

pub async fn upsert_note(
    conn: &Connection,
    path: &str,
    hash: &str,
    chunks: &[(String, Vec<f32>)],
) -> anyhow::Result<()> {
    let tx = conn.transaction().await?;

    tx.execute("DELETE FROM chunks WHERE path = ?1", [path])
        .await?;

    tx.execute(
        "INSERT OR REPLACE INTO notes (path, content_hash, updated_at) VALUES (?1, ?2, unixepoch())",
        [path, hash],
    )
    .await?;

    for (i, (content, embedding)) in chunks.iter().enumerate() {
        let embedding_json = serde_json::to_string(embedding)?;
        let chunk_index = i64::try_from(i).context("chunk index exceeds i64 range")?;

        tx.execute(
            "INSERT INTO chunks (path, chunk_index, content, embedding) VALUES (?1, ?2, ?3, vector32(?4))",
            libsql::params![path, chunk_index, content.as_str(), embedding_json],
        )
        .await?;
    }

    tx.commit().await?;
    Ok(())
}

pub async fn delete_note(conn: &Connection, path: &str) -> anyhow::Result<()> {
    let tx = conn.transaction().await?;
    tx.execute("DELETE FROM chunks WHERE path = ?1", [path])
        .await?;
    tx.execute("DELETE FROM notes WHERE path = ?1", [path])
        .await?;
    tx.commit().await?;
    Ok(())
}

pub async fn all_chunks(conn: &Connection) -> anyhow::Result<Vec<(String, String)>> {
    let mut rows = conn.query("SELECT path, content FROM chunks", ()).await?;
    let mut results = Vec::new();
    while let Some(row) = rows.next().await? {
        results.push((row.get(0)?, row.get(1)?));
    }
    Ok(results)
}

pub async fn search_semantic(
    conn: &Connection,
    query_embedding: &[f32],
    limit: usize,
) -> anyhow::Result<Vec<SearchResult>> {
    let embedding_json = serde_json::to_string(query_embedding)?;

    let mut rows = conn
        .query(
            "SELECT c.path, c.content, vector_distance_cos(c.embedding, vector32(?1)) AS distance
             FROM vector_top_k('chunks_idx', vector32(?1), ?2) AS v
             JOIN chunks c ON c.rowid = v.id
             ORDER BY distance ASC",
            libsql::params![
                embedding_json,
                i64::try_from(limit.saturating_mul(3)).unwrap_or(i64::MAX)
            ],
        )
        .await?;

    let mut seen = HashSet::new();
    let mut results = Vec::new();

    while let Some(row) = rows.next().await? {
        let path: String = row.get(0)?;
        let content: String = row.get(1)?;

        if seen.contains(&path) {
            continue;
        }
        seen.insert(path.clone());
        results.push(SearchResult {
            path,
            snippet: content,
        });
        if results.len() >= limit {
            break;
        }
    }

    Ok(results)
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

pub async fn search_related(
    conn: &Connection,
    query_embedding: &[f32],
    exclude_path: &str,
    limit: usize,
) -> anyhow::Result<Vec<RelatedResult>> {
    let embedding_json = serde_json::to_string(query_embedding)?;

    // Start with a modest candidate pool and double until we have enough distinct
    // non-excluded results or we hit the index ceiling. The excluded note's chunks
    // can dominate the top-K when it is long, so a fixed small multiplier silently
    // drops valid related documents.
    let mut k = limit.saturating_mul(5).max(20);

    loop {
        let k_param = i64::try_from(k).unwrap_or(i64::MAX);
        let mut rows = conn
            .query(
                "SELECT c.path, vector_distance_cos(c.embedding, vector32(?1)) AS distance
                 FROM vector_top_k('chunks_idx', vector32(?1), ?2) AS v
                 JOIN chunks c ON c.rowid = v.id
                 ORDER BY distance ASC",
                libsql::params![embedding_json.clone(), k_param],
            )
            .await?;

        let mut seen = HashSet::new();
        let mut results = Vec::new();
        let mut total_candidates = 0usize;

        while let Some(row) = rows.next().await? {
            total_candidates += 1;
            let path: String = row.get(0)?;
            let distance: f64 = row.get(1)?;

            if path == exclude_path || seen.contains(&path) {
                continue;
            }
            seen.insert(path.clone());
            results.push(RelatedResult {
                path,
                similarity: 1.0 - distance,
            });
            if results.len() >= limit {
                break;
            }
        }

        let exhausted_index = total_candidates < k;
        if results.len() >= limit || exhausted_index || k >= RELATED_MAX_K {
            return Ok(results);
        }

        k = k.saturating_mul(2).min(RELATED_MAX_K);
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

    fn dummy_embedding() -> Vec<f32> {
        vec![0.0; TEST_DIM]
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
        assert_eq!(chunks[0].1, "version two");
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
    async fn search_semantic_returns_results() {
        let (_dir, _db, conn) = test_db().await;
        let embedding = vec![1.0; TEST_DIM];
        let chunks = vec![("test content".to_owned(), embedding)];
        upsert_note(&conn, "note.md", "abc", &chunks)
            .await
            .expect("upsert failed");

        let query_embedding = vec![1.0; TEST_DIM];
        let results = search_semantic(&conn, &query_embedding, 10)
            .await
            .expect("search failed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "note.md");
        assert_eq!(results[0].snippet, "test content");
    }

    #[tokio::test]
    async fn search_semantic_deduplicates_by_path() {
        let (_dir, _db, conn) = test_db().await;
        let embedding = vec![1.0; TEST_DIM];
        let chunks = vec![
            ("chunk one".to_owned(), embedding.clone()),
            ("chunk two".to_owned(), embedding),
        ];
        upsert_note(&conn, "note.md", "abc", &chunks)
            .await
            .expect("upsert failed");

        let query_embedding = vec![1.0; TEST_DIM];
        let results = search_semantic(&conn, &query_embedding, 10)
            .await
            .expect("search failed");

        assert_eq!(results.len(), 1, "should deduplicate chunks from same path");
    }

    #[tokio::test]
    async fn search_semantic_respects_limit() {
        let (_dir, _db, conn) = test_db().await;
        for i in 0..5 {
            let embedding = vec![1.0; TEST_DIM];
            let chunks = vec![(format!("content {i}"), embedding)];
            upsert_note(&conn, &format!("note{i}.md"), &format!("h{i}"), &chunks)
                .await
                .expect("upsert failed");
        }

        let query_embedding = vec![1.0; TEST_DIM];
        let results = search_semantic(&conn, &query_embedding, 2)
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
        let (_dir, _db, conn) = test_db().await;
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

        let results = search_related(&conn, &embedding, "a.md", 10)
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
        let (_dir, _db, conn) = test_db().await;
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

        let results = search_related(&conn, &embedding, "source.md", 1)
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
        let (_dir, _db, conn) = test_db().await;
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

        let results = search_related(&conn, &embedding, "a.md", 10)
            .await
            .expect("search failed");

        assert_eq!(results.len(), 1, "should deduplicate chunks from same path");
    }
}
