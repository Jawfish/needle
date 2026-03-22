use std::{collections::HashMap, path::Path};

use libsql::Connection;

pub struct SearchResult {
    pub path: String,
    pub snippet: String,
}

pub async fn connect(db_path: &Path) -> anyhow::Result<Connection> {
    let db = libsql::Builder::new_local(db_path).build().await?;
    let conn = db.connect()?;
    init_schema(&conn).await?;
    Ok(conn)
}

async fn init_schema(conn: &Connection) -> anyhow::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS notes (
            path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            updated_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL REFERENCES notes(path) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding F32_BLOB(1024)
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);

        DROP TABLE IF EXISTS chunks_fts;",
    )
    .await?;

    // Vector index creation is separate since it uses special syntax
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

pub async fn upsert_note(
    conn: &Connection,
    path: &str,
    hash: &str,
    chunks: &[(String, Vec<f32>)],
) -> anyhow::Result<()> {
    conn.execute("DELETE FROM chunks WHERE path = ?1", [path])
        .await?;

    conn.execute(
        "INSERT OR REPLACE INTO notes (path, content_hash, updated_at) VALUES (?1, ?2, unixepoch())",
        [path, hash],
    )
    .await?;

    for (i, (content, embedding)) in chunks.iter().enumerate() {
        let embedding_json = serde_json::to_string(embedding)?;

        conn.execute(
            "INSERT INTO chunks (path, chunk_index, content, embedding) VALUES (?1, ?2, ?3, vector32(?4))",
            libsql::params![
                path,
                i64::try_from(i).expect("chunk index exceeds i64"),
                content.as_str(),
                embedding_json
            ],
        )
        .await?;
    }

    Ok(())
}

pub async fn delete_note(conn: &Connection, path: &str) -> anyhow::Result<()> {
    conn.execute("DELETE FROM chunks WHERE path = ?1", [path])
        .await?;
    conn.execute("DELETE FROM notes WHERE path = ?1", [path])
        .await?;
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
            libsql::params![embedding_json, i64::try_from(limit * 3).unwrap_or(i64::MAX)],
        )
        .await?;

    let mut seen = HashMap::new();
    let mut results = Vec::new();

    while let Some(row) = rows.next().await? {
        let path: String = row.get(0)?;
        let content: String = row.get(1)?;

        if seen.contains_key(&path) {
            continue;
        }
        seen.insert(path.clone(), true);
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

#[cfg(test)]
mod tests {
    use super::*;

    async fn test_conn() -> Connection {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let conn = connect(&db_path).await.expect("connect failed");
        // Keep dir alive by leaking it (tests are short-lived)
        std::mem::forget(dir);
        conn
    }

    fn dummy_embedding() -> Vec<f32> {
        vec![0.0; 1024]
    }

    fn make_chunks(texts: &[&str]) -> Vec<(String, Vec<f32>)> {
        texts
            .iter()
            .map(|t| ((*t).to_owned(), dummy_embedding()))
            .collect()
    }

    #[tokio::test]
    async fn connect_creates_schema() {
        let conn = test_conn().await;
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
        let conn = test_conn().await;
        let hashes = all_note_hashes(&conn).await.expect("query failed");
        assert!(hashes.is_empty());
    }

    #[tokio::test]
    async fn upsert_and_retrieve_hashes() {
        let conn = test_conn().await;
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
        let conn = test_conn().await;
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
        let conn = test_conn().await;
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
        let conn = test_conn().await;
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
        let conn = test_conn().await;
        let result = delete_note(&conn, "does_not_exist.md").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn all_chunks_returns_all_paths_and_content() {
        let conn = test_conn().await;
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

    #[tokio::test]
    async fn search_semantic_returns_results() {
        let conn = test_conn().await;
        let embedding = vec![1.0; 1024];
        let chunks = vec![("test content".to_owned(), embedding)];
        upsert_note(&conn, "note.md", "abc", &chunks)
            .await
            .expect("upsert failed");

        let query_embedding = vec![1.0; 1024];
        let results = search_semantic(&conn, &query_embedding, 10)
            .await
            .expect("search failed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "note.md");
        assert_eq!(results[0].snippet, "test content");
    }

    #[tokio::test]
    async fn search_semantic_deduplicates_by_path() {
        let conn = test_conn().await;
        let embedding = vec![1.0; 1024];
        let chunks = vec![
            ("chunk one".to_owned(), embedding.clone()),
            ("chunk two".to_owned(), embedding),
        ];
        upsert_note(&conn, "note.md", "abc", &chunks)
            .await
            .expect("upsert failed");

        let query_embedding = vec![1.0; 1024];
        let results = search_semantic(&conn, &query_embedding, 10)
            .await
            .expect("search failed");

        assert_eq!(results.len(), 1, "should deduplicate chunks from same path");
    }

    #[tokio::test]
    async fn search_semantic_respects_limit() {
        let conn = test_conn().await;
        for i in 0..5 {
            let embedding = vec![1.0; 1024];
            let chunks = vec![(format!("content {i}"), embedding)];
            upsert_note(&conn, &format!("note{i}.md"), &format!("h{i}"), &chunks)
                .await
                .expect("upsert failed");
        }

        let query_embedding = vec![1.0; 1024];
        let results = search_semantic(&conn, &query_embedding, 2)
            .await
            .expect("search failed");

        assert!(results.len() <= 2, "should respect limit");
    }
}
