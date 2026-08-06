use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use libsql::Connection;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use crate::db;

const CONNECTIVITY: usize = 16;
const INITIAL_CAPACITY: usize = 256;

pub struct VectorIndex {
    path: PathBuf,
    index: Index,
}

impl VectorIndex {
    pub async fn open_or_rebuild(conn: &Connection, path: PathBuf) -> anyhow::Result<Self> {
        let state = db::chunk_index_state(conn).await?;
        if let Some(index) = Self::open_current(conn, &path, state).await? {
            return Ok(index);
        }
        Self::rebuild(conn, path).await
    }

    async fn open_current(
        conn: &Connection,
        path: &Path,
        state: (usize, i64),
    ) -> anyhow::Result<Option<Self>> {
        let Some(recorded) = db::vector_index_state(conn).await? else {
            return Ok(None);
        };
        if recorded != state || !path.exists() {
            return Ok(None);
        }
        match Index::restore(&path.to_string_lossy()) {
            Ok(index) if index.size() == state.0 => Ok(Some(Self {
                path: path.to_owned(),
                index,
            })),
            Ok(_) | Err(_) => Ok(None),
        }
    }

    pub async fn rebuild(conn: &Connection, path: PathBuf) -> anyhow::Result<Self> {
        let state = db::chunk_index_state(conn).await?;
        let embeddings = db::all_chunk_embeddings_with_ids(conn).await?;
        let dimensions = match embeddings.first() {
            Some((_, embedding)) => embedding.len(),
            None => db::embedding_dimension(conn)
                .await?
                .context("missing embedding dimension")?,
        };
        anyhow::ensure!(dimensions > 0, "invalid vector dimensions");
        let index = Self::create_index(dimensions)?;
        if !embeddings.is_empty() {
            index
                .reserve(embeddings.len())
                .context("failed to reserve vector index")?;
            for (id, embedding) in embeddings {
                index
                    .add(u64::try_from(id).context("negative chunk id")?, &embedding)
                    .context("failed to add vector")?;
            }
        }
        let result = Self { path, index };
        result.save(conn, state).await?;
        Ok(result)
    }

    fn create_index(dimensions: usize) -> anyhow::Result<Index> {
        Index::new(&IndexOptions {
            dimensions,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F16,
            connectivity: CONNECTIVITY,
            ..IndexOptions::default()
        })
        .context("failed to create vector index")
    }

    pub fn add(&self, id: i64, embedding: &[f32]) -> anyhow::Result<()> {
        let required = self.index.size().saturating_add(1);
        if required > self.index.capacity() {
            let capacity = self
                .index
                .capacity()
                .max(INITIAL_CAPACITY)
                .saturating_mul(2)
                .max(required);
            self.index
                .reserve(capacity)
                .context("failed to grow vector index")?;
        }
        self.index
            .add(u64::try_from(id).context("negative chunk id")?, embedding)
            .context("failed to add vector")
    }

    pub fn remove(&self, id: i64) -> anyhow::Result<()> {
        self.index
            .remove(u64::try_from(id).context("negative chunk id")?)
            .context("failed to remove vector")?;
        Ok(())
    }

    pub fn search(&self, query: &[f32], count: usize) -> anyhow::Result<Vec<i64>> {
        let matches = self
            .index
            .search(query, count)
            .context("vector search failed")?;
        matches
            .keys
            .into_iter()
            .map(|key| i64::try_from(key).context("vector key exceeds i64"))
            .collect()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.index.size()
    }

    #[cfg(test)]
    pub fn contains(&self, id: i64) -> bool {
        u64::try_from(id).is_ok_and(|key| self.index.contains(key))
    }

    pub async fn save(&self, conn: &Connection, state: (usize, i64)) -> anyhow::Result<()> {
        let parent = self.path.parent().context("vector index has no parent")?;
        fs::create_dir_all(parent)?;
        let temporary = parent.join(format!(
            ".{}.tmp",
            self.path.file_name().unwrap_or_default().to_string_lossy()
        ));
        self.index
            .save(&temporary.to_string_lossy())
            .context("failed to save vector index")?;
        fs::rename(&temporary, &self.path).context("failed to atomically replace vector index")?;
        db::store_vector_index_state(conn, state).await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;
    use crate::document::PreparedChunk;

    const CORPUS_DIM: usize = 128;
    const OVER_FETCH: usize = 3;

    struct Rng(u64);

    impl Rng {
        fn next_unit(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            f32::from(u16::try_from(self.0 >> 48).expect("16 bits")) / 32_768.0 - 1.0
        }

        fn vector(&mut self, dimensions: usize) -> Vec<f32> {
            (0..dimensions).map(|_| self.next_unit()).collect()
        }

        fn near(&mut self, centroid: &[f32], spread: f32) -> Vec<f32> {
            centroid
                .iter()
                .map(|value| value + self.next_unit() * spread)
                .collect()
        }
    }

    const CLUSTERS: usize = 40;
    const SPREAD: f32 = 0.35;

    fn cosine(left: &[f32], right: &[f32]) -> f64 {
        let dot: f64 = left
            .iter()
            .zip(right)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let left_norm: f64 = left.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        let right_norm: f64 = right.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
        dot / (left_norm.sqrt() * right_norm.sqrt())
    }

    async fn corpus_db(
        dimensions: usize,
        count: usize,
    ) -> (
        tempfile::TempDir,
        libsql::Database,
        Connection,
        Vec<Vec<f32>>,
    ) {
        let dir = tempfile::tempdir().expect("tempdir");
        let (database, conn) = db::connect(&dir.path().join("test.db"), Some(dimensions))
            .await
            .expect("connect");
        let mut rng = Rng(0x5EED);
        let centroids: Vec<Vec<f32>> = (0..CLUSTERS).map(|_| rng.vector(dimensions)).collect();
        let mut vectors = Vec::with_capacity(count);
        for index in 0..count {
            let embedding = rng.near(&centroids[index % CLUSTERS], SPREAD);
            let chunk = PreparedChunk {
                content: format!("chunk {index}"),
                locator: None,
            };
            db::upsert_note(
                &conn,
                &format!("note{index}.md"),
                "hash",
                &[(chunk, embedding.clone())],
            )
            .await
            .expect("upsert");
            vectors.push(embedding);
        }
        (dir, database, conn, vectors)
    }

    async fn chunk_ids(conn: &Connection, path: &str) -> Vec<i64> {
        db::chunk_embeddings_for_paths(conn, &[path.to_owned()])
            .await
            .expect("ids")
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    }

    fn exhaustive_top_paths(query: &[f32], vectors: &[Vec<f32>], limit: usize) -> Vec<String> {
        let mut scored: Vec<_> = vectors
            .iter()
            .enumerate()
            .map(|(index, candidate)| (cosine(query, candidate), index))
            .collect();
        scored.sort_by(|left, right| right.0.total_cmp(&left.0));
        scored
            .into_iter()
            .take(limit)
            .map(|(_, index)| format!("note{index}.md"))
            .collect()
    }

    #[tokio::test]
    async fn ranks_like_an_exhaustive_cosine_scan() {
        let (dir, _database, conn, vectors) = corpus_db(CORPUS_DIM, 2_000).await;
        let index = VectorIndex::open_or_rebuild(&conn, dir.path().join("vectors.usearch"))
            .await
            .expect("build index");

        let mut rng = Rng(0x00C0_FFEE);
        let centroids: Vec<Vec<f32>> = (0..CLUSTERS).map(|_| rng.vector(CORPUS_DIM)).collect();
        let queries = 20;
        let limit = 10;
        let mut hits = 0;
        for query_index in 0..queries {
            let query = rng.near(&centroids[query_index % CLUSTERS], SPREAD);
            let candidates = index.search(&query, limit * OVER_FETCH).expect("search");
            let found: HashSet<String> = db::search_semantic_candidates(&conn, &query, candidates)
                .await
                .expect("hydrate")
                .into_iter()
                .take(limit)
                .map(|result| result.path)
                .collect();
            hits += exhaustive_top_paths(&query, &vectors, limit)
                .iter()
                .filter(|path| found.contains(*path))
                .count();
        }

        let recall = f64::from(u32::try_from(hits).expect("hits"))
            / f64::from(u32::try_from(queries * limit).expect("total"));
        assert!(recall >= 0.98, "recall@{limit} was {recall}");
    }

    #[tokio::test]
    async fn drops_rowids_that_an_update_reallocated() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (_database, conn) = db::connect(&dir.path().join("test.db"), Some(CORPUS_DIM))
            .await
            .expect("connect");
        let mut rng = Rng(7);
        let index = VectorIndex::open_or_rebuild(&conn, dir.path().join("vectors.usearch"))
            .await
            .expect("index");

        let first = rng.vector(CORPUS_DIM);
        let original = PreparedChunk {
            content: "original".to_owned(),
            locator: None,
        };
        db::upsert_note(&conn, "note.md", "h1", &[(original, first.clone())])
            .await
            .expect("insert");
        let inserted = chunk_ids(&conn, "note.md").await;
        for id in &inserted {
            index.add(*id, &first).expect("add");
        }
        assert_eq!(index.len(), inserted.len());

        let second = rng.vector(CORPUS_DIM);
        let revised = PreparedChunk {
            content: "revised".to_owned(),
            locator: None,
        };
        for id in &inserted {
            index.remove(*id).expect("remove stale");
        }
        db::upsert_note(&conn, "note.md", "h2", &[(revised, second.clone())])
            .await
            .expect("update");
        let reinserted = chunk_ids(&conn, "note.md").await;
        for id in &reinserted {
            index.add(*id, &second).expect("add");
        }

        assert_ne!(
            inserted, reinserted,
            "update must reallocate rowids for this test to be meaningful"
        );
        for stale in &inserted {
            assert!(
                !index.contains(*stale),
                "rowid {stale} was reallocated by the update and must not remain indexed"
            );
        }
        assert_eq!(index.len(), reinserted.len());

        for id in &reinserted {
            index.remove(*id).expect("remove");
        }
        db::delete_note(&conn, "note.md").await.expect("delete");
        assert_eq!(index.len(), 0);
        assert_eq!(
            db::chunk_index_state(&conn).await.expect("state").0,
            index.len()
        );
    }

    #[tokio::test]
    async fn rebuilds_a_missing_index_without_embedding_anything() {
        let (dir, _database, conn, _vectors) = corpus_db(CORPUS_DIM, 32).await;
        let path = dir.path().join("vectors.usearch");
        let index = VectorIndex::open_or_rebuild(&conn, path.clone())
            .await
            .expect("build");
        assert_eq!(index.len(), 32);
        drop(index);
        std::fs::remove_file(&path).expect("remove index file");

        let before = db::all_chunk_embeddings_with_ids(&conn)
            .await
            .expect("before");
        let rebuilt = VectorIndex::open_or_rebuild(&conn, path.clone())
            .await
            .expect("rebuild");
        let after = db::all_chunk_embeddings_with_ids(&conn)
            .await
            .expect("after");

        assert_eq!(rebuilt.len(), 32);
        assert!(path.exists());
        assert_eq!(
            before, after,
            "rebuilding must reuse stored embeddings rather than recomputing them"
        );
    }

    #[tokio::test]
    async fn rebuilds_a_diverged_index_without_embedding_anything() {
        let (dir, _database, conn, _vectors) = corpus_db(CORPUS_DIM, 16).await;
        let path = dir.path().join("vectors.usearch");
        VectorIndex::open_or_rebuild(&conn, path.clone())
            .await
            .expect("build");

        let mut rng = Rng(99);
        let chunk = PreparedChunk {
            content: "added behind the index".to_owned(),
            locator: None,
        };
        db::upsert_note(&conn, "extra.md", "h", &[(chunk, rng.vector(CORPUS_DIM))])
            .await
            .expect("upsert");

        let before = db::all_chunk_embeddings_with_ids(&conn)
            .await
            .expect("before");
        let reopened = VectorIndex::open_or_rebuild(&conn, path)
            .await
            .expect("reopen");
        let after = db::all_chunk_embeddings_with_ids(&conn)
            .await
            .expect("after");

        assert_eq!(reopened.len(), 17);
        assert_eq!(
            db::chunk_index_state(&conn).await.expect("state").0,
            reopened.len()
        );
        assert_eq!(
            before, after,
            "rebuilding must reuse stored embeddings rather than recomputing them"
        );
    }

    #[tokio::test]
    async fn stores_each_vector_within_the_size_budget() {
        let count = 2_000;
        let dimensions = 1_024;
        let dir = tempfile::tempdir().expect("tempdir");
        let (_database, conn) = db::connect(&dir.path().join("test.db"), Some(dimensions))
            .await
            .expect("connect");
        let mut rng = Rng(11);
        let index = VectorIndex::create_index(dimensions).expect("index");
        index.reserve(count).expect("reserve");
        for id in 0..count {
            index
                .add(u64::try_from(id).expect("id"), &rng.vector(dimensions))
                .expect("add");
        }
        let stored = VectorIndex {
            path: dir.path().join("vectors.usearch"),
            index,
        };
        stored.save(&conn, (count, 0)).await.expect("save");

        let bytes = std::fs::metadata(stored.path()).expect("metadata").len();
        let per_vector = bytes / u64::try_from(count).expect("count");
        let projected_mib = per_vector * 46_000 / (1024 * 1024);
        assert!(
            projected_mib < 200,
            "{per_vector} bytes per vector projects to {projected_mib} MiB at 46k chunks"
        );
    }
}
