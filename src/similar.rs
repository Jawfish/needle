use std::collections::HashMap;

use anyhow::bail;
use libsql::Connection;

use crate::db;

const EMBEDDING_DIM: usize = 1024;
const BYTES_PER_F32: usize = 4;
const EXPECTED_BLOB_SIZE: usize = EMBEDDING_DIM * BYTES_PER_F32;

pub struct SimilarPair {
    pub similarity: f64,
    pub path_a: String,
    pub path_b: String,
}

fn decode_embedding(blob: &[u8]) -> anyhow::Result<Vec<f32>> {
    if blob.len() != EXPECTED_BLOB_SIZE {
        bail!(
            "expected {EXPECTED_BLOB_SIZE} bytes for {EMBEDDING_DIM}-dim f32 embedding, got {}",
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

fn average_embeddings(embeddings: &[Vec<f32>]) -> Option<Vec<f32>> {
    if embeddings.is_empty() {
        return None;
    }
    let mut acc = vec![0.0_f64; EMBEDDING_DIM];
    for emb in embeddings {
        for (a, &v) in acc.iter_mut().zip(emb.iter()) {
            *a += f64::from(v);
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let count = embeddings.len() as f64;
    #[allow(clippy::cast_possible_truncation)]
    let result = acc.iter().map(|&v| (v / count) as f32).collect();
    Some(result)
}

fn normalize(v: &mut [f32]) {
    let norm_sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let norm = norm_sq.sqrt();
    if norm > f64::EPSILON {
        for x in v.iter_mut() {
            #[allow(clippy::cast_possible_truncation)]
            let normalized = (f64::from(*x) / norm) as f32;
            *x = normalized;
        }
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| f64::from(x) * f64::from(y))
        .sum()
}

fn find_similar_pairs(
    docs: &[(String, Vec<f32>)],
    threshold: f64,
    limit: Option<usize>,
) -> Vec<SimilarPair> {
    let mut pairs = Vec::new();
    for i in 0..docs.len() {
        for j in (i + 1)..docs.len() {
            let sim = cosine_similarity(&docs[i].1, &docs[j].1);
            if sim >= threshold {
                pairs.push(SimilarPair {
                    similarity: sim,
                    path_a: docs[i].0.clone(),
                    path_b: docs[j].0.clone(),
                });
            }
        }
    }
    pairs.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(n) = limit {
        pairs.truncate(n);
    }
    pairs
}

pub async fn find_similar(
    conn: &Connection,
    threshold: f64,
    limit: Option<usize>,
) -> anyhow::Result<Vec<SimilarPair>> {
    let raw_chunks = db::all_chunk_embeddings(conn).await?;

    let mut by_path: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    for (path, blob) in &raw_chunks {
        let embedding = decode_embedding(blob)?;
        by_path.entry(path.clone()).or_default().push(embedding);
    }

    let mut docs: Vec<(String, Vec<f32>)> = by_path
        .into_iter()
        .filter_map(|(path, embeddings)| average_embeddings(&embeddings).map(|avg| (path, avg)))
        .collect();

    for (_, emb) in &mut docs {
        normalize(emb);
    }

    docs.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(find_similar_pairs(&docs, threshold, limit))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_blob(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn decode_embedding_converts_known_bytes() {
        #[allow(clippy::cast_precision_loss)]
        let values: Vec<f32> = (0..EMBEDDING_DIM).map(|i| i as f32 * 0.1).collect();
        let blob = make_blob(&values);
        let decoded = decode_embedding(&blob).expect("decode failed");
        assert_eq!(decoded.len(), EMBEDDING_DIM);
        for (a, b) in decoded.iter().zip(values.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn decode_embedding_rejects_wrong_size() {
        let blob = vec![0u8; 100];
        assert!(decode_embedding(&blob).is_err());
    }

    #[test]
    fn decode_embedding_accepts_exact_4096_bytes() {
        let blob = vec![0u8; EXPECTED_BLOB_SIZE];
        let decoded = decode_embedding(&blob).expect("decode failed");
        assert_eq!(decoded.len(), EMBEDDING_DIM);
    }

    #[test]
    fn average_embeddings_returns_none_for_empty_slice() {
        assert!(average_embeddings(&[]).is_none());
    }

    #[test]
    fn average_of_single_vector_returns_that_vector() {
        let v = vec![1.0_f32; EMBEDDING_DIM];
        let avg = average_embeddings(std::slice::from_ref(&v)).expect("should return Some");
        for (a, b) in avg.iter().zip(v.iter()) {
            assert!((a - b).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn average_of_two_vectors_returns_midpoint() {
        let a = vec![0.0_f32; EMBEDDING_DIM];
        let b = vec![2.0_f32; EMBEDDING_DIM];
        let avg = average_embeddings(&[a, b]).expect("should return Some");
        for &v in &avg {
            assert!((v - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn normalize_produces_unit_vector() {
        let mut v = vec![3.0_f32; EMBEDDING_DIM];
        normalize(&mut v);
        let norm_sq: f64 = v.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-6,
            "norm squared should be ~1.0, got {norm_sq}"
        );
    }

    #[test]
    fn normalize_handles_zero_vector() {
        let mut v = vec![0.0_f32; EMBEDDING_DIM];
        normalize(&mut v);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn cosine_similarity_of_identical_normalized_vectors_is_one() {
        let mut v = vec![1.0_f32; EMBEDDING_DIM];
        normalize(&mut v);
        let sim = cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "identical normalized vectors should have similarity ~1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_of_orthogonal_vectors_is_zero() {
        let mut a = vec![0.0_f32; EMBEDDING_DIM];
        let mut b = vec![0.0_f32; EMBEDDING_DIM];
        a[0] = 1.0;
        b[1] = 1.0;
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "orthogonal vectors should have similarity ~0.0, got {sim}"
        );
    }

    #[test]
    fn find_similar_pairs_filters_by_threshold() {
        let mut high_a = vec![1.0_f32; EMBEDDING_DIM];
        let mut high_b = vec![1.0_f32; EMBEDDING_DIM];
        let mut different = vec![0.0_f32; EMBEDDING_DIM];
        different[0] = 1.0;

        normalize(&mut high_a);
        normalize(&mut high_b);

        let docs = vec![
            ("a.md".to_owned(), high_a),
            ("b.md".to_owned(), high_b),
            ("c.md".to_owned(), different),
        ];

        let pairs = find_similar_pairs(&docs, 0.9, None);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].path_a, "a.md");
        assert_eq!(pairs[0].path_b, "b.md");
    }

    #[test]
    fn find_similar_pairs_sorts_descending() {
        let mut v1 = vec![1.0_f32; EMBEDDING_DIM];
        let mut v2 = vec![1.0_f32; EMBEDDING_DIM];
        v2[0] = 0.99;
        let mut v3 = vec![1.0_f32; EMBEDDING_DIM];
        v3[0] = 0.95;

        normalize(&mut v1);
        normalize(&mut v2);
        normalize(&mut v3);

        let docs = vec![
            ("a.md".to_owned(), v1),
            ("b.md".to_owned(), v2),
            ("c.md".to_owned(), v3),
        ];

        let pairs = find_similar_pairs(&docs, 0.0, None);
        for window in pairs.windows(2) {
            assert!(
                window[0].similarity >= window[1].similarity,
                "pairs should be sorted descending"
            );
        }
    }

    #[test]
    fn find_similar_pairs_respects_limit() {
        let mut v1 = vec![1.0_f32; EMBEDDING_DIM];
        let mut v2 = vec![1.0_f32; EMBEDDING_DIM];
        let mut v3 = vec![1.0_f32; EMBEDDING_DIM];
        normalize(&mut v1);
        normalize(&mut v2);
        normalize(&mut v3);

        let docs = vec![
            ("a.md".to_owned(), v1),
            ("b.md".to_owned(), v2),
            ("c.md".to_owned(), v3),
        ];

        let pairs = find_similar_pairs(&docs, 0.0, Some(1));
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn find_similar_pairs_returns_empty_for_single_document() {
        let mut v = vec![1.0_f32; EMBEDDING_DIM];
        normalize(&mut v);
        let docs = vec![("a.md".to_owned(), v)];
        let pairs = find_similar_pairs(&docs, 0.0, None);
        assert!(pairs.is_empty());
    }

    #[test]
    fn find_similar_pairs_returns_empty_for_no_documents() {
        let pairs = find_similar_pairs(&[], 0.0, None);
        assert!(pairs.is_empty());
    }

    #[tokio::test]
    async fn find_similar_with_test_database() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let db_path = dir.path().join("test.db");
        let (_db, conn) = db::connect(&db_path).await.expect("connect failed");

        let emb_a = vec![1.0_f32; EMBEDDING_DIM];
        let emb_b = vec![1.0_f32; EMBEDDING_DIM];
        let mut emb_c = vec![0.0_f32; EMBEDDING_DIM];
        emb_c[0] = 1.0;

        db::upsert_note(&conn, "a.md", "h1", &[("content a".to_owned(), emb_a)])
            .await
            .expect("upsert failed");
        db::upsert_note(&conn, "b.md", "h2", &[("content b".to_owned(), emb_b)])
            .await
            .expect("upsert failed");
        db::upsert_note(&conn, "c.md", "h3", &[("content c".to_owned(), emb_c)])
            .await
            .expect("upsert failed");

        let pairs = find_similar(&conn, 0.9, None).await.expect("find failed");
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].path_a, "a.md");
        assert_eq!(pairs[0].path_b, "b.md");
        assert!(pairs[0].similarity > 0.99);
    }
}
