use std::collections::{HashMap, HashSet, VecDeque};

use libsql::Connection;

use crate::db;

pub struct SimilarPair {
    pub similarity: f64,
    pub path_a: String,
    pub path_b: String,
}

fn average_embeddings(embeddings: &[Vec<f32>]) -> Option<Vec<f32>> {
    let dim = embeddings.first()?.len();
    let mut acc = vec![0.0_f64; dim];
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

fn sort_pairs_descending(pairs: &mut [SimilarPair]) {
    pairs.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn find_similar_pairs(
    docs: &[(String, Vec<f32>)],
    threshold: f64,
    limit: Option<usize>,
) -> Vec<SimilarPair> {
    let mut pairs = Vec::new();
    let mut effective_threshold = threshold;
    let compact_at = limit.map(|n| n.saturating_mul(2).max(1));

    for i in 0..docs.len() {
        for j in (i + 1)..docs.len() {
            let sim = cosine_similarity(&docs[i].1, &docs[j].1);
            if sim >= effective_threshold {
                pairs.push(SimilarPair {
                    similarity: sim,
                    path_a: docs[i].0.clone(),
                    path_b: docs[j].0.clone(),
                });
                if let Some(cap) = compact_at
                    && pairs.len() >= cap
                {
                    let n = limit.unwrap_or(cap);
                    sort_pairs_descending(&mut pairs);
                    pairs.truncate(n);
                    effective_threshold = pairs.last().map_or(threshold, |p| p.similarity);
                }
            }
        }
    }
    sort_pairs_descending(&mut pairs);
    if let Some(n) = limit {
        pairs.truncate(n);
    }
    pairs
}

pub struct SimilarGroup {
    pub paths: Vec<String>,
    pub pairs: Vec<SimilarPair>,
}

pub fn group_pairs(pairs: Vec<SimilarPair>) -> Vec<SimilarGroup> {
    let mut adjacency: HashMap<String, HashSet<String>> = HashMap::new();
    for pair in &pairs {
        adjacency
            .entry(pair.path_a.clone())
            .or_default()
            .insert(pair.path_b.clone());
        adjacency
            .entry(pair.path_b.clone())
            .or_default()
            .insert(pair.path_a.clone());
    }

    let mut path_to_component: HashMap<String, usize> = HashMap::new();
    let mut component_id = 0;

    for node in adjacency.keys() {
        if path_to_component.contains_key(node) {
            continue;
        }
        let mut queue = VecDeque::new();
        queue.push_back(node.clone());
        path_to_component.insert(node.clone(), component_id);

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = adjacency.get(&current) {
                for neighbor in neighbors {
                    if !path_to_component.contains_key(neighbor) {
                        path_to_component.insert(neighbor.clone(), component_id);
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }
        component_id += 1;
    }

    let mut group_map: HashMap<usize, (HashSet<String>, Vec<SimilarPair>)> = HashMap::new();
    for pair in pairs {
        let cid = path_to_component.get(&pair.path_a).copied().unwrap_or(0);
        let entry = group_map.entry(cid).or_default();
        entry.0.insert(pair.path_a.clone());
        entry.0.insert(pair.path_b.clone());
        entry.1.push(pair);
    }

    let mut groups: Vec<SimilarGroup> = group_map
        .into_values()
        .map(|(path_set, mut gpairs)| {
            gpairs.sort_by(|a, b| {
                b.similarity
                    .partial_cmp(&a.similarity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut paths: Vec<String> = path_set.into_iter().collect();
            paths.sort();
            SimilarGroup {
                paths,
                pairs: gpairs,
            }
        })
        .collect();

    groups.sort_by(|a, b| {
        b.paths.len().cmp(&a.paths.len()).then_with(|| {
            let max_a = a.pairs.first().map_or(0.0, |p| p.similarity);
            let max_b = b.pairs.first().map_or(0.0, |p| p.similarity);
            max_b
                .partial_cmp(&max_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    groups
}

pub async fn find_similar(
    conn: &Connection,
    threshold: f64,
    limit: usize,
) -> anyhow::Result<Vec<SimilarPair>> {
    let chunks = db::all_chunk_embeddings(conn).await?;

    let mut by_path: HashMap<String, Vec<Vec<f32>>> = HashMap::new();
    for (path, embedding) in chunks {
        by_path.entry(path).or_default().push(embedding);
    }

    let mut docs: Vec<(String, Vec<f32>)> = by_path
        .into_iter()
        .filter_map(|(path, embeddings)| average_embeddings(&embeddings).map(|avg| (path, avg)))
        .collect();

    for (_, emb) in &mut docs {
        normalize(emb);
    }

    docs.sort_by(|a, b| a.0.cmp(&b.0));

    Ok(find_similar_pairs(&docs, threshold, Some(limit)))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EMBEDDING_DIM: usize = 1024;

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

    fn make_pair(sim: f64, a: &str, b: &str) -> SimilarPair {
        SimilarPair {
            similarity: sim,
            path_a: a.to_owned(),
            path_b: b.to_owned(),
        }
    }

    #[test]
    fn group_pairs_empty_input_returns_empty() {
        let groups = group_pairs(vec![]);
        assert!(groups.is_empty());
    }

    #[test]
    fn group_pairs_single_pair_produces_one_group() {
        let groups = group_pairs(vec![make_pair(0.9, "a.md", "b.md")]);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].paths.len(), 2);
        assert_eq!(groups[0].pairs.len(), 1);
    }

    #[test]
    fn group_pairs_disconnected_pairs_form_separate_groups() {
        let pairs = vec![
            make_pair(0.9, "a.md", "b.md"),
            make_pair(0.85, "c.md", "d.md"),
        ];
        let groups = group_pairs(pairs);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn group_pairs_chain_forms_single_group() {
        let pairs = vec![
            make_pair(0.9, "a.md", "b.md"),
            make_pair(0.85, "b.md", "c.md"),
        ];
        let groups = group_pairs(pairs);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].paths.len(), 3);
        assert_eq!(groups[0].pairs.len(), 2);
    }

    #[test]
    fn group_pairs_three_pairwise_connected_docs_form_one_group() {
        let pairs = vec![
            make_pair(0.95, "a.md", "b.md"),
            make_pair(0.90, "a.md", "c.md"),
            make_pair(0.88, "b.md", "c.md"),
        ];
        let groups = group_pairs(pairs);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].paths.len(), 3);
        assert_eq!(groups[0].pairs.len(), 3);
    }

    #[test]
    fn group_pairs_sorted_by_size_then_max_similarity() {
        let pairs = vec![
            make_pair(0.99, "x.md", "y.md"),
            make_pair(0.90, "a.md", "b.md"),
            make_pair(0.85, "b.md", "c.md"),
        ];
        let groups = group_pairs(pairs);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].paths.len(), 3, "larger group should come first");
        assert_eq!(groups[1].paths.len(), 2);
    }

    #[test]
    fn group_pairs_sorts_paths_alphabetically() {
        let groups = group_pairs(vec![make_pair(0.9, "z.md", "a.md")]);
        assert_eq!(groups[0].paths, vec!["a.md", "z.md"]);
    }

    #[test]
    fn group_pairs_sorts_pairs_by_similarity_descending() {
        let pairs = vec![
            make_pair(0.85, "a.md", "c.md"),
            make_pair(0.95, "a.md", "b.md"),
            make_pair(0.90, "b.md", "c.md"),
        ];
        let groups = group_pairs(pairs);
        let sims: Vec<f64> = groups[0].pairs.iter().map(|p| p.similarity).collect();
        assert_eq!(sims, vec![0.95, 0.90, 0.85]);
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

        let pairs = find_similar(&conn, 0.9, 50).await.expect("find failed");
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].path_a, "a.md");
        assert_eq!(pairs[0].path_b, "b.md");
        assert!(pairs[0].similarity > 0.99);
    }
}
