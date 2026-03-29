use std::collections::HashMap;

use libsql::Connection;
use nucleo::{
    Config as NucleoConfig, Matcher, Utf32Str,
    pattern::{Atom, AtomKind, CaseMatching, Normalization},
};

pub use crate::types::RrfWeights;
use crate::{db, embed::Embedder, error::NeedleError, fts::FtsIndex};

const RRF_K: f64 = 60.0;

pub struct FusedResult {
    pub path: String,
    pub score: f64,
    pub snippet: String,
}

pub async fn search(
    conn: &Connection,
    fts: &FtsIndex,
    embedder: Option<&Embedder>,
    query: &str,
    limit: usize,
    weights: &RrfWeights,
) -> anyhow::Result<Vec<FusedResult>> {
    let candidate_limit = limit.saturating_mul(5);

    let semantic_ranked = if weights.semantic > 0.0 {
        let embedder = embedder.ok_or(NeedleError::NoEmbeddingProvider)?;
        let embedding = embedder.embed_query(query).await?;
        db::search_semantic(conn, &embedding, candidate_limit).await?
    } else {
        Vec::new()
    };

    let fts_ranked = if weights.fts > 0.0 {
        search_fts(fts, query, candidate_limit).await?
    } else {
        Vec::new()
    };

    let all_paths = collect_all_paths(conn).await?;

    let filename_ranked = if weights.filename > 0.0 {
        rank_by_filename(query, &all_paths)
    } else {
        Vec::new()
    };

    let mut scores: HashMap<String, (f64, String)> = HashMap::new();

    let semantic_items: Vec<RankedItem> = semantic_ranked
        .into_iter()
        .map(|r| RankedItem {
            path: r.path,
            snippet: r.snippet,
        })
        .collect();

    accumulate_rrf(&mut scores, &semantic_items, weights.semantic);
    accumulate_rrf(&mut scores, &fts_ranked, weights.fts);
    accumulate_rrf(&mut scores, &filename_ranked, weights.filename);

    let mut results: Vec<FusedResult> = scores
        .into_iter()
        .map(|(path, (score, snippet))| FusedResult {
            path,
            score,
            snippet,
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(limit);

    Ok(results)
}

struct RankedItem {
    path: String,
    snippet: String,
}

fn accumulate_rrf(scores: &mut HashMap<String, (f64, String)>, items: &[RankedItem], weight: f64) {
    for (rank, item) in items.iter().enumerate() {
        #[allow(clippy::cast_precision_loss)]
        let rrf_score = weight / (RRF_K + (rank as f64) + 1.0);
        let entry = scores
            .entry(item.path.clone())
            .or_insert_with(|| (0.0, item.snippet.clone()));
        entry.0 += rrf_score;
    }
}

async fn search_fts(fts: &FtsIndex, query: &str, limit: usize) -> anyhow::Result<Vec<RankedItem>> {
    let results = fts.search(query, limit).await?;
    Ok(results
        .into_iter()
        .map(|r| RankedItem {
            path: r.path,
            snippet: r.snippet,
        })
        .collect())
}

fn rank_by_filename(query: &str, paths: &[String]) -> Vec<RankedItem> {
    let mut matcher = Matcher::new(NucleoConfig::DEFAULT);
    let atom = Atom::new(
        query,
        CaseMatching::Ignore,
        Normalization::Smart,
        AtomKind::Fuzzy,
        false,
    );

    let mut scored: Vec<(u16, &String)> = paths
        .iter()
        .filter_map(|p| {
            let mut buf = Vec::new();
            let haystack = Utf32Str::new(p, &mut buf);
            atom.score(haystack, &mut matcher).map(|s| (s, p))
        })
        .collect();

    scored.sort_by_key(|item| std::cmp::Reverse(item.0));

    scored
        .into_iter()
        .map(|(_, path)| RankedItem {
            path: path.clone(),
            snippet: path.clone(),
        })
        .collect()
}

async fn collect_all_paths(conn: &Connection) -> anyhow::Result<Vec<String>> {
    let mut rows = conn.query("SELECT path FROM notes", ()).await?;
    let mut paths = Vec::new();
    while let Some(row) = rows.next().await? {
        let path: String = row.get(0)?;
        paths.push(path);
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_weights_default_values() {
        let w = RrfWeights::default();
        assert!((w.semantic - 1.5).abs() < f64::EPSILON);
        assert!((w.fts - 1.0).abs() < f64::EPSILON);
        assert!((w.filename - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn accumulate_rrf_scores_first_item_highest() {
        let mut scores = HashMap::new();
        let items = vec![
            RankedItem {
                path: "first.md".to_owned(),
                snippet: "first".to_owned(),
            },
            RankedItem {
                path: "second.md".to_owned(),
                snippet: "second".to_owned(),
            },
        ];
        accumulate_rrf(&mut scores, &items, 1.0);

        let first = scores.get("first.md").expect("first should be scored").0;
        let second = scores.get("second.md").expect("second should be scored").0;
        assert!(
            first > second,
            "rank 0 should score higher than rank 1: {first} vs {second}"
        );
    }

    #[test]
    fn accumulate_rrf_applies_weight_multiplier() {
        let items = vec![RankedItem {
            path: "note.md".to_owned(),
            snippet: String::new(),
        }];

        let mut scores_low = HashMap::new();
        accumulate_rrf(&mut scores_low, &items, 1.0);

        let mut scores_high = HashMap::new();
        accumulate_rrf(&mut scores_high, &items, 2.0);

        let low = scores_low.get("note.md").expect("should exist").0;
        let high = scores_high.get("note.md").expect("should exist").0;
        assert!(
            2.0f64.mul_add(-low, high).abs() < f64::EPSILON,
            "doubling weight should double score"
        );
    }

    #[test]
    fn accumulate_rrf_zero_weight_produces_zero_score() {
        let mut scores = HashMap::new();
        let items = vec![RankedItem {
            path: "note.md".to_owned(),
            snippet: String::new(),
        }];
        accumulate_rrf(&mut scores, &items, 0.0);

        let score = scores.get("note.md").expect("should exist").0;
        assert!(
            score.abs() < f64::EPSILON,
            "zero weight should produce zero score"
        );
    }

    #[test]
    fn accumulate_rrf_sums_scores_for_same_path() {
        let mut scores = HashMap::new();

        let items_a = vec![RankedItem {
            path: "note.md".to_owned(),
            snippet: "from signal a".to_owned(),
        }];
        let items_b = vec![RankedItem {
            path: "note.md".to_owned(),
            snippet: "from signal b".to_owned(),
        }];

        accumulate_rrf(&mut scores, &items_a, 1.0);
        accumulate_rrf(&mut scores, &items_b, 1.0);

        let score = scores.get("note.md").expect("should exist").0;
        let expected = 2.0 / (RRF_K + 1.0);
        assert!(
            (score - expected).abs() < f64::EPSILON,
            "same path at rank 0 in two signals should sum: {score} vs {expected}"
        );
    }

    #[test]
    fn accumulate_rrf_formula_matches_specification() {
        let mut scores = HashMap::new();
        let items = vec![
            RankedItem {
                path: "a.md".to_owned(),
                snippet: String::new(),
            },
            RankedItem {
                path: "b.md".to_owned(),
                snippet: String::new(),
            },
        ];
        accumulate_rrf(&mut scores, &items, 1.0);

        let a = scores.get("a.md").expect("should exist").0;
        let b = scores.get("b.md").expect("should exist").0;

        let expected_a = 1.0 / (RRF_K + 0.0 + 1.0);
        let expected_b = 1.0 / (RRF_K + 1.0 + 1.0);
        assert!((a - expected_a).abs() < f64::EPSILON);
        assert!((b - expected_b).abs() < f64::EPSILON);
    }

    #[test]
    fn accumulate_rrf_keeps_first_snippet_for_path() {
        let mut scores = HashMap::new();
        let items_a = vec![RankedItem {
            path: "note.md".to_owned(),
            snippet: "first snippet".to_owned(),
        }];
        let items_b = vec![RankedItem {
            path: "note.md".to_owned(),
            snippet: "second snippet".to_owned(),
        }];

        accumulate_rrf(&mut scores, &items_a, 1.0);
        accumulate_rrf(&mut scores, &items_b, 1.0);

        let snippet = &scores.get("note.md").expect("should exist").1;
        assert_eq!(
            snippet, "first snippet",
            "should keep the first snippet seen"
        );
    }

    #[test]
    fn rank_by_filename_exact_match_ranks_first() {
        let paths = vec![
            "notes/unrelated.md".to_owned(),
            "notes/Tiger Style.md".to_owned(),
            "notes/another.md".to_owned(),
        ];
        let results = rank_by_filename("tiger style", &paths);
        assert!(!results.is_empty());
        assert_eq!(results[0].path, "notes/Tiger Style.md");
    }

    #[test]
    fn rank_by_filename_partial_match() {
        let paths = vec![
            "llm/Kubernetes Basics.md".to_owned(),
            "llm/Tiger Style.md".to_owned(),
        ];
        let results = rank_by_filename("kube", &paths);
        assert!(!results.is_empty());
        assert_eq!(results[0].path, "llm/Kubernetes Basics.md");
    }

    #[test]
    fn rank_by_filename_returns_empty_for_no_match() {
        let paths = vec!["notes/hello.md".to_owned()];
        let results = rank_by_filename("zzzzzzzzz", &paths);
        assert!(results.is_empty());
    }

    #[test]
    fn rank_by_filename_is_case_insensitive() {
        let paths = vec!["notes/UPPERCASE.md".to_owned()];
        let results = rank_by_filename("uppercase", &paths);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn rank_by_filename_empty_paths_returns_empty() {
        let results = rank_by_filename("anything", &[]);
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_fuses_fts_and_filename_signals() {
        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = crate::db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");

        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");

        let emb = vec![0.0f32; 1024];
        crate::db::upsert_note(
            &conn,
            "rust-guide.md",
            "h1",
            &[(
                "Rust is a systems programming language".to_owned(),
                emb.clone(),
            )],
        )
        .await
        .expect("upsert");
        crate::db::upsert_note(
            &conn,
            "python-guide.md",
            "h2",
            &[("Python is a scripting language".to_owned(), emb)],
        )
        .await
        .expect("upsert");

        fts.upsert(
            "rust-guide.md",
            &["Rust is a systems programming language".to_owned()],
        )
        .await
        .expect("fts upsert");
        fts.upsert(
            "python-guide.md",
            &["Python is a scripting language".to_owned()],
        )
        .await
        .expect("fts upsert");

        let weights = RrfWeights {
            semantic: 0.0,
            fts: 1.0,
            filename: 1.0,
        };
        let results = search(&conn, &fts, None, "rust", 10, &weights)
            .await
            .expect("search");

        assert!(!results.is_empty());
        assert_eq!(results[0].path, "rust-guide.md");
    }

    #[tokio::test]
    async fn search_works_without_api_key_when_semantic_is_zero() {
        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = crate::db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");

        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");

        let emb = vec![0.0f32; 1024];
        crate::db::upsert_note(
            &conn,
            "note.md",
            "h1",
            &[("some searchable content".to_owned(), emb)],
        )
        .await
        .expect("upsert");

        fts.upsert("note.md", &["some searchable content".to_owned()])
            .await
            .expect("fts upsert");

        let weights = RrfWeights {
            semantic: 0.0,
            fts: 1.0,
            filename: 0.0,
        };
        let results = search(&conn, &fts, None, "searchable", 10, &weights)
            .await
            .expect("search without api key");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "note.md");
    }

    #[tokio::test]
    async fn search_requires_api_key_when_semantic_is_positive() {
        let db_dir = tempfile::tempdir().expect("tempdir");
        let (_db, conn) = crate::db::connect(&db_dir.path().join("test.db"), Some(1024))
            .await
            .expect("connect");

        let fts_dir = tempfile::tempdir().expect("tempdir");
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");

        let weights = RrfWeights {
            semantic: 1.0,
            fts: 1.0,
            filename: 1.0,
        };
        let result = search(&conn, &fts, None, "anything", 10, &weights).await;
        assert!(result.is_err());
    }
}
