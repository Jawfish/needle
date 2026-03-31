use std::collections::HashMap;

use crate::{config::DirectoryStore, rank::FusedResult, similar::SimilarPair};

/// Merge per-store search result sets into a single ranked list.
///
/// Scores for the same path across stores are summed, then the combined
/// list is sorted descending by score and truncated to `limit`.
pub fn merge_fused_results(per_store: Vec<Vec<FusedResult>>, limit: usize) -> Vec<FusedResult> {
    let mut accumulator: HashMap<String, FusedResult> = HashMap::new();
    for results in per_store {
        for result in results {
            accumulator
                .entry(result.path.clone())
                .and_modify(|existing| existing.score += result.score)
                .or_insert(result);
        }
    }
    let mut merged: Vec<FusedResult> = accumulator.into_values().collect();
    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    merged.truncate(limit);
    merged
}

/// Merge per-store similar-pair lists into a single ranked list.
///
/// All pairs are concatenated, sorted descending by similarity, and
/// truncated to `pair_limit` when present (grouping mode passes `None`).
pub fn merge_similar_pairs(
    per_store: Vec<Vec<SimilarPair>>,
    pair_limit: Option<usize>,
) -> Vec<SimilarPair> {
    let mut merged: Vec<SimilarPair> = per_store.into_iter().flatten().collect();
    merged.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(limit) = pair_limit {
        merged.truncate(limit);
    }
    merged
}

/// Find the store whose `notes_dir` is the canonical parent of `path`.
///
/// For absolute paths: returns the store whose `notes_dir` is a prefix of
/// `path`. With non-overlapping roots enforced at config resolution time,
/// at most one store can match, so the result is deterministic.
///
/// For relative paths: returns the single configured store when there is
/// exactly one, otherwise returns `None`. Callers must surface an error
/// asking for an absolute path when `None` is returned with a relative path
/// and multiple stores configured.
pub fn owning_store<'a>(stores: &'a [DirectoryStore], path: &str) -> Option<&'a DirectoryStore> {
    let as_path = std::path::Path::new(path);
    if as_path.is_absolute() {
        stores.iter().find(|s| as_path.starts_with(&s.notes_dir))
    } else if stores.len() == 1 {
        stores.first()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::{config::DirectoryStore, rank::FusedResult, similar::SimilarPair};

    fn fused(path: &str, score: f64) -> FusedResult {
        FusedResult {
            path: path.to_owned(),
            score,
            snippet: String::new(),
        }
    }

    fn pair(a: &str, b: &str, similarity: f64) -> SimilarPair {
        SimilarPair {
            path_a: a.to_owned(),
            path_b: b.to_owned(),
            similarity,
        }
    }

    fn store(notes_dir: &str) -> DirectoryStore {
        let base = PathBuf::from(notes_dir);
        DirectoryStore {
            notes_dir: base.clone(),
            db_path: base.join("needle.db"),
            tantivy_dir: base.join("tantivy"),
        }
    }

    #[test]
    fn merge_fused_results_sums_scores_for_same_path() {
        let per_store = vec![
            vec![fused("a.md", 1.0), fused("b.md", 0.5)],
            vec![fused("a.md", 0.3)],
        ];
        let merged = merge_fused_results(per_store, 10);
        let a = merged.iter().find(|r| r.path == "a.md").expect("a.md");
        assert!((a.score - 1.3).abs() < f64::EPSILON);
    }

    #[test]
    fn merge_fused_results_sorts_descending_and_truncates() {
        let per_store = vec![vec![fused("low.md", 0.1), fused("high.md", 0.9)]];
        let merged = merge_fused_results(per_store, 1);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].path, "high.md");
    }

    #[test]
    fn merge_fused_results_empty_input_returns_empty() {
        assert!(merge_fused_results(vec![], 10).is_empty());
    }

    #[test]
    fn merge_similar_pairs_sorts_descending_and_truncates() {
        let per_store = vec![
            vec![pair("a", "b", 0.9), pair("c", "d", 0.5)],
            vec![pair("e", "f", 0.7)],
        ];
        let merged = merge_similar_pairs(per_store, Some(2));
        assert_eq!(merged.len(), 2);
        assert!((merged[0].similarity - 0.9).abs() < f64::EPSILON);
        assert!((merged[1].similarity - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn merge_similar_pairs_no_limit_returns_all() {
        let per_store = vec![vec![pair("a", "b", 0.9)], vec![pair("c", "d", 0.5)]];
        let merged = merge_similar_pairs(per_store, None);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn owning_store_returns_store_whose_notes_dir_is_prefix() {
        let stores = [store("/dir1"), store("/dir2")];
        let found = owning_store(&stores, "/dir2/note.md");
        assert_eq!(
            found.map(|s| s.notes_dir.as_path()),
            Some(std::path::Path::new("/dir2"))
        );
    }

    #[test]
    fn owning_store_returns_none_for_unmatched_absolute_path() {
        let stores = [store("/dir1"), store("/dir2")];
        assert!(owning_store(&stores, "/other/note.md").is_none());
    }

    #[test]
    fn owning_store_returns_none_for_relative_path_with_multiple_stores() {
        let stores = [store("/dir1"), store("/dir2")];
        assert!(
            owning_store(&stores, "relative/note.md").is_none(),
            "relative path with multiple stores must return None to force explicit routing"
        );
    }

    #[test]
    fn owning_store_returns_store_for_relative_path_with_single_store() {
        let stores = [store("/dir1")];
        let found = owning_store(&stores, "relative/note.md");
        assert_eq!(
            found.map(|s| s.notes_dir.as_path()),
            Some(std::path::Path::new("/dir1"))
        );
    }

    #[test]
    fn owning_store_returns_none_for_empty_stores() {
        assert!(owning_store(&[], "/any/path.md").is_none());
    }

    // --- Merge collision regression tests ---

    #[test]
    fn same_filename_from_different_stores_does_not_collapse_when_absolute() {
        // query_search absolutizes paths when multiple stores are configured, so
        // /dir1/note.md and /dir2/note.md are distinct keys even though both have
        // the relative filename "note.md".
        let per_store = vec![
            vec![fused("/dir1/note.md", 0.8)],
            vec![fused("/dir2/note.md", 0.7)],
        ];
        let merged = merge_fused_results(per_store, 10);
        assert_eq!(
            merged.len(),
            2,
            "same filename from different stores must remain two separate results"
        );
    }

    #[test]
    fn same_document_from_same_store_accumulates_across_ranking_signals() {
        // Different ranking signals (semantic + FTS) for the same document
        // within one store still share a key and should have their scores summed.
        let per_store = vec![vec![
            fused("note.md", 0.6), // semantic signal
            fused("note.md", 0.4), // FTS signal
        ]];
        let merged = merge_fused_results(per_store, 10);
        assert_eq!(merged.len(), 1, "same document must be merged");
        let note = &merged[0];
        assert!((note.score - 1.0).abs() < f64::EPSILON, "scores must sum");
    }

    #[test]
    fn similar_pairs_from_different_stores_are_distinct_when_absolute() {
        let per_store = vec![
            vec![pair("/dir1/a.md", "/dir1/b.md", 0.9)],
            vec![pair("/dir2/a.md", "/dir2/b.md", 0.8)],
        ];
        let merged = merge_similar_pairs(per_store, None);
        assert_eq!(
            merged.len(),
            2,
            "same relative pair names from different stores must remain two separate pairs"
        );
    }
}
