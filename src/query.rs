use std::path::PathBuf;

use crate::{
    embed::Embedder,
    rank::{self, FtsSource, FusedResult, PathSource, RrfWeights, SemanticSource},
    similar::{
        self, AllChunkEmbeddingsSource, NoteEmbeddingsSource, RelatedResult, RelatedSearchSource,
        SimilarPair,
    },
};

/// All query ports required to search one store, plus the store root for path rewriting.
pub struct SearchStorePorts<'a> {
    pub notes_dir: &'a PathBuf,
    pub semantic: &'a dyn SemanticSource,
    pub fts: &'a dyn FtsSource,
    pub paths: &'a dyn PathSource,
}

/// All query ports required to run `similar` against one store.
pub struct SimilarStorePorts<'a> {
    pub notes_dir: &'a PathBuf,
    pub embeddings: &'a dyn AllChunkEmbeddingsSource,
}

/// All query ports required to run `related` against one store.
pub struct RelatedStorePorts<'a> {
    pub note_embeddings: &'a dyn NoteEmbeddingsSource,
    pub related_search: &'a dyn RelatedSearchSource,
}

/// Query every store for search results, returning one `Vec<FusedResult>` per store.
///
/// When multiple stores are configured, result paths are converted to absolute
/// paths so that identical relative filenames from different stores produce
/// distinct, globally-unique keys for the merge step.
pub async fn query_search(
    stores: &[SearchStorePorts<'_>],
    embedder: Option<&Embedder>,
    query: &str,
    limit: usize,
    weights: &RrfWeights,
) -> anyhow::Result<Vec<Vec<FusedResult>>> {
    let use_absolute = stores.len() > 1;
    let mut per_store: Vec<Vec<FusedResult>> = Vec::with_capacity(stores.len());
    for store in stores {
        let mut results = rank::search(
            store.semantic,
            store.fts,
            store.paths,
            embedder,
            query,
            limit,
            weights,
        )
        .await?;
        if use_absolute {
            for result in &mut results {
                result.path = store
                    .notes_dir
                    .join(&result.path)
                    .to_string_lossy()
                    .into_owned();
            }
        }
        per_store.push(results);
    }
    Ok(per_store)
}

/// Query every store for similar pairs, returning one `Vec<SimilarPair>` per store.
///
/// When multiple stores are configured, `path_a` and `path_b` are converted to
/// absolute paths for the same reason as in `query_search`.
pub async fn query_similar(
    stores: &[SimilarStorePorts<'_>],
    threshold: f64,
    pair_limit: Option<usize>,
) -> anyhow::Result<Vec<Vec<SimilarPair>>> {
    let use_absolute = stores.len() > 1;
    let mut per_store: Vec<Vec<SimilarPair>> = Vec::with_capacity(stores.len());
    for store in stores {
        let mut pairs = similar::find_similar(store.embeddings, threshold, pair_limit).await?;
        if use_absolute {
            for pair in &mut pairs {
                pair.path_a = store
                    .notes_dir
                    .join(&pair.path_a)
                    .to_string_lossy()
                    .into_owned();
                pair.path_b = store
                    .notes_dir
                    .join(&pair.path_b)
                    .to_string_lossy()
                    .into_owned();
            }
        }
        per_store.push(pairs);
    }
    Ok(per_store)
}

/// Query a single store for documents related to `path`.
///
/// `path` must be store-relative (callers must strip any `notes_dir` prefix
/// before passing it here).  Result paths are store-relative; callers that
/// need absolute paths should call `store.to_absolute` on each result.
pub async fn query_related(
    ports: &RelatedStorePorts<'_>,
    path: &str,
    limit: usize,
) -> anyhow::Result<Vec<RelatedResult>> {
    similar::find_related(ports.note_embeddings, ports.related_search, path, limit).await
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::{
        rank::{Candidate, SearchFuture},
        similar::{RelatedResult, SimilarFuture},
    };

    // --- In-memory fakes ---

    struct FakeSemanticSource(Vec<Candidate>);
    impl SemanticSource for FakeSemanticSource {
        fn search_semantic<'a>(
            &'a self,
            _query_embedding: &'a [f32],
            _limit: usize,
        ) -> SearchFuture<'a, Vec<Candidate>> {
            let items: Vec<Candidate> = self
                .0
                .iter()
                .map(|c| Candidate {
                    path: c.path.clone(),
                    snippet: c.snippet.clone(),
                })
                .collect();
            Box::pin(async move { Ok(items) })
        }
    }

    struct FakeFtsSource(Vec<Candidate>);
    impl FtsSource for FakeFtsSource {
        fn search_fts<'a>(
            &'a self,
            _query: &'a str,
            _limit: usize,
        ) -> SearchFuture<'a, Vec<Candidate>> {
            let items: Vec<Candidate> = self
                .0
                .iter()
                .map(|c| Candidate {
                    path: c.path.clone(),
                    snippet: c.snippet.clone(),
                })
                .collect();
            Box::pin(async move { Ok(items) })
        }
    }

    struct FakePathSource(Vec<String>);
    impl PathSource for FakePathSource {
        fn all_paths(&self) -> SearchFuture<'_, Vec<String>> {
            let paths = self.0.clone();
            Box::pin(async move { Ok(paths) })
        }
    }

    fn fts_candidate(path: &str) -> Candidate {
        Candidate {
            path: path.to_owned(),
            snippet: path.to_owned(),
        }
    }

    fn default_weights() -> RrfWeights {
        RrfWeights {
            semantic: 0.0,
            fts: 1.0,
            filename: 0.0,
        }
    }

    #[tokio::test]
    async fn single_store_paths_are_not_absolutized() {
        let notes_dir = PathBuf::from("/docs/a");
        let semantic = FakeSemanticSource(vec![]);
        let fts = FakeFtsSource(vec![fts_candidate("note.md")]);
        let paths = FakePathSource(vec!["note.md".to_owned()]);

        let stores = [SearchStorePorts {
            notes_dir: &notes_dir,
            semantic: &semantic,
            fts: &fts,
            paths: &paths,
        }];

        let per_store = query_search(&stores, None, "anything", 10, &default_weights())
            .await
            .expect("query_search must succeed");

        assert_eq!(per_store.len(), 1);
        let results = &per_store[0];
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].path, "note.md",
            "single-store paths must remain store-relative"
        );
    }

    #[tokio::test]
    async fn multi_store_paths_are_absolutized_with_respective_notes_dir() {
        let dir_a = PathBuf::from("/docs/a");
        let dir_b = PathBuf::from("/docs/b");
        let semantic_a = FakeSemanticSource(vec![]);
        let semantic_b = FakeSemanticSource(vec![]);
        let fts_a = FakeFtsSource(vec![fts_candidate("note.md")]);
        let fts_b = FakeFtsSource(vec![fts_candidate("note.md")]);
        let paths_a = FakePathSource(vec!["note.md".to_owned()]);
        let paths_b = FakePathSource(vec!["note.md".to_owned()]);

        let stores = [
            SearchStorePorts {
                notes_dir: &dir_a,
                semantic: &semantic_a,
                fts: &fts_a,
                paths: &paths_a,
            },
            SearchStorePorts {
                notes_dir: &dir_b,
                semantic: &semantic_b,
                fts: &fts_b,
                paths: &paths_b,
            },
        ];

        let per_store = query_search(&stores, None, "anything", 10, &default_weights())
            .await
            .expect("query_search must succeed");

        assert_eq!(per_store.len(), 2);
        assert_eq!(
            per_store[0][0].path, "/docs/a/note.md",
            "first store results must be prefixed with /docs/a"
        );
        assert_eq!(
            per_store[1][0].path, "/docs/b/note.md",
            "second store results must be prefixed with /docs/b"
        );
    }

    #[tokio::test]
    async fn multi_store_unique_filenames_are_not_collapsed() {
        let dir_a = PathBuf::from("/docs/a");
        let dir_b = PathBuf::from("/docs/b");
        let semantic_a = FakeSemanticSource(vec![]);
        let semantic_b = FakeSemanticSource(vec![]);
        let fts_a = FakeFtsSource(vec![fts_candidate("note.md")]);
        let fts_b = FakeFtsSource(vec![fts_candidate("other.md")]);
        let paths_a = FakePathSource(vec!["note.md".to_owned()]);
        let paths_b = FakePathSource(vec!["other.md".to_owned()]);

        let stores = [
            SearchStorePorts {
                notes_dir: &dir_a,
                semantic: &semantic_a,
                fts: &fts_a,
                paths: &paths_a,
            },
            SearchStorePorts {
                notes_dir: &dir_b,
                semantic: &semantic_b,
                fts: &fts_b,
                paths: &paths_b,
            },
        ];

        let per_store = query_search(&stores, None, "anything", 10, &default_weights())
            .await
            .expect("query_search must succeed");

        let all_paths: Vec<&str> = per_store
            .iter()
            .flat_map(|v| v.iter().map(|r| r.path.as_str()))
            .collect();

        assert!(
            all_paths.contains(&"/docs/a/note.md"),
            "must contain /docs/a/note.md, got: {all_paths:?}"
        );
        assert!(
            all_paths.contains(&"/docs/b/other.md"),
            "must contain /docs/b/other.md, got: {all_paths:?}"
        );
    }

    // --- SimilarStorePorts fakes ---

    struct FakeAllChunkEmbeddingsSource {
        rows: Vec<(String, Vec<f32>)>,
    }

    impl AllChunkEmbeddingsSource for FakeAllChunkEmbeddingsSource {
        fn has_embeddings(&self) -> SimilarFuture<'_, bool> {
            let has = !self.rows.is_empty();
            Box::pin(async move { Ok(has) })
        }

        fn all_chunk_embeddings(&self) -> SimilarFuture<'_, Vec<(String, Vec<f32>)>> {
            let rows = self.rows.clone();
            Box::pin(async move { Ok(rows) })
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn unit_vec(dim: usize) -> Vec<f32> {
        let v = 1.0_f32 / (dim as f32).sqrt();
        vec![v; dim]
    }

    #[tokio::test]
    async fn similar_single_store_paths_are_not_absolutized() {
        let notes_dir = PathBuf::from("/docs/a");
        let embeddings = FakeAllChunkEmbeddingsSource {
            rows: vec![
                ("a.md".to_owned(), unit_vec(4)),
                ("b.md".to_owned(), unit_vec(4)),
            ],
        };

        let stores = [SimilarStorePorts {
            notes_dir: &notes_dir,
            embeddings: &embeddings,
        }];

        let per_store = query_similar(&stores, 0.9, None)
            .await
            .expect("query_similar must succeed");

        assert_eq!(per_store.len(), 1);
        let pairs = &per_store[0];
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].path_a, "a.md");
        assert_eq!(pairs[0].path_b, "b.md");
    }

    #[tokio::test]
    async fn similar_multi_store_paths_are_absolutized() {
        let dir_a = PathBuf::from("/docs/a");
        let dir_b = PathBuf::from("/docs/b");
        let embeddings_a = FakeAllChunkEmbeddingsSource {
            rows: vec![
                ("x.md".to_owned(), unit_vec(4)),
                ("y.md".to_owned(), unit_vec(4)),
            ],
        };
        let embeddings_b = FakeAllChunkEmbeddingsSource {
            rows: vec![
                ("x.md".to_owned(), unit_vec(4)),
                ("y.md".to_owned(), unit_vec(4)),
            ],
        };

        let stores = [
            SimilarStorePorts {
                notes_dir: &dir_a,
                embeddings: &embeddings_a,
            },
            SimilarStorePorts {
                notes_dir: &dir_b,
                embeddings: &embeddings_b,
            },
        ];

        let per_store = query_similar(&stores, 0.9, None)
            .await
            .expect("query_similar must succeed");

        assert_eq!(per_store.len(), 2);
        let pairs_a = &per_store[0];
        assert_eq!(pairs_a[0].path_a, "/docs/a/x.md");
        assert_eq!(pairs_a[0].path_b, "/docs/a/y.md");
        let pairs_b = &per_store[1];
        assert_eq!(pairs_b[0].path_a, "/docs/b/x.md");
        assert_eq!(pairs_b[0].path_b, "/docs/b/y.md");
    }

    // --- RelatedStorePorts fakes ---

    struct FakeNoteEmbeddingsSource(Vec<Vec<f32>>);
    impl NoteEmbeddingsSource for FakeNoteEmbeddingsSource {
        fn chunk_embeddings_for_path<'a>(
            &'a self,
            _path: &'a str,
        ) -> SimilarFuture<'a, Vec<Vec<f32>>> {
            let chunks = self.0.clone();
            Box::pin(async move { Ok(chunks) })
        }
    }

    struct FakeRelatedSearchSource(Vec<RelatedResult>);
    impl RelatedSearchSource for FakeRelatedSearchSource {
        fn search_related<'a>(
            &'a self,
            _embedding: &'a [f32],
            _exclude_path: &'a str,
            _limit: usize,
        ) -> SimilarFuture<'a, Vec<RelatedResult>> {
            let results: Vec<RelatedResult> = self
                .0
                .iter()
                .map(|r| RelatedResult {
                    path: r.path.clone(),
                    similarity: r.similarity,
                })
                .collect();
            Box::pin(async move { Ok(results) })
        }
    }

    #[tokio::test]
    async fn related_returns_results_from_injected_ports() {
        let note_embeddings = FakeNoteEmbeddingsSource(vec![unit_vec(4)]);
        let related_search = FakeRelatedSearchSource(vec![
            RelatedResult {
                path: "other.md".to_owned(),
                similarity: 0.95,
            },
            RelatedResult {
                path: "another.md".to_owned(),
                similarity: 0.80,
            },
        ]);

        let ports = RelatedStorePorts {
            note_embeddings: &note_embeddings,
            related_search: &related_search,
        };

        let results = query_related(&ports, "target.md", 10)
            .await
            .expect("query_related must succeed");

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].path, "other.md");
        assert!((results[0].similarity - 0.95).abs() < f64::EPSILON);
    }
}
