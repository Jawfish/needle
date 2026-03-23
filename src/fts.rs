use std::{path::Path, sync::Arc};

use tantivy::{
    IndexReader, IndexWriter, ReloadPolicy, TantivyDocument,
    collector::TopDocs,
    query::QueryParser,
    schema::{Field, STORED, STRING, Schema, TextFieldIndexing, TextOptions, Value},
    snippet::SnippetGenerator,
};
use tokio::sync::Mutex;

pub struct FtsResult {
    pub path: String,
    pub snippet: String,
}

pub struct FtsIndex {
    index: tantivy::Index,
    reader: IndexReader,
    writer: Arc<Mutex<IndexWriter>>,
    path_field: Field,
    content_field: Field,
}

impl FtsIndex {
    pub fn open_or_create(index_dir: &Path) -> anyhow::Result<Self> {
        let content_options = TextOptions::default()
            .set_indexing_options(
                TextFieldIndexing::default()
                    .set_tokenizer("en_stem")
                    .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions),
            )
            .set_stored();

        let mut schema_builder = Schema::builder();
        let path_field = schema_builder.add_text_field("path", STRING | STORED);
        let content_field = schema_builder.add_text_field("content", content_options);
        let schema = schema_builder.build();

        let index = if index_dir.join("meta.json").exists() {
            tantivy::Index::open_in_dir(index_dir)?
        } else {
            tantivy::Index::create_in_dir(index_dir, schema)?
        };

        index.tokenizers().register(
            "en_stem",
            tantivy::tokenizer::TextAnalyzer::builder(
                tantivy::tokenizer::SimpleTokenizer::default(),
            )
            .filter(tantivy::tokenizer::RemoveLongFilter::limit(40))
            .filter(tantivy::tokenizer::LowerCaser)
            .filter(tantivy::tokenizer::Stemmer::new(
                tantivy::tokenizer::Language::English,
            ))
            .build(),
        );

        let writer = index.writer(50_000_000)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        Ok(Self {
            index,
            reader,
            writer: Arc::new(Mutex::new(writer)),
            path_field,
            content_field,
        })
    }

    pub fn is_empty(&self) -> bool {
        let _ = self.reader.reload();
        let searcher = self.reader.searcher();
        searcher.num_docs() == 0
    }

    pub async fn upsert(&self, path: &str, chunks: &[String]) -> anyhow::Result<()> {
        let path_field = self.path_field;
        let content_field = self.content_field;
        let path_owned = path.to_owned();
        let chunks_owned: Vec<String> = chunks.to_vec();
        let writer = Arc::clone(&self.writer);

        tokio::task::spawn_blocking(move || {
            let mut guard = writer.blocking_lock();
            let path_term = tantivy::Term::from_field_text(path_field, &path_owned);
            guard.delete_term(path_term);

            for chunk in &chunks_owned {
                let mut doc = TantivyDocument::new();
                doc.add_text(path_field, &path_owned);
                doc.add_text(content_field, chunk);
                guard.add_document(doc)?;
            }

            guard.commit()?;
            drop(guard);
            anyhow::Ok(())
        })
        .await??;

        Ok(())
    }

    pub async fn delete(&self, path: &str) -> anyhow::Result<()> {
        let path_field = self.path_field;
        let path_owned = path.to_owned();
        let writer = Arc::clone(&self.writer);

        tokio::task::spawn_blocking(move || {
            let mut guard = writer.blocking_lock();
            let path_term = tantivy::Term::from_field_text(path_field, &path_owned);
            guard.delete_term(path_term);
            guard.commit()?;
            drop(guard);
            anyhow::Ok(())
        })
        .await??;

        Ok(())
    }

    pub async fn search(&self, query: &str, limit: usize) -> anyhow::Result<Vec<FtsResult>> {
        let index = self.index.clone();
        let reader = self.reader.clone();
        let content_field = self.content_field;
        let path_field = self.path_field;
        let query_owned = query.to_owned();

        tokio::task::spawn_blocking(move || {
            reader.reload()?;
            let searcher = reader.searcher();

            let query_parser = QueryParser::for_index(&index, vec![content_field]);
            let (parsed_query, _errors) = query_parser.parse_query_lenient(&query_owned);

            let candidate_limit = limit * 3;
            let top_docs = searcher.search(&parsed_query, &TopDocs::with_limit(candidate_limit))?;

            let snippet_generator =
                SnippetGenerator::create(&searcher, &parsed_query, content_field)?;

            let mut seen = std::collections::HashSet::new();
            let mut results = Vec::new();

            for (_, doc_address) in top_docs {
                let doc: TantivyDocument = searcher.doc(doc_address)?;

                let path = doc
                    .get_first(path_field)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_owned();

                if seen.contains(&path) {
                    continue;
                }

                let content = doc
                    .get_first(content_field)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default();

                let snippet = snippet_generator.snippet(content);
                let fragment = snippet.fragment();

                let snippet_text = if fragment.is_empty() {
                    content.lines().next().unwrap_or_default().to_owned()
                } else {
                    fragment.to_owned()
                };

                seen.insert(path.clone());
                results.push(FtsResult {
                    path,
                    snippet: snippet_text,
                });

                if results.len() >= limit {
                    break;
                }
            }

            anyhow::Ok(results)
        })
        .await?
    }

    pub async fn rebuild(&self, chunks: Vec<(String, String)>) -> anyhow::Result<()> {
        let path_field = self.path_field;
        let content_field = self.content_field;
        let writer = Arc::clone(&self.writer);

        tokio::task::spawn_blocking(move || {
            let mut guard = writer.blocking_lock();
            guard.delete_all_documents()?;

            for (path, content) in &chunks {
                let mut doc = TantivyDocument::new();
                doc.add_text(path_field, path);
                doc.add_text(content_field, content);
                guard.add_document(doc)?;
            }

            guard.commit()?;
            drop(guard);
            anyhow::Ok(())
        })
        .await??;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_index() -> (tempfile::TempDir, FtsIndex) {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let index = FtsIndex::open_or_create(dir.path()).expect("failed to create index");
        (dir, index)
    }

    #[tokio::test]
    async fn empty_index_reports_empty() {
        let (_dir, index) = temp_index();
        assert!(index.is_empty());
    }

    #[tokio::test]
    async fn upsert_makes_index_non_empty() {
        let (_dir, index) = temp_index();
        index
            .upsert("note.md", &["hello world".to_owned()])
            .await
            .expect("upsert failed");
        assert!(!index.is_empty());
    }

    #[tokio::test]
    async fn search_finds_indexed_content() {
        let (_dir, index) = temp_index();
        index
            .upsert(
                "rust.md",
                &["Rust is a systems programming language".to_owned()],
            )
            .await
            .expect("upsert failed");

        let results = index
            .search("systems programming", 10)
            .await
            .expect("search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "rust.md");
    }

    #[tokio::test]
    async fn search_returns_empty_for_no_match() {
        let (_dir, index) = temp_index();
        index
            .upsert(
                "rust.md",
                &["Rust is a systems programming language".to_owned()],
            )
            .await
            .expect("upsert failed");

        let results = index.search("elephants", 10).await.expect("search failed");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn stemming_matches_inflected_forms() {
        let (_dir, index) = temp_index();
        index
            .upsert("note.md", &["The runners were running quickly".to_owned()])
            .await
            .expect("upsert failed");

        let results = index.search("run", 10).await.expect("search failed");
        assert_eq!(
            results.len(),
            1,
            "stemming should match 'run' to 'running/runners'"
        );
    }

    #[tokio::test]
    async fn query_with_apostrophe_does_not_error() {
        let (_dir, index) = temp_index();
        index
            .upsert("note.md", &["The app won't start properly".to_owned()])
            .await
            .expect("upsert failed");

        let results = index
            .search("won't start", 10)
            .await
            .expect("search failed");
        assert!(
            !results.is_empty(),
            "apostrophe in query should not cause failure"
        );
    }

    #[tokio::test]
    async fn query_with_special_characters_does_not_error() {
        let (_dir, index) = temp_index();
        index
            .upsert(
                "note.md",
                &["Error: ERR_NETWORK_CHANGED in Chrome".to_owned()],
            )
            .await
            .expect("upsert failed");

        let cases = [
            "ERR_NETWORK_CHANGED",
            "error: something",
            "foo AND bar OR baz",
            "\"unclosed quote",
            "(unmatched paren",
            "file.txt",
            "c++ language",
            "user@example.com",
            "path/to/file",
        ];
        for query in &cases {
            let result = index.search(query, 10).await;
            assert!(result.is_ok(), "query {query:?} should not error");
        }
    }

    #[tokio::test]
    async fn delete_removes_document_from_results() {
        let (_dir, index) = temp_index();
        index
            .upsert("keep.md", &["Kubernetes deployment guide".to_owned()])
            .await
            .expect("upsert failed");
        index
            .upsert("remove.md", &["Kubernetes cluster setup".to_owned()])
            .await
            .expect("upsert failed");

        index.delete("remove.md").await.expect("delete failed");

        let results = index.search("kubernetes", 10).await.expect("search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "keep.md");
    }

    #[tokio::test]
    async fn upsert_replaces_previous_content() {
        let (_dir, index) = temp_index();
        index
            .upsert("note.md", &["old content about elephants".to_owned()])
            .await
            .expect("upsert failed");
        index
            .upsert("note.md", &["new content about giraffes".to_owned()])
            .await
            .expect("upsert failed");

        let old = index.search("elephants", 10).await.expect("search failed");
        assert!(old.is_empty(), "old content should be gone after upsert");

        let new = index.search("giraffes", 10).await.expect("search failed");
        assert_eq!(new.len(), 1);
        assert_eq!(new[0].path, "note.md");
    }

    #[tokio::test]
    async fn deduplicates_results_by_path() {
        let (_dir, index) = temp_index();
        index
            .upsert(
                "note.md",
                &[
                    "Kubernetes is a container orchestration platform".to_owned(),
                    "Kubernetes manages containerized workloads".to_owned(),
                    "Kubernetes provides service discovery".to_owned(),
                ],
            )
            .await
            .expect("upsert failed");

        let results = index.search("kubernetes", 10).await.expect("search failed");
        assert_eq!(
            results.len(),
            1,
            "multiple chunks from same path should deduplicate"
        );
        assert_eq!(results[0].path, "note.md");
    }

    #[tokio::test]
    async fn respects_limit() {
        let (_dir, index) = temp_index();
        for i in 0..10 {
            index
                .upsert(
                    &format!("note{i}.md"),
                    &[format!("Rust programming language tutorial part {i}")],
                )
                .await
                .expect("upsert failed");
        }

        let results = index
            .search("rust programming", 3)
            .await
            .expect("search failed");
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn rebuild_populates_from_chunk_pairs() {
        let (_dir, index) = temp_index();
        assert!(index.is_empty());

        let chunks = vec![
            (
                "alpha.md".to_owned(),
                "Alpha content about databases".to_owned(),
            ),
            (
                "beta.md".to_owned(),
                "Beta content about networking".to_owned(),
            ),
        ];
        index.rebuild(chunks).await.expect("rebuild failed");

        assert!(!index.is_empty());

        let results = index.search("databases", 10).await.expect("search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "alpha.md");
    }

    #[tokio::test]
    async fn rebuild_clears_existing_documents() {
        let (_dir, index) = temp_index();
        index
            .upsert("old.md", &["old content about databases".to_owned()])
            .await
            .expect("upsert failed");

        let chunks = vec![(
            "new.md".to_owned(),
            "new content about networking".to_owned(),
        )];
        index.rebuild(chunks).await.expect("rebuild failed");

        let old_results = index.search("databases", 10).await.expect("search failed");
        assert!(
            old_results.is_empty(),
            "old documents should be cleared by rebuild"
        );

        let new_results = index.search("networking", 10).await.expect("search failed");
        assert_eq!(new_results.len(), 1);
        assert_eq!(new_results[0].path, "new.md");
    }

    #[tokio::test]
    async fn reopen_persists_data() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");

        {
            let index = FtsIndex::open_or_create(dir.path()).expect("create failed");
            index
                .upsert("note.md", &["persistent data about compilers".to_owned()])
                .await
                .expect("upsert failed");
        }

        let index = FtsIndex::open_or_create(dir.path()).expect("reopen failed");
        let results = index.search("compilers", 10).await.expect("search failed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, "note.md");
    }

    #[tokio::test]
    async fn empty_query_returns_empty_results() {
        let (_dir, index) = temp_index();
        index
            .upsert("note.md", &["some content".to_owned()])
            .await
            .expect("upsert failed");

        let results = index.search("", 10).await.expect("search failed");
        assert!(results.is_empty());
    }
}
