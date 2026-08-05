use std::{net::IpAddr, path::PathBuf, sync::Arc};

use anyhow::Context as _;
use axum::{
    Router,
    extract::{Query, State},
    response::Html,
    routing::get,
};
use maud::{DOCTYPE, Markup, PreEscaped, html};
use serde::Deserialize;

use crate::{
    db::{DbPathSource, DbSemanticSource},
    embed::Embedder,
    fts::FtsFtsSource,
    query::{self, SearchStorePorts},
    rank::{FtsSource, PathSource, RrfWeights, SemanticSource},
    search_merge,
};

const RESULT_LIMIT: usize = 20;
const CSS: &str = "body{max-width:52rem;margin:2rem auto;padding:0 1rem;font-family:system-ui,sans-serif;color:#17202a;background:#fafafa}form{display:flex;gap:.5rem}input{flex:1;padding:.6rem;font:inherit}button{padding:.6rem 1rem;font:inherit}article{background:white;border:1px solid #d5d8dc;border-radius:.4rem;margin:1rem 0;padding:1rem}h1{margin-top:0}.path{font-family:monospace;overflow-wrap:anywhere}.locator{color:#52616b}.snippet{white-space:pre-wrap}.state{padding:1rem;background:#eef3f5;border-radius:.4rem}";

pub struct SearchAdapter {
    pub semantic: DbSemanticSource,
    pub fts: FtsFtsSource,
    pub paths: DbPathSource,
}

struct SearchStore {
    notes_dir: PathBuf,
    semantic: Arc<dyn SemanticSource>,
    fts: Arc<dyn FtsSource>,
    paths: Arc<dyn PathSource>,
}

struct AppState {
    stores: Vec<SearchStore>,
    embedder: Option<Arc<Embedder>>,
    weights: RrfWeights,
}

#[derive(Deserialize)]
struct SearchParams {
    q: Option<String>,
}

pub async fn run(
    host: IpAddr,
    port: u16,
    stores: &[crate::config::DirectoryStore],
    adapters: Vec<SearchAdapter>,
    embedder: Option<Embedder>,
    weights: RrfWeights,
) -> anyhow::Result<()> {
    let state = Arc::new(AppState::from_adapters(stores, adapters, embedder, weights));
    let listener = tokio::net::TcpListener::bind((host, port))
        .await
        .with_context(|| format!("binding HTTP server to {host}:{port}"))?;
    let address = listener
        .local_addr()
        .context("reading bound HTTP server address")?;
    if !address.ip().is_loopback() {
        tracing::warn!("WARNING: unauthenticated indexed content is exposed at http://{address}");
    }
    tracing::info!("serving indexed documents at http://{address}");

    let shutdown = tokio::signal::ctrl_c();
    tokio::pin!(shutdown);
    let server = axum::Server::from_tcp(
        listener
            .into_std()
            .context("preparing HTTP server listener")?,
    )?
    .serve(router(state).into_make_service())
    .with_graceful_shutdown(async move {
        if shutdown.await.is_ok() {
            tracing::info!("shutting down");
        }
    });
    server.await.context("running HTTP server")
}

impl AppState {
    fn from_adapters(
        stores: &[crate::config::DirectoryStore],
        adapters: Vec<SearchAdapter>,
        embedder: Option<Embedder>,
        weights: RrfWeights,
    ) -> Self {
        let stores = stores
            .iter()
            .zip(adapters)
            .map(|(store, adapter)| SearchStore {
                notes_dir: store.notes_dir.clone(),
                semantic: Arc::new(adapter.semantic),
                fts: Arc::new(adapter.fts),
                paths: Arc::new(adapter.paths),
            })
            .collect();
        Self {
            stores,
            embedder: embedder.map(Arc::new),
            weights,
        }
    }
}

fn router(state: Arc<AppState>) -> Router {
    Router::new().route("/", get(home)).with_state(state)
}

#[allow(clippy::option_if_let_else)]
async fn home(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchParams>,
) -> Html<String> {
    let query = params.q.unwrap_or_default();
    if query.trim().is_empty() {
        let content = html! { p class="state" { "Search your indexed documents." } };
        return Html(page(&query, &content));
    }

    let ports: Vec<SearchStorePorts<'_>> = state
        .stores
        .iter()
        .map(|store| SearchStorePorts {
            notes_dir: &store.notes_dir,
            semantic: store.semantic.as_ref(),
            fts: store.fts.as_ref(),
            paths: store.paths.as_ref(),
        })
        .collect();
    let content = match query::query_search(
        &ports,
        state.embedder.as_deref(),
        &query,
        RESULT_LIMIT,
        &state.weights,
    )
    .await
    {
        Ok(per_store) => {
            let results = search_merge::merge_fused_results(per_store, RESULT_LIMIT);
            if results.is_empty() {
                html! { p class="state" { "No indexed documents matched your search." } }
            } else {
                html! {
                    p { "Showing up to " (RESULT_LIMIT) " ranked results." }
                    @for result in results {
                        article {
                            div class="path" { (result.path) }
                            (result.locator.as_ref().map_or_else(
                                || html! {},
                                |locator| html! { div class="locator" { (locator) } },
                            ))
                            p class="snippet" { (result.snippet) }
                        }
                    }
                }
            }
        }
        Err(_) => {
            html! { p class="state" { "Search is temporarily unavailable. Please try again." } }
        }
    };
    Html(page(&query, &content))
}

fn page(query: &str, content: &Markup) -> String {
    html! {
        (DOCTYPE)
        html lang="en" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                title { "Needle search" }
                style { (PreEscaped(CSS)) }
            }
            body {
                h1 { "Needle" }
                form method="get" action="/" {
                    label for="q" { "Search indexed documents" }
                    input id="q" name="q" type="search" value=(query) autofocus;
                    button type="submit" { "Search" }
                }
                (content)
            }
        }
    }
    .into_string()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use axum::{body::Body, http::Request};
    use hyper::body::to_bytes;
    use tower::ServiceExt;

    use super::*;
    use crate::rank::{Candidate, SearchFuture};

    struct FakeSemantic;
    impl SemanticSource for FakeSemantic {
        fn search_semantic<'a>(
            &'a self,
            _: &'a [f32],
            _: usize,
        ) -> SearchFuture<'a, Vec<Candidate>> {
            Box::pin(async { Ok(vec![]) })
        }
    }

    struct FakeFts {
        results: Vec<Candidate>,
        fails: bool,
    }
    impl FtsSource for FakeFts {
        fn search_fts<'a>(&'a self, _: &'a str, limit: usize) -> SearchFuture<'a, Vec<Candidate>> {
            let results = self
                .results
                .iter()
                .take(limit)
                .map(copy_candidate)
                .collect();
            let fails = self.fails;
            Box::pin(async move {
                if fails {
                    anyhow::bail!("provider token secret and /private/index failed");
                }
                Ok(results)
            })
        }
    }

    struct FakePaths(Vec<String>);
    impl PathSource for FakePaths {
        fn all_paths(&self) -> SearchFuture<'_, Vec<String>> {
            let paths = self.0.clone();
            Box::pin(async move { Ok(paths) })
        }
    }

    fn copy_candidate(candidate: &Candidate) -> Candidate {
        Candidate {
            path: candidate.path.clone(),
            snippet: candidate.snippet.clone(),
            locator: candidate.locator.clone(),
        }
    }

    fn candidate(path: &str) -> Candidate {
        Candidate {
            path: path.to_owned(),
            snippet: format!("snippet {path}"),
            locator: Some(format!("locator {path}")),
        }
    }

    fn app(stores: Vec<(PathBuf, Vec<Candidate>, bool)>) -> Router {
        let stores = stores
            .into_iter()
            .map(|(notes_dir, results, fails)| SearchStore {
                paths: Arc::new(FakePaths(results.iter().map(|r| r.path.clone()).collect())),
                notes_dir,
                semantic: Arc::new(FakeSemantic),
                fts: Arc::new(FakeFts { results, fails }),
            })
            .collect();
        router(Arc::new(AppState {
            stores,
            embedder: None,
            weights: RrfWeights {
                semantic: 0.0,
                fts: 1.0,
                filename: 0.0,
            },
        }))
    }

    async fn get(app: Router, uri: &str) -> String {
        let response = app
            .oneshot(Request::get(uri).body(Body::empty()).expect("request"))
            .await
            .expect("response");
        String::from_utf8(to_bytes(response.into_body()).await.expect("body").to_vec())
            .expect("utf8")
    }

    #[tokio::test]
    async fn empty_query_renders_home_state() {
        let body = get(app(vec![]), "/?q=%20%20").await;
        assert!(body.contains("Search your indexed documents."));
    }

    #[tokio::test]
    async fn results_show_path_snippet_and_locator() {
        let body = get(
            app(vec![(
                PathBuf::from("/notes"),
                vec![candidate("note.md")],
                false,
            )]),
            "/?q=note",
        )
        .await;
        assert!(
            body.contains("note.md")
                && body.contains("snippet note.md")
                && body.contains("locator note.md")
        );
    }

    #[tokio::test]
    async fn multiple_stores_keep_same_relative_paths_distinct() {
        let body = get(
            app(vec![
                (PathBuf::from("/one"), vec![candidate("note.md")], false),
                (PathBuf::from("/two"), vec![candidate("note.md")], false),
            ]),
            "/?q=note",
        )
        .await;
        assert!(body.contains("/one/note.md") && body.contains("/two/note.md"));
    }

    #[tokio::test]
    async fn results_are_limited_to_twenty() {
        let results = (0..25)
            .map(|i| candidate(&format!("note-{i}.md")))
            .collect();
        let body = get(
            app(vec![(PathBuf::from("/notes"), results, false)]),
            "/?q=note",
        )
        .await;
        assert_eq!(body.matches("class=\"path\"").count(), RESULT_LIMIT);
    }

    #[tokio::test]
    async fn no_match_and_failure_are_distinct_and_sanitized() {
        let none = get(
            app(vec![(PathBuf::from("/notes"), vec![], false)]),
            "/?q=none",
        )
        .await;
        let failed = get(
            app(vec![(PathBuf::from("/notes"), vec![], true)]),
            "/?q=fail",
        )
        .await;
        assert!(none.contains("No indexed documents matched"));
        assert!(failed.contains("Search is temporarily unavailable"));
        assert!(!failed.contains("secret") && !failed.contains("/private/index"));
    }

    #[tokio::test]
    async fn dynamic_values_are_escaped() {
        let unsafe_value = "<script>alert(1)</script>\"";
        let result = Candidate {
            path: unsafe_value.to_owned(),
            snippet: unsafe_value.to_owned(),
            locator: Some(unsafe_value.to_owned()),
        };
        let body = get(
            app(vec![(PathBuf::from("/notes"), vec![result], false)]),
            "/?q=%3Cscript%3Ealert%281%29%3C%2Fscript%3E%22",
        )
        .await;
        assert!(!body.contains(unsafe_value));
        assert!(body.contains("&lt;script&gt;alert(1)&lt;/script&gt;&quot;"));
    }
}
