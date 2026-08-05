use std::{
    net::IpAddr,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

use anyhow::Context as _;
use axum::{
    Router,
    extract::{RawQuery, State},
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::get,
};
use maud::{DOCTYPE, Markup, PreEscaped, html};
use percent_encoding::{NON_ALPHANUMERIC, percent_decode_str, utf8_percent_encode};

use crate::{
    config::{self, Namespace},
    db::{DbDocumentChunksSource, DbPathSource, DbSemanticSource, DocumentChunksSource},
    document::PreparedChunk,
    embed::Embedder,
    fts::FtsFtsSource,
    query::{self, SearchStorePorts},
    rank::{FtsSource, PathSource, RrfWeights, SemanticSource},
    search_merge,
};

const RESULT_LIMIT: usize = 20;
const CSS: &str = "body{max-width:52rem;margin:2rem auto;padding:0 1rem;font-family:system-ui,sans-serif;color:#17202a;background:#fafafa}form{display:grid;gap:.75rem}.search-row{display:flex;gap:.5rem}.search-row input{flex:1;padding:.6rem;font:inherit}button{padding:.6rem 1rem;font:inherit}fieldset{display:flex;flex-wrap:wrap;gap:.5rem;border:1px solid #d5d8dc;border-radius:.4rem;padding:.5rem .75rem}legend{padding:0 .25rem}.namespace{display:flex;align-items:flex-start;gap:.4rem;padding:.25rem .4rem}.namespace input{margin-top:.25rem}.namespace-description{display:block;color:#52616b;font-size:.875rem}article{background:white;border:1px solid #d5d8dc;border-radius:.4rem;margin:1rem 0;padding:1rem}h1{margin-top:0}.path{font-family:monospace;overflow-wrap:anywhere}.locator{color:#52616b}.snippet{white-space:pre-wrap}.state{padding:1rem;background:#eef3f5;border-radius:.4rem}mark{background:#ffe082}";

pub struct SearchAdapter {
    pub semantic: DbSemanticSource,
    pub fts: FtsFtsSource,
    pub paths: DbPathSource,
    pub documents: DbDocumentChunksSource,
}

struct SearchStore {
    notes_dir: PathBuf,
    semantic: Arc<dyn SemanticSource>,
    fts: Arc<dyn FtsSource>,
    paths: Arc<dyn PathSource>,
    documents: Arc<dyn DocumentChunksSource>,
}

struct AppState {
    stores: Vec<SearchStore>,
    namespaces: Vec<Namespace>,
    embedder: Option<Arc<Embedder>>,
    weights: RrfWeights,
}

struct SearchParams {
    q: String,
    namespace: Vec<String>,
}

pub async fn run(
    host: IpAddr,
    port: u16,
    stores: &[crate::config::DirectoryStore],
    namespaces: &[Namespace],
    adapters: Vec<SearchAdapter>,
    embedder: Option<Embedder>,
    weights: RrfWeights,
) -> anyhow::Result<()> {
    let state = Arc::new(AppState::from_adapters(
        stores, namespaces, adapters, embedder, weights,
    ));
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
    axum::Server::from_tcp(
        listener
            .into_std()
            .context("preparing HTTP server listener")?,
    )?
    .serve(router(state).into_make_service())
    .with_graceful_shutdown(async move {
        if shutdown.await.is_ok() {
            tracing::info!("shutting down");
        }
    })
    .await
    .context("running HTTP server")
}

impl AppState {
    fn from_adapters(
        stores: &[crate::config::DirectoryStore],
        namespaces: &[Namespace],
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
                documents: Arc::new(adapter.documents),
            })
            .collect();
        Self {
            stores,
            namespaces: namespaces.to_vec(),
            embedder: embedder.map(Arc::new),
            weights,
        }
    }
}

fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(home))
        .route("/document", get(document))
        .with_state(state)
}

fn selected_stores<'a>(
    state: &'a AppState,
    selected_names: &[String],
) -> anyhow::Result<Vec<(usize, &'a SearchStore)>> {
    let selected_paths = config::selected_namespace_paths(&state.namespaces, selected_names)?;
    Ok(state
        .stores
        .iter()
        .enumerate()
        .filter(|(_, store)| {
            selected_paths.as_ref().is_none_or(|paths| {
                paths
                    .iter()
                    .any(|path| path.as_path() == store.notes_dir.as_path())
            })
        })
        .collect())
}

fn parse_search_query(raw: Option<&str>) -> Result<SearchParams, ()> {
    let mut query = None;
    let mut namespace = Vec::new();
    for pair in raw.unwrap_or_default().split('&') {
        if pair.is_empty() {
            continue;
        }
        let (key, value) = pair.split_once('=').ok_or(())?;
        let key = decode_query(key)?;
        let value = decode_query(value)?;
        match key.as_str() {
            "q" if query.is_none() => query = Some(value),
            "namespace" => namespace.push(value),
            _ => return Err(()),
        }
    }
    Ok(SearchParams {
        q: query.unwrap_or_default(),
        namespace,
    })
}

async fn home(State(state): State<Arc<AppState>>, RawQuery(raw): RawQuery) -> Response {
    let Ok(params) = parse_search_query(raw.as_deref()) else {
        return error_page(
            StatusCode::BAD_REQUEST,
            "Invalid search request.",
            "",
            &state.namespaces,
            &[],
        );
    };
    let query = params.q;
    let stores = match selected_stores(&state, &params.namespace) {
        Ok(stores) => stores,
        Err(error) => {
            return error_page(
                StatusCode::BAD_REQUEST,
                &error.to_string(),
                &query,
                &state.namespaces,
                &params.namespace,
            );
        }
    };
    if query.trim().is_empty() {
        return Html(page(
            &query,
            &state.namespaces,
            &params.namespace,
            &html! { p class="state" { "Search your indexed documents." } },
        ))
        .into_response();
    }
    let ports: Vec<SearchStorePorts<'_>> = stores
        .iter()
        .map(|(_, store)| SearchStorePorts {
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
                html! { p { "Showing up to " (RESULT_LIMIT) " ranked results." } @for result in results {
                    @if let Some((store, path)) = result_store(&stores, &result.path) {
                        article { div class="path" { a href=(document_url(store, path, &query, &params.namespace, result.locator.as_deref())) { (result.path) } }
                            (result.locator.as_ref().map_or_else(|| html! {}, |locator| html! { div class="locator" { (locator) } }))
                            p class="snippet" { (result.snippet) }
                        }
                    }
                }}
            }
        }
        Err(_) => {
            html! { p class="state" { "Search is temporarily unavailable. Please try again." } }
        }
    };
    Html(page(&query, &state.namespaces, &params.namespace, &content)).into_response()
}

async fn document(State(state): State<Arc<AppState>>, RawQuery(raw): RawQuery) -> Response {
    let Ok(params) = parse_document_query(raw.as_deref()) else {
        return error_page(
            StatusCode::BAD_REQUEST,
            "Invalid document request.",
            "",
            &state.namespaces,
            &[],
        );
    };
    let Some(store) = state.stores.get(params.store) else {
        return error_page(
            StatusCode::NOT_FOUND,
            "Document not found.",
            &params.query,
            &state.namespaces,
            &params.namespace,
        );
    };
    match store.documents.document_chunks(&params.path).await {
        Ok(Some(chunks)) => Html(document_page(
            &params.path,
            &params.query,
            &params.namespace,
            params.locator.as_deref(),
            &chunks,
            &state.namespaces,
        ))
        .into_response(),
        Ok(None) => error_page(
            StatusCode::NOT_FOUND,
            "Document not found.",
            &params.query,
            &state.namespaces,
            &params.namespace,
        ),
        Err(error) => {
            tracing::error!(%error, "reading indexed document failed");
            error_page(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Document is temporarily unavailable.",
                &params.query,
                &state.namespaces,
                &params.namespace,
            )
        }
    }
}

struct DocumentParams {
    store: usize,
    path: String,
    query: String,
    namespace: Vec<String>,
    locator: Option<String>,
}

fn parse_document_query(raw: Option<&str>) -> Result<DocumentParams, ()> {
    let mut store = None;
    let mut path = None;
    let mut query = None;
    let mut namespace = Vec::new();
    let mut locator = None;
    for pair in raw.unwrap_or_default().split('&') {
        let (key, value) = pair.split_once('=').ok_or(())?;
        let key = decode_query(key)?;
        let value = decode_query(value)?;
        match key.as_str() {
            "store" if store.is_none() => store = Some(value.parse().map_err(|_| ())?),
            "path" if path.is_none() => path = Some(value),
            "q" if query.is_none() => query = Some(value),
            "namespace" => namespace.push(value),
            "locator" if locator.is_none() => locator = Some(value),
            _ => return Err(()),
        }
    }
    let path = path.ok_or(())?;
    if !is_clean_relative_path(&path) {
        return Err(());
    }
    Ok(DocumentParams {
        store: store.ok_or(())?,
        path,
        query: query.unwrap_or_default(),
        namespace,
        locator,
    })
}

fn decode_query(value: &str) -> Result<String, ()> {
    if value.as_bytes().windows(1).enumerate().any(|(i, byte)| {
        byte == b"%"
            && (i + 2 >= value.len()
                || !value.as_bytes()[i + 1].is_ascii_hexdigit()
                || !value.as_bytes()[i + 2].is_ascii_hexdigit())
    }) {
        return Err(());
    }
    percent_decode_str(&value.replace('+', " "))
        .decode_utf8()
        .map(std::borrow::Cow::into_owned)
        .map_err(|_| ())
}

fn is_clean_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.contains('\0')
        && Path::new(path)
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
}

fn result_store<'a>(stores: &[(usize, &SearchStore)], path: &'a str) -> Option<(usize, &'a str)> {
    if let [(store, _)] = stores {
        return Some((*store, path));
    }
    stores.iter().find_map(|(id, store)| {
        Path::new(path)
            .strip_prefix(&store.notes_dir)
            .ok()
            .and_then(|relative| relative.to_str())
            .map(|relative| (*id, relative))
    })
}

fn document_url(
    store: usize,
    path: &str,
    query: &str,
    namespaces: &[String],
    locator: Option<&str>,
) -> String {
    let mut url = format!(
        "/document?store={store}&path={}&q={}",
        encode(path),
        encode(query)
    );
    append_namespace_params(&mut url, namespaces);
    if let Some(locator) = locator {
        url.push_str("&locator=");
        url.push_str(&encode(locator));
    }
    url.push_str("#match");
    url
}

fn search_url(query: &str, namespaces: &[String]) -> String {
    let mut url = format!("/?q={}", encode(query));
    append_namespace_params(&mut url, namespaces);
    url
}

fn append_namespace_params(url: &mut String, namespaces: &[String]) {
    for namespace in namespaces {
        url.push_str("&namespace=");
        url.push_str(&encode(namespace));
    }
}

fn encode(value: &str) -> String {
    utf8_percent_encode(value, NON_ALPHANUMERIC).to_string()
}

fn document_page(
    path: &str,
    query: &str,
    selected_names: &[String],
    locator: Option<&str>,
    chunks: &[PreparedChunk],
    namespaces: &[Namespace],
) -> String {
    let matched = locator
        .and_then(|needle| {
            chunks
                .iter()
                .position(|chunk| chunk.locator.as_deref() == Some(needle))
        })
        .or_else(|| {
            chunks
                .iter()
                .position(|chunk| contains_term(&chunk.content, query))
        });
    let content = html! { p { a href=(search_url(query, selected_names)) { "Back to results" } } h2 class="path" { (path) }
        @if chunks.is_empty() { p class="state" { "This indexed document has no chunks." } }
        @for (index, chunk) in chunks.iter().enumerate() { article id=[if Some(index) == matched { Some("match") } else { None }] {
            (chunk.locator.as_ref().map_or_else(|| html! {}, |locator| html! { div class="locator" { (locator) } }))
            p class="snippet" { (highlight(&chunk.content, query)) }
        }}
    };
    page(query, namespaces, selected_names, &content)
}

fn terms(query: &str) -> Vec<&str> {
    let mut terms: Vec<&str> = query.split_whitespace().collect();
    terms.sort_unstable_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    terms.dedup();
    terms
}

fn contains_term(content: &str, query: &str) -> bool {
    terms(query).iter().any(|term| content.contains(term))
}

fn highlight(content: &str, query: &str) -> Markup {
    let terms = terms(query);
    let mut parts = Vec::new();
    let mut offset = 0;
    while offset < content.len() {
        if let Some(term) = terms
            .iter()
            .find(|term| content[offset..].starts_with(**term))
        {
            parts.push((true, &content[offset..offset + term.len()]));
            offset += term.len();
        } else {
            let width = content[offset..].chars().next().map_or(0, char::len_utf8);
            parts.push((false, &content[offset..offset + width]));
            offset += width;
        }
    }
    html! { @for (is_match, part) in parts { @if is_match { mark { (part) } } @else { (part) } } }
}

fn error_page(
    status: StatusCode,
    message: &str,
    query: &str,
    namespaces: &[Namespace],
    selected_names: &[String],
) -> Response {
    (
        status,
        Html(page(
            query,
            namespaces,
            selected_names,
            &html! { p class="state" { (message) } },
        )),
    )
        .into_response()
}

fn page(
    query: &str,
    namespaces: &[Namespace],
    selected_names: &[String],
    content: &Markup,
) -> String {
    html! { (DOCTYPE) html lang="en" { head { meta charset="utf-8"; meta name="viewport" content="width=device-width, initial-scale=1"; title { "Needle search" } style { (PreEscaped(CSS)) } } body { h1 { "Needle" } form method="get" action="/" { label for="q" { "Search indexed documents" } div class="search-row" { input id="q" name="q" type="search" value=(query) autofocus; button type="submit" { "Search" } } @if !namespaces.is_empty() { fieldset { legend { "Documentation namespaces" } @for namespace in namespaces { label class="namespace" { input type="checkbox" name="namespace" value=(&namespace.name) checked[selected_names.iter().any(|name| name == &namespace.name)]; span { (namespace.name) @if let Some(description) = &namespace.description { span class="namespace-description" { (description) } } } } } } } } (content) } } }.into_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::Namespace,
        rank::{Candidate, SearchFuture},
    };
    use axum::{body::Body, http::Request};
    use hyper::body::to_bytes;
    use std::{collections::HashMap, path::PathBuf};
    use tower::ServiceExt;

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
    struct FakeDocuments {
        documents: HashMap<String, Vec<PreparedChunk>>,
        fails: bool,
    }
    impl DocumentChunksSource for FakeDocuments {
        fn document_chunks<'a>(
            &'a self,
            path: &'a str,
        ) -> SearchFuture<'a, Option<Vec<PreparedChunk>>> {
            let result = self.documents.get(path).cloned();
            let fails = self.fails;
            Box::pin(async move {
                if fails {
                    anyhow::bail!("secret /database/path");
                }
                Ok(result)
            })
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
    type FakeStore = (
        PathBuf,
        Vec<Candidate>,
        HashMap<String, Vec<PreparedChunk>>,
        bool,
    );

    fn app(stores: Vec<FakeStore>) -> Router {
        app_with_namespaces(stores, Vec::new())
    }

    fn app_with_namespaces(stores: Vec<FakeStore>, namespaces: Vec<Namespace>) -> Router {
        let stores = stores
            .into_iter()
            .map(|(notes_dir, results, documents, fails)| SearchStore {
                paths: Arc::new(FakePaths(results.iter().map(|r| r.path.clone()).collect())),
                notes_dir,
                semantic: Arc::new(FakeSemantic),
                fts: Arc::new(FakeFts { results, fails }),
                documents: Arc::new(FakeDocuments { documents, fails }),
            })
            .collect();
        router(Arc::new(AppState {
            stores,
            namespaces,
            embedder: None,
            weights: RrfWeights {
                semantic: 0.0,
                fts: 1.0,
                filename: 0.0,
            },
        }))
    }
    async fn response(app: Router, uri: &str) -> (StatusCode, String) {
        let response = app
            .oneshot(Request::get(uri).body(Body::empty()).expect("request"))
            .await
            .expect("response");
        let status = response.status();
        let body = String::from_utf8(to_bytes(response.into_body()).await.expect("body").to_vec())
            .expect("utf8");
        (status, body)
    }
    fn namespace(name: &str, description: Option<&str>, paths: &[&str]) -> Namespace {
        Namespace {
            name: name.to_owned(),
            description: description.map(str::to_owned),
            paths: paths.iter().map(PathBuf::from).collect(),
        }
    }

    fn store(
        path: &str,
        results: Vec<Candidate>,
        documents: &[(&str, Vec<PreparedChunk>)],
    ) -> (
        PathBuf,
        Vec<Candidate>,
        HashMap<String, Vec<PreparedChunk>>,
        bool,
    ) {
        (
            PathBuf::from(path),
            results,
            documents
                .iter()
                .map(|(path, chunks)| ((*path).to_owned(), chunks.clone()))
                .collect(),
            false,
        )
    }

    #[tokio::test]
    async fn empty_query_renders_home_state() {
        let body = response(app(vec![]), "/?q=%20%20").await.1;
        assert!(body.contains("Search your indexed documents."));
    }

    #[tokio::test]
    async fn namespaces_are_visible_and_selected_in_the_search_form() {
        let application = app_with_namespaces(
            vec![store("/alpha", vec![], &[]), store("/shared", vec![], &[])],
            vec![
                namespace("alpha", Some("Alpha documentation"), &["/alpha", "/shared"]),
                namespace("shared", None, &["/shared"]),
            ],
        );
        let body = response(application, "/?q=note&namespace=alpha&namespace=shared")
            .await
            .1;

        assert!(body.contains("Documentation namespaces"));
        assert!(body.contains("Alpha documentation"));
        assert!(body.contains("value=\"alpha\" checked"));
        assert!(body.contains("value=\"shared\" checked"));
    }

    #[tokio::test]
    async fn unscoped_browser_searches_all_stores() {
        let application = app_with_namespaces(
            vec![
                store("/alpha", vec![candidate("alpha.md")], &[]),
                store("/beta", vec![candidate("beta.md")], &[]),
            ],
            vec![
                namespace("alpha", None, &["/alpha"]),
                namespace("beta", None, &["/beta"]),
            ],
        );
        let body = response(application, "/?q=note").await.1;

        assert!(body.contains("/alpha/alpha.md"));
        assert!(body.contains("/beta/beta.md"));
    }

    #[tokio::test]
    async fn selected_namespaces_search_their_deduplicated_store_union() {
        let application = app_with_namespaces(
            vec![
                store("/alpha", vec![candidate("alpha.md")], &[]),
                store("/shared", vec![candidate("shared.md")], &[]),
                store("/beta", vec![candidate("beta.md")], &[]),
            ],
            vec![
                namespace("alpha", None, &["/alpha", "/shared"]),
                namespace("beta", None, &["/shared", "/beta"]),
            ],
        );
        let body = response(
            application,
            "/?q=note&namespace=alpha&namespace=beta&namespace=alpha",
        )
        .await
        .1;

        assert_eq!(body.matches("class=\"path\"").count(), 3);
        assert!(body.contains("/alpha/alpha.md"));
        assert!(body.contains("/shared/shared.md"));
        assert!(body.contains("/beta/beta.md"));
    }

    #[tokio::test]
    async fn unknown_namespace_returns_an_actionable_bad_request() {
        let application = app_with_namespaces(
            vec![store("/alpha", vec![candidate("alpha.md")], &[])],
            vec![
                namespace("beta", None, &["/beta"]),
                namespace("alpha", None, &["/alpha"]),
            ],
        );
        let (status, body) = response(application, "/?q=note&namespace=missing").await;

        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("unknown namespace") && body.contains("missing"));
        assert!(body.contains("available namespaces: alpha, beta"));
        assert!(!body.contains("alpha.md"));
    }

    #[tokio::test]
    async fn document_links_preserve_namespace_selection_and_store_routing() {
        let chunks = vec![PreparedChunk {
            content: "shared document".to_owned(),
            locator: None,
        }];
        let application = app_with_namespaces(
            vec![
                store("/alpha", vec![candidate("alpha.md")], &[]),
                store(
                    "/shared",
                    vec![candidate("note.md")],
                    &[("note.md", chunks)],
                ),
            ],
            vec![
                namespace("alpha", None, &["/alpha", "/shared"]),
                namespace("shared", None, &["/shared"]),
            ],
        );
        let body = response(
            application.clone(),
            "/?q=note&namespace=alpha&namespace=shared",
        )
        .await
        .1;
        assert!(body.contains("store=1"));
        assert!(body.contains("namespace=alpha&amp;namespace=shared"));

        let document = response(
            application,
            "/document?store=1&path=note.md&q=note&namespace=alpha&namespace=shared",
        )
        .await
        .1;
        assert!(document.contains("shared document"));
        assert!(document.contains("href=\"/?q=note&amp;namespace=alpha&amp;namespace=shared\""));
    }

    #[tokio::test]
    async fn results_show_path_snippet_and_locator() {
        let body = response(
            app(vec![store("/notes", vec![candidate("note.md")], &[])]),
            "/?q=note",
        )
        .await
        .1;
        assert!(
            body.contains("note.md")
                && body.contains("snippet note.md")
                && body.contains("locator note.md")
        );
    }

    #[tokio::test]
    async fn results_are_limited_to_twenty() {
        let results = (0..25)
            .map(|index| candidate(&format!("note-{index}.md")))
            .collect();
        let body = response(app(vec![store("/notes", results, &[])]), "/?q=note")
            .await
            .1;
        assert_eq!(body.matches("class=\"path\"").count(), RESULT_LIMIT);
    }

    #[tokio::test]
    async fn no_match_and_failure_states_are_distinct() {
        let no_match = response(app(vec![store("/notes", vec![], &[])]), "/?q=none")
            .await
            .1;
        let failure = response(
            app(vec![(
                PathBuf::from("/notes"),
                vec![],
                HashMap::new(),
                true,
            )]),
            "/?q=fail",
        )
        .await
        .1;
        assert!(no_match.contains("No indexed documents matched"));
        assert!(failure.contains("Search is temporarily unavailable"));
    }

    #[tokio::test]
    async fn document_renders_ordered_chunks_and_locators() {
        let chunks = vec![
            PreparedChunk {
                content: "first chunk".to_owned(),
                locator: Some("First heading".to_owned()),
            },
            PreparedChunk {
                content: "second chunk".to_owned(),
                locator: None,
            },
        ];
        let body = response(
            app(vec![store("/notes", vec![], &[("note.md", chunks)])]),
            "/document?store=0&path=note.md&q=second",
        )
        .await
        .1;
        let first = body.find("first chunk").expect("first chunk");
        let second = body
            .find("<mark>second</mark> chunk")
            .expect("highlighted second chunk");
        assert!(first < second);
        assert!(body.contains("First heading") && body.contains("id=\"match\""));
    }

    #[tokio::test]
    async fn results_link_to_encoded_store_relative_documents() {
        let body = response(
            app(vec![store(
                "/notes",
                vec![candidate("nested/a & b.md")],
                &[],
            )]),
            "/?q=a%20b",
        )
        .await
        .1;
        assert!(body.contains("href=\"/document?store=0&amp;path=nested%2Fa%20%26%20b%2Emd&amp;q=a%20b&amp;locator=locator%20nested%2Fa%20%26%20b%2Emd#match\""));
    }
    #[tokio::test]
    async fn duplicate_relative_paths_have_isolated_document_views() {
        let chunks_a = vec![PreparedChunk {
            content: "alpha".to_owned(),
            locator: Some("A".to_owned()),
        }];
        let chunks_b = vec![PreparedChunk {
            content: "beta".to_owned(),
            locator: Some("B".to_owned()),
        }];
        let application = app(vec![
            store("/one", vec![candidate("note.md")], &[("note.md", chunks_a)]),
            store("/two", vec![candidate("note.md")], &[("note.md", chunks_b)]),
        ]);
        let body = response(application.clone(), "/?q=note").await.1;
        assert!(body.contains("store=0") && body.contains("store=1"));
        assert!(
            response(
                application.clone(),
                "/document?store=0&path=note.md&q=alpha"
            )
            .await
            .1
            .contains("alpha")
        );
        assert!(
            response(application, "/document?store=1&path=note.md&q=beta")
                .await
                .1
                .contains("beta")
        );
    }
    #[tokio::test]
    async fn document_escapes_and_highlights_literal_terms_at_locator() {
        let chunks = vec![
            PreparedChunk {
                content: "<tag> a+b a+b".to_owned(),
                locator: Some("<loc>".to_owned()),
            },
            PreparedChunk {
                content: "later".to_owned(),
                locator: None,
            },
        ];
        let body = response(
            app(vec![store("/notes", vec![], &[("note.md", chunks)])]),
            "/document?store=0&path=note.md&q=a%2Bb&locator=%3Cloc%3E",
        )
        .await
        .1;
        assert!(
            body.contains("id=\"match\"")
                && body.contains("&lt;tag&gt; <mark>a+b</mark> <mark>a+b</mark>")
                && body.contains("&lt;loc&gt;")
        );
        assert!(!body.contains("<tag>"));
    }
    #[tokio::test]
    async fn document_uses_query_match_when_locator_is_absent() {
        let chunks = vec![
            PreparedChunk {
                content: "first".to_owned(),
                locator: None,
            },
            PreparedChunk {
                content: "needle [x]".to_owned(),
                locator: None,
            },
        ];
        let body = response(
            app(vec![store("/notes", vec![], &[("note.md", chunks)])]),
            "/document?store=0&path=note.md&q=needle%20%5Bx%5D",
        )
        .await
        .1;
        assert_eq!(body.matches("id=\"match\"").count(), 1);
        assert!(body.contains("<mark>needle</mark> <mark>[x]</mark>"));
    }
    #[tokio::test]
    async fn malformed_and_missing_documents_are_safe_4xx() {
        let application = app(vec![store("/notes", vec![], &[])]);
        for uri in [
            "/document?store=x&path=note.md",
            "/document?store=0&path=../note.md",
            "/document?store=0&path=%2Fetc%2Fpasswd",
            "/document?store=0&path=note%00md",
            "/document?store=3&path=note.md",
            "/document?store=0&path=stale.md",
        ] {
            let (status, body) = response(application.clone(), uri).await;
            assert!(status.is_client_error());
            assert!(!body.contains("/database/path"));
        }
    }
    #[tokio::test]
    async fn document_read_errors_are_sanitized() {
        let application = app(vec![(
            PathBuf::from("/notes"),
            vec![],
            HashMap::new(),
            true,
        )]);
        let (status, body) = response(application, "/document?store=0&path=note.md").await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert!(!body.contains("secret") && !body.contains("/database/path"));
    }
    #[tokio::test]
    async fn search_failure_is_sanitized() {
        let application = app(vec![(
            PathBuf::from("/notes"),
            vec![],
            HashMap::new(),
            true,
        )]);
        let body = response(application, "/?q=fail").await.1;
        assert!(body.contains("Search is temporarily unavailable") && !body.contains("secret"));
    }
}
