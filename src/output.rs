use std::io::Write;

use serde::Serialize;

use crate::{
    config::Namespace,
    control::StatusRoot,
    rank::FusedResult,
    similar::{RelatedResult, SimilarGroup, SimilarPair},
};

#[derive(Clone, Copy)]
pub enum OutputMode {
    Human { paths_only: bool },
    Json,
}

#[derive(Serialize)]
struct NamespaceOutput {
    name: String,
    description: Option<String>,
    paths: Vec<String>,
}

pub struct SearchResult {
    pub fused: FusedResult,
    pub namespaces: Vec<String>,
}

pub struct Failure {
    pub directory: String,
    pub path: String,
    pub error: String,
}

#[derive(Serialize)]
struct FailureOutput {
    directory: String,
    path: String,
    error: String,
}

#[derive(Serialize)]
struct JsonSearchResult<'a> {
    path: &'a str,
    score: f64,
    snippet: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    locator: Option<&'a str>,
    namespaces: &'a [String],
}

pub fn print_namespaces(
    namespaces: &[Namespace],
    mode: OutputMode,
    writer: &mut impl Write,
) -> anyhow::Result<()> {
    let namespaces = namespace_output(namespaces);
    match mode {
        OutputMode::Json => {
            let json = serde_json::to_string(&namespaces)?;
            writeln!(writer, "{json}")?;
        }
        OutputMode::Human { .. } => {
            for namespace in namespaces {
                if let Some(description) = namespace.description {
                    writeln!(writer, "{}\t{description}", namespace.name)?;
                } else {
                    writeln!(writer, "{}", namespace.name)?;
                }
                for path in namespace.paths {
                    writeln!(writer, "  {path}")?;
                }
            }
        }
    }
    Ok(())
}

pub fn print_failures(
    failures: &[Failure],
    mode: OutputMode,
    writer: &mut impl Write,
) -> anyhow::Result<()> {
    let failures = failure_output(failures);
    match mode {
        OutputMode::Json => {
            let json = serde_json::to_string(&failures)?;
            writeln!(writer, "{json}")?;
        }
        OutputMode::Human { .. } => {
            let mut directory = None;
            for failure in failures {
                if directory.as_deref() != Some(failure.directory.as_str()) {
                    writeln!(writer, "{}", failure.directory)?;
                    directory = Some(failure.directory.clone());
                }
                writeln!(writer, "  {}\t{}", failure.path, failure.error)?;
            }
        }
    }
    Ok(())
}

pub fn print_status(
    status: &[StatusRoot],
    mode: OutputMode,
    writer: &mut impl Write,
) -> anyhow::Result<()> {
    match mode {
        OutputMode::Json => {
            let json = serde_json::to_string(status)?;
            writeln!(writer, "{json}")?;
        }
        OutputMode::Human { .. } => {
            for root in status {
                let uptime = root
                    .uptime_seconds
                    .map_or_else(|| "-".to_owned(), |seconds| format!("{seconds}s"));
                writeln!(
                    writer,
                    "{}\twatcher: {}\tuptime: {uptime}\tdocuments: {}\tchunks: {}\tpreparation failures: {}",
                    root.directory,
                    if root.watcher_live {
                        "live"
                    } else {
                        "not live"
                    },
                    root.documents,
                    root.chunks,
                    root.preparation_failures,
                )?;
            }
        }
    }
    Ok(())
}

fn failure_output(failures: &[Failure]) -> Vec<FailureOutput> {
    let mut failures: Vec<&Failure> = failures.iter().collect();
    failures.sort_unstable_by(|left, right| {
        (&left.directory, &left.path, &left.error).cmp(&(
            &right.directory,
            &right.path,
            &right.error,
        ))
    });
    failures
        .into_iter()
        .map(|failure| FailureOutput {
            directory: failure.directory.clone(),
            path: failure.path.clone(),
            error: failure.error.clone(),
        })
        .collect()
}

fn namespace_output(namespaces: &[Namespace]) -> Vec<NamespaceOutput> {
    let mut namespaces: Vec<&Namespace> = namespaces.iter().collect();
    namespaces.sort_unstable_by(|left, right| left.name.cmp(&right.name));
    namespaces
        .into_iter()
        .map(|namespace| {
            let mut paths: Vec<String> = namespace
                .paths
                .iter()
                .map(|path| path.to_string_lossy().into_owned())
                .collect();
            paths.sort_unstable();
            NamespaceOutput {
                name: namespace.name.clone(),
                description: namespace.description.clone(),
                paths,
            }
        })
        .collect()
}

/// Print search results to `writer`.
///
/// # Errors
///
/// Returns an error if serialization fails or if writing to `writer` fails.
pub fn print_search(
    results: &[SearchResult],
    mode: OutputMode,
    writer: &mut impl Write,
) -> anyhow::Result<()> {
    match mode {
        OutputMode::Json => {
            let json_results: Vec<JsonSearchResult<'_>> = results
                .iter()
                .map(|result| JsonSearchResult {
                    path: &result.fused.path,
                    score: result.fused.score,
                    snippet: &result.fused.snippet,
                    locator: result.fused.locator.as_deref(),
                    namespaces: &result.namespaces,
                })
                .collect();
            let json = serde_json::to_string(&json_results)?;
            writeln!(writer, "{json}")?;
        }
        OutputMode::Human { paths_only } => {
            if paths_only {
                for search_result in results {
                    writeln!(writer, "{}", search_result.fused.path)?;
                }
            } else {
                for search_result in results {
                    let result = &search_result.fused;
                    if let Some(locator) = &result.locator {
                        writeln!(
                            writer,
                            "{:.4}\t{}\t{locator}: {}",
                            result.score,
                            result.path,
                            first_line(&result.snippet)
                        )?;
                    } else {
                        writeln!(
                            writer,
                            "{:.4}\t{}\t{}",
                            result.score,
                            result.path,
                            first_line(&result.snippet)
                        )?;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Print similar pairs or groups to `writer`.
///
/// # Errors
///
/// Returns an error if serialization fails or if writing to `writer` fails.
pub fn print_similar(
    pairs: Vec<SimilarPair>,
    limit: usize,
    group: bool,
    mode: OutputMode,
    writer: &mut impl Write,
) -> anyhow::Result<()> {
    match mode {
        OutputMode::Json => {
            if group {
                let mut groups: Vec<SimilarGroup> = crate::similar::group_pairs(pairs);
                groups.truncate(limit);
                let json_str = serde_json::to_string(&groups)?;
                writeln!(writer, "{json_str}")?;
            } else {
                let json_str = serde_json::to_string(&pairs)?;
                writeln!(writer, "{json_str}")?;
            }
        }
        OutputMode::Human { paths_only } => {
            if group {
                let mut groups = crate::similar::group_pairs(pairs);
                groups.truncate(limit);
                if paths_only {
                    for g in &groups {
                        for path in &g.paths {
                            writeln!(writer, "{path}")?;
                        }
                    }
                } else {
                    for (i, g) in groups.iter().enumerate() {
                        if i > 0 {
                            writeln!(writer)?;
                        }
                        writeln!(writer, "Group {} ({} documents):", i + 1, g.paths.len())?;
                        for pair in &g.pairs {
                            writeln!(
                                writer,
                                "  {:.4}  {} <> {}",
                                pair.similarity, pair.path_a, pair.path_b
                            )?;
                        }
                    }
                }
            } else if paths_only {
                for pair in &pairs {
                    writeln!(writer, "{}", pair.path_a)?;
                    writeln!(writer, "{}", pair.path_b)?;
                }
            } else {
                for pair in &pairs {
                    writeln!(
                        writer,
                        "{:.4}\t{}\t{}",
                        pair.similarity, pair.path_a, pair.path_b
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// Print related results to `writer`.
///
/// # Errors
///
/// Returns an error if serialization fails or if writing to `writer` fails.
pub fn print_related(
    results: &[RelatedResult],
    mode: OutputMode,
    writer: &mut impl Write,
) -> anyhow::Result<()> {
    match mode {
        OutputMode::Json => {
            let json_str = serde_json::to_string(results)?;
            writeln!(writer, "{json_str}")?;
        }
        OutputMode::Human { paths_only } => {
            if paths_only {
                for r in results {
                    writeln!(writer, "{}", r.path)?;
                }
            } else {
                for r in results {
                    writeln!(writer, "{:.4}\t{}", r.similarity, r.path)?;
                }
            }
        }
    }
    Ok(())
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn make_namespace(name: &str, description: Option<&str>, paths: &[&str]) -> Namespace {
        Namespace {
            name: name.to_owned(),
            description: description.map(str::to_owned),
            paths: paths.iter().map(PathBuf::from).collect(),
        }
    }

    fn make_failure(directory: &str, path: &str, error: &str) -> Failure {
        Failure {
            directory: directory.to_owned(),
            path: path.to_owned(),
            error: error.to_owned(),
        }
    }

    fn make_status(directory: &str, watcher_live: bool) -> StatusRoot {
        StatusRoot {
            directory: directory.to_owned(),
            watcher_live,
            uptime_seconds: watcher_live.then_some(12),
            documents: 3,
            chunks: 7,
            preparation_failures: 1,
        }
    }

    fn make_fused(path: &str, score: f64, snippet: &str) -> SearchResult {
        SearchResult {
            fused: FusedResult {
                path: path.to_owned(),
                score,
                snippet: snippet.to_owned(),
                locator: None,
            },
            namespaces: Vec::new(),
        }
    }

    fn make_fused_with_locator(
        path: &str,
        score: f64,
        snippet: &str,
        locator: &str,
    ) -> SearchResult {
        SearchResult {
            fused: FusedResult {
                path: path.to_owned(),
                score,
                snippet: snippet.to_owned(),
                locator: Some(locator.to_owned()),
            },
            namespaces: Vec::new(),
        }
    }

    fn make_pair(sim: f64, a: &str, b: &str) -> SimilarPair {
        SimilarPair {
            similarity: sim,
            path_a: a.to_owned(),
            path_b: b.to_owned(),
        }
    }

    fn make_related(path: &str, similarity: f64) -> RelatedResult {
        RelatedResult {
            path: path.to_owned(),
            similarity,
        }
    }

    fn write_to_string(f: impl FnOnce(&mut Vec<u8>) -> anyhow::Result<()>) -> String {
        let mut buf = Vec::new();
        f(&mut buf).expect("write should succeed");
        String::from_utf8(buf).expect("output should be valid UTF-8")
    }

    #[test]
    fn namespaces_json_is_stable_and_complete() {
        let namespaces = vec![
            make_namespace("zeta", None, &["/docs/shared"]),
            make_namespace(
                "alpha",
                Some("Alpha documentation"),
                &["/docs/z", "/docs/a"],
            ),
        ];
        let output =
            write_to_string(|writer| print_namespaces(&namespaces, OutputMode::Json, writer));
        assert_eq!(
            output,
            "[{\"name\":\"alpha\",\"description\":\"Alpha documentation\",\"paths\":[\"/docs/a\",\"/docs/z\"]},{\"name\":\"zeta\",\"description\":null,\"paths\":[\"/docs/shared\"]}]\n"
        );
    }

    #[test]
    fn namespaces_human_output_is_stable_and_readable() {
        let namespaces = vec![
            make_namespace("zeta", None, &["/docs/shared"]),
            make_namespace(
                "alpha",
                Some("Alpha documentation"),
                &["/docs/z", "/docs/a"],
            ),
        ];
        let output = write_to_string(|writer| {
            print_namespaces(&namespaces, OutputMode::Human { paths_only: false }, writer)
        });
        assert_eq!(
            output,
            "alpha\tAlpha documentation\n  /docs/a\n  /docs/z\nzeta\n  /docs/shared\n"
        );
    }

    #[test]
    fn namespaces_json_empty_output_is_an_array() {
        let output = write_to_string(|writer| print_namespaces(&[], OutputMode::Json, writer));
        assert_eq!(output, "[]\n");
    }

    #[test]
    fn failures_human_output_groups_sorted_entries_by_directory() {
        let failures = vec![
            make_failure("/docs/zeta", "z.md", "z error"),
            make_failure("/docs/alpha", "z.md", "second error"),
            make_failure("/docs/alpha", "a.md", "first error"),
        ];
        let output = write_to_string(|writer| {
            print_failures(&failures, OutputMode::Human { paths_only: false }, writer)
        });
        assert_eq!(
            output,
            "/docs/alpha\n  a.md\tfirst error\n  z.md\tsecond error\n/docs/zeta\n  z.md\tz error\n"
        );
    }

    #[test]
    fn failures_json_output_is_sorted_and_complete() {
        let failures = vec![
            make_failure("/docs/zeta", "z.md", "z error"),
            make_failure("/docs/alpha", "a.md", "first error"),
        ];
        let output = write_to_string(|writer| print_failures(&failures, OutputMode::Json, writer));
        assert_eq!(
            output,
            "[{\"directory\":\"/docs/alpha\",\"path\":\"a.md\",\"error\":\"first error\"},{\"directory\":\"/docs/zeta\",\"path\":\"z.md\",\"error\":\"z error\"}]\n"
        );
    }

    #[test]
    fn status_output_reports_watcher_and_index_counts() {
        let status = vec![make_status("/notes", true), make_status("/archive", false)];
        let human = write_to_string(|writer| {
            print_status(&status, OutputMode::Human { paths_only: false }, writer)
        });
        assert_eq!(
            human,
            "/notes\twatcher: live\tuptime: 12s\tdocuments: 3\tchunks: 7\tpreparation failures: 1\n/archive\twatcher: not live\tuptime: -\tdocuments: 3\tchunks: 7\tpreparation failures: 1\n"
        );

        let json = write_to_string(|writer| print_status(&status, OutputMode::Json, writer));
        let value: serde_json::Value =
            serde_json::from_str(json.trim()).expect("parse status JSON");
        assert_eq!(value[0]["watcher_live"], true);
        assert_eq!(value[1]["uptime_seconds"], serde_json::Value::Null);
        assert_eq!(value[0]["documents"], 3);
        assert_eq!(value[0]["chunks"], 7);
        assert_eq!(value[0]["preparation_failures"], 1);
    }

    #[test]
    fn search_json_output_is_valid_json_array() {
        let results = vec![
            make_fused("a.md", 0.9, "snippet a"),
            make_fused("b.md", 0.8, "snippet b"),
        ];
        let out = write_to_string(|w| print_search(&results, OutputMode::Json, w));
        let value: serde_json::Value =
            serde_json::from_str(out.trim()).expect("output should parse as JSON");
        assert!(value.is_array(), "output should be a JSON array");
    }

    #[test]
    fn search_json_includes_all_namespace_memberships() {
        let mut result = make_fused("a.md", 0.9, "snippet");
        result.namespaces = vec!["alpha".to_owned(), "shared".to_owned()];
        let out = write_to_string(|writer| print_search(&[result], OutputMode::Json, writer));
        let value: serde_json::Value =
            serde_json::from_str(out.trim()).expect("output should parse as JSON");
        let namespaces = value[0]["namespaces"].as_array().expect("namespaces array");
        assert_eq!(namespaces[0], "alpha");
        assert_eq!(namespaces[1], "shared");
    }

    #[test]
    fn search_json_includes_full_snippet() {
        let snippet = "line one\nline two\nline three";
        let results = vec![make_fused("a.md", 0.9, snippet)];
        let out = write_to_string(|w| print_search(&results, OutputMode::Json, w));
        let value: serde_json::Value =
            serde_json::from_str(out.trim()).expect("output should parse as JSON");
        let stored = value[0]["snippet"]
            .as_str()
            .expect("snippet should be a string");
        assert_eq!(
            stored, snippet,
            "JSON snippet should contain the full multi-line text"
        );
    }

    #[test]
    fn search_json_includes_locators_and_omits_them_when_absent() {
        let results = vec![
            make_fused_with_locator("semantic.md", 0.9, "semantic chunk", "Introduction"),
            make_fused_with_locator("fts.md", 0.8, "fts chunk", "p. 12"),
            make_fused_with_locator("fused.md", 0.7, "fused chunk", "Methods"),
            make_fused("plain.txt", 0.6, "plain chunk"),
            make_fused_with_locator("/store/note.md", 0.5, "store chunk", "Appendix"),
        ];
        let out = write_to_string(|w| print_search(&results, OutputMode::Json, w));
        let value: serde_json::Value =
            serde_json::from_str(out.trim()).expect("output should parse as JSON");

        assert_eq!(value[0]["locator"], "Introduction");
        assert_eq!(value[1]["locator"], "p. 12");
        assert_eq!(value[2]["locator"], "Methods");
        assert!(value[3].get("locator").is_none());
        assert_eq!(value[4]["locator"], "Appendix");
    }

    #[test]
    fn search_human_includes_locators_in_snippet_column() {
        let results = vec![
            make_fused_with_locator("semantic.md", 0.9, "semantic chunk", "Introduction"),
            make_fused_with_locator("fts.md", 0.8, "fts chunk", "p. 12"),
            make_fused_with_locator("fused.md", 0.7, "fused chunk", "Methods"),
            make_fused("plain.txt", 0.6, "plain chunk"),
            make_fused_with_locator("/store/note.md", 0.5, "store chunk", "Appendix"),
        ];
        let out =
            write_to_string(|w| print_search(&results, OutputMode::Human { paths_only: false }, w));
        let lines: Vec<&str> = out.lines().collect();

        assert_eq!(lines.len(), 5);
        assert_eq!(
            lines[0],
            "0.9000\tsemantic.md\tIntroduction: semantic chunk"
        );
        assert_eq!(lines[1], "0.8000\tfts.md\tp. 12: fts chunk");
        assert_eq!(lines[2], "0.7000\tfused.md\tMethods: fused chunk");
        assert_eq!(lines[3], "0.6000\tplain.txt\tplain chunk");
        assert_eq!(lines[4], "0.5000\t/store/note.md\tAppendix: store chunk");
        assert!(lines.iter().all(|line| line.matches('\t').count() == 2));
    }

    #[test]
    fn search_json_empty_results_produces_empty_array() {
        let out = write_to_string(|w| print_search(&[], OutputMode::Json, w));
        assert_eq!(out.trim(), "[]");
    }

    #[test]
    fn search_human_paths_only_still_prints_paths() {
        let results = vec![make_fused_with_locator(
            "notes/a.md",
            0.9,
            "some snippet",
            "Introduction",
        )];
        let out =
            write_to_string(|w| print_search(&results, OutputMode::Human { paths_only: true }, w));
        assert_eq!(out, "notes/a.md\n");
    }

    #[test]
    fn similar_flat_human_output_is_unchanged() {
        let pairs = vec![
            make_pair(0.95, "a.md", "b.md"),
            make_pair(0.90, "b.md", "c.md"),
        ];
        let output = write_to_string(|writer| {
            print_similar(
                pairs,
                10,
                false,
                OutputMode::Human { paths_only: false },
                writer,
            )
        });
        assert_eq!(output, "0.9500\ta.md\tb.md\n0.9000\tb.md\tc.md\n");
    }

    #[test]
    fn similar_grouped_human_output_is_unchanged() {
        let pairs = vec![
            make_pair(0.95, "a.md", "b.md"),
            make_pair(0.90, "b.md", "c.md"),
        ];
        let output = write_to_string(|writer| {
            print_similar(
                pairs,
                10,
                true,
                OutputMode::Human { paths_only: false },
                writer,
            )
        });
        assert_eq!(
            output,
            "Group 1 (3 documents):\n  0.9500  a.md <> b.md\n  0.9000  b.md <> c.md\n"
        );
    }

    #[test]
    fn similar_paths_only_outputs_are_unchanged() {
        let flat = write_to_string(|writer| {
            print_similar(
                vec![
                    make_pair(0.95, "a.md", "b.md"),
                    make_pair(0.90, "b.md", "c.md"),
                ],
                10,
                false,
                OutputMode::Human { paths_only: true },
                writer,
            )
        });
        let grouped = write_to_string(|writer| {
            print_similar(
                vec![
                    make_pair(0.95, "a.md", "b.md"),
                    make_pair(0.90, "b.md", "c.md"),
                ],
                10,
                true,
                OutputMode::Human { paths_only: true },
                writer,
            )
        });
        assert_eq!(flat, "a.md\nb.md\nb.md\nc.md\n");
        assert_eq!(grouped, "a.md\nb.md\nc.md\n");
    }

    #[test]
    fn similar_flat_json_has_correct_fields() {
        let pairs = vec![make_pair(0.95, "a.md", "b.md")];
        let out = write_to_string(|w| print_similar(pairs, 10, false, OutputMode::Json, w));
        let value: serde_json::Value =
            serde_json::from_str(out.trim()).expect("output should parse as JSON");
        assert!(value.is_array());
        let obj = &value[0];
        assert!(obj["path_a"].is_string(), "should have path_a key");
        assert!(obj["path_b"].is_string(), "should have path_b key");
        assert!(obj["similarity"].is_number(), "should have similarity key");
    }

    #[test]
    fn similar_grouped_json_has_paths_and_pairs() {
        let pairs = vec![
            make_pair(0.95, "a.md", "b.md"),
            make_pair(0.90, "b.md", "c.md"),
        ];
        let out = write_to_string(|w| print_similar(pairs, 10, true, OutputMode::Json, w));
        let value: serde_json::Value =
            serde_json::from_str(out.trim()).expect("output should parse as JSON");
        assert!(value.is_array());
        let group = &value[0];
        assert!(group["paths"].is_array(), "group should have paths array");
        assert!(group["pairs"].is_array(), "group should have pairs array");
        let pair = &group["pairs"][0];
        assert!(pair["path_a"].is_string());
        assert!(pair["path_b"].is_string());
        assert!(pair["similarity"].is_number());
    }

    #[test]
    fn related_json_has_correct_fields() {
        let results = vec![make_related("b.md", 0.97)];
        let out = write_to_string(|w| print_related(&results, OutputMode::Json, w));
        let value: serde_json::Value =
            serde_json::from_str(out.trim()).expect("output should parse as JSON");
        assert!(value.is_array());
        let obj = &value[0];
        assert!(obj["path"].is_string(), "should have path key");
        assert!(obj["similarity"].is_number(), "should have similarity key");
    }

    #[test]
    fn json_flag_takes_precedence_over_paths_only() {
        let results = vec![make_fused("a.md", 0.9, "snippet text")];
        let out = write_to_string(|w| print_search(&results, OutputMode::Json, w));
        let value: serde_json::Value =
            serde_json::from_str(out.trim()).expect("output should parse as JSON");
        let obj = &value[0];
        assert!(obj["path"].is_string(), "Json mode should include path");
        assert!(obj["score"].is_number(), "Json mode should include score");
        assert!(
            obj["snippet"].is_string(),
            "Json mode should include snippet"
        );
    }
}
