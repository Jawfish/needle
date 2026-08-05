use std::io::Write;

use serde::Serialize;

use crate::{
    config::Namespace,
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
    results: &[FusedResult],
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
                for result in results {
                    writeln!(writer, "{}", result.path)?;
                }
            } else {
                for result in results {
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

    fn make_fused(path: &str, score: f64, snippet: &str) -> FusedResult {
        FusedResult {
            path: path.to_owned(),
            score,
            snippet: snippet.to_owned(),
            locator: None,
        }
    }

    fn make_fused_with_locator(
        path: &str,
        score: f64,
        snippet: &str,
        locator: &str,
    ) -> FusedResult {
        FusedResult {
            path: path.to_owned(),
            score,
            snippet: snippet.to_owned(),
            locator: Some(locator.to_owned()),
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
