use std::io::Write;

use crate::{
    rank::FusedResult,
    similar::{RelatedResult, SimilarGroup, SimilarPair},
};

#[derive(Clone, Copy)]
pub enum OutputMode {
    Human { paths_only: bool },
    Json,
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
    use super::*;

    fn make_fused(path: &str, score: f64, snippet: &str) -> FusedResult {
        FusedResult {
            path: path.to_owned(),
            score,
            snippet: snippet.to_owned(),
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
    fn search_json_empty_results_produces_empty_array() {
        let out = write_to_string(|w| print_search(&[], OutputMode::Json, w));
        assert_eq!(out.trim(), "[]");
    }

    #[test]
    fn search_human_paths_only_still_prints_paths() {
        let results = vec![make_fused("notes/a.md", 0.9, "some snippet")];
        let out =
            write_to_string(|w| print_search(&results, OutputMode::Human { paths_only: true }, w));
        assert!(out.contains("notes/a.md"), "output should contain the path");
        assert!(
            !out.contains("0.9"),
            "paths-only output should not contain the score"
        );
        assert!(
            !out.contains("some snippet"),
            "paths-only output should not contain the snippet"
        );
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
