use std::path::Path;

#[cfg(feature = "documents")]
use anyhow::Context;

#[cfg(not(feature = "documents"))]
const CHUNK_TARGET_CHARS: usize = 4000;

pub trait DocumentPreparer: Send + Sync {
    fn supports_path(&self, source_path: &Path) -> bool;
    fn prepare(&self, source_path: &Path, source: &[u8]) -> anyhow::Result<Vec<String>>;
    fn profile(&self) -> &'static str;
}

#[cfg(not(feature = "documents"))]
#[derive(Default)]
pub struct MarkdownPreparer;

#[cfg(not(feature = "documents"))]
impl DocumentPreparer for MarkdownPreparer {
    fn supports_path(&self, source_path: &Path) -> bool {
        source_path
            .extension()
            .is_some_and(|extension| extension == "md")
    }

    fn prepare(&self, _source_path: &Path, source: &[u8]) -> anyhow::Result<Vec<String>> {
        let content = std::str::from_utf8(source)?;
        Ok(chunk_text(content))
    }

    fn profile(&self) -> &'static str {
        "markdown-v1"
    }
}

#[cfg(feature = "documents")]
#[derive(Default)]
pub struct XbergPreparer;

#[cfg(feature = "documents")]
impl DocumentPreparer for XbergPreparer {
    fn supports_path(&self, source_path: &Path) -> bool {
        source_path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .is_some_and(|extension| {
                ["md", "markdown", "txt"]
                    .iter()
                    .any(|supported| extension.eq_ignore_ascii_case(supported))
            })
    }

    fn prepare(&self, source_path: &Path, source: &[u8]) -> anyhow::Result<Vec<String>> {
        let mime_type = mime_type(source_path).expect("supported paths have a MIME type");
        let input = xberg::ExtractInput::from_bytes(
            source.to_vec(),
            mime_type,
            Some(source_path.to_string_lossy().to_string()),
        );
        let extraction = extract_with_xberg(input)
            .with_context(|| format!("extracting {} with Xberg", source_path.display()))?;

        if let Some(error) = extraction.errors.first() {
            anyhow::bail!(
                "extracting {} with Xberg: {}",
                source_path.display(),
                error.message
            );
        }

        Ok(extraction
            .results
            .into_iter()
            .flat_map(|document| {
                for warning in document.processing_warnings {
                    tracing::warn!(
                        path = %source_path.display(),
                        source = %warning.source,
                        message = %warning.message,
                        "Xberg processing warning"
                    );
                }
                document
                    .chunks
                    .unwrap_or_default()
                    .into_iter()
                    .map(|chunk| chunk.content)
            })
            .collect())
    }

    fn profile(&self) -> &'static str {
        "xberg-1.0.14-render-markdown-chunker-markdown-max-1000-overlap-200-trim-sizing-characters-heading-context-page-extraction-cache-off"
    }
}

#[cfg(feature = "documents")]
fn mime_type(source_path: &Path) -> Option<&'static str> {
    match source_path.extension().and_then(std::ffi::OsStr::to_str) {
        Some(extension) if extension.eq_ignore_ascii_case("md") => Some("text/markdown"),
        Some(extension) if extension.eq_ignore_ascii_case("markdown") => Some("text/markdown"),
        Some(extension) if extension.eq_ignore_ascii_case("txt") => Some("text/plain"),
        _ => None,
    }
}

#[cfg(feature = "documents")]
fn extract_with_xberg(input: xberg::ExtractInput) -> xberg::Result<xberg::ExtractionResult> {
    let config = xberg::ExtractionConfig {
        use_cache: false,
        output_format: xberg::OutputFormat::Markdown,
        chunking: Some(xberg::ChunkingConfig {
            chunker_type: xberg::ChunkerType::Markdown,
            prepend_heading_context: true,
            ..Default::default()
        }),
        pages: Some(xberg::PageConfig {
            extract_pages: true,
            ..Default::default()
        }),
        ..Default::default()
    };

    if tokio::runtime::Handle::try_current().is_ok() {
        return std::thread::scope(|scope| {
            scope
                .spawn(|| xberg_runtime().block_on(xberg::extract(input, &config)))
                .join()
                .map_err(|_| {
                    xberg::XbergError::Other("Xberg extraction thread panicked".to_owned())
                })?
        });
    }

    xberg_runtime().block_on(xberg::extract(input, &config))
}

#[cfg(feature = "documents")]
fn xberg_runtime() -> &'static tokio::runtime::Runtime {
    static RUNTIME: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("creating Xberg runtime must succeed")
    })
}

#[cfg(feature = "documents")]
pub type DefaultPreparer = XbergPreparer;

#[cfg(not(feature = "documents"))]
pub type DefaultPreparer = MarkdownPreparer;

#[cfg(not(feature = "documents"))]
fn chunk_text(content: &str) -> Vec<String> {
    let content = strip_frontmatter(content);
    let paragraphs: Vec<&str> = content.split("\n\n").collect();
    let mut chunks = Vec::new();
    let mut current = String::new();

    for paragraph in paragraphs {
        if current.len() + paragraph.len() > CHUNK_TARGET_CHARS && !current.is_empty() {
            chunks.push(std::mem::take(&mut current));
        }

        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(paragraph);
    }

    if !current.is_empty() {
        chunks.push(current);
    }

    if chunks.is_empty() {
        chunks.push(String::new());
    }

    chunks
}

#[cfg(not(feature = "documents"))]
fn strip_frontmatter(content: &str) -> &str {
    if !content.starts_with("---") {
        return content;
    }
    let after_open = &content[3..];
    let Some(close_pos) = after_open.find("\n---") else {
        return content;
    };

    let body = after_open[..close_pos].trim();
    if !is_yaml_frontmatter(body) {
        return content;
    }

    content[close_pos + 7..].trim_start()
}

#[cfg(not(feature = "documents"))]
fn is_yaml_frontmatter(block: &str) -> bool {
    block.is_empty() || block.lines().any(|line| line.trim().contains(": "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "documents"))]
    #[test]
    fn markdown_preparer_reports_stable_profile() {
        assert_eq!(MarkdownPreparer.profile(), "markdown-v1");
    }

    #[cfg(not(feature = "documents"))]
    #[test]
    fn markdown_preparer_supports_only_markdown_paths() {
        assert!(MarkdownPreparer.supports_path(Path::new("note.md")));
        assert!(!MarkdownPreparer.supports_path(Path::new("note.txt")));
    }

    #[cfg(not(feature = "documents"))]
    #[test]
    fn markdown_preparer_strips_frontmatter_before_chunking() {
        let source = b"---\ntitle: Note\n---\n\n# Heading\n\nBody text";
        let chunks = MarkdownPreparer
            .prepare(Path::new("note.md"), source)
            .expect("prepare");
        assert_eq!(chunks, vec!["# Heading\n\nBody text"]);
    }

    #[cfg(not(feature = "documents"))]
    #[test]
    fn markdown_preparer_rejects_invalid_utf8() {
        let result = MarkdownPreparer.prepare(Path::new("note.md"), &[0xff]);
        assert!(result.is_err());
    }

    #[cfg(feature = "documents")]
    #[test]
    fn xberg_preparer_supports_markdown_and_text_case_insensitively() {
        for path in ["note.md", "note.MARKDOWN", "note.Txt"] {
            assert!(XbergPreparer.supports_path(Path::new(path)), "{path}");
        }
        assert!(!XbergPreparer.supports_path(Path::new("note.html")));
    }

    #[cfg(feature = "documents")]
    #[test]
    fn xberg_preparer_uses_a_distinct_profile() {
        assert_ne!(XbergPreparer.profile(), "markdown-v1");
        assert!(XbergPreparer.profile().contains("xberg-1.0.14"));
    }

    #[cfg(feature = "documents")]
    #[test]
    fn xberg_preparer_returns_no_chunks_for_empty_documents() {
        let chunks = XbergPreparer
            .prepare(Path::new("empty.md"), b"")
            .expect("prepare");
        assert!(chunks.is_empty());
    }

    #[cfg(feature = "documents")]
    #[test]
    fn xberg_preparer_extracts_markdown_with_heading_context() {
        let chunks = XbergPreparer
            .prepare(Path::new("note.md"), b"# Heading\n\nBody text")
            .expect("prepare");
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("Heading"));
        assert!(chunks[0].contains("Body text"));
    }

    #[cfg(not(feature = "documents"))]
    #[test]
    fn large_file_splits_on_paragraph_boundaries() {
        let paragraph = "a".repeat(3000);
        let content = format!("{paragraph}\n\n{paragraph}\n\n{paragraph}");
        let chunks = chunk_text(&content);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| !chunk.is_empty()));
    }

    #[cfg(not(feature = "documents"))]
    #[test]
    fn empty_content_produces_one_chunk() {
        assert_eq!(chunk_text(""), vec![String::new()]);
    }

    #[cfg(not(feature = "documents"))]
    #[test]
    fn chunk_preserves_all_content() {
        let content = "paragraph one\n\nparagraph two\n\nparagraph three";
        assert_eq!(chunk_text(content).join("\n\n"), content);
    }

    #[cfg(not(feature = "documents"))]
    #[test]
    fn strip_frontmatter_preserves_non_yaml_block() {
        let content = "---\nthis is not yaml\n---\n\nBody";
        assert_eq!(strip_frontmatter(content), content);
    }
}
