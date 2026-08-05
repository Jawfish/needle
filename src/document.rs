use std::path::Path;

const CHUNK_TARGET_CHARS: usize = 4000;

pub trait DocumentPreparer: Send + Sync {
    fn supports_path(&self, source_path: &Path) -> bool;
    fn prepare(&self, source_path: &Path, source: &[u8]) -> anyhow::Result<Vec<String>>;
    fn profile(&self) -> &'static str;
}

pub struct MarkdownPreparer;

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

fn is_yaml_frontmatter(block: &str) -> bool {
    block.is_empty() || block.lines().any(|line| line.trim().contains(": "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markdown_preparer_reports_stable_profile() {
        assert_eq!(MarkdownPreparer.profile(), "markdown-v1");
    }

    #[test]
    fn markdown_preparer_supports_only_markdown_paths() {
        assert!(MarkdownPreparer.supports_path(Path::new("note.md")));
        assert!(!MarkdownPreparer.supports_path(Path::new("note.txt")));
    }

    #[test]
    fn markdown_preparer_strips_frontmatter_before_chunking() {
        let source = b"---\ntitle: Note\n---\n\n# Heading\n\nBody text";
        let chunks = MarkdownPreparer
            .prepare(Path::new("note.md"), source)
            .expect("prepare");
        assert_eq!(chunks, vec!["# Heading\n\nBody text"]);
    }

    #[test]
    fn markdown_preparer_rejects_invalid_utf8() {
        let result = MarkdownPreparer.prepare(Path::new("note.md"), &[0xff]);
        assert!(result.is_err());
    }

    #[test]
    fn large_file_splits_on_paragraph_boundaries() {
        let paragraph = "a".repeat(3000);
        let content = format!("{paragraph}\n\n{paragraph}\n\n{paragraph}");
        let chunks = chunk_text(&content);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| !chunk.is_empty()));
    }

    #[test]
    fn empty_content_produces_one_chunk() {
        assert_eq!(chunk_text(""), vec![String::new()]);
    }

    #[test]
    fn chunk_preserves_all_content() {
        let content = "paragraph one\n\nparagraph two\n\nparagraph three";
        assert_eq!(chunk_text(content).join("\n\n"), content);
    }

    #[test]
    fn strip_frontmatter_preserves_non_yaml_block() {
        let content = "---\nthis is not yaml\n---\n\nBody";
        assert_eq!(strip_frontmatter(content), content);
    }
}
