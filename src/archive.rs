use std::{
    fs::File,
    io::Read as _,
    path::{Path, PathBuf},
};

use anyhow::Context as _;

use crate::hash;

pub const MEMBER_SEPARATOR: char = '!';

const MAX_MEMBER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_COMPRESSION_RATIO: u64 = 200;
const RATIO_EXEMPT_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Member {
    pub index: usize,
    pub name: String,
    pub content_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedMember {
    pub name: String,
    pub content_hash: String,
    pub reason: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Scan {
    pub members: Vec<Member>,
    pub rejected: Vec<RejectedMember>,
}

pub fn is_archive(path: &Path) -> bool {
    path.extension()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|extension| extension.eq_ignore_ascii_case("zip"))
}

pub fn member_path(archive_rel_path: &str, member_name: &str) -> String {
    format!("{archive_rel_path}{MEMBER_SEPARATOR}{member_name}")
}

pub fn archive_prefix(archive_rel_path: &str) -> String {
    format!("{archive_rel_path}{MEMBER_SEPARATOR}")
}

pub fn scan(archive_path: &Path, is_supported: &dyn Fn(&Path) -> bool) -> anyhow::Result<Scan> {
    let file = File::open(archive_path)
        .with_context(|| format!("failed to open {}", archive_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("failed to read {}", archive_path.display()))?;

    let mut scan = Scan::default();
    for index in 0..archive.len() {
        let entry = archive.by_index_raw(index).with_context(|| {
            format!("failed to read entry {index} of {}", archive_path.display())
        })?;
        if entry.is_dir() {
            continue;
        }
        let name = entry.name().to_owned();
        if is_ignored(&name) || !is_supported(Path::new(&name)) {
            continue;
        }
        let content_hash = member_hash(&name, entry.crc32(), entry.size());
        let rejection = rejection_reason(&entry);
        drop(entry);

        match rejection {
            Some(reason) => scan.rejected.push(RejectedMember {
                name,
                content_hash,
                reason,
            }),
            None => scan.members.push(Member {
                index,
                name,
                content_hash,
            }),
        }
    }
    Ok(scan)
}

pub fn read_member(archive_path: &Path, index: usize, name: &str) -> anyhow::Result<Vec<u8>> {
    let file = File::open(archive_path)
        .with_context(|| format!("failed to open {}", archive_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("failed to read {}", archive_path.display()))?;
    let mut entry = archive
        .by_index(index)
        .with_context(|| format!("failed to read entry {index} of {}", archive_path.display()))?;
    anyhow::ensure!(
        entry.name() == name,
        "entry {index} of {} is {}, expected {name}",
        archive_path.display(),
        entry.name()
    );

    let mut source = Vec::with_capacity(usize::try_from(entry.size().min(MAX_MEMBER_BYTES))?);
    entry
        .by_ref()
        .take(MAX_MEMBER_BYTES)
        .read_to_end(&mut source)
        .with_context(|| format!("failed to extract {name} from {}", archive_path.display()))?;
    Ok(source)
}

fn member_hash(name: &str, crc32: u32, uncompressed_size: u64) -> String {
    hash::content_hash(&format!(
        "zip-member:{name}:{crc32:08x}:{uncompressed_size}"
    ))
}

fn rejection_reason<R: std::io::Read>(entry: &zip::read::ZipFile<'_, R>) -> Option<String> {
    if entry.encrypted() {
        return Some("member is encrypted".to_owned());
    }
    if !matches!(
        entry.compression(),
        zip::CompressionMethod::Stored | zip::CompressionMethod::Deflated
    ) {
        return Some(format!(
            "unsupported compression method {}",
            entry.compression()
        ));
    }
    let size = entry.size();
    if size > MAX_MEMBER_BYTES {
        return Some(format!(
            "member is {size} bytes, larger than the {MAX_MEMBER_BYTES} byte limit"
        ));
    }
    let compressed_size = entry.compressed_size();
    if size > RATIO_EXEMPT_BYTES && size / compressed_size.max(1) > MAX_COMPRESSION_RATIO {
        return Some(format!(
            "member expands {}x, beyond the {MAX_COMPRESSION_RATIO}x limit",
            size / compressed_size.max(1)
        ));
    }
    None
}

fn is_ignored(name: &str) -> bool {
    let path = PathBuf::from(name);
    path.components().any(|component| {
        let component = component.as_os_str().to_string_lossy();
        component.starts_with('.') || component == "__MACOSX"
    })
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use super::*;

    fn markdown_only(path: &Path) -> bool {
        path.extension().is_some_and(|extension| extension == "md")
    }

    fn write_archive(entries: &[(&str, &[u8])]) -> tempfile::NamedTempFile {
        let file = tempfile::Builder::new()
            .suffix(".zip")
            .tempfile()
            .expect("temp file");
        let mut writer = zip::write::ZipWriter::new(file.reopen().expect("reopen"));
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for (name, content) in entries {
            writer.start_file(*name, options).expect("start entry");
            writer.write_all(content).expect("write entry");
        }
        writer.finish().expect("finish archive");
        file
    }

    #[test]
    fn only_zip_extensions_are_archives() {
        assert!(is_archive(Path::new("docs.zip")));
        assert!(is_archive(Path::new("docs.ZIP")));
        assert!(!is_archive(Path::new("book.epub")));
        assert!(!is_archive(Path::new("notes.md")));
    }

    #[test]
    fn member_paths_join_archive_and_entry() {
        assert_eq!(member_path("docs.zip", "a/b.md"), "docs.zip!a/b.md");
        assert_eq!(archive_prefix("docs.zip"), "docs.zip!");
    }

    #[test]
    fn supported_members_are_listed_without_extraction() {
        let archive = write_archive(&[
            ("notes/one.md", b"# one"),
            ("image.png", b"not a document"),
            ("notes/two.md", b"# two"),
        ]);

        let scan = scan(archive.path(), &markdown_only).expect("scan");

        let names: Vec<&str> = scan.members.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["notes/one.md", "notes/two.md"]);
        assert!(scan.rejected.is_empty());
    }

    #[test]
    fn hidden_and_macos_metadata_members_are_skipped() {
        let archive = write_archive(&[
            ("__MACOSX/._one.md", b"junk"),
            (".hidden/two.md", b"junk"),
            ("three.md", b"# three"),
        ]);

        let scan = scan(archive.path(), &markdown_only).expect("scan");

        let names: Vec<&str> = scan.members.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["three.md"]);
    }

    #[test]
    fn member_hashes_change_only_when_the_member_changes() {
        let first = write_archive(&[("one.md", b"# one"), ("two.md", b"# two")]);
        let second = write_archive(&[("one.md", b"# one"), ("two.md", b"# two changed")]);

        let before = scan(first.path(), &markdown_only).expect("scan");
        let after = scan(second.path(), &markdown_only).expect("scan");

        assert_eq!(
            before.members[0].content_hash, after.members[0].content_hash,
            "an untouched member keeps its hash"
        );
        assert_ne!(
            before.members[1].content_hash, after.members[1].content_hash,
            "an edited member gets a new hash"
        );
    }

    #[test]
    fn members_are_extracted_one_at_a_time() {
        let archive = write_archive(&[("one.md", b"# one"), ("two.md", b"# two")]);
        let scan = scan(archive.path(), &markdown_only).expect("scan");

        let second = &scan.members[1];
        let source = read_member(archive.path(), second.index, &second.name).expect("read member");

        assert_eq!(source, b"# two");
    }

    #[test]
    fn reading_a_member_that_moved_fails_instead_of_indexing_the_wrong_text() {
        let archive = write_archive(&[("one.md", b"# one"), ("two.md", b"# two")]);

        let error = read_member(archive.path(), 1, "one.md").expect_err("mismatch should fail");

        assert!(
            error.to_string().contains("expected one.md"),
            "error should name the expected member: {error}"
        );
    }

    #[test]
    fn oversized_members_are_rejected_with_a_reason() {
        let mut content = Vec::new();
        content.resize(usize::try_from(RATIO_EXEMPT_BYTES).expect("fits") + 1, b'a');
        let archive = write_archive(&[("bomb.md", &content)]);

        let scan = scan(archive.path(), &markdown_only).expect("scan");

        assert!(scan.members.is_empty());
        assert_eq!(scan.rejected.len(), 1);
        assert!(
            scan.rejected[0].reason.contains("expands"),
            "reason should explain the rejection: {}",
            scan.rejected[0].reason
        );
    }

    #[test]
    fn a_corrupt_archive_reports_an_error() {
        let file = tempfile::Builder::new()
            .suffix(".zip")
            .tempfile()
            .expect("temp file");
        std::fs::write(file.path(), b"not an archive").expect("write");

        assert!(scan(file.path(), &markdown_only).is_err());
    }
}
