use sha2::{Digest, Sha256};

pub fn content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let result = hasher.finalize();
    hex_encode(&result)
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            use std::fmt::Write;
            let _ = write!(s, "{b:02x}");
            s
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_content_produces_same_hash() {
        let h1 = content_hash("hello world");
        let h2 = content_hash("hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_content_produces_different_hash() {
        let h1 = content_hash("hello");
        let h2 = content_hash("world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn hash_is_64_hex_characters() {
        let h = content_hash("test");
        assert_eq!(h.len(), 64, "SHA256 should produce 64 hex chars");
        assert!(
            h.chars().all(|c| c.is_ascii_hexdigit()),
            "hash should only contain hex characters"
        );
    }

    #[test]
    fn empty_string_produces_known_sha256() {
        let h = content_hash("");
        assert_eq!(
            h, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "empty string SHA256 should match known value"
        );
    }

    #[test]
    fn hash_is_lowercase_hex() {
        let h = content_hash("UPPERCASE");
        assert!(
            h.chars().all(|c| !c.is_ascii_uppercase()),
            "hex output should be lowercase"
        );
    }

    #[test]
    fn whitespace_changes_produce_different_hashes() {
        let h1 = content_hash("hello world");
        let h2 = content_hash("hello  world");
        let h3 = content_hash("hello world\n");
        assert_ne!(h1, h2);
        assert_ne!(h1, h3);
    }
}
