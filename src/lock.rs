use std::{
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::Context;
use fs2::FileExt;

/// An exclusive cross-process lock on the needle index for a given data directory.
///
/// The lock is backed by an OS-level `flock(2)` on a `.lock` file co-located
/// with the database.  The lock is held for as long as this value is alive and
/// is released automatically when it is dropped (the fd is closed).
///
/// Use `try_acquire` rather than a blocking acquire: if another process already
/// holds the lock (e.g. `needle watch`) the caller should report an error rather
/// than block indefinitely.
#[derive(Debug)]
pub struct IndexLock {
    file: File,
    path: PathBuf,
}

impl IndexLock {
    /// Attempt to acquire the exclusive index lock without blocking.
    ///
    /// Returns an error if the lock is already held by another process, or if
    /// the lock file cannot be created or opened.
    pub fn try_acquire(db_path: &Path) -> anyhow::Result<Self> {
        let path = lock_path(db_path);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&path)
            .with_context(|| format!("opening index lock file {}", path.display()))?;
        file.try_lock_exclusive().map_err(|e| {
            if e.kind() == std::io::ErrorKind::WouldBlock {
                anyhow::anyhow!(
                    "another needle process is already running; \
                     stop it before running reindex or watch"
                )
            } else {
                anyhow::Error::from(e).context(format!("acquiring index lock {}", path.display()))
            }
        })?;
        Ok(Self { file, path })
    }
}

impl Drop for IndexLock {
    fn drop(&mut self) {
        if let Err(e) = self.file.unlock() {
            tracing::warn!(path = %self.path.display(), err = %e, "failed to release index lock");
        }
    }
}

fn lock_path(db_path: &Path) -> PathBuf {
    db_path.with_extension("lock")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_creates_lock_file_and_reports_its_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("needle.db");
        let lock = IndexLock::try_acquire(&db_path).expect("acquire");
        assert!(
            lock.path.exists(),
            "lock file must exist while lock is held"
        );
        assert_eq!(lock.path, dir.path().join("needle.lock"));
    }

    #[test]
    fn lock_file_path_replaces_db_extension() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("needle.db");
        assert_eq!(lock_path(&db_path), dir.path().join("needle.lock"));
    }

    #[test]
    fn second_acquire_from_same_process_fails_with_helpful_message() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("needle.db");

        // flock on Linux is per open-file-description, so two separate opens
        // from the same process do conflict.
        let _first = IndexLock::try_acquire(&db_path).expect("first acquire");
        let second = IndexLock::try_acquire(&db_path);

        let err = second.expect_err("second acquire must fail while first is held");
        let msg = err.to_string();
        assert!(
            msg.contains("another needle process"),
            "error must name the conflict: {msg}"
        );
    }

    #[test]
    fn lock_is_released_on_drop() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("needle.db");

        {
            let _lock = IndexLock::try_acquire(&db_path).expect("first acquire");
        }

        IndexLock::try_acquire(&db_path).expect("re-acquire after drop must succeed");
    }
}
