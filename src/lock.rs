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
                contention_error(&path)
            } else {
                anyhow::Error::from(e).context(format!("acquiring index lock {}", path.display()))
            }
        })?;
        let mut lock = Self { file, path };
        if let Err(e) = lock.record_holder() {
            tracing::debug!(path = %lock.path.display(), err = %e, "could not record lock holder");
        }
        Ok(lock)
    }

    fn record_holder(&mut self) -> std::io::Result<()> {
        use std::io::{Seek, SeekFrom, Write};

        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        writeln!(self.file, "{} {}", std::process::id(), command_line())?;
        self.file.flush()
    }
}

fn command_line() -> String {
    std::env::args().collect::<Vec<_>>().join(" ")
}

/// Describe the process holding the lock, so the caller knows what to stop.
///
/// The record is advisory: `flock(2)` remains the authority, so content left
/// behind by a process that died without unlocking never blocks acquisition,
/// and an unreadable or half written record simply falls back to the plain
/// message.
fn contention_error(path: &Path) -> anyhow::Error {
    holder(path).map_or_else(
        || {
            anyhow::anyhow!(
                "another needle process is already running; \
                 stop it before running reindex or watch"
            )
        },
        |holder| {
            anyhow::anyhow!(
                "another needle process is already running ({holder}); \
                 stop it before running reindex or watch"
            )
        },
    )
}

fn holder(path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    let (pid, command) = content.lines().next()?.trim().split_once(' ')?;
    pid.parse::<u32>().ok()?;
    if command.is_empty() {
        return None;
    }
    Some(format!("pid {pid}: {command}"))
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
    fn the_holder_is_recorded_in_the_lock_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("needle.db");

        let lock = IndexLock::try_acquire(&db_path).expect("acquire");

        let record = std::fs::read_to_string(&lock.path).expect("read lock file");
        let (pid, command) = record.trim().split_once(' ').expect("pid and command line");
        assert_eq!(pid, std::process::id().to_string());
        assert!(
            !command.is_empty(),
            "holder record must name a command line"
        );
    }

    #[test]
    fn a_failed_acquire_names_the_process_holding_the_lock() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("needle.db");

        let _first = IndexLock::try_acquire(&db_path).expect("first acquire");
        let err = IndexLock::try_acquire(&db_path).expect_err("second acquire must fail");

        let msg = err.to_string();
        let pid = std::process::id();
        assert!(
            msg.contains(&format!("pid {pid}")),
            "must name the pid: {msg}"
        );
        assert!(
            msg.contains("stop it"),
            "must say what to do about it: {msg}"
        );
    }

    #[test]
    fn a_record_left_by_a_dead_process_does_not_block_acquisition() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("needle.db");
        std::fs::write(dir.path().join("needle.lock"), "4294967294 needle watch\n")
            .expect("write stale record");

        IndexLock::try_acquire(&db_path).expect("a stale record must not block acquisition");
    }

    #[test]
    fn an_unreadable_record_still_reports_the_conflict() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("needle.db");

        let first = IndexLock::try_acquire(&db_path).expect("first acquire");
        std::fs::write(&first.path, "garbage").expect("overwrite record");
        let err = IndexLock::try_acquire(&db_path).expect_err("second acquire must fail");

        assert!(
            err.to_string().contains("another needle process"),
            "error must still name the conflict: {err}"
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
