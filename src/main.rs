mod cli;
mod config;
mod db;
mod embed;
mod error;
mod fts;
mod hash;
mod index;
mod lock;
mod rank;
mod similar;
mod types;
mod watch;

use std::io::{IsTerminal, Read};

use anyhow::Context;
use clap::Parser;

use crate::{
    cli::{Cli, Command},
    config::{CliEmbedArgs, CliWeights, Config},
    embed::Embedder,
    error::NeedleError,
};

fn is_dimension_mismatch(err: &anyhow::Error) -> bool {
    err.downcast_ref::<NeedleError>()
        .is_some_and(|e| matches!(e, NeedleError::DimensionMismatch { .. }))
}

async fn open_db(
    db_path: &std::path::Path,
    dim: Option<usize>,
) -> anyhow::Result<(libsql::Database, libsql::Connection)> {
    db::connect(db_path, dim).await
}

/// Run a full reindex, recovering atomically when there is a dimension mismatch.
///
/// When dimensions match, opens the DB and FTS index normally and runs the
/// reindex in place.
///
/// When they differ, both the DB and FTS index are rebuilt into sibling temp
/// locations.  Only after the full reindex succeeds are both swapped over the
/// originals with rename(2).  On any failure both temp artifacts are removed
/// and the originals are left untouched.
async fn run_reindex(
    db_path: &std::path::Path,
    fts_dir: &std::path::Path,
    embedder: &embed::Embedder,
    notes_dir: &std::path::Path,
) -> anyhow::Result<index::IndexStats> {
    let dim = embedder.dim();

    match db::connect(db_path, Some(dim)).await {
        Ok((_db, conn)) => {
            let fts = fts::FtsIndex::open_or_create(fts_dir)?;
            let stats = index::index_directory(&conn, &fts, embedder, notes_dir).await?;
            Ok(stats)
        }
        Err(ref e) if is_dimension_mismatch(e) => {
            tracing::warn!("dimension mismatch detected; rebuilding index atomically");
            reindex_via_temp(db_path, fts_dir, embedder, notes_dir, dim).await
        }
        Err(e) => Err(e),
    }
}

async fn reindex_via_temp(
    db_path: &std::path::Path,
    fts_dir: &std::path::Path,
    embedder: &embed::Embedder,
    notes_dir: &std::path::Path,
    dim: usize,
) -> anyhow::Result<index::IndexStats> {
    let tmp_db_path = db_path.with_extension("reindex-tmp");
    let tmp_fts_dir = sibling_fts_dir(fts_dir, "reindex-tmp");
    let backup_fts_dir = sibling_fts_dir(fts_dir, "reindex-bak");

    // Crash recovery must run before cleanup: if backup exists and live FTS is
    // absent, the backup is the only copy of the live index.  Restore it now so
    // that a subsequent build failure does not leave the index permanently gone.
    recover_fts_from_interrupted_swap(fts_dir, &backup_fts_dir)?;

    clean_temp_artifacts(&tmp_db_path, &tmp_fts_dir);

    let result = build_index_in_temp(&tmp_db_path, &tmp_fts_dir, embedder, notes_dir, dim).await;

    match result {
        Ok(stats) => {
            promote_temp_artifacts(
                &tmp_db_path,
                db_path,
                &tmp_fts_dir,
                fts_dir,
                &backup_fts_dir,
            )?;
            Ok(stats)
        }
        Err(e) => {
            clean_temp_artifacts(&tmp_db_path, &tmp_fts_dir);
            Err(e)
        }
    }
}

/// Restore the FTS backup when the live FTS directory is missing.
///
/// `promote_temp_artifacts` moves the live FTS aside before swapping in the
/// new one.  If the process is killed between those two steps the backup holds
/// the only copy of the index.  This function detects that state and renames
/// the backup back to the live path so subsequent cleanup cannot delete it.
fn recover_fts_from_interrupted_swap(
    fts_dir: &std::path::Path,
    backup_fts_dir: &std::path::Path,
) -> anyhow::Result<()> {
    if backup_fts_dir.exists() && !fts_dir.exists() {
        tracing::warn!(
            backup = %backup_fts_dir.display(),
            "FTS backup found without live FTS dir; prior reindex was interrupted; restoring"
        );
        std::fs::rename(backup_fts_dir, fts_dir).with_context(|| {
            format!(
                "restoring FTS backup {} to {}",
                backup_fts_dir.display(),
                fts_dir.display()
            )
        })?;
    }
    Ok(())
}

/// Promote fully-built temp artifacts into the live locations.
///
/// Linux rename(2) cannot replace a non-empty directory, so the live FTS
/// directory must be vacated before the temp one can be renamed into place.
/// Steps are ordered so that failure at any point leaves a state that can be
/// rolled back to the original:
///
///   1. Move live FTS to a backup path (vacates the target path for step 2).
///   2. Move temp FTS into the now-vacant live FTS location.
///   3. Move temp DB over the live DB file (file-over-file rename is atomic).
///   4. Remove the FTS backup (best-effort cleanup).
///
/// The DB rename is last. If the FTS steps succeed but the DB rename fails,
/// rolling back the FTS restores full consistency without touching the DB.
fn promote_temp_artifacts(
    tmp_db_path: &std::path::Path,
    db_path: &std::path::Path,
    tmp_fts_dir: &std::path::Path,
    fts_dir: &std::path::Path,
    backup_fts_dir: &std::path::Path,
) -> anyhow::Result<()> {
    // Step 1: Move live FTS aside to vacate the target path.
    //
    // A stale backup can exist when a prior run crashed after step 2 (temp FTS
    // renamed into place) but before step 4 (backup removed).  In that state
    // both live and backup exist; the live FTS is correct and the backup is
    // obsolete.  Remove it before the rename so Linux rename(2) does not return
    // ENOTEMPTY when the backup directory is non-empty.
    if backup_fts_dir.exists() {
        std::fs::remove_dir_all(backup_fts_dir).with_context(|| {
            format!(
                "removing stale FTS backup {} before promotion",
                backup_fts_dir.display()
            )
        })?;
    }
    let live_fts_existed = fts_dir.exists();
    if live_fts_existed {
        std::fs::rename(fts_dir, backup_fts_dir).with_context(|| {
            format!(
                "backing up live FTS dir {} to {}",
                fts_dir.display(),
                backup_fts_dir.display()
            )
        })?;
    }

    // Step 2: Move temp FTS into the now-empty live location.
    if let Err(e) = std::fs::rename(tmp_fts_dir, fts_dir) {
        if live_fts_existed && let Err(rb) = std::fs::rename(backup_fts_dir, fts_dir) {
            tracing::error!(
                err = %rb,
                "failed to restore FTS backup after swap error; index may be inconsistent"
            );
        }
        return Err(e).with_context(|| {
            format!(
                "moving temp FTS {} into live location {}",
                tmp_fts_dir.display(),
                fts_dir.display()
            )
        });
    }

    // Step 3: Move temp DB over the live DB (file-over-file rename is atomic).
    if let Err(e) = std::fs::rename(tmp_db_path, db_path) {
        // Roll back FTS: move new FTS back to temp, then restore backup.
        if let Err(rb) = std::fs::rename(fts_dir, tmp_fts_dir) {
            tracing::error!(
                err = %rb,
                "failed to revert FTS during DB swap rollback; index may be inconsistent"
            );
        } else if live_fts_existed && let Err(rb) = std::fs::rename(backup_fts_dir, fts_dir) {
            tracing::error!(
                err = %rb,
                "failed to restore FTS backup during DB rollback; index may be inconsistent"
            );
        }
        return Err(e).with_context(|| {
            format!(
                "swapping temp DB {} over live DB {}",
                tmp_db_path.display(),
                db_path.display()
            )
        });
    }

    // Step 4: Remove FTS backup (best effort; state is already consistent).
    if backup_fts_dir.exists()
        && let Err(e) = std::fs::remove_dir_all(backup_fts_dir)
    {
        tracing::warn!(
            path = %backup_fts_dir.display(),
            err = %e,
            "failed to remove FTS backup after successful swap"
        );
    }

    Ok(())
}

fn sibling_fts_dir(fts_dir: &std::path::Path, suffix: &str) -> std::path::PathBuf {
    let mut name = fts_dir
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("fts"))
        .to_os_string();
    name.push(".");
    name.push(suffix);
    fts_dir.with_file_name(name)
}

fn clean_temp_artifacts(tmp_db_path: &std::path::Path, tmp_fts_dir: &std::path::Path) {
    if tmp_db_path.exists()
        && let Err(e) = std::fs::remove_file(tmp_db_path)
    {
        tracing::warn!(path = %tmp_db_path.display(), err = %e, "failed to remove temp DB");
    }
    if tmp_fts_dir.exists()
        && let Err(e) = std::fs::remove_dir_all(tmp_fts_dir)
    {
        tracing::warn!(
            path = %tmp_fts_dir.display(),
            err = %e,
            "failed to remove temp FTS dir"
        );
    }
}
async fn build_index_in_temp(
    tmp_db_path: &std::path::Path,
    tmp_fts_dir: &std::path::Path,
    embedder: &embed::Embedder,
    notes_dir: &std::path::Path,
    dim: usize,
) -> anyhow::Result<index::IndexStats> {
    std::fs::create_dir_all(tmp_fts_dir)
        .with_context(|| format!("creating temp FTS dir {}", tmp_fts_dir.display()))?;
    let (_tmp_db, tmp_conn) = db::connect(tmp_db_path, Some(dim)).await?;
    let tmp_fts = fts::FtsIndex::open_or_create(tmp_fts_dir)?;
    index::index_directory(&tmp_conn, &tmp_fts, embedder, notes_dir).await
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    run(Cli::parse()).await
}

fn search_needs_embedder(command: &Command, weights: &rank::RrfWeights) -> bool {
    match command {
        Command::Watch | Command::Reindex => true,
        Command::Search { .. } => weights.semantic > 0.0,
        _ => false,
    }
}

const fn extract_cli_weights(command: &Command) -> CliWeights {
    match command {
        Command::Search {
            w_semantic,
            w_fts,
            w_filename,
            ..
        } => CliWeights {
            semantic: *w_semantic,
            fts: *w_fts,
            filename: *w_filename,
        },
        _ => CliWeights {
            semantic: None,
            fts: None,
            filename: None,
        },
    }
}

async fn run_watch(config: &config::Config, embedder: Option<Embedder>) -> anyhow::Result<()> {
    let _lock = lock::IndexLock::try_acquire(&config.db_path)?;
    let dim = embedder.as_ref().map(Embedder::dim);
    let (_db, conn) = open_db(&config.db_path, dim).await?;
    let fts = fts::FtsIndex::open_or_create(&config.tantivy_dir)?;
    let embedder = embedder.ok_or(NeedleError::NoEmbeddingProvider)?;
    watch::run_watcher(conn, fts, &embedder, config.notes_dir.clone()).await
}

async fn run_reindex_command(
    config: &config::Config,
    embedder: Option<Embedder>,
) -> anyhow::Result<()> {
    let embedder = embedder.ok_or(NeedleError::NoEmbeddingProvider)?;
    let _lock = lock::IndexLock::try_acquire(&config.db_path)?;
    let stats = run_reindex(
        &config.db_path,
        &config.tantivy_dir,
        &embedder,
        &config.notes_dir,
    )
    .await?;
    tracing::info!(%stats, "reindex complete");
    Ok(())
}

async fn run(cli: Cli) -> anyhow::Result<()> {
    let cli_weights = extract_cli_weights(&cli.command);
    let cli_embed = CliEmbedArgs {
        provider: cli.provider,
        model: cli.model,
        api_base: cli.api_base,
    };
    let config = Config::resolve(cli.notes_dir, cli_weights, cli_embed)?;

    let embedder = if search_needs_embedder(&cli.command, &config.weights) {
        Some(Embedder::from_config(&config.embed)?)
    } else {
        None
    };

    if matches!(cli.command, Command::Watch) {
        return run_watch(&config, embedder).await;
    }

    if matches!(cli.command, Command::Reindex) {
        return run_reindex_command(&config, embedder).await;
    }

    let dim = embedder.as_ref().map(Embedder::dim);
    let (_db, conn) = open_db(&config.db_path, dim).await?;

    match cli.command {
        Command::Watch | Command::Reindex => unreachable!("handled above"),
        Command::Search {
            query,
            limit,
            paths_only,
            ..
        } => {
            let query = resolve_query(
                query,
                &mut std::io::stdin().lock(),
                std::io::stdin().is_terminal(),
            )?;
            let fts = fts::FtsIndex::open_or_create(&config.tantivy_dir)?;
            let results = rank::search(
                &conn,
                &fts,
                embedder.as_ref(),
                &query,
                limit,
                &config.weights,
            )
            .await?;
            if paths_only {
                for result in &results {
                    println!("{}", result.path);
                }
            } else {
                for result in &results {
                    println!(
                        "{:.4}\t{}\t{}",
                        result.score,
                        result.path,
                        first_line(&result.snippet)
                    );
                }
            }
        }
        Command::Similar {
            threshold,
            limit,
            group,
            paths_only,
        } => {
            let pair_limit = if group { None } else { Some(limit) };
            let pairs = similar::find_similar(&conn, threshold, pair_limit).await?;
            print_similar(pairs, limit, group, paths_only);
        }
        Command::Related {
            path,
            limit,
            paths_only,
        } => {
            let results = similar::find_related(&conn, &path, limit).await?;
            if paths_only {
                for r in &results {
                    println!("{}", r.path);
                }
            } else {
                for r in &results {
                    println!("{:.4}\t{}", r.similarity, r.path);
                }
            }
        }
    }

    Ok(())
}

fn first_line(s: &str) -> &str {
    s.lines().next().unwrap_or("")
}

fn print_similar(pairs: Vec<similar::SimilarPair>, limit: usize, group: bool, paths_only: bool) {
    if group {
        let mut groups = similar::group_pairs(pairs);
        groups.truncate(limit);
        if paths_only {
            for g in &groups {
                for path in &g.paths {
                    println!("{path}");
                }
            }
        } else {
            for (i, g) in groups.iter().enumerate() {
                if i > 0 {
                    println!();
                }
                println!("Group {} ({} documents):", i + 1, g.paths.len());
                for pair in &g.pairs {
                    println!(
                        "  {:.4}  {} <> {}",
                        pair.similarity, pair.path_a, pair.path_b
                    );
                }
            }
        }
    } else if paths_only {
        for pair in &pairs {
            println!("{}", pair.path_a);
            println!("{}", pair.path_b);
        }
    } else {
        for pair in &pairs {
            println!("{:.4}\t{}\t{}", pair.similarity, pair.path_a, pair.path_b);
        }
    }
}

const STDIN_QUERY_LIMIT_BYTES: u64 = 1024 * 1024; // 1 MiB

fn resolve_query(
    explicit: Option<String>,
    reader: &mut impl Read,
    is_terminal: bool,
) -> anyhow::Result<String> {
    if let Some(q) = explicit {
        return Ok(q);
    }
    if is_terminal {
        anyhow::bail!("no query provided; pass as argument or pipe to stdin");
    }
    let mut buf = String::new();
    reader
        .take(STDIN_QUERY_LIMIT_BYTES)
        .read_to_string(&mut buf)?;
    let trimmed = buf.trim().to_owned();
    if trimmed.is_empty() {
        anyhow::bail!("empty query from stdin");
    }
    Ok(trimmed)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn resolve_query_returns_explicit_argument() {
        let mut reader = Cursor::new(b"");
        let result = resolve_query(Some("hello".to_owned()), &mut reader, false);
        assert_eq!(result.expect("should succeed"), "hello");
    }

    #[test]
    fn resolve_query_reads_from_stdin_when_no_argument() {
        let mut reader = Cursor::new(b"from stdin");
        let result = resolve_query(None, &mut reader, false);
        assert_eq!(result.expect("should succeed"), "from stdin");
    }

    #[test]
    fn resolve_query_trims_whitespace_from_stdin() {
        let mut reader = Cursor::new(b"  hello world  \n");
        let result = resolve_query(None, &mut reader, false);
        assert_eq!(result.expect("should succeed"), "hello world");
    }

    #[test]
    fn resolve_query_errors_when_terminal_and_no_argument() {
        let mut reader = Cursor::new(b"");
        let result = resolve_query(None, &mut reader, true);
        assert!(result.is_err());
        let msg = result.expect_err("should fail").to_string();
        assert!(msg.contains("no query provided"), "got: {msg}");
    }

    #[test]
    fn resolve_query_errors_on_empty_stdin() {
        let mut reader = Cursor::new(b"");
        let result = resolve_query(None, &mut reader, false);
        assert!(result.is_err());
        let msg = result.expect_err("should fail").to_string();
        assert!(msg.contains("empty query"), "got: {msg}");
    }

    #[test]
    fn resolve_query_errors_on_whitespace_only_stdin() {
        let mut reader = Cursor::new(b"   \n  ");
        let result = resolve_query(None, &mut reader, false);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_query_truncates_stdin_at_limit_and_still_returns_content() {
        // Input larger than STDIN_QUERY_LIMIT_BYTES should not OOM or error;
        // the leading content within the limit is accepted as the query.
        let limit_bytes = usize::try_from(STDIN_QUERY_LIMIT_BYTES).expect("limit fits usize");
        let big_input = vec![b'x'; limit_bytes + 1];
        let mut reader = Cursor::new(big_input);
        let result = resolve_query(None, &mut reader, false);
        assert!(result.is_ok(), "oversized stdin should not error");
        let query = result.expect("should succeed");
        assert_eq!(query.len(), limit_bytes);
    }

    fn search_command(w_semantic: Option<f64>) -> Command {
        Command::Search {
            query: None,
            limit: 10,
            paths_only: false,
            w_semantic,
            w_fts: None,
            w_filename: None,
        }
    }

    #[test]
    fn search_with_positive_semantic_weight_needs_embedder() {
        let weights = rank::RrfWeights {
            semantic: 1.5,
            fts: 1.0,
            filename: 0.7,
        };
        assert!(search_needs_embedder(&search_command(None), &weights));
    }

    #[test]
    fn search_with_zero_semantic_weight_does_not_need_embedder() {
        let weights = rank::RrfWeights {
            semantic: 0.0,
            fts: 1.0,
            filename: 0.7,
        };
        assert!(!search_needs_embedder(&search_command(None), &weights));
    }

    #[test]
    fn watch_always_needs_embedder() {
        let weights = rank::RrfWeights {
            semantic: 0.0,
            fts: 0.0,
            filename: 0.0,
        };
        assert!(search_needs_embedder(&Command::Watch, &weights));
    }

    #[test]
    fn reindex_always_needs_embedder() {
        let weights = rank::RrfWeights {
            semantic: 0.0,
            fts: 0.0,
            filename: 0.0,
        };
        assert!(search_needs_embedder(&Command::Reindex, &weights));
    }

    #[test]
    fn similar_never_needs_embedder() {
        let weights = rank::RrfWeights {
            semantic: 1.5,
            fts: 1.0,
            filename: 0.7,
        };
        assert!(!search_needs_embedder(
            &Command::Similar {
                threshold: 0.85,
                limit: 50,
                group: false,
                paths_only: false,
            },
            &weights
        ));
    }

    #[tokio::test]
    async fn open_db_succeeds_when_dimensions_match() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.db");
        let result = open_db(&path, Some(384)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn open_db_propagates_mismatch_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.db");

        let (db, conn) = db::connect(&path, Some(384)).await.expect("first connect");
        drop(conn);
        drop(db);

        let result = open_db(&path, Some(1024)).await;
        assert!(result.is_err());
        assert!(
            is_dimension_mismatch(&result.expect_err("should fail")),
            "error must be DimensionMismatch"
        );
    }

    #[tokio::test]
    async fn run_reindex_recovers_from_dimension_mismatch_when_reindex_succeeds() {
        let notes_dir = tempfile::tempdir().expect("notes tempdir");
        let db_dir = tempfile::tempdir().expect("db tempdir");
        let fts_dir = tempfile::tempdir().expect("fts tempdir");
        let db_path = db_dir.path().join("needle.db");

        std::fs::write(notes_dir.path().join("note.md"), "# Hello\n\nContent.")
            .expect("write note");

        let (db, conn) = db::connect(&db_path, Some(384))
            .await
            .expect("first connect");
        drop(conn);
        drop(db);

        let embedder = embed::Embedder::create_null(1024);
        let result = run_reindex(&db_path, fts_dir.path(), &embedder, notes_dir.path()).await;
        assert!(
            result.is_ok(),
            "run_reindex must succeed after mismatch: {:?}",
            result.err()
        );

        let (_db, conn) = db::connect(&db_path, Some(1024))
            .await
            .expect("DB must be openable with the new dimension after successful reindex");
        assert!(
            db::chunks_table_exists(&conn).await.expect("check failed"),
            "schema must be initialised with the new dimension"
        );
    }

    /// Dimension mismatch + reindex failure must leave the original DB intact.
    ///
    /// Steps:
    ///   1. Build a populated index at dimension 384.
    ///   2. Drop a non-UTF8 file into the notes dir so `index_directory` fails.
    ///   3. Call `run_reindex` with dimension 1024 (triggers mismatch path).
    ///   4. Assert the original DB file still exists and still contains the old
    ///      data (the prior index is preserved, not destroyed).
    #[tokio::test]
    async fn reindex_failure_preserves_original_db_on_dimension_mismatch() {
        use std::io::Write;

        let notes_dir = tempfile::tempdir().expect("notes tempdir");
        let db_dir = tempfile::tempdir().expect("db tempdir");
        let fts_dir = tempfile::tempdir().expect("fts tempdir");
        let db_path = db_dir.path().join("needle.db");

        // Populate the original index with dim=384.
        {
            let good_md = notes_dir.path().join("good.md");
            std::fs::write(&good_md, "# Good note\n\nSome content.").expect("write note");

            let (_db, conn) = db::connect(&db_path, Some(384)).await.expect("connect");
            let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
            let embedder = crate::embed::Embedder::create_null(384);
            index::index_directory(&conn, &fts, &embedder, notes_dir.path())
                .await
                .expect("initial index");
        }

        assert!(db_path.exists(), "original DB must exist before test");

        // Place a non-UTF8 file that will cause `read_to_string` to fail.
        let bad_file = notes_dir.path().join("corrupt.md");
        {
            let mut f = std::fs::File::create(&bad_file).expect("create corrupt file");
            f.write_all(&[0xFF, 0xFE, 0xFD]).expect("write bad bytes");
        }

        // Run reindex with dim=1024 (mismatch). The corrupt file should cause failure.
        let embedder = crate::embed::Embedder::create_null(1024);
        let result = run_reindex(&db_path, fts_dir.path(), &embedder, notes_dir.path()).await;
        assert!(result.is_err(), "reindex must fail due to corrupt file");

        // The original DB file must still be present and contain the prior data.
        assert!(
            db_path.exists(),
            "original DB must be preserved after a failed reindex"
        );
        let (_db, conn) = db::connect(&db_path, Some(384))
            .await
            .expect("original DB must still be openable with old dimension");
        let hashes = db::all_note_hashes(&conn).await.expect("query hashes");
        assert!(
            hashes.contains_key("good.md"),
            "prior index content must be intact after failed reindex"
        );
    }

    /// Dimension mismatch + reindex failure must not corrupt the live FTS index.
    ///
    /// Reproduction of the divergence bug: a temp reindex build fails (corrupt
    /// file causes `read_to_string` to error), but the old code called fts.rebuild
    /// on the live index before propagating the error, wiping it. After the fix
    /// the live FTS must still answer queries for content indexed before the
    /// failed reindex.
    #[tokio::test]
    async fn reindex_failure_preserves_fts_index_on_dimension_mismatch() {
        use std::io::Write;

        let notes_dir = tempfile::tempdir().expect("notes tempdir");
        let db_dir = tempfile::tempdir().expect("db tempdir");
        let fts_dir = tempfile::tempdir().expect("fts tempdir");
        let db_path = db_dir.path().join("needle.db");

        // Build the initial index with a known searchable term.
        {
            let good_md = notes_dir.path().join("good.md");
            std::fs::write(&good_md, "# Good note\n\nzebra content").expect("write note");

            let (_db, conn) = db::connect(&db_path, Some(384)).await.expect("connect");
            let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
            let embedder = crate::embed::Embedder::create_null(384);
            index::index_directory(&conn, &fts, &embedder, notes_dir.path())
                .await
                .expect("initial index");
        }

        // Confirm FTS finds the term before attempting the failing reindex.
        {
            let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
            let pre_results = fts.search("zebra", 10).await.expect("pre-reindex search");
            assert_eq!(
                pre_results.len(),
                1,
                "FTS must contain 'zebra' before the failing reindex"
            );
        }

        // Drop a corrupt file so the mismatch reindex fails before completing.
        let bad_file = notes_dir.path().join("corrupt.md");
        {
            let mut f = std::fs::File::create(&bad_file).expect("create corrupt file");
            f.write_all(&[0xFF, 0xFE, 0xFD]).expect("write bad bytes");
        }

        // Run reindex with dim=1024 (mismatch). Must fail.
        let embedder = crate::embed::Embedder::create_null(1024);
        let result = run_reindex(&db_path, fts_dir.path(), &embedder, notes_dir.path()).await;
        assert!(result.is_err(), "reindex must fail due to corrupt file");

        // The live FTS index must still answer queries for the original content.
        let fts = crate::fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
        let post_results = fts.search("zebra", 10).await.expect("post-reindex search");
        assert_eq!(
            post_results.len(),
            1,
            "FTS must still contain 'zebra' after a failed reindex; live index must not be mutated"
        );
    }

    /// Mismatch reindex must succeed and update FTS when the live FTS dir is
    /// already populated with Tantivy files.
    ///
    /// On Linux, rename(2) over a non-empty directory returns ENOTEMPTY.  A
    /// populated live FTS dir is normal operation, so `reindex_via_temp` must
    /// vacate the live dir first (backup) and only then rename the temp dir
    /// into place.
    #[tokio::test]
    async fn mismatch_reindex_succeeds_with_populated_fts_dir() {
        let notes_dir = tempfile::tempdir().expect("notes tempdir");
        let db_dir = tempfile::tempdir().expect("db tempdir");
        let fts_dir = tempfile::tempdir().expect("fts tempdir");
        let db_path = db_dir.path().join("needle.db");

        // Build an initial populated index at dim=384.  This writes Tantivy
        // files into fts_dir, making it non-empty.
        {
            std::fs::write(notes_dir.path().join("note.md"), "# Note\n\nquasar content")
                .expect("write note");

            let (_db, conn) = db::connect(&db_path, Some(384)).await.expect("connect");
            let fts = fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
            let embedder = embed::Embedder::create_null(384);
            index::index_directory(&conn, &fts, &embedder, notes_dir.path())
                .await
                .expect("initial index");
        }

        // Sanity: fts_dir must be non-empty (Tantivy has written files).
        let fts_entry_count = std::fs::read_dir(fts_dir.path())
            .expect("read fts_dir")
            .count();
        assert!(
            fts_entry_count > 0,
            "fts_dir must be non-empty before mismatch reindex to trigger the bug"
        );

        // Run a successful mismatch reindex (dim=1024, no corrupt files).
        let embedder = embed::Embedder::create_null(1024);
        let result = run_reindex(&db_path, fts_dir.path(), &embedder, notes_dir.path()).await;
        assert!(
            result.is_ok(),
            "mismatch reindex must succeed even when live FTS dir is non-empty: {:?}",
            result.err()
        );

        // Both DB and FTS must reflect the new index.
        let (_db, conn) = db::connect(&db_path, Some(1024))
            .await
            .expect("DB must be openable with the new dimension");
        assert!(
            db::chunks_table_exists(&conn)
                .await
                .expect("check chunks table"),
            "DB must have chunks table with new dimension"
        );

        let fts = fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
        let results = fts.search("quasar", 10).await.expect("fts search");
        assert_eq!(
            results.len(),
            1,
            "FTS must contain the reindexed content after a successful mismatch reindex"
        );
    }

    /// A prior run may crash after promote step 2 (temp FTS is now live) but before
    /// step 4 removes the backup.  The next mismatch reindex reaches step 1 and
    /// tries rename(live, backup), which fails on Linux when backup is non-empty.
    /// The fix must remove the stale backup before step 1 so the rename succeeds.
    #[tokio::test]
    async fn mismatch_reindex_succeeds_when_stale_backup_and_live_fts_both_exist() {
        let notes_dir = tempfile::tempdir().expect("notes tempdir");
        let db_dir = tempfile::tempdir().expect("db tempdir");
        let fts_dir = tempfile::tempdir().expect("fts tempdir");
        let db_path = db_dir.path().join("needle.db");

        // Build an initial populated index at dim=384.
        {
            std::fs::write(notes_dir.path().join("note.md"), "# Note\n\nphoton content")
                .expect("write note");

            let (_db, conn) = db::connect(&db_path, Some(384)).await.expect("connect");
            let fts = fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
            let embedder = embed::Embedder::create_null(384);
            index::index_directory(&conn, &fts, &embedder, notes_dir.path())
                .await
                .expect("initial index");
        }

        // Simulate crash after step 2 but before step 4: both live and backup exist.
        let backup_fts_dir = sibling_fts_dir(fts_dir.path(), "reindex-bak");
        std::fs::create_dir_all(&backup_fts_dir).expect("create stale backup dir");
        std::fs::write(backup_fts_dir.join("stale.marker"), b"stale").expect("write stale marker");

        assert!(
            fts_dir.path().exists(),
            "live FTS must exist to reproduce the dual-directory crash state"
        );
        assert!(
            backup_fts_dir.exists(),
            "stale backup must exist to reproduce the dual-directory crash state"
        );

        // Run a successful mismatch reindex (dim=1024, no corrupt files).
        let embedder = embed::Embedder::create_null(1024);
        let result = run_reindex(&db_path, fts_dir.path(), &embedder, notes_dir.path()).await;
        assert!(
            result.is_ok(),
            "mismatch reindex must succeed even when stale backup and live FTS both exist: {:?}",
            result.err()
        );

        // DB must be openable with the new dimension.
        let (_db, conn) = db::connect(&db_path, Some(1024))
            .await
            .expect("DB must be openable with the new dimension");
        assert!(
            db::chunks_table_exists(&conn)
                .await
                .expect("check chunks table"),
            "DB must have chunks table with new dimension"
        );

        // FTS must contain the reindexed content.
        let fts = fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
        let results = fts.search("photon", 10).await.expect("fts search");
        assert_eq!(
            results.len(),
            1,
            "FTS must contain the reindexed content after recovery from dual-directory state"
        );
    }

    /// A prior run may crash between promote step 1 (backup live FTS) and step 2
    /// (rename temp FTS into place).  That leaves the backup as the only copy of
    /// the live FTS.  When the next reindex attempt starts, it must restore the
    /// backup before cleaning temp artifacts; otherwise deleting the backup
    /// permanently destroys FTS availability even if the new build also fails.
    #[tokio::test]
    async fn interrupted_reindex_restores_fts_backup_on_next_attempt() {
        use std::io::Write;

        let notes_dir = tempfile::tempdir().expect("notes tempdir");
        let db_dir = tempfile::tempdir().expect("db tempdir");
        let fts_dir = tempfile::tempdir().expect("fts tempdir");
        let db_path = db_dir.path().join("needle.db");

        // Build initial index with known searchable content at dim=384.
        {
            std::fs::write(
                notes_dir.path().join("note.md"),
                "# Note\n\nneutron content",
            )
            .expect("write note");

            let (_db, conn) = db::connect(&db_path, Some(384)).await.expect("connect");
            let fts = fts::FtsIndex::open_or_create(fts_dir.path()).expect("fts");
            let embedder = embed::Embedder::create_null(384);
            index::index_directory(&conn, &fts, &embedder, notes_dir.path())
                .await
                .expect("initial index");
        }

        // Simulate a crash between promote step 1 (live FTS moved to backup) and
        // step 2 (temp FTS renamed into place): the backup exists and live FTS is
        // absent.
        let backup_fts_dir = sibling_fts_dir(fts_dir.path(), "reindex-bak");
        std::fs::rename(fts_dir.path(), &backup_fts_dir)
            .expect("simulate: move live FTS to backup");
        assert!(
            !fts_dir.path().exists(),
            "live FTS must be absent to reproduce the crash state"
        );

        // Add a corrupt file so the new build deterministically fails, verifying
        // that recovery holds even when the subsequent rebuild does not succeed.
        let bad_file = notes_dir.path().join("corrupt.md");
        {
            let mut f = std::fs::File::create(&bad_file).expect("create corrupt file");
            f.write_all(&[0xFF, 0xFE, 0xFD]).expect("write bad bytes");
        }

        // Trigger mismatch reindex (dim=1024). Must fail due to corrupt file.
        let embedder = embed::Embedder::create_null(1024);
        let result = run_reindex(&db_path, fts_dir.path(), &embedder, notes_dir.path()).await;
        assert!(result.is_err(), "reindex must fail due to corrupt file");

        // The backup must have been restored to the live FTS location before the
        // build started, so the original content is intact regardless of the
        // build failure.
        assert!(
            fts_dir.path().exists(),
            "live FTS must be restored from backup even after a failed reindex"
        );
        assert!(
            !backup_fts_dir.exists(),
            "backup must be gone after restoration"
        );

        let fts = fts::FtsIndex::open_or_create(fts_dir.path()).expect("open restored fts");
        let results = fts.search("neutron", 10).await.expect("fts search");
        assert_eq!(
            results.len(),
            1,
            "original FTS content must be intact after crash recovery and failed reindex"
        );
    }
}
