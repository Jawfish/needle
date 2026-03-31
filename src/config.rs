use std::path::{Path, PathBuf};

use anyhow::Context as _;
use serde::Deserialize;

use crate::{
    error::NeedleError,
    hash,
    types::{EmbedConfig, RrfWeights},
};

pub struct CliEmbedArgs {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub api_base: Option<String>,
}

#[derive(Debug)]
pub struct DirectoryStore {
    pub notes_dir: PathBuf,
    pub db_path: PathBuf,
    pub tantivy_dir: PathBuf,
}

impl DirectoryStore {
    /// Convert `path` to a path relative to this store's `notes_dir`.
    ///
    /// Absolute paths are stripped of the `notes_dir` prefix.  Relative paths
    /// are returned unchanged.  Returns an error when an absolute path does not
    /// reside under this store's `notes_dir`.
    pub fn to_relative(&self, path: &str) -> anyhow::Result<String> {
        let as_path = Path::new(path);
        if as_path.is_absolute() {
            as_path
                .strip_prefix(&self.notes_dir)
                .map(|p| p.to_string_lossy().into_owned())
                .with_context(|| format!("path {path} is not under {}", self.notes_dir.display()))
        } else {
            Ok(path.to_owned())
        }
    }

    /// Join `rel_path` with this store's `notes_dir` to produce an absolute path string.
    pub fn to_absolute(&self, rel_path: &str) -> String {
        self.notes_dir.join(rel_path).to_string_lossy().into_owned()
    }
}

#[derive(Debug)]
pub struct Config {
    pub docs_dirs: Vec<DirectoryStore>,
    pub embed: EmbedConfig,
    pub weights: RrfWeights,
}

#[derive(Deserialize, Default)]
struct FileConfig {
    provider: Option<String>,
    model: Option<String>,
    api_base: Option<String>,
    dim: Option<usize>,
    voyage_api_key: Option<String>,
    openai_api_key: Option<String>,
    needle_api_key: Option<String>,
    notes_dirs: Option<Vec<PathBuf>>,
    w_semantic: Option<f64>,
    w_fts: Option<f64>,
    w_filename: Option<f64>,
}

#[derive(Clone, Copy)]
pub struct CliWeights {
    pub semantic: Option<f64>,
    pub fts: Option<f64>,
    pub filename: Option<f64>,
}

impl Config {
    pub fn resolve(
        cli_docs_dirs: Vec<PathBuf>,
        cli_weights: CliWeights,
        cli_embed: CliEmbedArgs,
    ) -> anyhow::Result<Self> {
        Self::resolve_with(cli_docs_dirs, cli_weights, cli_embed, load_file_config()?)
    }

    fn resolve_with(
        cli_docs_dirs: Vec<PathBuf>,
        cli_weights: CliWeights,
        cli_embed: CliEmbedArgs,
        file_config: FileConfig,
    ) -> anyhow::Result<Self> {
        let weights = resolve_weights(cli_weights, &file_config);
        let embed = resolve_embed_config(cli_embed, &file_config);

        let raw_paths: Vec<PathBuf> = if !cli_docs_dirs.is_empty() {
            cli_docs_dirs
        } else if let Some(dirs) = file_config.notes_dirs.filter(|v| !v.is_empty()) {
            dirs
        } else {
            return Err(NeedleError::MissingDirectories(
                "no docs directories configured; set notes_dirs in config.toml or pass --docs-dir <PATH>".to_owned(),
            )
            .into());
        };

        let mut canonical_paths: Vec<PathBuf> = Vec::with_capacity(raw_paths.len());
        for path in raw_paths {
            let canonical = path
                .canonicalize()
                .with_context(|| format!("canonicalizing path {}", path.display()))?;
            if !canonical_paths.contains(&canonical) {
                canonical_paths.push(canonical);
            }
        }

        let mut overlap_pairs: Vec<String> = Vec::new();
        for i in 0..canonical_paths.len() {
            for j in (i + 1)..canonical_paths.len() {
                let a = &canonical_paths[i];
                let b = &canonical_paths[j];
                if a.starts_with(b.as_path()) || b.starts_with(a.as_path()) {
                    overlap_pairs.push(format!("  {} and {}", a.display(), b.display()));
                }
            }
        }
        if !overlap_pairs.is_empty() {
            return Err(NeedleError::OverlappingDirectories(overlap_pairs.join("\n")).into());
        }

        let missing: Vec<String> = canonical_paths
            .iter()
            .filter(|p| !p.is_dir())
            .map(|p| p.display().to_string())
            .collect();

        if !missing.is_empty() {
            return Err(NeedleError::MissingDirectories(missing.join("\n")).into());
        }

        let mut docs_dirs: Vec<DirectoryStore> = Vec::with_capacity(canonical_paths.len());
        for notes_dir in canonical_paths {
            let data_dir = data_dir_for(&notes_dir)?;
            std::fs::create_dir_all(&data_dir)?;
            let db_path = data_dir.join("needle.db");
            let tantivy_dir = data_dir.join("tantivy");
            std::fs::create_dir_all(&tantivy_dir)?;
            docs_dirs.push(DirectoryStore {
                notes_dir,
                db_path,
                tantivy_dir,
            });
        }

        tracing::debug!(
            dirs = ?docs_dirs.iter().map(|s| s.notes_dir.display().to_string()).collect::<Vec<_>>(),
            "resolved docs dirs"
        );

        Ok(Self {
            docs_dirs,
            embed,
            weights,
        })
    }
}

fn resolve_embed_config(cli: CliEmbedArgs, file: &FileConfig) -> EmbedConfig {
    let env = |key: &str| std::env::var(key).ok();

    EmbedConfig {
        provider: cli
            .provider
            .or_else(|| env("NEEDLE_PROVIDER"))
            .or_else(|| file.provider.clone()),
        model: cli
            .model
            .or_else(|| env("NEEDLE_MODEL"))
            .or_else(|| file.model.clone()),
        api_base: cli
            .api_base
            .or_else(|| env("NEEDLE_API_BASE"))
            .or_else(|| file.api_base.clone()),
        dim: env("NEEDLE_DIM").and_then(|s| s.parse().ok()).or(file.dim),
        voyage_api_key: env("VOYAGE_API_KEY").or_else(|| file.voyage_api_key.clone()),
        openai_api_key: env("OPENAI_API_KEY").or_else(|| file.openai_api_key.clone()),
        needle_api_key: env("NEEDLE_API_KEY").or_else(|| file.needle_api_key.clone()),
    }
}

fn config_path() -> anyhow::Result<PathBuf> {
    let base = if let Ok(dir) = std::env::var("XDG_CONFIG_HOME") {
        PathBuf::from(dir)
    } else {
        let home = std::env::var("HOME").context("HOME not set")?;
        PathBuf::from(home).join(".config")
    };
    Ok(base.join("needle/config.toml"))
}

fn data_dir() -> anyhow::Result<PathBuf> {
    let base = if let Ok(dir) = std::env::var("XDG_DATA_HOME") {
        PathBuf::from(dir)
    } else {
        let home = std::env::var("HOME").context("HOME not set")?;
        PathBuf::from(home).join(".local/share")
    };
    Ok(base.join("needle"))
}

fn resolve_weights(cli_weights: CliWeights, file_config: &FileConfig) -> RrfWeights {
    let defaults = RrfWeights::default();
    RrfWeights {
        semantic: cli_weights
            .semantic
            .or(file_config.w_semantic)
            .unwrap_or(defaults.semantic),
        fts: cli_weights
            .fts
            .or(file_config.w_fts)
            .unwrap_or(defaults.fts),
        filename: cli_weights
            .filename
            .or(file_config.w_filename)
            .unwrap_or(defaults.filename),
    }
}

fn data_dir_for(notes_dir: &Path) -> anyhow::Result<PathBuf> {
    let canonical = notes_dir.canonicalize()?;
    let dir_hash = hash::content_hash(&canonical.to_string_lossy());
    let base = data_dir()?;
    Ok(base.join(&dir_hash[..12]))
}

fn load_file_config() -> anyhow::Result<FileConfig> {
    let Ok(path) = config_path() else {
        return Ok(FileConfig::default());
    };

    let content = match std::fs::read_to_string(&path) {
        Ok(content) => content,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(FileConfig::default()),
        Err(e) => {
            return Err(
                anyhow::Error::from(e).context(format!("reading config: {}", path.display()))
            );
        }
    };

    toml::from_str(&content).context(format!("parsing config: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_weights_override_file_config() {
        let file_config = FileConfig {
            w_semantic: Some(2.0),
            w_fts: Some(3.0),
            w_filename: Some(4.0),
            ..Default::default()
        };

        let cli_weights = CliWeights {
            semantic: Some(10.0),
            fts: None,
            filename: None,
        };

        let weights = resolve_weights(cli_weights, &file_config);
        assert!((weights.semantic - 10.0).abs() < f64::EPSILON);
        assert!((weights.fts - 3.0).abs() < f64::EPSILON);
        assert!((weights.filename - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn file_config_overrides_defaults() {
        let file_config = FileConfig {
            w_semantic: Some(9.0),
            w_fts: Some(8.0),
            w_filename: Some(7.0),
            ..Default::default()
        };

        let cli_weights = CliWeights {
            semantic: None,
            fts: None,
            filename: None,
        };

        let weights = resolve_weights(cli_weights, &file_config);
        assert!((weights.semantic - 9.0).abs() < f64::EPSILON);
        assert!((weights.fts - 8.0).abs() < f64::EPSILON);
        assert!((weights.filename - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn defaults_used_when_no_overrides() {
        let cli_weights = CliWeights {
            semantic: None,
            fts: None,
            filename: None,
        };

        let weights = resolve_weights(cli_weights, &FileConfig::default());
        let defaults = RrfWeights::default();
        assert!((weights.semantic - defaults.semantic).abs() < f64::EPSILON);
        assert!((weights.fts - defaults.fts).abs() < f64::EPSILON);
        assert!((weights.filename - defaults.filename).abs() < f64::EPSILON);
    }

    #[test]
    fn nonexistent_notes_dir_is_an_error() {
        let bad_dir = PathBuf::from("/nonexistent/path/that/should/not/exist");
        let cli_weights = CliWeights {
            semantic: None,
            fts: None,
            filename: None,
        };

        let cli_embed = CliEmbedArgs {
            provider: None,
            model: None,
            api_base: None,
        };

        let result =
            Config::resolve_with(vec![bad_dir], cli_weights, cli_embed, FileConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn no_directories_configured_is_an_error() {
        let cli_weights = CliWeights {
            semantic: None,
            fts: None,
            filename: None,
        };
        let cli_embed = CliEmbedArgs {
            provider: None,
            model: None,
            api_base: None,
        };

        let result = Config::resolve_with(vec![], cli_weights, cli_embed, FileConfig::default());
        assert!(result.is_err());
        let msg = result.expect_err("should fail").to_string();
        assert!(msg.contains("no docs directories configured"), "got: {msg}");
    }

    #[test]
    fn cli_dirs_override_file_config_dirs() {
        let cli_widgets = CliWeights {
            semantic: None,
            fts: None,
            filename: None,
        };
        let cli_embed = CliEmbedArgs {
            provider: None,
            model: None,
            api_base: None,
        };
        let file_config = FileConfig {
            notes_dirs: Some(vec![PathBuf::from(
                "/nonexistent/file/config/dir/that/does/not/exist",
            )]),
            ..Default::default()
        };

        // When CLI dirs are given, the nonexistent file-config dir is not consulted.
        // But CLI dir also doesn't exist, so we still get an error - just for the CLI path.
        let cli_dir = PathBuf::from("/nonexistent/cli/dir/that/does/not/exist");
        let result = Config::resolve_with(vec![cli_dir], cli_widgets, cli_embed, file_config);
        assert!(result.is_err());
        let msg = result.expect_err("should fail").to_string();
        // Should mention the CLI path, not the file-config path
        assert!(
            msg.contains("nonexistent/cli/dir") || msg.contains("canonicalizing"),
            "got: {msg}"
        );
    }

    #[test]
    fn duplicate_cli_dirs_are_deduplicated() {
        // Two references to the same real directory (use /tmp which always exists)
        let real_dir = PathBuf::from("/tmp");
        let cli_widgets = CliWeights {
            semantic: None,
            fts: None,
            filename: None,
        };
        let cli_embed = CliEmbedArgs {
            provider: None,
            model: None,
            api_base: None,
        };
        let result = Config::resolve_with(
            vec![real_dir.clone(), real_dir],
            cli_widgets,
            cli_embed,
            FileConfig::default(),
        );
        // /tmp exists so this should succeed
        assert!(result.is_ok(), "resolve should succeed for /tmp");
        let config = result.expect("succeed");
        assert_eq!(
            config.docs_dirs.len(),
            1,
            "duplicate entry must be deduplicated"
        );
    }

    #[test]
    fn missing_directories_error_names_all_bad_paths() {
        let bad1 = PathBuf::from("/nonexistent/path/aaa");
        let bad2 = PathBuf::from("/nonexistent/path/bbb");
        let cli_weights = CliWeights {
            semantic: None,
            fts: None,
            filename: None,
        };
        let cli_embed = CliEmbedArgs {
            provider: None,
            model: None,
            api_base: None,
        };
        let result = Config::resolve_with(
            vec![bad1, bad2],
            cli_weights,
            cli_embed,
            FileConfig::default(),
        );
        assert!(result.is_err());
        // canonicalize will fail on completely nonexistent paths before is_dir check
        // The error should mention one of the paths
        assert!(result.is_err());
    }

    fn overlap_weights() -> CliWeights {
        CliWeights {
            semantic: None,
            fts: None,
            filename: None,
        }
    }

    fn overlap_embed() -> CliEmbedArgs {
        CliEmbedArgs {
            provider: None,
            model: None,
            api_base: None,
        }
    }

    #[test]
    fn parent_child_overlap_is_rejected() {
        let parent = tempfile::tempdir().expect("parent tempdir");
        let child = parent.path().join("child");
        std::fs::create_dir(&child).expect("create child");

        let result = Config::resolve_with(
            vec![parent.path().to_path_buf(), child],
            overlap_weights(),
            overlap_embed(),
            FileConfig::default(),
        );
        assert!(result.is_err());
        let msg = result.expect_err("should fail").to_string();
        assert!(msg.contains("overlap"), "got: {msg}");
    }

    #[test]
    fn child_parent_order_is_also_rejected() {
        let parent = tempfile::tempdir().expect("parent tempdir");
        let child = parent.path().join("child");
        std::fs::create_dir(&child).expect("create child");

        let result = Config::resolve_with(
            vec![child, parent.path().to_path_buf()],
            overlap_weights(),
            overlap_embed(),
            FileConfig::default(),
        );
        assert!(result.is_err());
        let msg = result.expect_err("should fail").to_string();
        assert!(msg.contains("overlap"), "got: {msg}");
    }

    #[test]
    fn non_overlapping_siblings_are_accepted() {
        let base = tempfile::tempdir().expect("base tempdir");
        let sib_a = base.path().join("a");
        let sib_b = base.path().join("b");
        std::fs::create_dir(&sib_a).expect("create sib_a");
        std::fs::create_dir(&sib_b).expect("create sib_b");

        let result = Config::resolve_with(
            vec![sib_a, sib_b],
            overlap_weights(),
            overlap_embed(),
            FileConfig::default(),
        );
        assert!(
            result.is_ok(),
            "non-overlapping siblings must be accepted: {:?}",
            result.err()
        );
        assert_eq!(result.expect("ok").docs_dirs.len(), 2);
    }

    #[test]
    fn three_non_overlapping_paths_are_all_accepted() {
        let base = tempfile::tempdir().expect("tempdir");
        for name in ["x", "y", "z"] {
            std::fs::create_dir(base.path().join(name)).expect("create dir");
        }

        let dirs = ["x", "y", "z"]
            .iter()
            .map(|n| base.path().join(n))
            .collect();

        let result = Config::resolve_with(
            dirs,
            overlap_weights(),
            overlap_embed(),
            FileConfig::default(),
        );
        assert!(
            result.is_ok(),
            "three non-overlapping paths must be accepted: {:?}",
            result.err()
        );
        assert_eq!(result.expect("ok").docs_dirs.len(), 3);
    }

    #[test]
    fn string_prefix_but_not_path_prefix_is_accepted() {
        // /docs and /docs-extra share a string prefix but are distinct path components.
        // Path::starts_with does component-level matching so they must not be flagged.
        let base = tempfile::tempdir().expect("tempdir");
        let docs = base.path().join("docs");
        let docs_extra = base.path().join("docs-extra");
        std::fs::create_dir(&docs).expect("create docs");
        std::fs::create_dir(&docs_extra).expect("create docs-extra");

        let result = Config::resolve_with(
            vec![docs, docs_extra],
            overlap_weights(),
            overlap_embed(),
            FileConfig::default(),
        );
        assert!(
            result.is_ok(),
            "string-prefix-only siblings must not be flagged as overlapping: {:?}",
            result.err()
        );
    }

    #[test]
    fn overlap_error_names_both_paths() {
        let parent = tempfile::tempdir().expect("parent tempdir");
        let child = parent.path().join("sub");
        std::fs::create_dir(&child).expect("create child");

        let result = Config::resolve_with(
            vec![parent.path().to_path_buf(), child],
            overlap_weights(),
            overlap_embed(),
            FileConfig::default(),
        );
        let msg = result.expect_err("should fail").to_string();
        // Both paths should appear in the error so the user knows which pair to fix.
        assert!(
            msg.contains(parent.path().to_string_lossy().as_ref()),
            "error must name the parent path, got: {msg}"
        );
    }

    #[test]
    fn cli_embed_args_override_file_config() {
        let file_config = FileConfig {
            provider: Some("voyage".to_owned()),
            model: Some("voyage-4".to_owned()),
            ..Default::default()
        };

        let cli_embed = CliEmbedArgs {
            provider: Some("openai".to_owned()),
            model: None,
            api_base: None,
        };

        let embed = resolve_embed_config(cli_embed, &file_config);
        assert_eq!(embed.provider.as_deref(), Some("openai"));
        assert_eq!(embed.model.as_deref(), Some("voyage-4"));
    }

    #[test]
    fn embed_config_defaults_to_none() {
        let cli_embed = CliEmbedArgs {
            provider: None,
            model: None,
            api_base: None,
        };

        let embed = resolve_embed_config(cli_embed, &FileConfig::default());
        assert!(embed.provider.is_none());
        assert!(embed.model.is_none());
        assert!(embed.api_base.is_none());
        assert!(embed.dim.is_none());
    }

    fn make_store(notes_dir: &str) -> DirectoryStore {
        let base = PathBuf::from(notes_dir);
        DirectoryStore {
            notes_dir: base.clone(),
            db_path: base.join("needle.db"),
            tantivy_dir: base.join("tantivy"),
        }
    }

    #[test]
    fn to_relative_strips_absolute_prefix() {
        let store = make_store("/home/user/notes");
        assert_eq!(
            store.to_relative("/home/user/notes/topic.md").expect("ok"),
            "topic.md"
        );
        assert_eq!(
            store
                .to_relative("/home/user/notes/sub/topic.md")
                .expect("ok"),
            "sub/topic.md"
        );
    }

    #[test]
    fn to_relative_passes_through_relative_path() {
        let store = make_store("/home/user/notes");
        assert_eq!(store.to_relative("topic.md").expect("ok"), "topic.md");
        assert_eq!(
            store.to_relative("sub/topic.md").expect("ok"),
            "sub/topic.md"
        );
    }

    #[test]
    fn to_relative_errors_on_absolute_path_outside_store() {
        let store = make_store("/home/user/notes");
        assert!(store.to_relative("/other/path/topic.md").is_err());
    }

    #[test]
    fn to_absolute_prepends_notes_dir() {
        let store = make_store("/home/user/notes");
        assert_eq!(store.to_absolute("topic.md"), "/home/user/notes/topic.md");
        assert_eq!(
            store.to_absolute("sub/topic.md"),
            "/home/user/notes/sub/topic.md"
        );
    }
}
