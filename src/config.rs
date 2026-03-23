use std::path::PathBuf;

use anyhow::Context;
use serde::Deserialize;

use crate::{error::NeedleError, rank::RrfWeights};

pub struct Config {
    pub notes_dir: PathBuf,
    pub db_path: PathBuf,
    pub tantivy_dir: PathBuf,
    pub voyage_api_key: Option<String>,
    pub weights: RrfWeights,
}

#[derive(Deserialize, Default)]
struct FileConfig {
    voyage_api_key: Option<String>,
    notes_dir: Option<PathBuf>,
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
        cli_notes_dir: Option<PathBuf>,
        cli_weights: CliWeights,
    ) -> anyhow::Result<Self> {
        Self::resolve_with(cli_notes_dir, cli_weights, load_file_config()?)
    }

    fn resolve_with(
        cli_notes_dir: Option<PathBuf>,
        cli_weights: CliWeights,
        file_config: FileConfig,
    ) -> anyhow::Result<Self> {
        let notes_dir = cli_notes_dir
            .or_else(|| std::env::var("ZK_NOTEBOOK_DIR").ok().map(PathBuf::from))
            .or(file_config.notes_dir)
            .context("notes directory not specified: use --notes-dir, set ZK_NOTEBOOK_DIR, or set notes_dir in config.toml")?;

        if !notes_dir.is_dir() {
            return Err(NeedleError::NotesDirectoryNotFound(notes_dir).into());
        }

        let voyage_api_key = std::env::var("VOYAGE_API_KEY")
            .ok()
            .or(file_config.voyage_api_key);

        let db_dir = data_dir()?;
        std::fs::create_dir_all(&db_dir)?;
        let db_path = db_dir.join("needle.db");
        let tantivy_dir = db_dir.join("tantivy");
        std::fs::create_dir_all(&tantivy_dir)?;

        let defaults = RrfWeights::default();
        let weights = RrfWeights {
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
        };

        Ok(Self {
            notes_dir,
            db_path,
            tantivy_dir,
            voyage_api_key,
            weights,
        })
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
        let dir = tempfile::tempdir().expect("tempdir");

        let file_config = FileConfig {
            w_semantic: Some(2.0),
            w_fts: Some(3.0),
            w_filename: Some(4.0),
            notes_dir: Some(dir.path().to_owned()),
            ..Default::default()
        };

        let cli_weights = CliWeights {
            semantic: Some(10.0),
            fts: None,
            filename: None,
        };

        let config = Config::resolve_with(None, cli_weights, file_config).expect("resolve");
        assert!((config.weights.semantic - 10.0).abs() < f64::EPSILON);
        assert!((config.weights.fts - 3.0).abs() < f64::EPSILON);
        assert!((config.weights.filename - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn file_config_overrides_defaults() {
        let dir = tempfile::tempdir().expect("tempdir");

        let file_config = FileConfig {
            w_semantic: Some(9.0),
            w_fts: Some(8.0),
            w_filename: Some(7.0),
            notes_dir: Some(dir.path().to_owned()),
            ..Default::default()
        };

        let cli_weights = CliWeights {
            semantic: None,
            fts: None,
            filename: None,
        };

        let config = Config::resolve_with(None, cli_weights, file_config).expect("resolve");
        assert!((config.weights.semantic - 9.0).abs() < f64::EPSILON);
        assert!((config.weights.fts - 8.0).abs() < f64::EPSILON);
        assert!((config.weights.filename - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn defaults_used_when_no_overrides() {
        let dir = tempfile::tempdir().expect("tempdir");

        let file_config = FileConfig {
            notes_dir: Some(dir.path().to_owned()),
            ..Default::default()
        };

        let cli_weights = CliWeights {
            semantic: None,
            fts: None,
            filename: None,
        };

        let config = Config::resolve_with(None, cli_weights, file_config).expect("resolve");
        let defaults = RrfWeights::default();
        assert!((config.weights.semantic - defaults.semantic).abs() < f64::EPSILON);
        assert!((config.weights.fts - defaults.fts).abs() < f64::EPSILON);
        assert!((config.weights.filename - defaults.filename).abs() < f64::EPSILON);
    }

    #[test]
    fn nonexistent_notes_dir_is_an_error() {
        let bad_dir = PathBuf::from("/nonexistent/path/that/should/not/exist");
        let cli_weights = CliWeights {
            semantic: None,
            fts: None,
            filename: None,
        };

        let result = Config::resolve_with(Some(bad_dir), cli_weights, FileConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn cli_notes_dir_overrides_file_config() {
        let cli_dir = tempfile::tempdir().expect("tempdir");
        let file_dir = tempfile::tempdir().expect("tempdir");

        let file_config = FileConfig {
            notes_dir: Some(file_dir.path().to_owned()),
            ..Default::default()
        };

        let cli_weights = CliWeights {
            semantic: None,
            fts: None,
            filename: None,
        };

        let config =
            Config::resolve_with(Some(cli_dir.path().to_owned()), cli_weights, file_config)
                .expect("resolve");
        assert_eq!(config.notes_dir, cli_dir.path());
    }
}
