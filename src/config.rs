use std::path::PathBuf;

use anyhow::Context;
use serde::Deserialize;

use crate::{error::NeedleError, rank::RrfWeights};

pub struct Config {
    pub notes_dir: PathBuf,
    pub db_path: PathBuf,
    pub tantivy_dir: PathBuf,
    pub voyage_api_key: String,
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
        let file_config = load_file_config()?;

        let notes_dir = cli_notes_dir
            .or_else(|| std::env::var("ZK_NOTEBOOK_DIR").ok().map(PathBuf::from))
            .or(file_config.notes_dir)
            .context("notes directory not specified: use --notes-dir, set ZK_NOTEBOOK_DIR, or set notes_dir in config.toml")?;

        if !notes_dir.is_dir() {
            return Err(NeedleError::NotesDirectoryNotFound(notes_dir).into());
        }

        let voyage_api_key = std::env::var("VOYAGE_API_KEY")
            .ok()
            .or(file_config.voyage_api_key)
            .ok_or(NeedleError::MissingApiKey)?;

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
            )
        }
    };

    toml::from_str(&content).context(format!("parsing config: {}", path.display()))
}
