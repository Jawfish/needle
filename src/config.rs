use std::path::PathBuf;

use anyhow::Context;
use serde::Deserialize;

use crate::error::SemError;

pub struct Config {
    pub notes_dir: PathBuf,
    pub db_path: PathBuf,
    pub tantivy_dir: PathBuf,
    pub voyage_api_key: String,
    pub w_semantic: f64,
    pub w_fts: f64,
    pub w_filename: f64,
}

const DEFAULT_W_SEMANTIC: f64 = 1.5;
const DEFAULT_W_FTS: f64 = 1.0;
const DEFAULT_W_FILENAME: f64 = 0.7;

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
        let file_config = load_file_config();

        let notes_dir = cli_notes_dir
            .or_else(|| std::env::var("ZK_NOTEBOOK_DIR").ok().map(PathBuf::from))
            .or(file_config.notes_dir)
            .context("notes directory not specified: use --notes-dir, set ZK_NOTEBOOK_DIR, or set notes_dir in config.toml")?;

        if !notes_dir.is_dir() {
            return Err(SemError::NotesDirectoryNotFound(notes_dir).into());
        }

        let voyage_api_key = std::env::var("VOYAGE_API_KEY")
            .ok()
            .or(file_config.voyage_api_key)
            .ok_or(SemError::MissingApiKey)?;

        let db_dir = data_dir();
        std::fs::create_dir_all(&db_dir)?;
        let db_path = db_dir.join("needle.db");
        let tantivy_dir = db_dir.join("tantivy");
        std::fs::create_dir_all(&tantivy_dir)?;

        let w_semantic = cli_weights
            .semantic
            .or(file_config.w_semantic)
            .unwrap_or(DEFAULT_W_SEMANTIC);
        let w_fts = cli_weights
            .fts
            .or(file_config.w_fts)
            .unwrap_or(DEFAULT_W_FTS);
        let w_filename = cli_weights
            .filename
            .or(file_config.w_filename)
            .unwrap_or(DEFAULT_W_FILENAME);

        Ok(Self {
            notes_dir,
            db_path,
            tantivy_dir,
            voyage_api_key,
            w_semantic,
            w_fts,
            w_filename,
        })
    }
}

fn config_path() -> PathBuf {
    std::env::var("XDG_CONFIG_HOME")
        .map_or_else(
            |_| {
                let home = std::env::var("HOME").expect("HOME not set");
                PathBuf::from(home).join(".config")
            },
            PathBuf::from,
        )
        .join("needle/config.toml")
}

fn data_dir() -> PathBuf {
    std::env::var("XDG_DATA_HOME")
        .map_or_else(
            |_| {
                let home = std::env::var("HOME").expect("HOME not set");
                PathBuf::from(home).join(".local/share")
            },
            PathBuf::from,
        )
        .join("needle")
}

fn load_file_config() -> FileConfig {
    let path = config_path();
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|content| toml::from_str(&content).ok())
        .unwrap_or_default()
}
