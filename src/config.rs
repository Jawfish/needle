use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::Deserialize;

use crate::{error::NeedleError, hash, rank::RrfWeights};

pub struct CliEmbedArgs {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub api_base: Option<String>,
}

pub struct Config {
    pub notes_dir: PathBuf,
    pub db_path: PathBuf,
    pub tantivy_dir: PathBuf,
    pub embed: EmbedConfig,
    pub weights: RrfWeights,
}

pub struct EmbedConfig {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub api_base: Option<String>,
    pub dim: Option<usize>,
    pub voyage_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub needle_api_key: Option<String>,
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
        cli_embed: CliEmbedArgs,
    ) -> anyhow::Result<Self> {
        Self::resolve_with(cli_notes_dir, cli_weights, cli_embed, load_file_config()?)
    }

    fn resolve_with(
        cli_notes_dir: Option<PathBuf>,
        cli_weights: CliWeights,
        cli_embed: CliEmbedArgs,
        file_config: FileConfig,
    ) -> anyhow::Result<Self> {
        let weights = resolve_weights(cli_weights, &file_config);
        let embed = resolve_embed_config(cli_embed, &file_config);

        let notes_dir = cli_notes_dir
            .or_else(|| std::env::var("NEEDLE_DOCS_DIR").ok().map(PathBuf::from))
            .or(file_config.notes_dir)
            .context("notes directory not specified: use --notes-dir, set NEEDLE_DOCS_DIR, or set notes_dir in config.toml")?;

        if !notes_dir.is_dir() {
            return Err(NeedleError::NotesDirectoryNotFound(notes_dir).into());
        }

        let db_dir = data_dir_for(&notes_dir)?;
        std::fs::create_dir_all(&db_dir)?;
        let db_path = db_dir.join("needle.db");
        let tantivy_dir = db_dir.join("tantivy");
        std::fs::create_dir_all(&tantivy_dir)?;

        Ok(Self {
            notes_dir,
            db_path,
            tantivy_dir,
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
            Config::resolve_with(Some(bad_dir), cli_weights, cli_embed, FileConfig::default());
        assert!(result.is_err());
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
}
