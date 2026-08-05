use std::{
    collections::HashSet,
    path::{Path, PathBuf},
};

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
pub struct Namespace {
    pub name: String,
    pub description: Option<String>,
    pub paths: Vec<PathBuf>,
}

#[derive(Debug)]
pub struct Config {
    pub namespaces: Vec<Namespace>,
    pub docs_dirs: Vec<DirectoryStore>,
    pub embed: EmbedConfig,
    pub weights: RrfWeights,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    provider: Option<String>,
    model: Option<String>,
    api_base: Option<String>,
    dim: Option<usize>,
    voyage_api_key: Option<String>,
    openai_api_key: Option<String>,
    needle_api_key: Option<String>,
    namespaces: Option<Vec<FileNamespace>>,
    w_semantic: Option<f64>,
    w_fts: Option<f64>,
    w_filename: Option<f64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileNamespace {
    name: Option<String>,
    description: Option<String>,
    paths: Option<Vec<PathBuf>>,
}

#[derive(Clone, Copy)]
pub struct CliWeights {
    pub semantic: Option<f64>,
    pub fts: Option<f64>,
    pub filename: Option<f64>,
}

impl Config {
    pub fn resolve(cli_weights: CliWeights, cli_embed: CliEmbedArgs) -> anyhow::Result<Self> {
        let file_config = load_file_config()?;
        Self::resolve_with(cli_weights, cli_embed, &file_config)
    }

    fn resolve_with(
        cli_weights: CliWeights,
        cli_embed: CliEmbedArgs,
        file_config: &FileConfig,
    ) -> anyhow::Result<Self> {
        let weights = resolve_weights(cli_weights, file_config);
        let embed = resolve_embed_config(cli_embed, file_config);
        let (namespaces, canonical_paths) = resolve_namespaces(file_config)?;
        let docs_dirs = build_directory_stores(canonical_paths)?;

        tracing::debug!(
            dirs = ?docs_dirs.iter().map(|store| store.notes_dir.display().to_string()).collect::<Vec<_>>(),
            "resolved docs dirs"
        );

        Ok(Self {
            namespaces,
            docs_dirs,
            embed,
            weights,
        })
    }
}

fn resolve_namespaces(file_config: &FileConfig) -> anyhow::Result<(Vec<Namespace>, Vec<PathBuf>)> {
    let definitions = file_config
        .namespaces
        .as_deref()
        .filter(|definitions| !definitions.is_empty())
        .ok_or(NeedleError::NoNamespaces)?;

    let mut names = HashSet::with_capacity(definitions.len());
    for (index, definition) in definitions.iter().enumerate() {
        let name = namespace_name(definition, index)?;
        if !names.insert(name) {
            return Err(NeedleError::InvalidNamespace(format!(
                "namespace '{name}' is configured more than once"
            ))
            .into());
        }
        namespace_paths(definition, name)?;
    }

    let mut namespaces = Vec::with_capacity(definitions.len());
    let mut canonical_paths = Vec::new();
    for (index, definition) in definitions.iter().enumerate() {
        let name = namespace_name(definition, index)?;
        let mut paths = Vec::new();
        for path in namespace_paths(definition, name)? {
            let canonical = path.canonicalize().with_context(|| {
                format!(
                    "canonicalizing path {} in namespace '{name}'",
                    path.display()
                )
            })?;
            if !canonical.is_dir() {
                return Err(NeedleError::MissingDirectories(format!(
                    "{} (namespace '{name}')",
                    canonical.display()
                ))
                .into());
            }
            if !paths.contains(&canonical) {
                paths.push(canonical.clone());
            }
            if !canonical_paths.contains(&canonical) {
                canonical_paths.push(canonical);
            }
        }
        namespaces.push(Namespace {
            name: name.to_owned(),
            description: definition.description.clone(),
            paths,
        });
    }

    reject_overlapping_directories(&canonical_paths)?;
    Ok((namespaces, canonical_paths))
}

fn namespace_name(definition: &FileNamespace, index: usize) -> anyhow::Result<&str> {
    let name = definition.name.as_deref().ok_or_else(|| {
        NeedleError::InvalidNamespace(format!("namespace entry {} is missing a name", index + 1))
    })?;
    if name.trim().is_empty() {
        return Err(NeedleError::InvalidNamespace(format!(
            "namespace entry {} has an empty name",
            index + 1
        ))
        .into());
    }
    Ok(name)
}

fn namespace_paths<'a>(definition: &'a FileNamespace, name: &str) -> anyhow::Result<&'a [PathBuf]> {
    definition
        .paths
        .as_deref()
        .filter(|paths| !paths.is_empty())
        .ok_or_else(|| {
            NeedleError::InvalidNamespace(format!(
                "namespace '{name}' must define one or more paths"
            ))
            .into()
        })
}

fn reject_overlapping_directories(paths: &[PathBuf]) -> anyhow::Result<()> {
    let mut overlap_pairs = Vec::new();
    for (index, path) in paths.iter().enumerate() {
        for other in &paths[index + 1..] {
            if path.starts_with(other) || other.starts_with(path) {
                overlap_pairs.push(format!("  {} and {}", path.display(), other.display()));
            }
        }
    }
    if overlap_pairs.is_empty() {
        Ok(())
    } else {
        Err(NeedleError::OverlappingDirectories(overlap_pairs.join("\n")).into())
    }
}

fn build_directory_stores(paths: Vec<PathBuf>) -> anyhow::Result<Vec<DirectoryStore>> {
    let mut docs_dirs = Vec::with_capacity(paths.len());
    for notes_dir in paths {
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
    Ok(docs_dirs)
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
        dim: env("NEEDLE_DIM")
            .and_then(|value| value.parse().ok())
            .or(file.dim),
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
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(FileConfig::default());
        }
        Err(error) => {
            return Err(
                anyhow::Error::from(error).context(format!("reading config: {}", path.display()))
            );
        }
    };

    toml::from_str(&content).context(format!("parsing config: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights() -> CliWeights {
        CliWeights {
            semantic: None,
            fts: None,
            filename: None,
        }
    }

    fn embed() -> CliEmbedArgs {
        CliEmbedArgs {
            provider: None,
            model: None,
            api_base: None,
        }
    }

    fn namespace(name: Option<&str>, paths: Option<Vec<PathBuf>>) -> FileNamespace {
        FileNamespace {
            name: name.map(str::to_owned),
            description: None,
            paths,
        }
    }

    fn resolve(namespaces: Vec<FileNamespace>) -> anyhow::Result<Config> {
        let file_config = FileConfig {
            namespaces: Some(namespaces),
            ..Default::default()
        };
        Config::resolve_with(weights(), embed(), &file_config)
    }

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
    fn file_config_overrides_weight_defaults() {
        let file_config = FileConfig {
            w_semantic: Some(9.0),
            w_fts: Some(8.0),
            w_filename: Some(7.0),
            ..Default::default()
        };

        let resolved = resolve_weights(weights(), &file_config);
        assert!((resolved.semantic - 9.0).abs() < f64::EPSILON);
        assert!((resolved.fts - 8.0).abs() < f64::EPSILON);
        assert!((resolved.filename - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn no_namespaces_configured_is_an_error() {
        let file_config = FileConfig::default();
        let result = Config::resolve_with(weights(), embed(), &file_config);
        let message = result.expect_err("must fail").to_string();
        assert!(message.contains("no documentation namespaces configured"));
    }

    #[test]
    fn namespace_requires_a_name() {
        let result = resolve(vec![namespace(None, Some(vec![PathBuf::from("/tmp")]))]);
        let message = result.expect_err("must fail").to_string();
        assert!(message.contains("missing a name"));
    }

    #[test]
    fn namespace_name_cannot_be_blank() {
        let result = resolve(vec![namespace(
            Some("  "),
            Some(vec![PathBuf::from("/tmp")]),
        )]);
        let message = result.expect_err("must fail").to_string();
        assert!(message.contains("empty name"));
    }

    #[test]
    fn namespace_names_must_be_unique() {
        let result = resolve(vec![
            namespace(Some("work"), Some(vec![PathBuf::from("/tmp")])),
            namespace(Some("work"), Some(vec![PathBuf::from("/var/tmp")])),
        ]);
        let message = result.expect_err("must fail").to_string();
        assert!(message.contains("configured more than once"));
    }

    #[test]
    fn namespace_names_are_case_sensitive() {
        let base = tempfile::tempdir().expect("tempdir");
        let first = base.path().join("first");
        let second = base.path().join("second");
        std::fs::create_dir(&first).expect("create first");
        std::fs::create_dir(&second).expect("create second");

        let config = resolve(vec![
            namespace(Some("work"), Some(vec![first])),
            namespace(Some("Work"), Some(vec![second])),
        ])
        .expect("must resolve");
        assert_eq!(config.namespaces.len(), 2);
    }

    #[test]
    fn namespace_requires_one_or_more_paths() {
        let result = resolve(vec![namespace(Some("work"), Some(Vec::new()))]);
        let message = result.expect_err("must fail").to_string();
        assert!(message.contains("one or more paths"));
    }

    #[test]
    fn shared_path_uses_one_store_and_preserves_membership() {
        let directory = tempfile::tempdir().expect("tempdir");
        let expected_path = directory.path().canonicalize().expect("canonicalize");
        let mut first = namespace(
            Some("project-a"),
            Some(vec![directory.path().to_path_buf()]),
        );
        first.description = Some("Project A documentation".to_owned());
        let second = namespace(Some("shared"), Some(vec![expected_path.clone()]));

        let config = resolve(vec![first, second]).expect("must resolve");
        assert_eq!(config.docs_dirs.len(), 1);
        assert_eq!(config.docs_dirs[0].notes_dir, expected_path);
        assert_eq!(
            config.namespaces[0].description.as_deref(),
            Some("Project A documentation")
        );
        assert_eq!(config.namespaces[0].paths, config.namespaces[1].paths);
    }

    #[test]
    fn duplicate_paths_within_a_namespace_use_one_store() {
        let directory = tempfile::tempdir().expect("tempdir");
        let config = resolve(vec![namespace(
            Some("project-a"),
            Some(vec![
                directory.path().to_path_buf(),
                directory.path().to_path_buf(),
            ]),
        )])
        .expect("must resolve");

        assert_eq!(config.docs_dirs.len(), 1);
        assert_eq!(config.namespaces[0].paths.len(), 1);
    }

    #[test]
    fn distinct_overlapping_paths_are_rejected() {
        let parent = tempfile::tempdir().expect("tempdir");
        let child = parent.path().join("child");
        std::fs::create_dir(&child).expect("create child");

        let result = resolve(vec![
            namespace(Some("parent"), Some(vec![parent.path().to_path_buf()])),
            namespace(Some("child"), Some(vec![child])),
        ]);
        let message = result.expect_err("must fail").to_string();
        assert!(message.contains("overlap"));
        assert!(message.contains(parent.path().to_string_lossy().as_ref()));
    }

    #[test]
    fn non_overlapping_paths_are_accepted() {
        let base = tempfile::tempdir().expect("tempdir");
        let first = base.path().join("first");
        let second = base.path().join("second");
        std::fs::create_dir(&first).expect("create first");
        std::fs::create_dir(&second).expect("create second");

        let config = resolve(vec![
            namespace(Some("one"), Some(vec![first])),
            namespace(Some("two"), Some(vec![second])),
        ])
        .expect("must resolve");
        assert_eq!(config.docs_dirs.len(), 2);
    }

    #[test]
    fn nonexistent_namespace_path_is_actionable() {
        let path = PathBuf::from("/nonexistent/needle-namespace-test");
        let result = resolve(vec![namespace(Some("work"), Some(vec![path.clone()]))]);
        let message = result.expect_err("must fail").to_string();
        assert!(message.contains("canonicalizing path"));
        assert!(message.contains(path.to_string_lossy().as_ref()));
        assert!(message.contains("namespace 'work'"));
    }

    #[test]
    fn file_namespace_path_is_rejected() {
        let file = tempfile::NamedTempFile::new().expect("tempfile");
        let result = resolve(vec![namespace(
            Some("work"),
            Some(vec![file.path().to_path_buf()]),
        )]);
        let message = result.expect_err("must fail").to_string();
        assert!(message.contains("docs directories not found"));
        assert!(message.contains("namespace 'work'"));
    }

    #[test]
    fn notes_dirs_is_not_accepted_in_configuration() {
        let result = toml::from_str::<FileConfig>("notes_dirs = [\"/tmp\"]");
        assert!(result.is_err());
        let error = result.err().expect("must fail");
        assert!(error.to_string().contains("notes_dirs"));
    }

    #[test]
    fn namespace_toml_parses() {
        let config = toml::from_str::<FileConfig>(
            r#"
[[namespaces]]
name = "project-a"
description = "Project A documentation"
paths = ["/tmp"]
"#,
        )
        .expect("must parse");
        let definitions = config.namespaces.expect("namespaces");
        assert_eq!(definitions.len(), 1);
        assert_eq!(definitions[0].name.as_deref(), Some("project-a"));
        assert_eq!(
            definitions[0].description.as_deref(),
            Some("Project A documentation")
        );
        assert_eq!(
            definitions[0].paths.as_deref(),
            Some([PathBuf::from("/tmp")].as_slice())
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
        let embed = resolve_embed_config(embed(), &FileConfig::default());
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
            store
                .to_relative("/home/user/notes/topic.md")
                .expect("must convert"),
            "topic.md"
        );
        assert_eq!(
            store
                .to_relative("/home/user/notes/sub/topic.md")
                .expect("must convert"),
            "sub/topic.md"
        );
    }

    #[test]
    fn to_relative_passes_through_relative_path() {
        let store = make_store("/home/user/notes");
        assert_eq!(
            store.to_relative("topic.md").expect("must convert"),
            "topic.md"
        );
        assert_eq!(
            store.to_relative("sub/topic.md").expect("must convert"),
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
