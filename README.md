# Needle

Semantic search for directories of Markdown, plain-text, PDF, EPUB, HTML, and Word (`.docx`) documents, including documents stored inside `.zip` archives. Needle fuses vector similarity, full-text search, and filename matching into one ranked result list. It reads queries from stdin and writes tab-separated lines, so it fits into shell pipelines.

## Install

```bash
cargo install --path .
```

Tell needle where your documents live by adding a namespace to `~/.config/needle/config.toml`:

```toml
[[namespaces]]
name = "notes"
description = "Personal notes"
paths = ["/path/to/notes"]
```

Embeddings run locally by default using [fastembed](https://github.com/Anush008/fastembed-rs) (all-MiniLM-L6-v2). It needs no API key. The model downloads on first run.

The default build enables the `local` and `documents` Cargo features. `documents` uses Xberg to prepare Markdown (`.md`, `.markdown`), plain text (`.txt`), PDF (`.pdf`), EPUB (`.epub`), HTML (`.html`, `.htm`), and Word (`.docx`) files. For a smaller binary without the local model and document preparation, build with `--no-default-features` and use an API provider instead. That build indexes Markdown only.

## Background service

`needle service` prints a definition but never installs, enables, or starts it. The generated definition uses the running executable's path; pass `--exec-path` to use a different path.

### Watch for changes

On Linux, install the systemd user service and follow its logs with journald:

```bash
needle service watch > ~/.config/systemd/user/needle-watch.service
systemctl --user daemon-reload
systemctl --user enable --now needle-watch
journalctl --user -u needle-watch -f
```

On macOS, install the launchd agent:

```bash
needle service watch > ~/Library/LaunchAgents/dev.needle.watch.plist
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/dev.needle.watch.plist
```

A systemd user service stops at logout and does not start at boot unless you enable lingering:

```bash
loginctl enable-linger $USER
```

### Reindex periodically

Prefer a periodic reindex when some staleness is acceptable. Incremental indexing skips unchanged files by hash, so a timer costs less and holds the index lock only while it runs. In return, search results can be stale for up to the interval and each run performs a full in-memory vector rebuild.

On Linux, the timer activates a separate reindex service, so install both files:

```bash
needle service reindex > ~/.config/systemd/user/needle-reindex.service
needle service timer > ~/.config/systemd/user/needle-reindex.timer
systemctl --user daemon-reload
systemctl --user enable --now needle-reindex.timer
```

On macOS, one launchd agent performs the periodic reindex:

```bash
needle service timer > ~/Library/LaunchAgents/dev.needle.timer.plist
launchctl bootstrap gui/$UID ~/Library/LaunchAgents/dev.needle.timer.plist
```

Nix users should not paste a `/nix/store` path into a unit. `current_exe()` resolves profile symlinks to a store path that garbage collection can remove; wait for the Home Manager module.

## Usage

Index your documents, then search:

```bash
needle reindex
needle search "error handling patterns"
```

Each result is one line: `score \t path \t snippet`. That makes results easy to pipe:

```bash
needle search "authentication" -p | xargs bat
echo "query from clipboard" | needle search
```

Search and similarity cover every configured directory by default. Repeat `--namespace` to target the union of selected groups:

```bash
needle search "authentication" --namespace notes --namespace work
needle similar --namespace notes --namespace work
```

### List namespaces

See the configured documentation groups before searching:

```bash
needle namespaces
needle --json namespaces
```

### Browse indexed documents

Start the browser interface (default `127.0.0.1:8080`, adjustable with `--host` and `--port`). You can select any combination of configured namespaces, and search links preserve the selection:

```bash
needle serve
```

### Find related documents

Given a document, find others like it using the vector index:

```bash
needle related "design/auth-flow.md"
needle related "design/auth-flow.md" -p | head -5
```

### Find duplicates and clusters

Compare all documents pairwise to surface near-duplicates. The similarity threshold defaults to 0.85:

```bash
needle similar
needle similar --threshold 0.9 --group
needle similar -p | sort -u | wc -l
```

### Watch for changes

Keep the index current as you edit:

```bash
needle watch
```

Needle watches every configured directory at once. A change re-indexes only the affected file.

### Archives

Supported documents inside a `.zip` archive are indexed as separate results, identified as `archive.zip!path/inside.md`. Needle reads the archive index to decide which members changed, so a reindex extracts only new or edited members, one at a time. Nested archives, encrypted members, and members that expand beyond internal limits are reported by `needle failures` instead of being indexed.

## Output and flags

- `-p` / `--paths-only` on `search`, `similar`, and `related` prints bare paths, one per line.
- `--json` on any command prints a JSON array instead of tab-separated lines.
- `-l` / `--limit` controls result count: default 10 for `search` and `related`, 50 for `similar`.

Search blends three rankings (semantic, full-text, and filename) with weights you can tune per query:

```bash
needle search "topic" --w-semantic 2.0 --w-fts 0.5 --w-filename 0
```

The defaults are 1.5 semantic, 1.0 full-text, 0.7 filename.

## Config

The config file lives at `~/.config/needle/config.toml`. Needle requires at least one namespace. Everything else is optional:

```toml
provider = "openai"
model = "text-embedding-3-small"
api_base = "http://localhost:11434/v1"
dim = 768
openai_api_key = "sk-..."
needle_api_key = "my-gateway-key"
w_semantic = 1.5
w_fts = 1.0
w_filename = 0.7

[[namespaces]]
name = "notes"
description = "Personal notes"
paths = ["/home/you/notes"]
```

Environment variables override the config file. CLI flags override everything.

| Setting                  | Env var                 | Config key       |
| ------------------------ | ----------------------- | ---------------- |
| Documentation namespaces | No environment variable | `[[namespaces]]` |
| Provider                 | `NEEDLE_PROVIDER`       | `provider`       |
| Model                    | `NEEDLE_MODEL`          | `model`          |
| API base URL             | `NEEDLE_API_BASE`       | `api_base`       |
| Dimension override       | `NEEDLE_DIM`            | `dim`            |
| Voyage API key           | `VOYAGE_API_KEY`        | `voyage_api_key` |
| OpenAI API key           | `OPENAI_API_KEY`        | `openai_api_key` |
| Custom endpoint key      | `NEEDLE_API_KEY`        | `needle_api_key` |
| Semantic weight          | `NEEDLE_W_SEMANTIC`     | `w_semantic`     |
| Full-text weight         | `NEEDLE_W_FTS`          | `w_fts`          |
| Filename weight          | `NEEDLE_W_FILENAME`     | `w_filename`     |
| Log verbosity            | `NEEDLE_LOG`            | No config key    |

Each namespace has a unique, case-sensitive `name`, an optional `description`, and one or more `paths`. A directory may belong to more than one namespace, but Needle creates and searches one index per canonical path. Distinct paths must not overlap: configuring both `/docs` and `/docs/project` is invalid.

Index data lives under `~/.local/share/needle/` (or `$XDG_DATA_HOME/needle/`), one store per directory.

### Logging

Needle logs its own activity at `info` and stays quiet about its dependencies. Raise its level with `-v` for debug or `-vv` for trace, or set `NEEDLE_LOG` to a level such as `debug`. Both affect Needle only, so a noisy dependency cannot bury the output.

`RUST_LOG` keeps its conventional meaning: it is passed through verbatim and outranks both `-v` and `NEEDLE_LOG`, so `RUST_LOG=debug` enables every dependency as usual.

### Embedding providers

Needle supports three embedding backends. It picks one from the API keys it finds, or you can set `NEEDLE_PROVIDER` to `local`, `openai`, or `voyage` explicitly.

**Local (default):** No setup. Runs fastembed with ONNX models in-process.

**OpenAI-compatible:** Works with OpenAI, Ollama, vLLM, text-embeddings-inference, or any server that speaks the `/v1/embeddings` API.

```bash
export OPENAI_API_KEY=sk-...
needle reindex
```

For a local server like Ollama:

```bash
export NEEDLE_PROVIDER=openai
export NEEDLE_API_BASE=http://localhost:11434/v1
export NEEDLE_MODEL=nomic-embed-text
export NEEDLE_DIM=768
needle reindex
```

For an authenticated OpenAI-compatible endpoint other than `api.openai.com`, use `NEEDLE_API_KEY` instead of `OPENAI_API_KEY`. Needle scopes `OPENAI_API_KEY` to the default OpenAI base URL and never sends it to a custom `NEEDLE_API_BASE`:

```bash
export NEEDLE_PROVIDER=openai
export NEEDLE_API_BASE=https://my-gateway.example/v1
export NEEDLE_API_KEY=my-gateway-key
export NEEDLE_MODEL=text-embedding-3-small
needle reindex
```

**Voyage AI:**

```bash
export VOYAGE_API_KEY=your-key
needle reindex
```

## License

MIT
