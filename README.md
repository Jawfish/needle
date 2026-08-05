# Needle

Semantic search for directories of Markdown, plain-text, PDF, EPUB, HTML, and Word (`.docx`) documents. Combines vector similarity, full-text search, and filename matching into a single ranked result set. Reads queries from stdin and emits tab-separated results, so it composes naturally in pipelines.

## Install

```bash
cargo install --path .
```

Add a documentation namespace to `~/.config/needle/config.toml`:

```toml
[[namespaces]]
name = "notes"
description = "Personal notes"
paths = ["/path/to/notes"]
```

By default, needle uses [fastembed](https://github.com/Anush008/fastembed-rs) for local embeddings (all-MiniLM-L6-v2). No API key needed. The model downloads automatically on first run.

The default build enables the `local` and `documents` Cargo features. `documents` uses Xberg to prepare Markdown (`.md`, `.markdown`), plain text (`.txt`), PDF (`.pdf`), EPUB (`.epub`), HTML (`.html`, `.htm`), and Word (`.docx`) documents. For a smaller binary without the local model and document preparation, build with `--no-default-features` and use an API provider instead. This build indexes Markdown only.

## Usage

Index your notes, then search:

```bash
needle reindex
needle search "error handling patterns"
```

Search output is tab-separated (`score \t path \t snippet`), so it works well in pipelines:

```bash
needle search "authentication" -p | xargs bat
echo "query from clipboard" | needle search
```

Search and similarity include every configured directory by default. Repeat `--namespace` to target the union of selected groups:

```bash
needle search "authentication" --namespace notes --namespace work
needle similar --namespace notes --namespace work
```

### Discover documentation namespaces

List the configured documentation groups before searching:

```bash
needle namespaces
needle --json namespaces
```

### Find related documents

Given a note, find others like it using the vector index:

```bash
needle related "design/auth-flow.md"
needle related "design/auth-flow.md" -p | head -5
```

### Find duplicates and clusters

Compare all documents pairwise to surface near-duplicates:

```bash
needle similar
needle similar --threshold 0.9 --group
needle similar -p | sort -u | wc -l
```

### Watch for changes

Keep the index up to date as you edit:

```bash
needle watch
```

All configured directories are watched simultaneously. Changes in any directory trigger re-indexing of the affected file only.

## Flags

`-p` / `--paths-only` on `search`, `similar`, and `related` emits bare paths, one per line.

`-l` / `--limit` controls result count (default 10 for search/related, 50 for similar).

Search ranking weights are tunable per-query or through config:

```bash
needle search "topic" --w-semantic 2.0 --w-fts 0.5 --w-filename 0
```

## Config

Optional config file at `~/.config/needle/config.toml`:

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
| Provider                 | `NEEDLE_PROVIDER`       | `provider`       |
| Model                    | `NEEDLE_MODEL`          | `model`          |
| API base URL             | `NEEDLE_API_BASE`       | `api_base`       |
| Dimension override       | `NEEDLE_DIM`            | `dim`            |
| Voyage API key           | `VOYAGE_API_KEY`        | `voyage_api_key` |
| OpenAI API key           | `OPENAI_API_KEY`        | `openai_api_key` |
| Custom endpoint key      | `NEEDLE_API_KEY`        | `needle_api_key` |
| Documentation namespaces | No environment variable | `[[namespaces]]` |

Each namespace has a unique, case-sensitive `name`, optional `description`, and one or more `paths`. A directory may belong to several namespaces, but Needle creates and searches one index for its canonical path. Distinct paths must not overlap: for example, configuring both `/docs` and `/docs/project` is invalid.

### Migrating to namespaces

Replace the removed `notes_dirs` setting and `--docs-dir` flag with a namespace:

```toml
[[namespaces]]
name = "notes"
paths = ["/path"]
```

### Embedding Providers

Needle supports three embedding backends. It infers which to use from available API keys, or you can set `NEEDLE_PROVIDER` explicitly.

**Local (default):** No setup needed. Uses fastembed with ONNX models in-process.

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

For an authenticated OpenAI-compatible endpoint (not `api.openai.com`), use `NEEDLE_API_KEY` instead of `OPENAI_API_KEY`. `OPENAI_API_KEY` is scoped to the default OpenAI base URL and is never forwarded to a custom `NEEDLE_API_BASE`:

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
