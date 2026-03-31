# Needle

Semantic search for a directory of markdown files. Combines vector similarity, full-text search, and filename matching into a single ranked result set. Reads queries from stdin and emits tab-separated results, so it composes naturally in pipelines.

## Install

```bash
cargo install --path .
```

Add your docs directories to `~/.config/needle/config.toml`:

```toml
notes_dirs = ["/path/to/notes"]
```

Or pass `--docs-dir` before the subcommand to override for one invocation:

```bash
needle --docs-dir ~/notes search "query"
needle --docs-dir ~/notes --docs-dir ~/work/docs search "query"
```

By default, needle uses [fastembed](https://github.com/Anush008/fastembed-rs) for local embeddings (all-MiniLM-L6-v2). No API key needed. The model downloads automatically on first run.

For a smaller binary without the local model, build with `--no-default-features` and use an API provider instead.

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
notes_dirs = ["/home/you/notes"]
provider = "openai"
model = "text-embedding-3-small"
api_base = "http://localhost:11434/v1"
dim = 768
openai_api_key = "sk-..."
needle_api_key = "my-gateway-key"
w_semantic = 1.5
w_fts = 1.0
w_filename = 0.7
```

Environment variables override the config file. CLI flags override everything.

| Setting             | Env var           | Config key        |
| ------------------- | ----------------- | ----------------- |
| Provider            | `NEEDLE_PROVIDER` | `provider`        |
| Model               | `NEEDLE_MODEL`    | `model`           |
| API base URL        | `NEEDLE_API_BASE` | `api_base`        |
| Dimension override  | `NEEDLE_DIM`      | `dim`             |
| Voyage API key      | `VOYAGE_API_KEY`  | `voyage_api_key`  |
| OpenAI API key      | `OPENAI_API_KEY`  | `openai_api_key`  |
| Custom endpoint key | `NEEDLE_API_KEY`  | `needle_api_key`  |
| Docs directories    | (use `--docs-dir`)| `notes_dirs`      |

Docs directories can be specified as a TOML array in `notes_dirs`, or overridden per-invocation with one or more `--docs-dir <PATH>` flags. When `--docs-dir` is passed, the config file directories are ignored for that invocation.

### Migrating from `notes_dir`

If you previously used `notes_dir = "/path"` in config or `NEEDLE_DOCS_DIR`, replace with:

```toml
notes_dirs = ["/path"]
```

Remove any `NEEDLE_DOCS_DIR` exports from your shell profile.

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
