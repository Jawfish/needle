# Needle

Semantic search for a directory of markdown files. Combines vector similarity, full-text search, and filename matching into a single ranked result set.

Useful if you keep a folder of notes (Zettelkasten, engineering journal, docs repo) and want to find things by meaning rather than exact keywords.

## Install

Requires a [Voyage AI](https://www.voyageai.com/) API key for embeddings.

```
cargo install --path .
export VOYAGE_API_KEY=your-key
export ZK_NOTEBOOK_DIR=/path/to/notes
```

## Usage

Index your notes, then search:

```
needle reindex
needle search "error handling patterns"
```

Search output is tab-separated (`score \t path \t snippet`), so it works well in pipelines:

```
needle search "authentication" -p | xargs bat
echo "query from clipboard" | needle search
```

### Find related documents

Given a note, find others like it using the vector index:

```
needle related "design/auth-flow.md"
needle related "design/auth-flow.md" -p | head -5
```

### Find duplicates and clusters

Compare all documents pairwise to surface near-duplicates:

```
needle similar
needle similar --threshold 0.9 --group
needle similar -p | sort -u | wc -l
```

### Watch for changes

Keep the index up to date as you edit:

```
needle watch
```

## Flags

`-p` / `--paths-only` on `search`, `similar`, and `related` emits bare paths, one per line.

`-l` / `--limit` controls result count (default 10 for search/related, 50 for similar).

Search ranking weights are tunable per-query or through config:

```
needle search "topic" --w-semantic 2.0 --w-fts 0.5 --w-filename 0
```

## Config

Optional config file at `~/.config/needle/config.toml`:

```toml
notes_dir = "/home/you/notes"
voyage_api_key = "your-key"
w_semantic = 1.5
w_fts = 1.0
w_filename = 0.7
```

Environment variables (`VOYAGE_API_KEY`, `ZK_NOTEBOOK_DIR`, `NEEDLE_W_*`) override the config file. CLI flags override everything.

## How it works

Needle chunks each markdown file, embeds it via Voyage AI (`voyage-4`, 1024 dimensions), and stores the vectors in a local SQLite database with a vector index (libsql). Full-text search uses Tantivy. The search command fuses all ranking signals with reciprocal rank fusion.

## License

MIT
