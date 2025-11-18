# CLI Usage

VectorFlow exposes a Typer-based CLI via the `vectorflow` entry point.

```bash
vectorflow --help
```

!!! tip "See help for any command"
    Run `vectorflow <command> --help` to view available options and examples.

## Processing PDFs

```bash
vectorflow process path/to/file.pdf \
  --model all-MiniLM-L6-v2 \
  --collection pdf_documents \
  --chunk-size 1500
```

Key options for `process`:

- `--model / -m`: embedding model name (Sentence Transformers or OpenAI models).
- `--collection / -c`: collection name in the vector database.
- `--batch-size / -b`: batch size for embedding generation.
- `--chunk-size`: text chunk size in characters.
- `--clean / --no-clean`: enable/disable text cleaning.
- `--progress / --no-progress`: enable/disable progress UI.
- `--verbose / -v`: verbose output.
- `--debug`: enable debug logging.

## Searching

```bash
vectorflow search "machine learning" --results 5
```

Useful options:

- `--results / -n`: number of results to return.
- `--document / -d`: filter by document ID.
- `--page / -p`: filter by page number.
- `--collection / -c`: override collection to search.
- `--threshold / -t`: minimum similarity score (0.0–1.0).
- `--verbose / -v`: print full chunk content.

## Collection utilities

Helpful maintenance commands:

- `vectorflow stats` – show collection-level statistics.
- `vectorflow list-docs` – list stored documents.
- `vectorflow doc-info <document_id>` – show stats for a single document.
- `vectorflow backend` – show current vector DB backend information.
- `vectorflow clear-collection` – remove all chunks in the current collection.
- `vectorflow create-collection` – create the collection if it does not exist.
- `vectorflow delete-collection <name>` – delete a collection.
- `vectorflow count` – count chunks in the current collection.
- `vectorflow similar <chunk_id>` – find chunks similar to a given chunk.
- `vectorflow search-meta key=value ...` – search by metadata filters.

Run any command with `--help` to see full options:

```bash
vectorflow process --help
```

## Common workflows

### 1. Index then search

```bash
vectorflow process data/papers/*.pdf --model all-MiniLM-L6-v2
vectorflow search "transformer architecture" --results 10 --verbose
```

### 2. Separate collections

```bash
# Technical docs
vectorflow process docs/tech/*.pdf --collection tech_docs

# Legal docs
vectorflow process docs/legal/*.pdf --collection legal_docs

# Search only in legal documents
vectorflow search "non-disclosure" --collection legal_docs
```

### 3. Inspect and maintain collections

```bash
# View high-level stats
vectorflow stats

# List documents and inspect one of them
vectorflow list-docs --limit 20
vectorflow doc-info my_document_id

# Clean up
vectorflow clear-collection --yes
```

### 4. Metadata and similarity-based queries

```bash
# Find chunks similar to a known chunk ID
vectorflow similar some_chunk_id --limit 5

# Search by metadata (e.g. page_number=1)
vectorflow search-meta page_number=1 --limit 20
```

## Command reference

> All commands are invoked via the `vectorflow` entry point, for example
> `vectorflow process ...`.

### `process`

Process one or more PDF files and store their embeddings in the configured
vector database.

**Example:**

```bash
vectorflow process docs/*.pdf \
  --model all-MiniLM-L6-v2 \
  --collection pdf_documents \
  --chunk-size 1500
```

**Key options:**

- `files...` (argument): one or more PDF paths to process.
- `--model, -m`: embedding model name (Sentence Transformers, OpenAI, etc.).
- `--collection, -c`: collection name (overrides configured `collection_name`).
- `--batch-size, -b`: batch size for embedding generation.
- `--chunk-size`: text chunk size in characters.
- `--clean / --no-clean`: enable or disable text cleaning.
- `--progress / --no-progress`: show or hide progress bars.
- `--verbose, -v`: more verbose console output.
- `--debug`: enable debug logging.

---

### `search`

Run a semantic search against the current collection.

```bash
vectorflow search "machine learning" --results 10 --verbose
```

**Key options:**

- `query` (argument): query text.
- `--results, -n`: number of results to return (default `10`).
- `--document, -d`: filter by `document_id`.
- `--page, -p`: filter by page number.
- `--collection, -c`: collection to search.
- `--threshold, -t`: minimum similarity score (`0.0`–`1.0`).
- `--verbose, -v`: print full chunk content instead of a truncated preview.
- `--debug`: enable debug logging.

---

### `stats`

Show summary statistics for the current collection.

```bash
vectorflow stats --collection pdf_documents
```

**Options:**

- `--collection, -c`: collection name (defaults to the configured one).
- `--debug`: enable debug logging.

---

### `list-docs`

List documents stored in the collection with basic statistics.

```bash
vectorflow list-docs --limit 50
```

**Options:**

- `--collection, -c`: collection name.
- `--limit, -l`: maximum number of documents to show (default `50`).
- `--debug`: enable debug logging.

---

### `doc-info`

Show detailed statistics for a single document.

```bash
vectorflow doc-info my_document_id
```

**Arguments and options:**

- `document_id` (argument): document ID to inspect.
- `--collection, -c`: collection name.
- `--debug`: enable debug logging.

---

### `collections`

List available vector database collections and their status.

```bash
vectorflow collections
```

**Options:**

- `--debug`: enable debug logging.

---

### `backend`

Display information about the current vector database backend (type, metric,
dimension, etc.).

```bash
vectorflow backend
```

**Options:**

- `--collection, -c`: collection name (used by some backends).
- `--debug`: enable debug logging.

---

### `create-collection`

Create a collection if it does not already exist.

```bash
vectorflow create-collection --collection pdf_documents
```

**Options:**

- `--collection, -c`: collection name to create.
- `--debug`: enable debug logging.

---

### `clear-collection`

Remove all chunks from a collection.

!!! warning "Destructive operation"
    This command irreversibly removes all chunks from the target collection. Double-check the collection name before using `--yes`.

```bash
vectorflow clear-collection --collection pdf_documents --yes
```

**Options:**

- `--collection, -c`: collection name.
- `--yes, -y`: skip confirmation prompt.
- `--debug`: enable debug logging.

---

### `delete-collection`

Delete a collection by name.

!!! warning "Deletes the entire collection"
    This removes the collection and all of its contents. Consider backing up or inspecting statistics before running this command.

```bash
vectorflow delete-collection pdf_documents --yes
```

**Arguments and options:**

- `collection_name` (argument): name of the collection to delete.
- `--yes, -y`: skip confirmation.
- `--debug`: enable debug logging.

---

### `count`

Count the total number of chunks in the collection.

```bash
vectorflow count --collection pdf_documents
```

**Options:**

- `--collection, -c`: collection name.
- `--debug`: enable debug logging.

---

### `delete`

Delete a single document and all of its chunks from the collection.

!!! warning "Removes a single document"
    This permanently deletes the document and its associated chunks from the vector database. Use with care.

```bash
vectorflow delete my_document_id --yes
```

**Arguments and options:**

- `document_id` (argument): document ID to delete.
- `--collection, -c`: collection name.
- `--yes, -y`: skip confirmation.
- `--debug`: enable debug logging.

---

### `similar`

Find chunks similar to a given chunk ID.

```bash
vectorflow similar some_chunk_id --limit 5 --verbose
```

**Arguments and options:**

- `chunk_id` (argument): reference chunk ID.
- `--limit, -n`: number of results to return (default `10`).
- `--collection, -c`: collection name.
- `--verbose, -v`: print full content, otherwise truncate.
- `--debug`: enable debug logging.

---

### `search-meta`

Search by metadata filters (for example `page_number=1`).

```bash
vectorflow search-meta page_number=1 document_id=my_doc --limit 20
```

**Arguments and options:**

- `filters...` (arguments): one or more `key=value` filters.
- `--limit, -n`: maximum number of results to return.
- `--collection, -c`: collection name.
- `--debug`: enable debug logging.

---

### `health`

Check the health of the embedding service, vector database and pipeline.

```bash
vectorflow health
```

**Options:**

- `--collection, -c`: collection name (used by some backends).
- `--debug`: enable debug logging.

---

### `config`

Display the current configuration, with an option to show all details.

```bash
vectorflow config            # key settings
vectorflow config --show-all  # all settings
```

**Options:**

- `--show-all, -a`: show all configuration fields (including defaults and
  performance settings).
- `--debug`: enable debug logging.
