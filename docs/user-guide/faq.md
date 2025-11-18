# FAQ & Troubleshooting

This page collects frequently asked questions and common issues when working with VectorFlow.

If you run into a problem that is not covered here, consider enabling debug logging and opening an issue with logs and configuration details.

## Installation & build issues

### `mkdocs build` fails with missing plugin errors

#### Symptoms

- Error messages mention `mkdocstrings` or other MkDocs plugins not found.

#### Cause

- The docs extras are not installed in the current Python environment.

#### Fix

```bash
pip install "vectorflow[docs]"
mkdocs build
```

Make sure you are running `mkdocs` from the same environment where `vectorflow` and its docs extras are installed.

### YAML linter complains about `python/name:` tags in `mkdocs.yml`

#### Symptoms

- Your editor or YAML linter reports "unresolved tag" errors for lines like:
  - `!!python/name:material.extensions.emoji.to_svg`
  - `!!python/name:pymdownx.superfences.fence_code_format`

#### Cause

- These tags are part of the MkDocs Material and pymdownx configuration and are interpreted by MkDocs, not by generic YAML linters.

#### Fix

- You can safely ignore these warnings in your editor.
- Alternatively, configure your YAML linter to skip `mkdocs.yml` or to allow `!!python/name` tags.

MkDocs itself will handle these correctly as long as the required packages are installed.

## Configuration & environment variables

### Configuration does not pick up environment variables

#### Symptoms

- Changing environment variables has no visible effect.
- `Config()` seems to always use default values.

#### Cause

- Environment variables may not be using the nested key format or the `.env` file is not in the project root.

#### Fix

- Ensure nested keys use the `__` delimiter:

```bash
EMBEDDING__MODEL_TYPE=openai
EMBEDDING__MODEL_NAME=text-embedding-3-small
TEXT_PROCESSING__CHUNK_SIZE=1200
```

- Place your `.env` file in the project root (same directory as `mkdocs.yml` and `pyproject.toml`).
- Restart your shell or IDE so that environment changes take effect.

See also: [Configuration](../getting-started/configuration.md) and [Environment Variables](../api/environment-variables.md).

## Embedding provider issues

### OpenAI embeddings: authentication or 401 errors

#### Symptoms

- Errors mentioning invalid API key or unauthorized requests when using `EmbeddingModelType.OPENAI`.

#### Cause

- Missing or incorrect `OPENAI_API_KEY` / `EMBEDDING__OPENAI_API_KEY`.

#### Fix

- Set the API key via environment variable:

```bash
export OPENAI_API_KEY="sk-..."  # pragma: allowlist secret
```

- Alternatively, use nested config variable:

```bash
EMBEDDING__OPENAI_API_KEY=sk-...
```

- Double-check that your `model_name` matches the available embeddings (for example `text-embedding-3-small`).

### Google Gemini or Azure OpenAI errors

#### Symptoms

- Errors about missing project/location for Gemini, or invalid endpoint/version for Azure OpenAI.

#### Fix

- For Google Gemini:

```bash
EMBEDDING__GOOGLE_GEMINI_API_KEY=...
EMBEDDING__GOOGLE_GEMINI_PROJECT_ID=your-gcp-project
EMBEDDING__GOOGLE_GEMINI_LOCATION=us-central1
```

- For Azure OpenAI:

```bash
EMBEDDING__AZURE_OPENAI_API_KEY=...
EMBEDDING__AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
EMBEDDING__AZURE_OPENAI_API_VERSION=2024-02-01
EMBEDDING__AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment
```

See also: [Embeddings API](../api/embeddings.md) and [Environment Variables](../api/environment-variables.md).

## Vector database issues

### Dimension mismatch between embeddings and vector index

#### Symptoms

- Errors mentioning vector dimension mismatch when inserting or querying embeddings.

#### Cause

- The vector database index was created with a different dimension than the embedding model.

#### Fix

- Confirm the embedding dimension for your model (for many OpenAI models this is `1536`).
- Configure your backend accordingly, for example with Pinecone:

```python
from vectorflow.core.vector_db.config import PineconeConfig

config.vector_db = PineconeConfig(
    api_key="your-pinecone-key",  # pragma: allowlist secret
    environment="your-pinecone-env",
    index_name="pdf-documents",
    dimension=1536,
    collection_name="pdf_documents",
)
```

- If the existing index has the wrong dimension, recreate it with the correct dimension.

See also: [Vector Database API](../api/vector-db.md).

### No results returned from search

#### Symptoms

- Searches return empty results even though documents were processed.

#### Possible causes & checks

- Ensure that `process_pdf` completed successfully (check `ProcessingResult.success`).
- Verify the collection name matches between ingestion and search (`collection_name` / `--collection`).
- Check that you are searching in the correct backend (local ChromaDB vs remote Pinecone/Qdrant).
- Try a very generic query without filters to rule out over-restrictive filters.

## CLI & GUI issues

### `vectorflow` or `vectorflow-gui` not found

#### Symptoms

- Shell reports `command not found` for `vectorflow` or `vectorflow-gui`.

#### Fix

- Ensure `vectorflow` is installed in the active virtual environment:

```bash
pip install vectorflow
```

- On some systems you may need to run via Python explicitly:

```bash
python -m vectorflow.cli.main --help
```

- If you have multiple Python versions, ensure the one on your PATH is the one where `vectorflow` is installed.

### GUI fails to start or shows a blank window

#### Checks

- Run from a terminal and watch for error output.
- Confirm that all required Qt / PySide6 dependencies are installed.
- Ensure your configuration is valid (especially vector database settings and embedding provider credentials).

The GUI uses the same `Config` as the CLI and Python API, so the [Configuration](../getting-started/configuration.md) and [Environment Variables](../api/environment-variables.md) pages apply here as well.
