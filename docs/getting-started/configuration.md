# Configuration

VectorFlow uses a Pydantic-based `Config` class that can be controlled by:

- Environment variables or a `.env` file
- Python code, by mutating the `Config` object before creating the pipeline

!!! info "Configuration layers"
    Values are loaded from environment variables and `.env` files first, then overridden by anything you set in Python code before instantiating the pipeline.

## Loading defaults

```python
from vectorflow import Config

config = Config()
```

This loads values from the environment via `pydantic-settings`. See the README for a full list of supported environment variables.

## Common fields

```python
from vectorflow import Config
from vectorflow.core.config.settings import EmbeddingModelType

config = Config()

# Embedding provider
config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
config.embedding.model_name = "all-MiniLM-L6-v2"

# Text chunking
config.text_processing.chunk_size = 1000

# Local ChromaDB backend
config.chroma_db.persist_directory = "./chroma_db"
config.chroma_db.collection_name = "pdf_documents"
```

## Environment variables

`Config` reads settings from environment variables using the nested key format
defined by `env_nested_delimiter="__"` in
`vectorflow.core.config.settings.Config`.

Some common examples:

```bash
# Embeddings
EMBEDDING__MODEL_TYPE=sentence-transformers
EMBEDDING__MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING__BATCH_SIZE=32

# Text processing
TEXT_PROCESSING__CHUNK_SIZE=1000
TEXT_PROCESSING__CHUNK_OVERLAP=100

# Local ChromaDB backend (legacy config)
CHROMA_DB__PERSIST_DIRECTORY=./chroma_db
CHROMA_DB__COLLECTION_NAME=pdf_documents
CHROMA_DB__DISTANCE_METRIC=cosine
CHROMA_DB__MAX_RESULTS=100

# Logging
LOGGING__LEVEL=INFO
```

You can put these into a `.env` file in your project root so they are loaded
automatically.

!!! tip "Use a `.env` file"
    For local development, a `.env` file keeps your shell environment clean while allowing the CLI, GUI and Python API to share the same configuration.

## Config sections overview

At a high level, `Config` groups settings into the following sections:

- `pdf`: PDF processing limits (max file size, timeouts, whether to extract images).
- `text_processing`: chunk size, overlap and separators.
- `embedding`: provider type, model name, batch size, retries and provider-specific keys.
- `chroma_db`: legacy ChromaDB backend configuration (still supported).
- `vector_db`: optional multi-backend configuration (ChromaDB, Pinecone, Weaviate, Qdrant, Milvus).
- `logging`: log level, format and file rotation.
- global flags: `debug`, `max_workers`.

When you assign `config.vector_db` explicitly (for example with a `PineconeConfig`
instance), it overrides the legacy `chroma_db` section and is used by the
pipeline as the active vector backend.

For a more exhaustive list of environment variables, see the [Environment Variables](../api/environment-variables.md) reference.
