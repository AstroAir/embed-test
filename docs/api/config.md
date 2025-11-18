# Configuration API

The central configuration model is `vectorflow.core.config.settings.Config`.

```python
from vectorflow.core.config.settings import Config, EmbeddingModelType
from vectorflow.core.config import ChromaDBConfig, EmbeddingConfig
```

!!! note "High-level configuration guides"
    For a gentler introduction, see the [Configuration](../getting-started/configuration.md) and [Environment Variables](environment-variables.md) pages. This section focuses on the Python-level API.

The top-level `Config` includes nested sections such as:

- `embedding: EmbeddingConfig`
- `chroma_db: ChromaDBConfig`
- `pdf`: PDF-related settings
- `text_processing`: chunking and normalization
- `logging`: logging configuration

All fields are Pydantic models and support environment variable overrides.

For a complete list of fields and default values, see the source in `vectorflow/core/config/settings.py`.

## Top-level fields (summary)

- `pdf: PDFConfig`
  - `max_file_size_mb` – maximum input PDF size.
  - `timeout_seconds` – per-document processing timeout.
  - `extract_images` – whether to extract images (for future use).
- `text_processing: TextProcessingConfig`
  - `chunk_size`, `chunk_overlap`, `min_chunk_size`, `separators`.
- `embedding: EmbeddingConfig`
  - `model_type`, `model_name`, `batch_size`, `max_retries`, `timeout_seconds`.
  - Provider-specific keys such as `openai_api_key`, `cohere_api_key`,
    `azure_openai_api_key`, `google_gemini_api_key`, etc.
- `chroma_db: ChromaDBConfig`
  - `persist_directory`, `collection_name`, `distance_metric`, `max_results`.
- `vector_db: Optional[VectorDBConfigType]`
  - When set (e.g. to `PineconeConfig`), enables the new multi-backend system.
  - When `None`, the legacy `chroma_db` configuration is automatically wrapped
    into a `ChromaDBConfig` and used as the active backend.
- `logging: LoggingConfig`
  - `level`, `format`, `file_path`, `rotation`, `retention`.
- `debug: bool`
  - When `True`, forces log level to `DEBUG`.
- `max_workers: int`
  - Number of worker threads used by the pipeline for embedding generation.

## Environment loading

The `Config` class extends `BaseSettings` and is configured to load:

- From a `.env` file in the project root (UTF-8).
- From process environment variables.

Nested fields use the `__` delimiter. For example:

```bash
EMBEDDING__MODEL_TYPE=openai
EMBEDDING__MODEL_NAME=text-embedding-3-small
TEXT_PROCESSING__CHUNK_SIZE=1500
```

## API reference

::: vectorflow.core.config.settings.Config

::: vectorflow.core.config.settings.EmbeddingConfig

::: vectorflow.core.config.settings.PDFConfig

::: vectorflow.core.config.settings.TextProcessingConfig

::: vectorflow.core.config.settings.LoggingConfig
