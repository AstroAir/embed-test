# Environment Variables

VectorFlow uses a Pydantic `Config` (`vectorflow.core.config.settings.Config`)
which is a `BaseSettings` subclass. This means configuration can be supplied via:

- A `.env` file in the project root (UTF-8)
- Process environment variables

Nested configuration fields use the `__` delimiter (see `env_nested_delimiter="__"`).
For example, the field `config.embedding.model_name` is controlled by
`EMBEDDING__MODEL_NAME`.

!!! note "Most common variables"
    The tables below list the most commonly used and recommended environment
    variables. Additional fields follow the same naming pattern
    `SECTION__FIELD`.

## Global settings

| Field        | Env var      | Default | Description                             |
| ------------ | ------------ | ------- | --------------------------------------- |
| `debug`      | `DEBUG`      | `False` | Enable debug mode (forces log level)    |
| `max_workers`| `MAX_WORKERS`| `4`     | Max worker threads for embeddings       |

## PDF settings (`pdf` section)

| Field              | Env var                    | Default | Description                        |
| ------------------ | -------------------------- | ------- | ---------------------------------- |
| `max_file_size_mb` | `PDF__MAX_FILE_SIZE_MB`    | `100`   | Max PDF file size in MB           |
| `timeout_seconds`  | `PDF__TIMEOUT_SECONDS`     | `300`   | Per-document processing timeout   |
| `extract_images`   | `PDF__EXTRACT_IMAGES`      | `False` | Whether to extract images         |

## Text processing (`text_processing` section)

| Field           | Env var                             | Default | Description                                  |
| --------------- | ----------------------------------- | ------- | -------------------------------------------- |
| `chunk_size`    | `TEXT_PROCESSING__CHUNK_SIZE`       | `1000`  | Chunk size in characters                     |
| `chunk_overlap` | `TEXT_PROCESSING__CHUNK_OVERLAP`    | `100`   | Overlap between chunks                       |
| `min_chunk_size`| `TEXT_PROCESSING__MIN_CHUNK_SIZE`   | `50`    | Minimum chunk size                           |
| `separators`    | `TEXT_PROCESSING__SEPARATORS`       | `\n\n,\n, ,` | Preferred split separators (advanced) |

## Embeddings (`embedding` section)

### Core fields

| Field            | Env var                           | Default                     | Description                             |
| ---------------- | --------------------------------- | --------------------------- | --------------------------------------- |
| `model_type`     | `EMBEDDING__MODEL_TYPE`           | `sentence-transformers`     | Provider type (see `EmbeddingModelType`)|
| `model_name`     | `EMBEDDING__MODEL_NAME`           | `all-MiniLM-L6-v2`          | Embedding model name                    |
| `batch_size`     | `EMBEDDING__BATCH_SIZE`           | `32`                        | Batch size for embedding generation     |
| `max_retries`    | `EMBEDDING__MAX_RETRIES`          | `3`                         | Max retries for API calls               |
| `timeout_seconds`| `EMBEDDING__TIMEOUT_SECONDS`      | `60`                        | Timeout for embedding requests          |

### Provider-specific fields

OpenAI:

| Field             | Env var                              | Default | Description           |
| ----------------- | ------------------------------------ | ------- | --------------------- |
| `openai_api_key`  | `EMBEDDING__OPENAI_API_KEY`          | `None`  | OpenAI API key        |
| `openai_base_url` | `EMBEDDING__OPENAI_BASE_URL`         | `None`  | OpenAI base URL       |

Cohere:

| Field             | Env var                              | Default | Description           |
| ----------------- | ------------------------------------ | ------- | --------------------- |
| `cohere_api_key`  | `EMBEDDING__COHERE_API_KEY`          | `None`  | Cohere API key        |
| `cohere_base_url` | `EMBEDDING__COHERE_BASE_URL`         | `None`  | Cohere base URL       |

Azure OpenAI:

| Field                       | Env var                                     | Default        | Description                         |
| --------------------------- | ------------------------------------------- | -------------- | ----------------------------------- |
| `azure_openai_api_key`      | `EMBEDDING__AZURE_OPENAI_API_KEY`           | `None`         | Azure OpenAI API key               |
| `azure_openai_endpoint`     | `EMBEDDING__AZURE_OPENAI_ENDPOINT`          | `None`         | Azure OpenAI endpoint              |
| `azure_openai_api_version`  | `EMBEDDING__AZURE_OPENAI_API_VERSION`       | `2024-02-01`   | Azure OpenAI API version           |
| `azure_openai_deployment_name`| `EMBEDDING__AZURE_OPENAI_DEPLOYMENT_NAME` | `None`         | Azure deployment name              |

Hugging Face:

| Field                        | Env var                                      | Default | Description                              |
| ---------------------------- | -------------------------------------------- | ------- | ---------------------------------------- |
| `huggingface_cache_dir`      | `EMBEDDING__HUGGINGFACE_CACHE_DIR`           | `None`  | HF cache directory                       |
| `huggingface_device`         | `EMBEDDING__HUGGINGFACE_DEVICE`              | `None`  | Device (`cpu`, `cuda`, etc.)            |
| `huggingface_trust_remote_code` | `EMBEDDING__HUGGINGFACE_TRUST_REMOTE_CODE`| `False` | Whether to trust remote code            |

Google USE:

| Field               | Env var                                  | Default | Description                    |
| ------------------- | ---------------------------------------- | ------- | ------------------------------ |
| `google_use_cache_dir` | `EMBEDDING__GOOGLE_USE_CACHE_DIR`     | `None`  | USE cache directory           |
| `google_use_version`   | `EMBEDDING__GOOGLE_USE_VERSION`       | `4`     | USE model version (`4` or `5`)|

Google Gemini:

| Field                    | Env var                                        | Default      | Description                           |
| ------------------------ | ---------------------------------------------- | ------------ | ------------------------------------- |
| `google_gemini_api_key`  | `EMBEDDING__GOOGLE_GEMINI_API_KEY`             | `None`       | Gemini / Vertex AI API key           |
| `google_gemini_base_url` | `EMBEDDING__GOOGLE_GEMINI_BASE_URL`            | `None`       | Gemini base URL                      |
| `google_gemini_project_id`| `EMBEDDING__GOOGLE_GEMINI_PROJECT_ID`         | `None`       | GCP project ID                       |
| `google_gemini_location` | `EMBEDDING__GOOGLE_GEMINI_LOCATION`           | `us-central1`| GCP region for Vertex AI             |

Batch/Resource tuning:

| Field                | Env var                                  | Default | Description                          |
| -------------------- | ---------------------------------------- | ------- | ------------------------------------ |
| `adaptive_batch_sizing` | `EMBEDDING__ADAPTIVE_BATCH_SIZING`    | `True`  | Enable adaptive batch sizing        |
| `memory_limit_mb`    | `EMBEDDING__MEMORY_LIMIT_MB`            | `None`  | Memory limit for batch processing   |
| `parallel_batches`   | `EMBEDDING__PARALLEL_BATCHES`           | `None`  | Number of parallel batches          |

### Provider-global environment variables

In addition to the nested settings, some providers are also picked up from
well-known global variables when `Config` is initialized:

- `OPENAI_API_KEY`
- `COHERE_API_KEY`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`

These are used when the corresponding nested embedding fields are not set.

## ChromaDB (`chroma_db` section)

| Field            | Env var                              | Default          | Description                            |
| ---------------- | ------------------------------------ | ---------------- | -------------------------------------- |
| `persist_directory` | `CHROMA_DB__PERSIST_DIRECTORY`    | `./chroma_db`    | Directory to persist ChromaDB data    |
| `collection_name`   | `CHROMA_DB__COLLECTION_NAME`      | `pdf_documents`  | Default collection name               |
| `distance_metric`   | `CHROMA_DB__DISTANCE_METRIC`      | `cosine`         | Distance metric (`cosine`, `l2`, `ip`)|
| `max_results`       | `CHROMA_DB__MAX_RESULTS`          | `100`            | Max number of search results          |

!!! info "Multi-backend configuration"
    For multi-backend vector databases (Pinecone, Qdrant, etc.), it is usually
    easier to configure `config.vector_db` from Python code. See
    [Vector Database API](vector-db.md) for details.

## Logging (`logging` section)

| Field       | Env var                   | Default                                   | Description                 |
| ----------- | ------------------------- | ----------------------------------------- | --------------------------- |
| `level`     | `LOGGING__LEVEL`          | `INFO`                                    | Log level                   |
| `format`    | `LOGGING__FORMAT`         | `{time} \| {level} \| {name} \| {message}` | Log format string           |
| `file_path` | `LOGGING__FILE_PATH`      | `None`                                    | Path to log file (if any)   |
| `rotation`  | `LOGGING__ROTATION`       | `10 MB`                                   | Log rotation policy         |
| `retention` | `LOGGING__RETENTION`      | `30 days`                                 | Log retention policy        |

## Usage example

Create a `.env` file in your project root:

```bash
# Embeddings
EMBEDDING__MODEL_TYPE=sentence-transformers
EMBEDDING__MODEL_NAME=all-MiniLM-L6-v2

# Text processing
TEXT_PROCESSING__CHUNK_SIZE=1200
TEXT_PROCESSING__CHUNK_OVERLAP=150

# ChromaDB
CHROMA_DB__PERSIST_DIRECTORY=./chroma_db
CHROMA_DB__COLLECTION_NAME=pdf_documents

# Logging
LOGGING__LEVEL=INFO

# Debug
DEBUG=false
```

Then simply create `Config()` in Python or let the CLI / GUI load it:

```python
from vectorflow import Config

config = Config()
```
