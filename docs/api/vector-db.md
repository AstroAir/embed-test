# Vector Database API

Vector database integration lives in `vectorflow.core.vector_db`.

Key building blocks:

- `VectorDBFactory` – creates a backend client from a configuration object.
- Backend-specific config models in `vectorflow.core.vector_db.config` (one per backend such as ChromaDB, Pinecone, Weaviate, Qdrant and Milvus).
- Core models in `vectorflow.core.vector_db.models`:
  - `DocumentChunk`
  - `SearchQuery`
  - `SearchResult`

The factory and config models provide a unified interface over multiple vector databases. `PDFVectorPipeline` uses these internally to store and query embeddings.

!!! tip "Choosing a backend"
    For local experimentation, `ChromaDBConfig` is a good default. For production workloads, consider a managed service such as Pinecone, Weaviate, Qdrant Cloud or Milvus depending on your requirements.

## Backend configuration classes

- `ChromaDBConfig`
  - Fields: `persist_directory`, `collection_name`, `distance_metric`, `max_results`.
  - Suitable for local development and small deployments.
- `PineconeConfig`
  - Fields: `api_key`, `environment`, `index_name`, `dimension`, `metric`,
    `collection_name`, `max_results`.
  - Wraps a Pinecone index as the backing vector store.
- `WeaviateConfig`
  - Fields: `url`, `api_key`, `class_name`, `collection_name`, `max_results`.
- `QdrantConfig`
  - Fields: `url`, `host`, `port`, `api_key`, `distance_metric`,
    `collection_name`, `max_results`.
- `MilvusConfig`
  - Fields: `host`, `port`, `user`, `password`, `dimension`, `metric_type`,
    `collection_name`, `max_results`.

All backend configs inherit from `VectorDBConfig` and implement `get_client_kwargs()`,
which is used by `VectorDBFactory` to initialize the correct client.

## VectorDBType enum

The `VectorDBType` enum in `vectorflow.core.vector_db.config` describes the supported backends:

- `CHROMADB` – `"chromadb"`
- `PINECONE` – `"pinecone"`
- `WEAVIATE` – `"weaviate"`
- `QDRANT` – `"qdrant"`
- `MILVUS` – `"milvus"`

## Convenience helpers

The `vectorflow.core.vector_db.factory` module also exposes helper functions:

- `create_vector_db(config)` – thin wrapper around `VectorDBFactory.create_client`.
- `get_available_backends()` – returns a mapping of available backends and descriptions.
- `create_chromadb_config(**kwargs)` – construct a `ChromaDBConfig` with sensible defaults.

## Using vector_db with Config

To enable the multi-backend system, assign a backend config instance to
`config.vector_db` before creating the pipeline:

```python
from vectorflow import Config, PDFVectorPipeline
from vectorflow.core.vector_db.config import PineconeConfig

config = Config()
config.vector_db = PineconeConfig(
    api_key="your-pinecone-key",  # pragma: allowlist secret
    environment="your-pinecone-env",
    index_name="pdf-documents",
    dimension=1536,  # must match the embedding dimension
    collection_name="pdf_documents",
)

pipeline = PDFVectorPipeline(config)
```

If `vector_db` is left as `None`, the pipeline falls back to the legacy `chroma_db`
section on `Config`, which is automatically converted into a `ChromaDBConfig`.

!!! warning "Match embedding dimensions"
    When configuring remote backends such as Pinecone or Milvus, make sure the index/vector dimension matches the embedding model dimension (for example `1536` for many OpenAI models). Mismatches will cause runtime errors.

## API reference

::: vectorflow.core.vector_db.config.VectorDBConfig

::: vectorflow.core.vector_db.config.ChromaDBConfig

::: vectorflow.core.vector_db.config.PineconeConfig

::: vectorflow.core.vector_db.config.WeaviateConfig

::: vectorflow.core.vector_db.config.QdrantConfig

::: vectorflow.core.vector_db.config.MilvusConfig

::: vectorflow.core.vector_db.factory.VectorDBFactory

::: vectorflow.core.vector_db.models.DocumentChunk

::: vectorflow.core.vector_db.models.SearchQuery

::: vectorflow.core.vector_db.models.SearchResult

::: vectorflow.core.vector_db.models.CollectionInfo

::: vectorflow.core.vector_db.models.DocumentInfo
