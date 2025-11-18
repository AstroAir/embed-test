# Cookbook

<!-- markdownlint-disable MD046 -->

This page collects common VectorFlow usage patterns as concrete, copy-pasteable recipes.

Each recipe focuses on a specific scenario and links back to the relevant API pages when you need more detail.

## Switching vector database backends

VectorFlow can work with different vector database backends through the `vector_db` configuration.

=== "ChromaDB (local default)"

    ```python
    from vectorflow import Config, PDFVectorPipeline

    # Default local ChromaDB setup
    config = Config()

    # Optional: customize ChromaDB location and collection name
    config.chroma_db.persist_directory = "./chroma_db"
    config.chroma_db.collection_name = "pdf_documents"

    pipeline = PDFVectorPipeline(config)
    result = pipeline.process_pdf("docs/example.pdf")
    print(result.success, result.chunks_processed)
    ```

=== "Pinecone (managed)"

    ```python
    from vectorflow import Config, PDFVectorPipeline
    from vectorflow.core.vector_db.config import PineconeConfig

    config = Config()

    # Configure Pinecone backend
    config.vector_db = PineconeConfig(
        api_key="your-pinecone-key",  # pragma: allowlist secret
        environment="your-pinecone-env",
        index_name="pdf-documents",
        # Make sure this matches your embedding model dimension
        dimension=1536,
        collection_name="pdf_documents",
    )

    pipeline = PDFVectorPipeline(config)
    result = pipeline.process_pdf("docs/example.pdf")
    print(result.success, result.chunks_processed)
    ```

!!! warning "Match embedding dimensions"
    When switching to remote backends such as Pinecone or Milvus, ensure the index/vector dimension matches the embedding model (for example `1536` for many OpenAI models). See the [Vector Database API](../api/vector-db.md) for details.

## Working with multiple collections

You can separate different document domains into different collections using the CLI.

```bash
# Index technical documents
vectorflow process docs/tech/*.pdf --collection tech_docs

# Index legal documents
vectorflow process docs/legal/*.pdf --collection legal_docs

# Search only in legal documents
vectorflow search "non-disclosure" --collection legal_docs --results 5
```

!!! tip "Inspect collections"
    Use `vectorflow collections`, `vectorflow stats` and `vectorflow backend` to inspect which collections exist and how many chunks they contain.

See also: [CLI Usage](../user-guide/cli.md) and [Vector Database API](../api/vector-db.md).

## Searching with document and page filters

At the Python level, `PDFVectorPipeline.search` accepts optional `document_id` and `page_number` filters:

```python
from vectorflow import Config, PDFVectorPipeline

config = Config()
pipeline = PDFVectorPipeline(config)

# Search only within a specific document
results = pipeline.search(
    "transformer architecture",
    n_results=5,
    document_id="paper_2024_01",
)

for r in results:
    print(r.score, r.document_id, r.page_number, r.content[:120])
```

With the CLI you can use metadata-based search helpers:

```bash
# Filter by document id and page number
vectorflow search "attention" --document paper_2024_01 --page 1 --results 5

# Search by metadata (for example page_number=1)
vectorflow search-meta page_number=1 --limit 20
```

See also: [Python API](api.md) and [CLI Usage](../user-guide/cli.md).

## Performance tuning recipes

A few configuration knobs can significantly affect performance:

- **Batching embeddings**

  ```python
  from vectorflow import Config

  config = Config()
  config.embedding.batch_size = 64
  config.embedding.adaptive_batch_sizing = True
  config.max_workers = 8
  ```

  - Larger `batch_size` generally improves throughput but uses more memory.
  - `adaptive_batch_sizing` lets VectorFlow reduce batch size automatically if memory is tight.
  - `max_workers` controls parallelism in `BatchEmbeddingProcessor`.

- **Chunking strategy**

  ```python
  from vectorflow import Config

  config = Config()
  config.text_processing.chunk_size = 1200
  config.text_processing.chunk_overlap = 150
  config.text_processing.min_chunk_size = 80
  ```

  - Larger `chunk_size` means fewer, longer chunks (better recall, more tokens per embedding).
  - Smaller `chunk_size` means more, shorter chunks (more granular results, more embeddings).

- **Logging and progress**

  ```python
  from vectorflow import Config, PDFVectorPipeline

  config = Config()
  config.logging.level = "INFO"  # or "DEBUG" for more verbose logs

  pipeline = PDFVectorPipeline(config)
  result = pipeline.process_pdf("docs/large.pdf")
  ```

  - Use `logging.level = "DEBUG"` while tuning.
  - Progress bars are controlled via `show_progress` in `process_pdf`.

For a full list of options, see [Configuration](../getting-started/configuration.md) and the [Configuration API](../api/config.md).
