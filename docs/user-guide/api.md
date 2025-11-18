# Python API

The high-level Python API lives in the `vectorflow` package and is designed to make it easy to go from PDF files to a searchable vector database with a few lines of code.

## Core classes

- `vectorflow.Config` – Pydantic settings model that controls the pipeline.
- `vectorflow.PDFVectorPipeline` – orchestrates PDF extraction, chunking, embedding and vector DB storage.

!!! note "Configuration via environment variables"
    `Config` reads from environment variables and `.env` files. This means CLI, GUI and Python API can all share the same configuration without duplicating settings.

## Quick start

```python
from vectorflow import Config, PDFVectorPipeline

config = Config()
pipeline = PDFVectorPipeline(config)

result = pipeline.process_pdf("document.pdf")
print(result.success, result.chunks_processed, result.embeddings_generated)

results = pipeline.search("machine learning", n_results=10)
for r in results:
    print(r.score, r.content[:200], r.metadata.get("document_id"))
```

## Processing results

`result` is a `ProcessingResult` dataclass with fields such as:

- `document_id`
- `file_path`
- `success`
- `chunks_processed`
- `embeddings_generated`
- `chunks_stored`
- `processing_time`
- `metadata` (includes PDF metadata and embedding info)
- `error_message` (optional error details when `success` is `False`)

In addition, `ProcessingResult` exposes a computed property `chunks_per_second` and a `to_dict()` helper for convenient logging or serialization.

## Search results

`pipeline.search(...)` returns a list of `SearchResult` objects from `vectorflow.core.vector_db.models`.

Each `SearchResult` has fields such as:

- `id` – chunk identifier in the vector store.
- `content` – text content of the matched chunk.
- `score` – similarity score (higher is more similar).
- `metadata` – metadata dictionary (includes `document_id`, `page_number`, etc.).

Convenience properties:

- `document_id` – shorthand for `metadata.get("document_id")`.
- `page_number` – shorthand for `metadata.get("page_number")`.

## Collection and document utilities

- `pipeline.get_documents()` – list summary information about all documents.
- `pipeline.get_document_info(document_id)` – detailed stats for a single document.
- `pipeline.delete_document(document_id)` – delete a document from the vector DB.
- `pipeline.get_collection_stats()` – get collection statistics.
- `pipeline.get_collection_info()` – get current collection information.
- `pipeline.get_vector_db_info()` – get backend information.
- `pipeline.health_check()` – run a health check on embedding service and vector DB.

These methods are implemented in `vectorflow.core.pipeline`.
