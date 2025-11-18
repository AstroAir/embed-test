# Pipeline API

This page gives an overview of the `PDFVectorPipeline`.

```python
from vectorflow import Config, PDFVectorPipeline
```

The main entry point is `PDFVectorPipeline`, defined in `vectorflow.core.pipeline`.

!!! note "High-level Python examples"
    For end-to-end examples that combine configuration, processing and search, see the [Python API](../user-guide/api.md) page in the User Guide.

## Key methods

- `process_pdf(pdf_path, document_id=None, clean_text=True, show_progress=True)`
- `search(query_text, n_results=10, document_id=None, page_number=None)`
- `get_documents()`
- `get_document_info(document_id)`
- `delete_document(document_id)`
- `get_collection_stats()`
- `get_collection_info()`
- `get_vector_db_info()`
- `health_check()`

See the "Python API" section in the User Guide for example code snippets.

## API reference

::: vectorflow.core.pipeline.PDFVectorPipeline
