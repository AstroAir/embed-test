# Quick Start

<!-- markdownlint-disable MD046 -->

This page shows the minimum steps to process a PDF and run a semantic search.

## Quick start (Python & CLI)

=== "Python"

    ```python
    from vectorflow import Config, PDFVectorPipeline
    from vectorflow.core.config.settings import EmbeddingModelType

    config = Config()
    config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
    config.embedding.model_name = "all-MiniLM-L6-v2"

    pipeline = PDFVectorPipeline(config)

    # Process a PDF
    result = pipeline.process_pdf("document.pdf")

    # Run a search
    results = pipeline.search("machine learning", n_results=5)
    for r in results:
        print(r.score, r.content[:200])
    ```

=== "CLI"

    Process one or more PDFs and store them in the vector database:

    ```bash
    vectorflow process docs/*.pdf --model all-MiniLM-L6-v2
    ```

    Search in the processed collection:

    ```bash
    vectorflow search "machine learning" --results 5
    ```

    List processed documents:

    ```bash
    vectorflow list-docs
    ```

## Next steps

- Configure environment variables and `.env` files (see [Configuration](configuration.md)).
- Explore CLI workflows (see [CLI Usage](../user-guide/cli.md)).
- Try the GUI application (see [GUI Application](../user-guide/gui.md)).
