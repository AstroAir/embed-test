# PDF Processing API

PDF-related utilities live in `vectorflow.core.pdf`.

Key classes:

- `vectorflow.core.pdf.processor.PDFProcessor` – loads PDF files and extracts text, metadata and page-level information.
- `vectorflow.core.pdf.text_processor.TextProcessor` – cleans text and splits it into chunks (`TextChunk` instances) with rich metadata.

!!! info "Used indirectly by the pipeline"
    These classes are composed by `PDFVectorPipeline` and are rarely used directly. Import them only if you are building custom pipelines or advanced integrations.

## API reference

::: vectorflow.core.pdf.processor.PDFProcessor

::: vectorflow.core.pdf.text_processor.TextProcessor
