# GUI Application

VectorFlow ships with a desktop GUI built on PySide6 and QFluentWidgets. It is useful when you want to process and search PDFs without writing code.

## Launching the GUI

After installing `vectorflow`, you can start the GUI with the console entry point:

```bash
vectorflow-gui
```

Alternatively, you can launch it from Python:

```python
from vectorflow.gui_main import main

if __name__ == "__main__":
    raise SystemExit(main())
```

!!! note "Shared configuration"
    The GUI uses the same `Config` model as the CLI and Python API, so your `.env` and environment variables are respected.

## Main features

- **PDF processing**: drag-and-drop one or more PDF files to process them into the configured vector database backend.
- **Progress monitoring**: see real-time status for extraction, chunking, embedding and storage.
- **Search interface**: run semantic search over processed documents with filters (document, page, etc.).
- **Document management**: inspect document statistics and manage collections.
- **Configuration view**: inspect and adjust key configuration options used by the pipeline.
- **Health and logs**: view basic health status and logs for troubleshooting.

## Typical workflow

1. Start the application with `vectorflow-gui`.
2. Review or adjust configuration (embedding model, vector DB backend, collection name) in the settings view if available.
3. Drag-and-drop one or more PDF files into the main window to start processing.
4. Monitor progress as pages are extracted, chunked, embedded and stored.
5. Use the search view to run semantic queries, filter by document or page, and inspect results.
6. Use document/collection views to inspect statistics or clean up data.

The GUI uses the same underlying `PDFVectorPipeline` as the CLI and Python API, so any changes to `Config` (via environment variables or `.env`) are shared across all entry points.

## End-to-end example: build and query a small knowledge base

This example walks through a complete session using the GUI only.

### 1. Prepare configuration

1. Make sure your `.env` and environment variables are set (embedding provider, API keys, vector DB backend, etc.).
2. Launch the GUI with `vectorflow-gui`.
3. Open the **Settings** tab and verify:
   - Embedding model and provider (for example Sentence Transformers or OpenAI).
   - Vector database backend (local ChromaDB or a remote backend such as Pinecone).
   - Collection name (for example `pdf_documents` or a domain-specific name).

!!! tip "Sharing configuration"
    The same `Config` is used by the CLI and Python API. If you already have a working CLI configuration, the GUI will usually work with it out of the box.

### 2. Index a batch of PDFs

1. Switch to the **Process** tab.
2. Drag a folder of PDFs (or select a few files) into the main area.
3. Watch the queue as files move through the stages:
   - Extraction
   - Chunking
   - Embedding
   - Storage
4. If any file fails, note the error status and later check the **Logs** tab for details.

When processing finishes, the **Documents** view will show the newly indexed documents.

### 3. Explore documents and run searches

1. Switch to the **Documents** tab and confirm that your PDFs appear with reasonable statistics (chunk counts, total characters, etc.).
2. Go to the **Search** tab.
3. Enter a query (for example, a concept or question that should appear in your PDFs).
4. Optionally set filters:
   - Document ID (to restrict to a specific document).
   - Page number, if you want to narrow results.
5. Run the search and scroll through the results list:
   - Expand items to view the full chunk content.
   - Check metadata such as `document_id` and page number.

If you see no results, try a broader query and double-check that the correct collection and backend are configured in **Settings**.

## Troubleshooting workflow in the GUI

When something goes wrong in the GUI, you can follow this sequence:

1. **Check Status tab**
   - Open **Status** and see if there are any warnings about the embedding service or vector database.
   - If available, trigger a health check from the GUI controller.
2. **Inspect Logs tab**
   - Switch to **Logs**.
   - Scroll to recent entries around the time of the failure (processing or search).
   - Look for tracebacks or connection errors.
3. **Verify configuration in Settings**
   - Ensure the embedding model, API keys and backend configuration match what you expect.
   - Confirm collection name and any backend-specific fields (for example Pinecone environment / index name).
4. **Cross-check with CLI**
   - Run `vectorflow health` in a terminal.
   - Optionally run a small `vectorflow process` and `vectorflow search` to see whether the issue is specific to the GUI or to the whole environment.

!!! warning "Do not ignore repeated failures"
    If multiple PDFs fail in the same way, or searches always return no results, it is usually a configuration or backend issue rather than a single broken file. Use the combination of **Status**, **Logs**, **Settings** and CLI commands to pinpoint the root cause.

## Interface overview

The main window uses a left-side navigation bar (QFluentWidgets `FluentWindow`):

- Top (primary work area):
  - **Process** (processing tab)
  - **Search** (search tab)
  - **Documents** (document management tab)
- Bottom (utilities and settings):
  - **Status** (system status)
  - **Settings** (configuration)
  - **Logs** (log viewer)

### Processing tab (Process)

The primary entry point for importing PDFs into the vector database.

- Drag and drop one or more PDF files into the main area to start processing.
- Uses the current `Config` (embedding settings and vector DB backend) to run the
  full pipeline.
- Typically displays the queue of files, current progress and processing
  statistics.
- When processing completes, it notifies the **Documents** view to refresh the
  document list.

**Typical usage:**

1. Open **Settings** and verify the embedding model and vector DB backend.
2. Switch to the **Process** tab and drag PDF files into the window.
3. Wait for the progress to complete and review success/failure messages.

### Search tab (Search)

Run semantic search via a graphical interface.

- Enter the query text.
- Optionally filter by document ID, page number, or other metadata (similar in
  spirit to the CLI `--document` and `--page` flags).
- Browse the results list, expanding items to see chunk content and metadata.

**Typical usage:**

1. Process one or more PDFs in the **Process** tab.
2. Switch to **Search**, type a query.
3. Adjust filters as needed, run the search and inspect the results.

### Documents tab (Documents)

Inspect and manage processed documents.

- Show a list of documents with basic statistics (number of chunks, characters,
  etc.).
- Provide a refresh action to pull the latest state after new processing runs.
- Depending on the implementation, may expose actions such as delete or other
  management tools.

**Typical usage:**

1. After processing a batch of PDFs, switch to **Documents**.
2. Check statistics per document to confirm successful ingestion.
3. If deeper cleanup is needed, combine with CLI commands such as
   `vectorflow delete`.

### Status tab (Status)

Show high-level system status and health.

- Receive status messages from the main controller.
- Useful to monitor background checks (e.g. health checks) or long-running
  tasks.

**Typical usage:**

1. Run `vectorflow health` in the CLI and observe the corresponding status in
   the GUI.
2. Keep an eye on whether the embedding service and vector database appear
   healthy (depending on the exact implementation).

### Settings tab (Settings)

Graphical view and editor for configuration.

- Display key `Config` fields: embedding configuration, vector DB configuration,
  text processing parameters, etc.
- Changes are propagated via a `config_changed` signal to other tabs so their
  controllers can update.

**Typical usage:**

1. Switch embedding model or vector DB backend (for example, from local Chroma
   to Pinecone).
2. Adjust `chunk_size`, `chunk_overlap`, and related parameters, then re-process
   PDFs to compare behaviour.

### Logs tab (Logs)

Log viewer for debugging and troubleshooting.

- Show log output corresponding to the configured `logging` settings.
- Pairs well with CLI `--debug` mode to quickly locate issues.

**Typical usage:**

1. When processing or search fails, switch to **Logs** to inspect stack traces
   or detailed error messages.
2. During development or performance testing, monitor performance-related
   logging.

## Keyboard shortcuts

The GUI registers a few useful keyboard shortcuts to speed up common actions:

- `Ctrl+Q`: quit the application.
- `Ctrl+O`: switch to the **Process** tab.
- `Ctrl+,`: switch to the **Settings** tab (on platforms that support this
  standard shortcut).

These shortcuts are registered in the main window's `_setup_shortcuts()` method
and wired to the corresponding tabs.

## Examples

For more advanced GUI usage patterns (custom integration, automation, plugins), see the repository examples:

- `examples/04_gui_applications/` – architecture, integration patterns and automation.
