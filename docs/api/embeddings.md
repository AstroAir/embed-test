# Embeddings API

<!-- markdownlint-disable MD046 -->

Embedding-related utilities live in `vectorflow.core.embeddings`.

Important components:

- `EmbeddingServiceFactory` – creates an embedding service instance from an `EmbeddingConfig`.
- `BatchEmbeddingProcessor` – batches text and calls the underlying embedding service efficiently.
- `EmbeddingModelType` – enum describing supported providers (Sentence Transformers, OpenAI, Cohere, HuggingFace, Google USE, Google Gemini, Azure OpenAI, etc.).

The pipeline automatically uses these via `PDFVectorPipeline(config)`. For advanced use cases you can instantiate them directly from `vectorflow.core.embeddings.factory`.

!!! warning "Keep API keys secret"
    All API keys and credentials (OpenAI, Cohere, Azure, Google, etc.) should be provided via environment variables or secret management solutions, not hard-coded in source control.

## Supported providers

The `EmbeddingModelType` enum currently includes:

- `SENTENCE_TRANSFORMERS` – local Sentence Transformers models.
- `OPENAI` – OpenAI / Azure-compatible text embedding models.
- `COHERE` – Cohere embedding models.
- `HUGGINGFACE` – models loaded via Hugging Face `transformers`.
- `GOOGLE_USE` – Universal Sentence Encoder models from TensorFlow Hub.
- `GOOGLE_GEMINI` – Gemini / Vertex AI text embedding models.
- `AZURE_OPENAI` – Azure-hosted OpenAI deployments.

Provider-specific settings are stored on `EmbeddingConfig`, for example:

- `openai_api_key`, `openai_base_url`
- `cohere_api_key`, `cohere_base_url`
- `azure_openai_api_key`, `azure_openai_endpoint`,
  `azure_openai_api_version`, `azure_openai_deployment_name`
- `huggingface_cache_dir`, `huggingface_device`,
  `huggingface_trust_remote_code`
- `google_use_cache_dir`, `google_use_version`
- `google_gemini_api_key`, `google_gemini_base_url`,
  `google_gemini_project_id`, `google_gemini_location`

## Model discovery helpers

The `EmbeddingServiceFactory` also provides helper methods for discovering supported models:

- `get_supported_models()` – returns a mapping from `EmbeddingModelType` to a list of recommended model names.
- `get_available_models()` – returns a mapping from provider string keys (for example `"openai"`, `"cohere"`) to available model identifiers.

These are useful when building CLIs or GUIs that allow users to choose from a curated list of embedding models.

## Example configurations

=== "Sentence Transformers"

    ```python
    from vectorflow import Config
    from vectorflow.core.config.settings import EmbeddingModelType

    config = Config()
    config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
    config.embedding.model_name = "all-MiniLM-L6-v2"
    ```

=== "OpenAI"

    ```python
    from vectorflow import Config
    from vectorflow.core.config.settings import EmbeddingModelType

    config = Config()
    config.embedding.model_type = EmbeddingModelType.OPENAI
    config.embedding.model_name = "text-embedding-3-small"

    # API key can be set via environment variable OPENAI_API_KEY,
    # or directly on the config:
    config.embedding.openai_api_key = "sk-..."  # pragma: allowlist secret
    ```

=== "Google Gemini"

    ```python
    from vectorflow import Config
    from vectorflow.core.config.settings import EmbeddingModelType

    config = Config()
    config.embedding.model_type = EmbeddingModelType.GOOGLE_GEMINI
    config.embedding.model_name = "gemini-embedding-001"

    # The following values can be loaded from env or set explicitly:
    config.embedding.google_gemini_api_key = "..."
    config.embedding.google_gemini_project_id = "your-gcp-project"
    config.embedding.google_gemini_location = "us-central1"
    ```

## API reference

::: vectorflow.core.embeddings.factory.EmbeddingServiceFactory

::: vectorflow.core.embeddings.factory.BatchEmbeddingProcessor
