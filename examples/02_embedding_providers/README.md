# Embedding Provider Examples

This directory demonstrates all available embedding providers and their configuration options.

## Examples

### `sentence_transformers_examples.py`
Local embedding generation with various sentence-transformer models.

### `openai_embeddings.py`
OpenAI API-based embeddings with different models and configurations.

### `cohere_embeddings.py`
Cohere embedding service integration and usage patterns.

### `huggingface_embeddings.py`
Hugging Face transformers for custom embedding models.

### `google_use_embeddings.py`
Google Universal Sentence Encoder integration.

### `gemini_embeddings.py`
Google Gemini embedding service examples.

### `provider_comparison.py`
Side-by-side comparison of different embedding providers.

### `provider_health_monitoring.py`
Health checking and monitoring for embedding providers.

### `adaptive_batch_processing.py`
Advanced batch processing with adaptive sizing and performance optimization.

## Prerequisites

- Python 3.9+
- PDF Vector System installed
- Provider-specific dependencies (install as needed)
- API keys for cloud providers (set as environment variables)

## API Keys Required

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-key"

# For Cohere
export COHERE_API_KEY="your-cohere-key"

# For Google Gemini
export GOOGLE_GEMINI_API_KEY="your-gemini-key"

# For Vertex AI
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
```

## What You'll Learn

- How to configure different embedding providers
- Performance characteristics of each provider
- Cost considerations for API-based providers
- Local vs. cloud embedding trade-offs
- Provider health monitoring and failover strategies
