# Examples

The `examples/` directory in the repository contains runnable scripts that demonstrate common usage patterns.

## Directory overview

- `01_basic_usage/` – basic pipeline and CLI usage.
- `02_embedding_providers/` – configuring different embedding providers.
- `03_cli_usage/` – more detailed CLI workflows.
- `04_gui_applications/` – using the desktop GUI.
- `05_vector_database/` – working with different vector DB backends.
- `06_text_processing/` – text cleaning and chunking.
- `07_configuration/` – advanced configuration and environments.
- `08_integration/` – integrating VectorFlow into other systems.
- `09_performance/` – performance and batching tips.
- `10_production/` – patterns for production deployments.

## Running example tests

You can run the tests that exercise these examples:

```bash
pytest examples/test_all_examples.py
```

## Running examples manually

Run individual scripts with your own API keys and configuration:

```bash
python examples/01_basic_usage/first_search.py
```

!!! note "Environment and API keys"
    Many examples interact with external embedding providers or vector databases. Make sure your `.env` and environment variables are configured correctly before running them.
