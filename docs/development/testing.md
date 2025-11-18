# Testing

<!-- markdownlint-disable MD046 -->

This project uses `pytest` for testing and `coverage.py` for coverage reporting. The test suite is configured in `pyproject.toml`.

## Quick start

Install dev dependencies:

```bash
pip install "vectorflow[dev]"
```

Then run all tests:

```bash
pytest
```

!!! tip "Run tests faster with xdist"
    The `dev` extra includes `pytest-xdist`, so you can run tests in parallel on multiple cores:

    ```bash
    pytest -n auto
    ```

## Selecting tests with markers

Useful markers (see `pyproject.toml`):

- `pytest -m "unit"` – run only unit tests.
- `pytest -m "integration"` – run integration tests.
- `pytest -m "gui"` – run GUI-related tests.

You can also combine markers, for example:

```bash
pytest -m "unit and not slow"
```

## Coverage reports

Coverage reports are written to the `htmlcov/` directory and `coverage.xml` (see the coverage configuration in `pyproject.toml`).
