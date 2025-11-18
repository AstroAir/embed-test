# Release Process

VectorFlow uses `hatchling` as its build backend (see `pyproject.toml`). This page is primarily intended for maintainers preparing a new release.

!!! warning "Releasing to PyPI"
    Make sure you are publishing from a clean, tagged commit and that you really intend to upload to the main PyPI index. For experiments, prefer TestPyPI.

A typical manual release workflow looks like:

1. Ensure tests and linters pass:

```bash
pip install "vectorflow[dev]"
pytest
ruff check .
black --check .
```

1. Build distributions (for example, using Hatch or the `build` package):

```bash
# using hatch (if installed)
hatch build

# or using the build package
python -m build
```

1. Upload artifacts to PyPI with your preferred tool (e.g. `twine`):

```bash
twine upload dist/*
```

Actual release automation may differ; see the CI workflows in `.github/workflows/` for more details.
