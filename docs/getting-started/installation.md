# Installation

VectorFlow is published on PyPI and can be installed with `pip`. This page shows the recommended setup for most users.

## Requirements

- Python 3.9–3.12

!!! note "Use a virtual environment"
    Using `venv`, `virtualenv`, or `conda` is recommended to isolate VectorFlow's dependencies from your global Python installation.

## Install core package

```bash
pip install vectorflow
```

To upgrade an existing installation:

```bash
pip install --upgrade vectorflow
```

This installs:

- Python APIs (`Config`, `PDFVectorPipeline`)
- CLI entry points: `vectorflow`, `vectorflow-gui`

## Install docs extras (optional)

If you want to build the documentation locally:

```bash
pip install "vectorflow[docs]"
```

## Verify installation

Run a few quick commands to confirm everything is installed correctly:

```bash
python -m pip show vectorflow
vectorflow --help
vectorflow-gui --help
```

If any of these commands fail, double-check that your virtual environment is activated and that `vectorflow` is installed in the expected Python environment.
