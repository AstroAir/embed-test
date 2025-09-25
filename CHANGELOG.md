# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete packaging and distribution workflow
- CI/CD pipeline with GitHub Actions
- Automated testing on multiple Python versions (3.9-3.12)
- Code quality checks with ruff, black, isort, mypy
- Security scanning with bandit and safety
- Test coverage reporting
- Automated PyPI publishing
- Pre-commit hooks for development
- Documentation building with MkDocs

### Changed
- Updated pyproject.toml with comprehensive packaging configuration
- Enhanced dependency management with version constraints
- Improved test configuration with pytest

### Fixed
- Package structure for proper distribution
- Version management with hatch-vcs

## [0.1.0] - 2024-01-XX

### Added
- Initial release of PDF Vector System
- PDF text extraction using PyMuPDF
- Text processing and chunking capabilities
- Embedding generation with sentence-transformers and OpenAI
- ChromaDB integration for vector storage
- Similarity search and retrieval functionality
- Command-line interface with Typer
- Configuration management with Pydantic
- Comprehensive logging with Loguru
- Rich console output and progress bars
- Example scripts and usage documentation
- Comprehensive test suite with pytest
- Type hints throughout the codebase

### Features
- Support for multiple embedding models (local and API-based)
- Configurable text chunking strategies
- Batch processing for efficient embedding generation
- Health checks for system components
- Document management (add, search, delete, list)
- Metadata support for enhanced search capabilities
- Error handling and recovery mechanisms
- Performance monitoring and timing

[Unreleased]: https://github.com/your-username/pdf-vector-system/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-username/pdf-vector-system/releases/tag/v0.1.0
