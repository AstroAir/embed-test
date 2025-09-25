"""Pytest configuration and fixtures for configuration tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import patch
import pytest

from pdf_vector_system.config.settings import (
    Config, PDFConfig, TextProcessingConfig, EmbeddingConfig,
    ChromaDBConfig, LoggingConfig, EmbeddingModelType, LogLevel
)


@pytest.fixture
def config_temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory specifically for config tests."""
    temp_path = Path(tempfile.mkdtemp(prefix="config_test_"))
    try:
        yield temp_path
    finally:
        import shutil
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def minimal_config() -> Config:
    """Create a minimal configuration for testing."""
    return Config(
        debug=False,
        max_workers=1
    )


@pytest.fixture
def test_env_vars() -> Generator[Dict[str, str], None, None]:
    """Provide test environment variables."""
    test_env = {
        "OPENAI_API_KEY": "test-api-key",
        "PDF__MAX_FILE_SIZE_MB": "50",
        "TEXT_PROCESSING__CHUNK_SIZE": "500",
        "EMBEDDING__BATCH_SIZE": "16",
        "CHROMA_DB__MAX_RESULTS": "50",
        "DEBUG": "true",
        "MAX_WORKERS": "2"
    }
    
    # Store original values
    original_values = {}
    for key, value in test_env.items():
        original_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        yield test_env
    finally:
        # Restore original values
        for key, original_value in original_values.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


@pytest.fixture
def config_with_all_sections(config_temp_dir: Path) -> Config:
    """Create a configuration with all sections configured."""
    return Config(
        pdf=PDFConfig(
            max_file_size_mb=25,
            timeout_seconds=120,
            extract_images=False
        ),
        text_processing=TextProcessingConfig(
            chunk_size=800,
            chunk_overlap=80,
            min_chunk_size=20,
            separators=["\n\n", "\n", " "]
        ),
        embedding=EmbeddingConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            batch_size=24
        ),
        chroma_db=ChromaDBConfig(
            persist_directory=config_temp_dir / "chroma_test",
            collection_name="test_collection",
            max_results=25
        ),
        logging=LoggingConfig(
            level=LogLevel.DEBUG,
            file_path=config_temp_dir / "test.log"
        ),
        debug=True,
        max_workers=3
    )
