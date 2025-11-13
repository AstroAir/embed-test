"""Backward compatibility layer for existing ChromaDB usage patterns."""

import warnings
from pathlib import Path
from typing import Any, Optional, Union

from vectorflow.core.config.settings import ChromaDBConfig as OldChromaDBConfig
from vectorflow.core.vector_db.chroma_client import ChromaDBClient
from vectorflow.core.vector_db.config import ChromaDBConfig as NewChromaDBConfig
from vectorflow.core.vector_db.models import DocumentChunk, SearchQuery, SearchResult


class CompatibilityWarning(UserWarning):
    """Warning for deprecated usage patterns."""


def create_chroma_client_legacy(
    persist_directory: Union[str, Path] = "./chroma_db",
    collection_name: str = "pdf_documents",
    distance_metric: str = "cosine",
    max_results: int = 10,
) -> ChromaDBClient:
    """
    Create ChromaDB client using legacy parameter style.

    This function provides backward compatibility for code that creates
    ChromaDB clients directly with individual parameters.

    Args:
        persist_directory: Directory to persist ChromaDB data
        collection_name: Default collection name
        distance_metric: Distance metric for similarity search
        max_results: Maximum number of search results

    Returns:
        ChromaDBClient instance

    Deprecated:
        Use VectorDBFactory.create_client() with ChromaDBConfig instead.
    """
    warnings.warn(
        "Direct ChromaDBClient creation is deprecated. "
        "Use VectorDBFactory.create_client() with ChromaDBConfig instead.",
        CompatibilityWarning,
        stacklevel=2,
    )

    config = NewChromaDBConfig(
        persist_directory=Path(persist_directory),
        collection_name=collection_name,
        distance_metric=distance_metric,
        max_results=max_results,
    )

    return ChromaDBClient(config)


def convert_old_config_to_new(old_config: OldChromaDBConfig) -> NewChromaDBConfig:
    """
    Convert old ChromaDBConfig to new format.

    Args:
        old_config: Old ChromaDBConfig from settings.py

    Returns:
        New ChromaDBConfig from vector_db.config
    """
    return NewChromaDBConfig(
        persist_directory=old_config.persist_directory,
        collection_name=old_config.collection_name,
        distance_metric=old_config.distance_metric,
        max_results=old_config.max_results,
    )


class LegacyChromaDBClient:
    """
    Legacy wrapper for ChromaDBClient that maintains old API patterns.

    This class provides backward compatibility for existing code that uses
    the old ChromaDBClient API patterns.

    Deprecated:
        Use ChromaDBClient directly or VectorDBFactory.create_client().
    """

    def __init__(
        self,
        persist_directory: Union[str, Path] = "./chroma_db",
        collection_name: str = "pdf_documents",
        distance_metric: str = "cosine",
        max_results: int = 10,
    ):
        """Initialize legacy ChromaDB client."""
        warnings.warn(
            "LegacyChromaDBClient is deprecated. "
            "Use ChromaDBClient with proper configuration instead.",
            CompatibilityWarning,
            stacklevel=2,
        )

        config = NewChromaDBConfig(
            persist_directory=Path(persist_directory),
            collection_name=collection_name,
            distance_metric=distance_metric,
            max_results=max_results,
        )

        self._client = ChromaDBClient(config)

    def __getattr__(self, name: str) -> Any:
        """Delegate all attribute access to the wrapped client."""
        return getattr(self._client, name)

    # Legacy method aliases for backward compatibility
    def add_documents(
        self, chunks: list[DocumentChunk], collection_name: Optional[str] = None
    ) -> None:
        """Legacy alias for add_chunks."""
        warnings.warn(
            "add_documents() is deprecated. Use add_chunks() instead.",
            CompatibilityWarning,
            stacklevel=2,
        )
        return self._client.add_chunks(chunks, collection_name)

    def search_documents(
        self,
        query_text: str,
        n_results: int = 10,
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Legacy alias for search with simplified parameters."""
        warnings.warn(
            "search_documents() is deprecated. Use search() with SearchQuery instead.",
            CompatibilityWarning,
            stacklevel=2,
        )

        query = SearchQuery(query_text=query_text, n_results=n_results, **kwargs)

        return self._client.search(query, collection_name=collection_name)


def migrate_chroma_config(config_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Migrate old ChromaDB configuration dictionary to new format.

    Args:
        config_dict: Old configuration dictionary

    Returns:
        New configuration dictionary compatible with VectorDBFactory
    """
    # Map old keys to new keys
    key_mapping = {
        "persist_directory": "persist_directory",
        "collection_name": "collection_name",
        "distance_metric": "distance_metric",
        "max_results": "max_results",
    }

    new_config = {"db_type": "chromadb"}

    for old_key, new_key in key_mapping.items():
        if old_key in config_dict:
            new_config[new_key] = config_dict[old_key]

    return new_config


class CompatibilityHelper:
    """Helper class for managing backward compatibility."""

    @staticmethod
    def check_legacy_usage(func_name: str, old_pattern: str, new_pattern: str) -> None:
        """Issue deprecation warning for legacy usage patterns."""
        warnings.warn(
            f"{func_name}: {old_pattern} is deprecated. Use {new_pattern} instead.",
            CompatibilityWarning,
            stacklevel=3,
        )

    @staticmethod
    def convert_search_params(
        query_text: str,
        n_results: int = 10,
        where: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> SearchQuery:
        """Convert legacy search parameters to SearchQuery object."""
        return SearchQuery(
            query_text=query_text, n_results=n_results, where=where, **kwargs
        )

    @staticmethod
    def ensure_new_config(
        config: Union[OldChromaDBConfig, NewChromaDBConfig],
    ) -> NewChromaDBConfig:
        """Ensure configuration is in new format."""
        if isinstance(config, OldChromaDBConfig):
            return convert_old_config_to_new(config)
        return config


# Backward compatibility aliases
ChromaClient = LegacyChromaDBClient  # Common alias used in old code
create_chroma_client = create_chroma_client_legacy  # Function alias


def setup_compatibility_warnings() -> None:
    """Setup compatibility warnings to be shown by default."""
    warnings.filterwarnings("default", category=CompatibilityWarning)


def disable_compatibility_warnings() -> None:
    """Disable compatibility warnings."""
    warnings.filterwarnings("ignore", category=CompatibilityWarning)


# Migration utilities
class ConfigMigrator:
    """Utility class for migrating configurations."""

    @staticmethod
    def migrate_env_file(env_file_path: str, backup: bool = True) -> None:
        """
        Migrate .env file from old to new configuration format.

        Args:
            env_file_path: Path to .env file
            backup: Whether to create backup before migration
        """
        from pathlib import Path

        env_path = Path(env_file_path)
        if not env_path.exists():
            raise FileNotFoundError(f"Environment file not found: {env_file_path}")

        if backup:
            backup_path = env_path.with_suffix(env_path.suffix + ".backup")
            backup_path.write_text(env_path.read_text())

        # Read current content
        content = env_path.read_text()
        lines = content.split("\n")

        # Migrate ChromaDB settings to new vector database format
        new_lines = []
        for line in lines:
            if line.startswith("CHROMA_DB__"):
                # Convert CHROMA_DB__ prefix to VECTOR_DB__
                new_line = line.replace("CHROMA_DB__", "VECTOR_DB__")
                new_lines.append(new_line)
                new_lines.append(f"# Migrated from: {line}")
            else:
                new_lines.append(line)

        # Add vector database type if not present
        if not any(line.startswith("VECTOR_DB__DB_TYPE") for line in new_lines):
            new_lines.insert(0, "VECTOR_DB__DB_TYPE=chromadb")

        # Write updated content
        env_path.write_text("\n".join(new_lines))

    @staticmethod
    def generate_migration_guide() -> str:
        """Generate a migration guide for updating code."""
        return """
# Migration Guide: ChromaDB to Multi-Backend Vector Database

## Configuration Changes

### Old (settings.py):
```python
from vectorflow.core.config.settings import Config
config = Config()
client = ChromaDBClient(config.chroma_db)
```

### New (recommended):
```python
from vectorflow.core.vector_db import VectorDBFactory, ChromaDBConfig
config = ChromaDBConfig(persist_directory="./chroma_db")
client = VectorDBFactory.create_client(config)
```

## API Changes

### Search Operations:
```python
# Old
results = client.search_documents("query", n_results=10)

# New
from vectorflow.core.vector_db import SearchQuery
query = SearchQuery(query_text="query", n_results=10)
results = client.search(query)
```

### Document Operations:
```python
# Old
client.add_documents(chunks)

# New
client.add_chunks(chunks)
```

## Environment Variables

### Old:
```
CHROMA_DB__PERSIST_DIRECTORY=./chroma_db
CHROMA_DB__COLLECTION_NAME=pdf_documents
```

### New:
```
VECTOR_DB__DB_TYPE=chromadb
VECTOR_DB__PERSIST_DIRECTORY=./chroma_db
VECTOR_DB__COLLECTION_NAME=pdf_documents
```
"""


# Initialize compatibility warnings by default
setup_compatibility_warnings()
