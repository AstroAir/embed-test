"""Tests for backward compatibility layer."""

import warnings
from unittest.mock import Mock, patch

import pytest

from pdf_vector_system.core.config.settings import ChromaDBConfig as OldChromaDBConfig
from pdf_vector_system.core.vector_db.compatibility import (
    ChromaClient,
    CompatibilityHelper,
    CompatibilityWarning,
    ConfigMigrator,
    LegacyChromaDBClient,
    convert_old_config_to_new,
    create_chroma_client,
    create_chroma_client_legacy,
    disable_compatibility_warnings,
    migrate_chroma_config,
    setup_compatibility_warnings,
)
from pdf_vector_system.core.vector_db.config import ChromaDBConfig as NewChromaDBConfig
from pdf_vector_system.core.vector_db.models import SearchQuery


class TestCompatibilityWarning:
    """Test CompatibilityWarning class."""

    def test_is_user_warning_subclass(self):
        """Test that CompatibilityWarning is a UserWarning subclass."""
        assert issubclass(CompatibilityWarning, UserWarning)

    def test_can_create_warning(self):
        """Test that CompatibilityWarning can be created."""
        warning = CompatibilityWarning("Test warning message")
        assert str(warning) == "Test warning message"


class TestCreateChromaClientLegacy:
    """Test create_chroma_client_legacy function."""

    @patch("pdf_vector_system.vector_db.compatibility.ChromaDBClient")
    def test_creates_client_with_defaults(self, mock_client_class):
        """Test creating client with default parameters."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = create_chroma_client_legacy()

            assert result == mock_client
            assert len(w) == 1
            assert issubclass(w[0].category, CompatibilityWarning)
            assert "deprecated" in str(w[0].message).lower()

    @patch("pdf_vector_system.vector_db.compatibility.ChromaDBClient")
    def test_creates_client_with_custom_params(
        self, mock_client_class, vector_db_temp_dir
    ):
        """Test creating client with custom parameters."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            result = create_chroma_client_legacy(
                persist_directory=vector_db_temp_dir / "custom",
                collection_name="custom_collection",
                distance_metric="l2",
                max_results=25,
            )

            assert result == mock_client

            # Check that ChromaDBClient was called with correct config
            mock_client_class.assert_called_once()
            config_arg = mock_client_class.call_args[0][0]
            assert isinstance(config_arg, NewChromaDBConfig)
            assert config_arg.collection_name == "custom_collection"
            assert config_arg.distance_metric == "l2"
            assert config_arg.max_results == 25

    def test_warning_stacklevel(self):
        """Test that warning has correct stack level."""
        with patch("pdf_vector_system.vector_db.compatibility.ChromaDBClient"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                create_chroma_client_legacy()

                assert len(w) == 1
                # Warning should point to the caller, not the function itself
                assert w[0].filename != __file__


class TestConvertOldConfigToNew:
    """Test convert_old_config_to_new function."""

    def test_converts_old_config(self, vector_db_temp_dir):
        """Test converting old config to new format."""
        old_config = OldChromaDBConfig(
            persist_directory=vector_db_temp_dir / "old_chroma",
            collection_name="old_collection",
            distance_metric="l2",
            max_results=15,
        )

        new_config = convert_old_config_to_new(old_config)

        assert isinstance(new_config, NewChromaDBConfig)
        assert new_config.persist_directory == old_config.persist_directory
        assert new_config.collection_name == old_config.collection_name
        assert new_config.distance_metric == old_config.distance_metric
        assert new_config.max_results == old_config.max_results

    def test_preserves_all_fields(self, vector_db_temp_dir):
        """Test that all fields are preserved during conversion."""
        old_config = OldChromaDBConfig(
            persist_directory=vector_db_temp_dir / "test",
            collection_name="test_collection",
            distance_metric="cosine",
            max_results=50,
        )

        new_config = convert_old_config_to_new(old_config)

        # Check all fields are preserved
        assert new_config.persist_directory == old_config.persist_directory
        assert new_config.collection_name == old_config.collection_name
        assert new_config.distance_metric == old_config.distance_metric
        assert new_config.max_results == old_config.max_results


class TestLegacyChromaDBClient:
    """Test LegacyChromaDBClient class."""

    @patch("pdf_vector_system.vector_db.compatibility.ChromaDBClient")
    def test_initialization_with_defaults(self, mock_client_class):
        """Test LegacyChromaDBClient initialization with defaults."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            legacy_client = LegacyChromaDBClient()

            assert legacy_client._client == mock_client
            assert len(w) == 1
            assert issubclass(w[0].category, CompatibilityWarning)

    @patch("pdf_vector_system.vector_db.compatibility.ChromaDBClient")
    def test_initialization_with_custom_params(
        self, mock_client_class, vector_db_temp_dir
    ):
        """Test LegacyChromaDBClient initialization with custom parameters."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            legacy_client = LegacyChromaDBClient(
                persist_directory=vector_db_temp_dir / "legacy",
                collection_name="legacy_collection",
                distance_metric="l2",
                max_results=30,
            )

            assert legacy_client._client == mock_client

    @patch("pdf_vector_system.vector_db.compatibility.ChromaDBClient")
    def test_attribute_delegation(self, mock_client_class):
        """Test that attributes are delegated to wrapped client."""
        mock_client = Mock()
        mock_client.some_method.return_value = "test_result"
        mock_client.some_property = "test_property"
        mock_client_class.return_value = mock_client

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")

            legacy_client = LegacyChromaDBClient()

            # Test method delegation
            result = legacy_client.some_method("arg1", "arg2")
            assert result == "test_result"
            mock_client.some_method.assert_called_once_with("arg1", "arg2")

            # Test property delegation
            assert legacy_client.some_property == "test_property"

    @patch("pdf_vector_system.vector_db.compatibility.ChromaDBClient")
    def test_add_documents_legacy_method(
        self, mock_client_class, sample_document_chunks
    ):
        """Test legacy add_documents method."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            legacy_client = LegacyChromaDBClient()
            legacy_client.add_documents(sample_document_chunks, "test_collection")

            # Should call add_chunks on wrapped client
            mock_client.add_chunks.assert_called_once_with(
                sample_document_chunks, "test_collection"
            )

            # Should issue deprecation warning
            assert any(
                issubclass(warning.category, CompatibilityWarning) for warning in w
            )
            assert any(
                "add_documents() is deprecated" in str(warning.message) for warning in w
            )

    @patch("pdf_vector_system.vector_db.compatibility.ChromaDBClient")
    def test_search_documents_legacy_method(
        self, mock_client_class, sample_search_results
    ):
        """Test legacy search_documents method."""
        mock_client = Mock()
        mock_client.search.return_value = sample_search_results
        mock_client_class.return_value = mock_client

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            legacy_client = LegacyChromaDBClient()
            result = legacy_client.search_documents(
                "test query", n_results=5, collection_name="test_collection"
            )

            assert result == sample_search_results

            # Should call search on wrapped client with SearchQuery
            mock_client.search.assert_called_once()
            call_args = mock_client.search.call_args
            query_arg = call_args[0][0]
            assert isinstance(query_arg, SearchQuery)
            assert query_arg.query_text == "test query"
            assert query_arg.n_results == 5

            # Should issue deprecation warning
            assert any(
                issubclass(warning.category, CompatibilityWarning) for warning in w
            )
            assert any(
                "search_documents() is deprecated" in str(warning.message)
                for warning in w
            )


class TestMigrateChromaConfig:
    """Test migrate_chroma_config function."""

    def test_migrates_config_dict(self):
        """Test migrating configuration dictionary."""
        old_config_dict = {
            "persist_directory": "./old_chroma",
            "collection_name": "old_collection",
            "distance_metric": "l2",
            "max_results": 20,
            "extra_field": "should_be_ignored",
        }

        new_config_dict = migrate_chroma_config(old_config_dict)

        assert new_config_dict["db_type"] == "chromadb"
        assert new_config_dict["persist_directory"] == "./old_chroma"
        assert new_config_dict["collection_name"] == "old_collection"
        assert new_config_dict["distance_metric"] == "l2"
        assert new_config_dict["max_results"] == 20
        assert "extra_field" not in new_config_dict

    def test_migrates_partial_config(self):
        """Test migrating partial configuration dictionary."""
        partial_config = {"collection_name": "partial_collection", "max_results": 15}

        new_config_dict = migrate_chroma_config(partial_config)

        assert new_config_dict["db_type"] == "chromadb"
        assert new_config_dict["collection_name"] == "partial_collection"
        assert new_config_dict["max_results"] == 15
        assert "persist_directory" not in new_config_dict
        assert "distance_metric" not in new_config_dict

    def test_migrates_empty_config(self):
        """Test migrating empty configuration dictionary."""
        empty_config = {}

        new_config_dict = migrate_chroma_config(empty_config)

        assert new_config_dict == {"db_type": "chromadb"}


class TestCompatibilityHelper:
    """Test CompatibilityHelper class."""

    def test_check_legacy_usage(self):
        """Test check_legacy_usage method."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            CompatibilityHelper.check_legacy_usage(
                "test_function", "old_pattern", "new_pattern"
            )

            assert len(w) == 1
            assert issubclass(w[0].category, CompatibilityWarning)
            assert "test_function" in str(w[0].message)
            assert "old_pattern" in str(w[0].message)
            assert "new_pattern" in str(w[0].message)

    def test_convert_search_params(self):
        """Test convert_search_params method."""
        query = CompatibilityHelper.convert_search_params(
            query_text="test query", n_results=15, where={"document_id": "doc1"}
        )

        assert isinstance(query, SearchQuery)
        assert query.query_text == "test query"
        assert query.n_results == 15
        assert query.where == {"document_id": "doc1"}

    def test_ensure_new_config_with_old(self, vector_db_temp_dir):
        """Test ensure_new_config with old config."""
        old_config = OldChromaDBConfig(
            persist_directory=vector_db_temp_dir / "test", collection_name="test"
        )

        new_config = CompatibilityHelper.ensure_new_config(old_config)

        assert isinstance(new_config, NewChromaDBConfig)
        assert new_config.persist_directory == old_config.persist_directory
        assert new_config.collection_name == old_config.collection_name

    def test_ensure_new_config_with_new(self, vector_db_temp_dir):
        """Test ensure_new_config with new config."""
        new_config = NewChromaDBConfig(
            persist_directory=vector_db_temp_dir / "test", collection_name="test"
        )

        result = CompatibilityHelper.ensure_new_config(new_config)

        assert result is new_config  # Should return same object


class TestCompatibilityAliases:
    """Test backward compatibility aliases."""

    def test_chroma_client_alias(self):
        """Test ChromaClient alias."""
        assert ChromaClient is LegacyChromaDBClient

    def test_create_chroma_client_alias(self):
        """Test create_chroma_client alias."""
        assert create_chroma_client is create_chroma_client_legacy


class TestWarningManagement:
    """Test warning management functions."""

    def test_setup_compatibility_warnings(self):
        """Test setup_compatibility_warnings function."""
        # Reset warnings filter
        warnings.resetwarnings()

        setup_compatibility_warnings()

        # Test that CompatibilityWarning is shown
        with warnings.catch_warnings(record=True) as w:
            warnings.warn("Test warning", CompatibilityWarning, stacklevel=2)
            assert len(w) == 1

    def test_disable_compatibility_warnings(self):
        """Test disable_compatibility_warnings function."""
        # Ensure warnings are enabled first
        setup_compatibility_warnings()

        disable_compatibility_warnings()

        # Test that CompatibilityWarning is ignored
        with warnings.catch_warnings(record=True) as w:
            warnings.warn("Test warning", CompatibilityWarning, stacklevel=2)
            assert len(w) == 0


class TestConfigMigrator:
    """Test ConfigMigrator class."""

    def test_generate_migration_guide(self):
        """Test generate_migration_guide method."""
        guide = ConfigMigrator.generate_migration_guide()

        assert isinstance(guide, str)
        assert "Migration Guide" in guide
        assert "Old" in guide
        assert "New" in guide
        assert "ChromaDB" in guide
        assert "VectorDBFactory" in guide

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.write_text")
    def test_migrate_env_file(
        self, mock_write, mock_read, mock_exists, vector_db_temp_dir
    ):
        """Test migrate_env_file method."""
        mock_exists.return_value = True
        mock_read.return_value = """
CHROMA_DB__PERSIST_DIRECTORY=./chroma_db
CHROMA_DB__COLLECTION_NAME=documents
OTHER_SETTING=value
"""

        env_file_path = str(vector_db_temp_dir / ".env")
        ConfigMigrator.migrate_env_file(env_file_path, backup=False)

        # Check that write_text was called
        mock_write.assert_called_once()
        written_content = mock_write.call_args[0][0]

        assert "VECTOR_DB__DB_TYPE=chromadb" in written_content
        assert "VECTOR_DB__PERSIST_DIRECTORY=./chroma_db" in written_content
        assert "VECTOR_DB__COLLECTION_NAME=documents" in written_content
        assert "OTHER_SETTING=value" in written_content

    @patch("pathlib.Path.exists")
    def test_migrate_env_file_not_found(self, mock_exists):
        """Test migrate_env_file with non-existent file."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError):
            ConfigMigrator.migrate_env_file("nonexistent.env")
