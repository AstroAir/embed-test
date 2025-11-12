"""Integration tests between vector database components."""

from unittest.mock import Mock, patch

import pytest

from pdf_vector_system.core.vector_db.config import (
    ChromaDBConfig,
    MilvusConfig,
    PineconeConfig,
    QdrantConfig,
    WeaviateConfig,
)
from pdf_vector_system.core.vector_db.converters import VectorDBConverter
from pdf_vector_system.core.vector_db.error_handler import VectorDBErrorHandler
from pdf_vector_system.core.vector_db.factory import VectorDBFactory
from pdf_vector_system.core.vector_db.health_check import VectorDBHealthManager
from pdf_vector_system.core.vector_db.models import (
    CollectionNotFoundError,
    SearchResult,
    VectorDBError,
)


class TestFactoryIntegration:
    """Test integration between factory and various components."""

    @pytest.mark.parametrize(
        ("backend_type", "config_class"),
        [
            ("chromadb", ChromaDBConfig),
            ("pinecone", PineconeConfig),
            ("weaviate", WeaviateConfig),
            ("qdrant", QdrantConfig),
            ("milvus", MilvusConfig),
        ],
    )
    def test_factory_creates_correct_client_type(
        self, backend_type, config_class, vector_db_temp_dir
    ):
        """Test that factory creates correct client type for each backend."""
        # Create appropriate config for each backend
        if backend_type == "chromadb":
            config = config_class(persist_directory=vector_db_temp_dir / "chroma")
        elif backend_type == "pinecone":
            config = config_class(api_key="test_key", index_name="test_index")
        elif backend_type == "weaviate":
            config = config_class(url="http://localhost:8080", class_name="TestClass")
        elif backend_type == "qdrant":
            config = config_class(
                url="http://localhost:6333", collection_name="test_collection"
            )
        elif backend_type == "milvus":
            config = config_class(
                host="localhost", port=19530, collection_name="test_collection"
            )

        # Mock the specific client class for each backend
        client_module_map = {
            "chromadb": "pdf_vector_system.vector_db.chroma_client.ChromaDBClient",
            "pinecone": "pdf_vector_system.vector_db.pinecone_client.PineconeClient",
            "weaviate": "pdf_vector_system.vector_db.weaviate_client.WeaviateClient",
            "qdrant": "pdf_vector_system.vector_db.qdrant_client.QdrantClient",
            "milvus": "pdf_vector_system.vector_db.milvus_client.MilvusClient",
        }

        with patch(client_module_map[backend_type]) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            client = VectorDBFactory.create_client(config)

            assert client == mock_client
            mock_client_class.assert_called_once_with(config)

    def test_factory_with_converter_integration(
        self, sample_document_chunks, vector_db_temp_dir
    ):
        """Test factory integration with converter for data format handling."""
        config = ChromaDBConfig(persist_directory=vector_db_temp_dir / "chroma")

        with patch(
            "pdf_vector_system.vector_db.chroma_client.ChromaDBClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.add_chunks.return_value = None
            mock_client_class.return_value = mock_client

            client = VectorDBFactory.create_client(config)
            converter = VectorDBConverter()

            # Convert chunks to ChromaDB format
            converter.to_chromadb_format(sample_document_chunks)

            # Add chunks through client
            client.add_chunks(sample_document_chunks)

            # Verify client was called
            mock_client.add_chunks.assert_called_once_with(sample_document_chunks)

    def test_factory_with_error_handler_integration(self, vector_db_temp_dir):
        """Test factory integration with error handler."""
        config = ChromaDBConfig(persist_directory=vector_db_temp_dir / "chroma")

        with patch(
            "pdf_vector_system.vector_db.chroma_client.ChromaDBClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.list_collections.side_effect = Exception("Connection failed")
            mock_client_class.return_value = mock_client

            client = VectorDBFactory.create_client(config)
            error_handler = VectorDBErrorHandler()

            # Test error handling integration
            try:
                client.list_collections()
            except Exception as e:
                wrapped_error = error_handler.handle_error(
                    e, "list_collections", {"config": config}
                )
                assert isinstance(wrapped_error, VectorDBError)


class TestConverterIntegration:
    """Test integration between converter and different backends."""

    def test_converter_with_all_backends(self, sample_document_chunks):
        """Test converter works with all backend formats."""
        converter = VectorDBConverter()

        # Test conversion to all backend formats
        chromadb_format = converter.to_chromadb_format(sample_document_chunks)
        pinecone_format = converter.to_pinecone_format(sample_document_chunks)
        weaviate_format = converter.to_weaviate_format(sample_document_chunks)
        qdrant_format = converter.to_qdrant_format(sample_document_chunks)
        milvus_format = converter.to_milvus_format(sample_document_chunks)

        # Verify all formats are valid
        assert isinstance(chromadb_format, dict)
        assert "ids" in chromadb_format
        assert len(chromadb_format["ids"]) == len(sample_document_chunks)

        assert isinstance(pinecone_format, list)
        assert len(pinecone_format) == len(sample_document_chunks)

        assert isinstance(weaviate_format, list)
        assert len(weaviate_format) == len(sample_document_chunks)

        assert isinstance(qdrant_format, list)
        assert len(qdrant_format) == len(sample_document_chunks)

        assert isinstance(milvus_format, list)
        assert len(milvus_format) == len(sample_document_chunks)

    def test_converter_query_integration(self, sample_search_query):
        """Test converter integration with search queries."""
        converter = VectorDBConverter()

        # Convert query for different backends
        chromadb_query = converter.convert_search_query(sample_search_query, "chromadb")
        pinecone_query = converter.convert_search_query(sample_search_query, "pinecone")
        weaviate_query = converter.convert_search_query(sample_search_query, "weaviate")
        qdrant_query = converter.convert_search_query(sample_search_query, "qdrant")
        milvus_query = converter.convert_search_query(sample_search_query, "milvus")

        # Verify all query formats are valid
        assert isinstance(chromadb_query, dict)
        assert isinstance(pinecone_query, dict)
        assert isinstance(weaviate_query, dict)
        assert isinstance(qdrant_query, dict)
        assert isinstance(milvus_query, dict)

    def test_converter_round_trip_conversion(self, sample_document_chunks):
        """Test round-trip conversion preserves data integrity."""
        converter = VectorDBConverter()

        # Convert to ChromaDB format and back
        chromadb_format = converter.to_chromadb_format(sample_document_chunks)
        reconstructed_chunks = converter.from_chromadb_format(chromadb_format)

        # Verify data integrity
        assert len(reconstructed_chunks) == len(sample_document_chunks)
        for original, reconstructed in zip(
            sample_document_chunks, reconstructed_chunks
        ):
            assert original.id == reconstructed.id
            assert original.content == reconstructed.content
            assert original.embedding == reconstructed.embedding
            assert original.metadata == reconstructed.metadata


class TestHealthCheckIntegration:
    """Test integration between health check system and backends."""

    def test_health_manager_with_all_backends(self, vector_db_temp_dir):
        """Test health manager integration with all backend types."""
        configs = [
            ChromaDBConfig(persist_directory=vector_db_temp_dir / "chroma"),
            PineconeConfig(api_key="test_key", index_name="test_index"),
            WeaviateConfig(url="http://localhost:8080", class_name="TestClass"),
            QdrantConfig(
                url="http://localhost:6333", collection_name="test_collection"
            ),
            MilvusConfig(
                host="localhost", port=19530, collection_name="test_collection"
            ),
        ]

        health_manager = VectorDBHealthManager()

        for config in configs:
            with patch.object(VectorDBFactory, "create_client") as mock_create:
                mock_client = Mock()
                mock_client.health_check.return_value = True
                mock_create.return_value = mock_client

                # Register client with health manager
                health_manager.register_client(config.db_type, mock_client)

                # Check health
                health_status = health_manager.check_health(config.db_type)
                assert health_status.is_healthy is True

    def test_health_check_with_error_handling(self, vector_db_temp_dir):
        """Test health check integration with error handling."""
        ChromaDBConfig(persist_directory=vector_db_temp_dir / "chroma")
        health_manager = VectorDBHealthManager()
        VectorDBErrorHandler()

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.health_check.side_effect = Exception("Health check failed")
            mock_create.return_value = mock_client

            health_manager.register_client("chromadb", mock_client)

            # Health check should handle errors gracefully
            health_status = health_manager.check_health("chromadb")
            assert health_status.is_healthy is False
            assert "Health check failed" in health_status.error_message


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows across components."""

    def test_complete_document_lifecycle(
        self, sample_document_chunks, sample_search_query, vector_db_temp_dir
    ):
        """Test complete document lifecycle: create, add, search, update, delete."""
        config = ChromaDBConfig(persist_directory=vector_db_temp_dir / "chroma")

        with patch(
            "pdf_vector_system.vector_db.chroma_client.ChromaDBClient"
        ) as mock_client_class:
            mock_client = Mock()

            # Mock all operations
            mock_client.create_collection.return_value = True
            mock_client.add_chunks.return_value = None
            mock_client.search.return_value = [
                SearchResult(
                    id=sample_document_chunks[0].id,
                    content=sample_document_chunks[0].content,
                    score=0.95,
                    metadata=sample_document_chunks[0].metadata,
                )
            ]
            mock_client.update_chunk.return_value = None
            mock_client.delete_chunks.return_value = None
            mock_client.delete_collection.return_value = True

            mock_client_class.return_value = mock_client

            # Create client through factory
            client = VectorDBFactory.create_client(config)

            # 1. Create collection
            assert client.create_collection("test_collection") is True

            # 2. Add documents
            client.add_chunks(sample_document_chunks)

            # 3. Search documents
            results = client.search(sample_search_query)
            assert len(results) == 1
            assert isinstance(results[0], SearchResult)

            # 4. Update document
            updated_chunk = sample_document_chunks[0]
            updated_chunk.content = "Updated content"
            client.update_chunk(updated_chunk)

            # 5. Delete documents
            chunk_ids = [chunk.id for chunk in sample_document_chunks]
            client.delete_chunks(chunk_ids)

            # 6. Delete collection
            assert client.delete_collection("test_collection") is True

            # Verify all operations were called
            mock_client.create_collection.assert_called_once()
            mock_client.add_chunks.assert_called_once()
            mock_client.search.assert_called_once()
            mock_client.update_chunk.assert_called_once()
            mock_client.delete_chunks.assert_called_once()
            mock_client.delete_collection.assert_called_once()

    def test_multi_backend_workflow(self, sample_document_chunks, vector_db_temp_dir):
        """Test workflow across multiple backends."""
        configs = [
            ChromaDBConfig(persist_directory=vector_db_temp_dir / "chroma"),
            PineconeConfig(api_key="test_key", index_name="test_index"),
        ]

        clients = []

        for config in configs:
            client_module_map = {
                "chromadb": "pdf_vector_system.vector_db.chroma_client.ChromaDBClient",
                "pinecone": "pdf_vector_system.vector_db.pinecone_client.PineconeClient",
            }

            with patch(client_module_map[config.db_type]) as mock_client_class:
                mock_client = Mock()
                mock_client.add_chunks.return_value = None
                mock_client.list_collections.return_value = ["test_collection"]
                mock_client_class.return_value = mock_client

                client = VectorDBFactory.create_client(config)
                clients.append(client)

        # Add same chunks to both backends
        for client in clients:
            client.add_chunks(sample_document_chunks)

        # Verify both backends received the chunks
        for client in clients:
            client.add_chunks.assert_called_with(sample_document_chunks)

    def test_error_propagation_across_components(
        self, sample_document_chunks, vector_db_temp_dir
    ):
        """Test error propagation across integrated components."""
        config = ChromaDBConfig(persist_directory=vector_db_temp_dir / "chroma")

        with patch(
            "pdf_vector_system.vector_db.chroma_client.ChromaDBClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.add_chunks.side_effect = CollectionNotFoundError(
                "Collection not found"
            )
            mock_client_class.return_value = mock_client

            client = VectorDBFactory.create_client(config)
            converter = VectorDBConverter()
            error_handler = VectorDBErrorHandler()

            # Convert chunks
            converter.to_chromadb_format(sample_document_chunks)

            # Try to add chunks - should propagate error
            try:
                client.add_chunks(sample_document_chunks)
            except CollectionNotFoundError as e:
                # Error should be properly handled
                wrapped_error = error_handler.handle_error(
                    e, "add_chunks", {"chunks_count": len(sample_document_chunks)}
                )
                assert isinstance(wrapped_error, CollectionNotFoundError)
                assert "Collection not found" in str(wrapped_error)


class TestConfigurationIntegration:
    """Test integration between configuration and other components."""

    def test_config_validation_with_factory(self, vector_db_temp_dir):
        """Test configuration validation integration with factory."""
        # Valid config
        valid_config = ChromaDBConfig(persist_directory=vector_db_temp_dir / "chroma")

        with patch(
            "pdf_vector_system.vector_db.chroma_client.ChromaDBClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Should create client successfully
            client = VectorDBFactory.create_client(valid_config)
            assert client == mock_client

        # Invalid config (missing required field for Pinecone)
        with pytest.raises(ValueError, match="api_key|cannot be empty"):
            # Empty API key
            PineconeConfig(api_key="", index_name="test_index")

    def test_config_serialization_integration(self, vector_db_temp_dir):
        """Test configuration serialization/deserialization integration."""
        original_config = ChromaDBConfig(
            persist_directory=vector_db_temp_dir / "chroma",
            collection_name="test_collection",
            distance_metric="cosine",
            max_results=100,
        )

        # Serialize config
        config_dict = original_config.model_dump()

        # Deserialize config
        restored_config = ChromaDBConfig(**config_dict)

        # Verify configs are equivalent
        assert original_config.persist_directory == restored_config.persist_directory
        assert original_config.collection_name == restored_config.collection_name
        assert original_config.distance_metric == restored_config.distance_metric
        assert original_config.max_results == restored_config.max_results

        # Both configs should create equivalent clients
        with patch(
            "pdf_vector_system.vector_db.chroma_client.ChromaDBClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            client1 = VectorDBFactory.create_client(original_config)
            client2 = VectorDBFactory.create_client(restored_config)

            # Both should create the same type of client
            assert isinstance(client1, type(client2))
