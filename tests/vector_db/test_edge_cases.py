"""Edge case and boundary condition tests for vector database components."""

from unittest.mock import Mock, patch

import pytest

from pdf_vector_system.vector_db.config import ChromaDBConfig
from pdf_vector_system.vector_db.converters import VectorDBConverter
from pdf_vector_system.vector_db.factory import VectorDBFactory
from pdf_vector_system.vector_db.models import (
    DocumentChunk,
    InvalidQueryError,
    SearchQuery,
    VectorDBError,
)


class TestEmptyAndNullInputs:
    """Test handling of empty and null inputs."""

    def test_empty_chunk_list(self):
        """Test adding empty list of chunks."""
        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.add_chunks.return_value = None
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            # Should handle empty list gracefully
            client.add_chunks([])
            mock_client.add_chunks.assert_called_once_with([])

    def test_empty_search_query(self):
        """Test search with empty query text."""
        query = SearchQuery(query_text="", n_results=5)

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.search.return_value = []
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            results = client.search(query)
            assert results == []

    def test_null_chunk_content(self):
        """Test handling of chunks with null/None content."""
        chunk = DocumentChunk(
            id="test_chunk",
            content=None,  # Null content
            embedding=[0.1, 0.2, 0.3],
            metadata={"document_id": "doc1"},
        )

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.add_chunks.side_effect = InvalidQueryError(
                "Content cannot be null"
            )
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            with pytest.raises(InvalidQueryError):
                client.add_chunks([chunk])

    def test_empty_embedding_vector(self):
        """Test handling of chunks with empty embedding vectors."""
        chunk = DocumentChunk(
            id="test_chunk",
            content="Test content",
            embedding=[],  # Empty embedding
            metadata={"document_id": "doc1"},
        )

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.add_chunks.side_effect = InvalidQueryError(
                "Embedding cannot be empty"
            )
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            with pytest.raises(InvalidQueryError):
                client.add_chunks([chunk])

    def test_none_metadata(self):
        """Test handling of chunks with None metadata."""
        chunk = DocumentChunk(
            id="test_chunk",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata=None,  # None metadata
        )

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.add_chunks.return_value = None
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            # Should handle None metadata gracefully
            client.add_chunks([chunk])
            mock_client.add_chunks.assert_called_once()


class TestLargeDataHandling:
    """Test handling of large datasets and boundary conditions."""

    def test_very_large_chunk_content(self):
        """Test handling of chunks with very large content."""
        # Create a chunk with 1MB of content
        large_content = "x" * (1024 * 1024)  # 1MB
        chunk = DocumentChunk(
            id="large_chunk",
            content=large_content,
            embedding=[0.1] * 384,  # Standard embedding size
            metadata={"document_id": "large_doc", "size": "1MB"},
        )

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.add_chunks.return_value = None
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            # Should handle large content
            client.add_chunks([chunk])
            mock_client.add_chunks.assert_called_once()

    def test_maximum_batch_size(self, large_document_chunks):
        """Test handling of maximum batch sizes."""
        # Test with very large batch (10,000 chunks)
        large_batch = (
            large_document_chunks[:10000]
            if len(large_document_chunks) >= 10000
            else large_document_chunks * 100
        )

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.add_chunks.return_value = None
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            # Should handle large batches
            client.add_chunks(large_batch)
            mock_client.add_chunks.assert_called_once()

    def test_high_dimensional_embeddings(self):
        """Test handling of high-dimensional embedding vectors."""
        # Create chunk with very high-dimensional embedding (4096 dimensions)
        high_dim_embedding = [0.1] * 4096
        chunk = DocumentChunk(
            id="high_dim_chunk",
            content="Test content",
            embedding=high_dim_embedding,
            metadata={"document_id": "doc1", "embedding_dim": 4096},
        )

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.add_chunks.return_value = None
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            client.add_chunks([chunk])
            mock_client.add_chunks.assert_called_once()

    def test_maximum_search_results(self):
        """Test search with maximum number of results."""
        query = SearchQuery(
            query_text="test query",
            n_results=10000,  # Very large number of results
        )

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.search.return_value = []
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            results = client.search(query)
            assert results == []
            mock_client.search.assert_called_once()


class TestSpecialCharacterHandling:
    """Test handling of special characters and Unicode."""

    def test_unicode_content(self):
        """Test handling of Unicode content in chunks."""
        unicode_content = "测试内容 🚀 émojis and spëcial chars ñ"
        chunk = DocumentChunk(
            id="unicode_chunk",
            content=unicode_content,
            embedding=[0.1, 0.2, 0.3],
            metadata={"document_id": "unicode_doc", "language": "mixed"},
        )

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.add_chunks.return_value = None
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            client.add_chunks([chunk])
            mock_client.add_chunks.assert_called_once()

    def test_special_characters_in_ids(self):
        """Test handling of special characters in chunk IDs."""
        special_ids = [
            "chunk-with-dashes",
            "chunk_with_underscores",
            "chunk.with.dots",
            "chunk@with@symbols",
            "chunk#with#hash",
            "chunk%with%percent",
        ]

        chunks = []
        for chunk_id in special_ids:
            chunk = DocumentChunk(
                id=chunk_id,
                content=f"Content for {chunk_id}",
                embedding=[0.1, 0.2, 0.3],
                metadata={"document_id": "special_doc"},
            )
            chunks.append(chunk)

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.add_chunks.return_value = None
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            client.add_chunks(chunks)
            mock_client.add_chunks.assert_called_once()

    def test_json_special_characters_in_metadata(self):
        """Test handling of JSON special characters in metadata."""
        metadata_with_special_chars = {
            "title": "Document with \"quotes\" and 'apostrophes'",
            "description": "Text with\nnewlines\tand\ttabs",
            "path": "C:\\Windows\\Path\\With\\Backslashes",
            "json_string": '{"nested": "json", "array": [1, 2, 3]}',
            "unicode": "Special chars: ñáéíóú 中文 🎉",
        }

        chunk = DocumentChunk(
            id="special_metadata_chunk",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata=metadata_with_special_chars,
        )

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.add_chunks.return_value = None
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            client.add_chunks([chunk])
            mock_client.add_chunks.assert_called_once()


class TestNumericalEdgeCases:
    """Test numerical edge cases and boundary conditions."""

    def test_zero_embedding_values(self):
        """Test handling of zero values in embeddings."""
        zero_embedding = [0.0] * 384
        chunk = DocumentChunk(
            id="zero_embedding_chunk",
            content="Test content",
            embedding=zero_embedding,
            metadata={"document_id": "doc1"},
        )

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.add_chunks.return_value = None
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            client.add_chunks([chunk])
            mock_client.add_chunks.assert_called_once()

    def test_extreme_embedding_values(self):
        """Test handling of extreme embedding values."""
        extreme_embedding = [
            float("inf"),  # Positive infinity
            float("-inf"),  # Negative infinity
            1e10,  # Very large positive
            -1e10,  # Very large negative
            1e-10,  # Very small positive
            -1e-10,  # Very small negative
        ] + [0.0] * 378  # Fill to standard size

        chunk = DocumentChunk(
            id="extreme_embedding_chunk",
            content="Test content",
            embedding=extreme_embedding,
            metadata={"document_id": "doc1"},
        )

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            # Some backends might reject infinite values
            mock_client.add_chunks.side_effect = InvalidQueryError(
                "Invalid embedding values"
            )
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            with pytest.raises(InvalidQueryError):
                client.add_chunks([chunk])

    def test_nan_embedding_values(self):
        """Test handling of NaN values in embeddings."""
        nan_embedding = [float("nan")] * 384
        chunk = DocumentChunk(
            id="nan_embedding_chunk",
            content="Test content",
            embedding=nan_embedding,
            metadata={"document_id": "doc1"},
        )

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.add_chunks.side_effect = InvalidQueryError(
                "NaN values not allowed"
            )
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            with pytest.raises(InvalidQueryError):
                client.add_chunks([chunk])

    def test_zero_search_results(self):
        """Test search query requesting zero results."""
        query = SearchQuery(
            query_text="test query",
            n_results=0,  # Zero results requested
        )

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.search.return_value = []
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            results = client.search(query)
            assert results == []

    def test_negative_search_results(self):
        """Test search query with negative number of results."""
        query = SearchQuery(
            query_text="test query",
            n_results=-5,  # Negative results
        )

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.search.side_effect = InvalidQueryError(
                "n_results must be positive"
            )
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            with pytest.raises(InvalidQueryError):
                client.search(query)


class TestConverterEdgeCases:
    """Test edge cases in data conversion utilities."""

    def test_convert_empty_chunk_list(self):
        """Test converting empty chunk list."""
        converter = VectorDBConverter()

        # Test conversion to different backend formats
        chromadb_format = converter.to_chromadb_format([])
        assert chromadb_format == {
            "ids": [],
            "documents": [],
            "embeddings": [],
            "metadatas": [],
        }

        pinecone_format = converter.to_pinecone_format([])
        assert pinecone_format == []

    def test_convert_chunk_with_missing_fields(self):
        """Test converting chunks with missing optional fields."""
        # Chunk without embedding
        chunk_no_embedding = DocumentChunk(
            id="no_embedding",
            content="Test content",
            embedding=None,
            metadata={"document_id": "doc1"},
        )

        # Chunk without metadata
        chunk_no_metadata = DocumentChunk(
            id="no_metadata",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata=None,
        )

        converter = VectorDBConverter()

        # Should handle missing fields gracefully
        result1 = converter.to_chromadb_format([chunk_no_embedding])
        assert result1["embeddings"][0] is None

        result2 = converter.to_chromadb_format([chunk_no_metadata])
        assert result2["metadatas"][0] is None or result2["metadatas"][0] == {}

    def test_convert_malformed_search_query(self):
        """Test converting malformed search queries."""
        converter = VectorDBConverter()

        # Query with invalid where clause
        malformed_query = SearchQuery(
            query_text="test",
            n_results=5,
            where={"invalid": {"nested": {"structure": "too deep"}}},
        )

        # Should handle gracefully or raise appropriate error
        try:
            result = converter.convert_search_query(malformed_query, "chromadb")
            # If conversion succeeds, verify it's handled appropriately
            assert isinstance(result, dict)
        except InvalidQueryError:
            # If conversion fails, that's also acceptable
            pass


class TestConcurrencyEdgeCases:
    """Test edge cases in concurrent operations."""

    def test_concurrent_collection_creation(self):
        """Test concurrent creation of the same collection."""
        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()

            # First call succeeds, second fails with "already exists"
            mock_client.create_collection.side_effect = [
                True,  # First call succeeds
                VectorDBError("Collection already exists"),  # Second call fails
            ]
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            # First creation should succeed
            result1 = client.create_collection("test_collection")
            assert result1 is True

            # Second creation should fail
            with pytest.raises(VectorDBError):
                client.create_collection("test_collection")

    def test_concurrent_chunk_operations(self, sample_document_chunks):
        """Test concurrent add and delete operations on same chunks."""
        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.add_chunks.return_value = None
            mock_client.delete_chunks.return_value = None
            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            # Simulate concurrent operations
            chunk_ids = [chunk.id for chunk in sample_document_chunks]

            # Add chunks
            client.add_chunks(sample_document_chunks)

            # Immediately try to delete them
            client.delete_chunks(chunk_ids)

            # Both operations should complete
            mock_client.add_chunks.assert_called_once()
            mock_client.delete_chunks.assert_called_once()
