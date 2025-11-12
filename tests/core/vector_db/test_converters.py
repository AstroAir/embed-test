"""Tests for VectorDBConverter universal conversion utilities."""

import pytest

from pdf_vector_system.core.vector_db.config import VectorDBType
from pdf_vector_system.core.vector_db.converters import VectorDBConverter
from pdf_vector_system.core.vector_db.models import DocumentChunk, SearchQuery


class TestVectorDBConverter:
    """Test VectorDBConverter class."""

    def test_chunks_to_chromadb_format(self, sample_document_chunks):
        """Test conversion of chunks to ChromaDB format."""
        result = VectorDBConverter.chunks_to_chromadb_format(sample_document_chunks)

        assert isinstance(result, dict)
        assert "ids" in result
        assert "documents" in result
        assert "embeddings" in result
        assert "metadatas" in result

        # Check data integrity
        assert len(result["ids"]) == len(sample_document_chunks)
        assert len(result["documents"]) == len(sample_document_chunks)
        assert len(result["embeddings"]) == len(sample_document_chunks)
        assert len(result["metadatas"]) == len(sample_document_chunks)

        # Check specific values
        assert result["ids"][0] == sample_document_chunks[0].id
        assert result["documents"][0] == sample_document_chunks[0].content
        assert result["embeddings"][0] == sample_document_chunks[0].embedding
        assert result["metadatas"][0] == sample_document_chunks[0].metadata

    def test_chunks_to_chromadb_format_empty(self):
        """Test conversion with empty chunk list."""
        result = VectorDBConverter.chunks_to_chromadb_format([])

        assert result == {"ids": [], "documents": [], "embeddings": [], "metadatas": []}

    def test_query_to_chromadb_params(self, sample_search_query):
        """Test conversion of SearchQuery to ChromaDB parameters."""
        result = VectorDBConverter.query_to_chromadb_params(sample_search_query)

        assert isinstance(result, dict)
        assert "n_results" in result
        assert "include" in result

        # Check default includes
        expected_includes = ["distances", "metadatas", "documents"]
        for include in expected_includes:
            assert include in result["include"]

        # Check query parameters
        assert result["n_results"] == sample_search_query.n_results

        if sample_search_query.where:
            assert result["where"] == sample_search_query.where

    def test_query_to_chromadb_params_minimal(self):
        """Test conversion with minimal SearchQuery."""
        query = SearchQuery(
            query_text="test",
            include_distances=False,
            include_metadata=False,
            include_documents=False,
        )

        result = VectorDBConverter.query_to_chromadb_params(query)

        assert result["include"] == []
        assert result["n_results"] == 10  # default

    def test_chunks_to_pinecone_format(self, sample_document_chunks):
        """Test conversion of chunks to Pinecone format."""
        result = VectorDBConverter.chunks_to_pinecone_format(sample_document_chunks)

        assert isinstance(result, list)
        assert len(result) == len(sample_document_chunks)

        for i, vector in enumerate(result):
            assert "id" in vector
            assert "values" in vector
            assert "metadata" in vector

            assert vector["id"] == sample_document_chunks[i].id
            assert vector["values"] == sample_document_chunks[i].embedding
            assert "content" in vector["metadata"]
            assert vector["metadata"]["content"] == sample_document_chunks[i].content

    def test_query_to_pinecone_params(self, sample_search_query):
        """Test conversion of SearchQuery to Pinecone parameters."""
        result = VectorDBConverter.query_to_pinecone_params(sample_search_query)

        assert isinstance(result, dict)
        assert "top_k" in result
        assert "include_metadata" in result
        assert "include_values" in result

        assert result["top_k"] == sample_search_query.n_results
        assert result["include_metadata"] == sample_search_query.include_metadata
        assert result["include_values"] == sample_search_query.include_distances

        if sample_search_query.where:
            assert result["filter"] == sample_search_query.where

    def test_chunks_to_weaviate_format(self, sample_document_chunks):
        """Test conversion of chunks to Weaviate format."""
        class_name = "TestDocument"
        result = VectorDBConverter.chunks_to_weaviate_format(
            sample_document_chunks, class_name
        )

        assert isinstance(result, list)
        assert len(result) == len(sample_document_chunks)

        for i, obj in enumerate(result):
            assert "class" in obj
            assert "id" in obj
            assert "properties" in obj
            assert "vector" in obj

            assert obj["class"] == class_name
            assert obj["id"] == sample_document_chunks[i].id
            assert obj["properties"]["content"] == sample_document_chunks[i].content
            assert obj["vector"] == sample_document_chunks[i].embedding

    def test_query_to_weaviate_params(self, sample_search_query):
        """Test conversion of SearchQuery to Weaviate parameters."""
        result = VectorDBConverter.query_to_weaviate_params(sample_search_query)

        assert isinstance(result, dict)
        assert "limit" in result
        assert result["limit"] == sample_search_query.n_results

        if (
            sample_search_query.include_distances
            or sample_search_query.include_metadata
        ):
            assert "additional" in result
            assert isinstance(result["additional"], list)

        if sample_search_query.where:
            assert result["where"] == sample_search_query.where

    def test_chunks_to_qdrant_format(self, sample_document_chunks):
        """Test conversion of chunks to Qdrant format."""
        result = VectorDBConverter.chunks_to_qdrant_format(sample_document_chunks)

        assert isinstance(result, list)
        assert len(result) == len(sample_document_chunks)

        for i, point in enumerate(result):
            assert "id" in point
            assert "vector" in point
            assert "payload" in point

            assert point["id"] == sample_document_chunks[i].id
            assert point["vector"] == sample_document_chunks[i].embedding
            assert point["payload"]["content"] == sample_document_chunks[i].content

    def test_query_to_qdrant_params(self, sample_search_query):
        """Test conversion of SearchQuery to Qdrant parameters."""
        result = VectorDBConverter.query_to_qdrant_params(sample_search_query)

        assert isinstance(result, dict)
        assert "limit" in result
        assert "with_payload" in result
        assert "with_vectors" in result

        assert result["limit"] == sample_search_query.n_results
        assert result["with_payload"] == sample_search_query.include_metadata
        assert result["with_vectors"] == sample_search_query.include_distances

        if sample_search_query.where:
            assert result["filter"] == sample_search_query.where

    def test_chunks_to_milvus_format(self, sample_document_chunks):
        """Test conversion of chunks to Milvus format."""
        result = VectorDBConverter.chunks_to_milvus_format(sample_document_chunks)

        assert isinstance(result, dict)
        assert "id" in result
        assert "content" in result
        assert "embedding" in result

        assert len(result["id"]) == len(sample_document_chunks)
        assert len(result["content"]) == len(sample_document_chunks)
        assert len(result["embedding"]) == len(sample_document_chunks)

        # Check metadata fields are flattened
        if sample_document_chunks:
            metadata_keys = sample_document_chunks[0].metadata.keys()
            for key in metadata_keys:
                assert f"metadata_{key}" in result

    def test_chunks_to_milvus_format_empty(self):
        """Test Milvus conversion with empty chunks."""
        result = VectorDBConverter.chunks_to_milvus_format([])

        assert result == {"id": [], "content": [], "embedding": []}

    def test_query_to_milvus_params(self, sample_search_query):
        """Test conversion of SearchQuery to Milvus parameters."""
        result = VectorDBConverter.query_to_milvus_params(sample_search_query)

        assert isinstance(result, dict)
        assert "limit" in result
        assert "output_fields" in result

        assert result["limit"] == sample_search_query.n_results

        if sample_search_query.include_metadata:
            assert result["output_fields"] == ["*"]
        else:
            assert result["output_fields"] == ["id", "content"]

        if sample_search_query.where:
            assert "expr" in result

    def test_convert_chunks_for_backend_chromadb(self, sample_document_chunks):
        """Test universal chunk conversion for ChromaDB."""
        result = VectorDBConverter.convert_chunks_for_backend(
            sample_document_chunks, VectorDBType.CHROMADB
        )

        assert isinstance(result, dict)
        assert "ids" in result
        assert "documents" in result
        assert "embeddings" in result
        assert "metadatas" in result

    def test_convert_chunks_for_backend_pinecone(self, sample_document_chunks):
        """Test universal chunk conversion for Pinecone."""
        result = VectorDBConverter.convert_chunks_for_backend(
            sample_document_chunks, VectorDBType.PINECONE
        )

        assert isinstance(result, list)
        assert all("id" in item and "values" in item for item in result)

    def test_convert_chunks_for_backend_weaviate(self, sample_document_chunks):
        """Test universal chunk conversion for Weaviate."""
        result = VectorDBConverter.convert_chunks_for_backend(
            sample_document_chunks, VectorDBType.WEAVIATE, class_name="TestDoc"
        )

        assert isinstance(result, list)
        assert all("class" in item and item["class"] == "TestDoc" for item in result)

    def test_convert_chunks_for_backend_qdrant(self, sample_document_chunks):
        """Test universal chunk conversion for Qdrant."""
        result = VectorDBConverter.convert_chunks_for_backend(
            sample_document_chunks, VectorDBType.QDRANT
        )

        assert isinstance(result, list)
        assert all(
            "id" in item and "vector" in item and "payload" in item for item in result
        )

    def test_convert_chunks_for_backend_milvus(self, sample_document_chunks):
        """Test universal chunk conversion for Milvus."""
        result = VectorDBConverter.convert_chunks_for_backend(
            sample_document_chunks, VectorDBType.MILVUS
        )

        assert isinstance(result, dict)
        assert "id" in result
        assert "content" in result
        assert "embedding" in result

    def test_convert_chunks_for_backend_unsupported(self, sample_document_chunks):
        """Test universal chunk conversion with unsupported backend."""
        with pytest.raises(ValueError, match="Unsupported backend type"):
            VectorDBConverter.convert_chunks_for_backend(
                sample_document_chunks, "unsupported_backend"
            )

    def test_convert_query_for_backend_all_types(self, sample_search_query):
        """Test universal query conversion for all backend types."""
        backends = [
            VectorDBType.CHROMADB,
            VectorDBType.PINECONE,
            VectorDBType.WEAVIATE,
            VectorDBType.QDRANT,
            VectorDBType.MILVUS,
        ]

        for backend in backends:
            result = VectorDBConverter.convert_query_for_backend(
                sample_search_query, backend
            )
            assert isinstance(result, dict)
            # Each backend should have some form of limit/results parameter
            limit_keys = ["n_results", "top_k", "limit"]
            assert any(key in result for key in limit_keys)

    def test_convert_query_for_backend_unsupported(self, sample_search_query):
        """Test universal query conversion with unsupported backend."""
        with pytest.raises(ValueError, match="Unsupported backend type"):
            VectorDBConverter.convert_query_for_backend(
                sample_search_query, "unsupported_backend"
            )


class TestConverterEdgeCases:
    """Test edge cases and error conditions for converters."""

    def test_chunks_with_missing_fields(self):
        """Test conversion with chunks that have missing fields."""
        # Create chunk with minimal data
        minimal_chunk = DocumentChunk(
            id="minimal_chunk",
            content="Minimal content",
            embedding=[0.1, 0.2, 0.3],
            metadata={},
        )

        # Should work for all backends
        backends = [
            VectorDBType.CHROMADB,
            VectorDBType.PINECONE,
            VectorDBType.QDRANT,
            VectorDBType.MILVUS,
        ]

        for backend in backends:
            result = VectorDBConverter.convert_chunks_for_backend(
                [minimal_chunk], backend
            )
            assert result is not None

    def test_query_with_complex_filters(self):
        """Test query conversion with complex filter conditions."""
        complex_query = SearchQuery(
            query_text="test query",
            n_results=5,
            where={
                "document_id": {"$in": ["doc1", "doc2", "doc3"]},
                "page_number": {"$gte": 1, "$lte": 10},
                "category": "research",
            },
            where_document={"$contains": "important"},
        )

        # Test with backends that support complex filters
        chromadb_params = VectorDBConverter.query_to_chromadb_params(complex_query)
        assert chromadb_params["where"] == complex_query.where
        assert chromadb_params["where_document"] == complex_query.where_document

        pinecone_params = VectorDBConverter.query_to_pinecone_params(complex_query)
        assert pinecone_params["filter"] == complex_query.where

    def test_large_batch_conversion(self, sample_embeddings_5d):
        """Test conversion with large batches of data."""
        # Create a large number of chunks
        large_chunks = []
        for i in range(1000):
            chunk = DocumentChunk(
                id=f"chunk_{i}",
                content=f"Content for chunk {i}",
                embedding=sample_embeddings_5d[i % len(sample_embeddings_5d)],
                metadata={"batch_id": i // 100, "index": i},
            )
            large_chunks.append(chunk)

        # Test conversion for different backends
        chromadb_result = VectorDBConverter.chunks_to_chromadb_format(large_chunks)
        assert len(chromadb_result["ids"]) == 1000

        pinecone_result = VectorDBConverter.chunks_to_pinecone_format(large_chunks)
        assert len(pinecone_result) == 1000

        qdrant_result = VectorDBConverter.chunks_to_qdrant_format(large_chunks)
        assert len(qdrant_result) == 1000

    def test_unicode_content_handling(self):
        """Test conversion with Unicode content."""
        unicode_chunk = DocumentChunk(
            id="unicode_chunk",
            content="Content with émojis 🚀 and spëcial chäractërs",
            embedding=[0.1, 0.2, 0.3],
            metadata={"language": "mixed", "special": "tëst"},
        )

        # Test with all backends
        for backend in VectorDBType:
            try:
                result = VectorDBConverter.convert_chunks_for_backend(
                    [unicode_chunk], backend
                )
                assert result is not None
            except ValueError as e:
                # Expected for unsupported backends
                assert "Unsupported backend type" in str(e)
