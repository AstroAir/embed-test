"""Tests for data model classes."""

import time

import pytest

from vectorflow.core.vector_db.models import (
    CollectionInfo,
    CollectionNotFoundError,
    DocumentChunk,
    DocumentInfo,
    DocumentNotFoundError,
    InvalidQueryError,
    SearchQuery,
    SearchResult,
    VectorDBError,
)


class TestDocumentChunk:
    """Test DocumentChunk class."""

    def test_basic_creation(self):
        """Test basic DocumentChunk creation."""
        chunk = DocumentChunk(
            id="test_chunk_1",
            content="This is test content.",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            metadata={"document_id": "test_doc", "page": 1},
        )

        assert chunk.id == "test_chunk_1"
        assert chunk.content == "This is test content."
        assert chunk.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert chunk.metadata["document_id"] == "test_doc"
        assert chunk.metadata["page"] == 1

    def test_validation_empty_id(self):
        """Test validation for empty ID."""
        with pytest.raises(ValueError, match="Chunk ID cannot be empty"):
            DocumentChunk(id="", content="Test content", embedding=[0.1, 0.2, 0.3])

    def test_validation_empty_content(self):
        """Test validation for empty content."""
        with pytest.raises(ValueError, match="Chunk content cannot be empty"):
            DocumentChunk(id="test_chunk", content="", embedding=[0.1, 0.2, 0.3])

    def test_validation_empty_embedding(self):
        """Test validation for empty embedding."""
        with pytest.raises(ValueError, match="Chunk embedding cannot be empty"):
            DocumentChunk(id="test_chunk", content="Test content", embedding=[])

    def test_create_chunk_factory_method(self):
        """Test the create_chunk factory method."""
        chunk = DocumentChunk.create_chunk(
            document_id="test_doc",
            chunk_index=0,
            content="Test content for chunk creation.",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            page_number=1,
            start_char=0,
            end_char=30,
        )

        assert chunk.id == "test_doc_chunk_0"
        assert chunk.content == "Test content for chunk creation."
        assert chunk.embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert chunk.metadata["document_id"] == "test_doc"
        assert chunk.metadata["chunk_index"] == 0
        assert chunk.metadata["page_number"] == 1
        assert chunk.metadata["start_char"] == 0
        assert chunk.metadata["end_char"] == 30
        assert "created_at" in chunk.metadata
        assert "content_length" in chunk.metadata

    def test_create_chunk_with_additional_metadata(self):
        """Test create_chunk with additional metadata."""
        additional_metadata = {"author": "Test Author", "category": "Science"}

        chunk = DocumentChunk.create_chunk(
            document_id="test_doc",
            chunk_index=1,
            content="Test content with metadata.",
            embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
            additional_metadata=additional_metadata,
        )

        assert chunk.metadata["author"] == "Test Author"
        assert chunk.metadata["category"] == "Science"
        assert chunk.metadata["document_id"] == "test_doc"
        assert chunk.metadata["chunk_index"] == 1

    def test_to_chroma_format(self):
        """Test conversion to ChromaDB format."""
        chunk = DocumentChunk(
            id="test_chunk",
            content="Test content",
            embedding=[0.1, 0.2, 0.3],
            metadata={"test": True},
        )

        chroma_format = chunk.to_chroma_format()

        assert chroma_format["ids"] == ["test_chunk"]
        assert chroma_format["documents"] == ["Test content"]
        assert chroma_format["embeddings"] == [[0.1, 0.2, 0.3]]
        assert chroma_format["metadatas"] == [{"test": True}]

    def test_default_metadata(self):
        """Test default metadata behavior."""
        chunk = DocumentChunk(
            id="test_chunk", content="Test content", embedding=[0.1, 0.2, 0.3]
        )

        assert isinstance(chunk.metadata, dict)
        assert len(chunk.metadata) == 0


class TestSearchResult:
    """Test SearchResult class."""

    def test_basic_creation(self):
        """Test basic SearchResult creation."""
        result = SearchResult(
            id="result_1",
            content="This is a search result.",
            score=0.95,
            metadata={"document_id": "doc_1", "chunk_index": 0},
        )

        assert result.id == "result_1"
        assert result.content == "This is a search result."
        assert result.score == 0.95
        assert result.metadata["document_id"] == "doc_1"
        assert result.metadata["chunk_index"] == 0

    def test_property_document_id(self):
        """Test document_id property."""
        result = SearchResult(
            id="result_1",
            content="Test content",
            score=0.8,
            metadata={"document_id": "test_doc"},
        )

        assert result.document_id == "test_doc"

        # Test with missing document_id
        result_no_doc_id = SearchResult(
            id="result_2", content="Test content", score=0.7, metadata={}
        )

        assert result_no_doc_id.document_id is None

    def test_property_chunk_index(self):
        """Test chunk_index property."""
        result = SearchResult(
            id="result_1",
            content="Test content",
            score=0.8,
            metadata={"chunk_index": 5},
        )

        assert result.chunk_index == 5

        # Test with missing chunk_index
        result_no_index = SearchResult(
            id="result_2", content="Test content", score=0.7, metadata={}
        )

        assert result_no_index.chunk_index is None

    def test_property_page_number(self):
        """Test page_number property."""
        result = SearchResult(
            id="result_1",
            content="Test content",
            score=0.8,
            metadata={"page_number": 3},
        )

        assert result.page_number == 3

        # Test with missing page_number
        result_no_page = SearchResult(
            id="result_2", content="Test content", score=0.7, metadata={}
        )

        assert result_no_page.page_number is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SearchResult(
            id="result_1",
            content="Test content",
            score=0.85,
            metadata={"document_id": "doc_1", "page_number": 2},
        )

        result_dict = result.to_dict()

        assert result_dict["id"] == "result_1"
        assert result_dict["content"] == "Test content"
        assert result_dict["score"] == 0.85
        assert result_dict["document_id"] == "doc_1"
        assert result_dict["chunk_index"] is None
        assert result_dict["page_number"] == 2
        assert result_dict["metadata"] == {"document_id": "doc_1", "page_number": 2}

    def test_default_metadata(self):
        """Test default metadata behavior."""
        result = SearchResult(id="result_1", content="Test content", score=0.8)

        assert isinstance(result.metadata, dict)
        assert len(result.metadata) == 0


class TestSearchQuery:
    """Test SearchQuery class."""

    def test_basic_creation(self):
        """Test basic SearchQuery creation."""
        query = SearchQuery(query_text="test query", n_results=5)

        assert query.query_text == "test query"
        assert query.n_results == 5
        assert query.where is None
        assert query.where_document is None
        assert query.include_distances is True
        assert query.include_metadata is True
        assert query.include_documents is True

    def test_custom_parameters(self):
        """Test SearchQuery with custom parameters."""
        where_clause = {"document_id": "test_doc"}
        where_document = {"$contains": "machine learning"}

        query = SearchQuery(
            query_text="advanced search",
            n_results=10,
            where=where_clause,
            where_document=where_document,
            include_distances=False,
            include_metadata=False,
            include_documents=True,
        )

        assert query.query_text == "advanced search"
        assert query.n_results == 10
        assert query.where == where_clause
        assert query.where_document == where_document
        assert query.include_distances is False
        assert query.include_metadata is False
        assert query.include_documents is True

    def test_validation_empty_query_text(self):
        """Test validation for empty query text."""
        with pytest.raises(ValueError, match="Query text cannot be empty"):
            SearchQuery(query_text="")

        with pytest.raises(ValueError, match="Query text cannot be empty"):
            SearchQuery(query_text="   ")

    def test_validation_negative_n_results(self):
        """Test validation for negative n_results."""
        with pytest.raises(ValueError, match="Number of results must be positive"):
            SearchQuery(query_text="test", n_results=0)

        with pytest.raises(ValueError, match="Number of results must be positive"):
            SearchQuery(query_text="test", n_results=-5)

    def test_to_chroma_params(self):
        """Test conversion to ChromaDB parameters."""
        query = SearchQuery(
            query_text="test query",
            n_results=8,
            where={"document_id": "test_doc"},
            where_document={"$contains": "test"},
            include_distances=True,
            include_metadata=True,
            include_documents=False,
        )

        params = query.to_chroma_params()

        assert params["n_results"] == 8
        assert params["where"] == {"document_id": "test_doc"}
        assert params["where_document"] == {"$contains": "test"}
        assert "distances" in params["include"]
        assert "metadatas" in params["include"]
        assert "documents" not in params["include"]

    def test_to_chroma_params_minimal(self):
        """Test conversion to ChromaDB parameters with minimal settings."""
        query = SearchQuery(
            query_text="simple query",
            include_distances=False,
            include_metadata=False,
            include_documents=False,
        )

        params = query.to_chroma_params()

        assert params["n_results"] == 10  # default
        assert params["include"] == []
        assert "where" not in params
        assert "where_document" not in params


class TestCollectionInfo:
    """Test CollectionInfo class."""

    def test_basic_creation(self):
        """Test basic CollectionInfo creation."""
        info = CollectionInfo(
            name="test_collection", count=100, metadata={"created_by": "test"}
        )

        assert info.name == "test_collection"
        assert info.count == 100
        assert info.metadata["created_by"] == "test"

    def test_is_empty_property(self):
        """Test is_empty property."""
        empty_collection = CollectionInfo(name="empty", count=0)
        non_empty_collection = CollectionInfo(name="non_empty", count=50)

        assert empty_collection.is_empty is True
        assert non_empty_collection.is_empty is False

    def test_to_dict(self):
        """Test conversion to dictionary."""
        info = CollectionInfo(
            name="test_collection", count=75, metadata={"version": "1.0"}
        )

        info_dict = info.to_dict()

        assert info_dict["name"] == "test_collection"
        assert info_dict["count"] == 75
        assert info_dict["metadata"] == {"version": "1.0"}
        assert info_dict["is_empty"] is False

    def test_default_metadata(self):
        """Test default metadata behavior."""
        info = CollectionInfo(name="test", count=10)

        assert isinstance(info.metadata, dict)
        assert len(info.metadata) == 0


class TestDocumentInfo:
    """Test DocumentInfo class."""

    def test_basic_creation(self):
        """Test basic DocumentInfo creation."""
        info = DocumentInfo(
            document_id="test_doc",
            chunk_count=10,
            total_characters=5000,
            page_count=5,
            created_at=time.time(),
            metadata={"title": "Test Document"},
        )

        assert info.document_id == "test_doc"
        assert info.chunk_count == 10
        assert info.total_characters == 5000
        assert info.page_count == 5
        assert info.created_at is not None
        assert info.metadata["title"] == "Test Document"

    def test_average_chunk_size_property(self):
        """Test average_chunk_size property."""
        info = DocumentInfo(
            document_id="test_doc", chunk_count=4, total_characters=1000
        )

        assert info.average_chunk_size == 250.0

        # Test with zero chunks
        empty_info = DocumentInfo(
            document_id="empty_doc", chunk_count=0, total_characters=0
        )

        assert empty_info.average_chunk_size == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        created_time = time.time()
        info = DocumentInfo(
            document_id="test_doc",
            chunk_count=8,
            total_characters=2400,
            page_count=3,
            created_at=created_time,
            metadata={"author": "Test Author"},
        )

        info_dict = info.to_dict()

        assert info_dict["document_id"] == "test_doc"
        assert info_dict["chunk_count"] == 8
        assert info_dict["total_characters"] == 2400
        assert info_dict["page_count"] == 3
        assert info_dict["created_at"] == created_time
        assert info_dict["average_chunk_size"] == 300.0
        assert info_dict["metadata"] == {"author": "Test Author"}

    def test_optional_fields(self):
        """Test optional fields."""
        info = DocumentInfo(
            document_id="minimal_doc", chunk_count=5, total_characters=1500
        )

        assert info.page_count is None
        assert info.created_at is None
        assert isinstance(info.metadata, dict)
        assert len(info.metadata) == 0


class TestExceptions:
    """Test custom exception classes."""

    def test_vector_db_error(self):
        """Test VectorDBError exception."""
        with pytest.raises(VectorDBError, match="Database error"):
            raise VectorDBError("Database error")

    def test_collection_not_found_error(self):
        """Test CollectionNotFoundError exception."""
        with pytest.raises(CollectionNotFoundError, match="Collection not found"):
            raise CollectionNotFoundError("Collection not found")

        # Should also be caught as VectorDBError
        with pytest.raises(VectorDBError):
            raise CollectionNotFoundError("Collection not found")

    def test_document_not_found_error(self):
        """Test DocumentNotFoundError exception."""
        with pytest.raises(DocumentNotFoundError, match="Document not found"):
            raise DocumentNotFoundError("Document not found")

        # Should also be caught as VectorDBError
        with pytest.raises(VectorDBError):
            raise DocumentNotFoundError("Document not found")

    def test_invalid_query_error(self):
        """Test InvalidQueryError exception."""
        with pytest.raises(InvalidQueryError, match="Invalid query"):
            raise InvalidQueryError("Invalid query")

        # Should also be caught as VectorDBError
        with pytest.raises(VectorDBError):
            raise InvalidQueryError("Invalid query")
