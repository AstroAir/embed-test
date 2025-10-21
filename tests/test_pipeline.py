"""Tests for PDFVectorPipeline class."""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pdf_vector_system.pipeline import (
    PDFVectorPipeline,
    PipelineError,
    ProcessingResult,
)
from pdf_vector_system.vector_db.models import DocumentChunk, SearchQuery, SearchResult


class TestProcessingResult:
    """Test ProcessingResult dataclass."""

    def test_creation(self):
        """Test ProcessingResult creation."""
        result = ProcessingResult(
            file_path="test.pdf",
            document_id="doc_1",
            chunks_processed=5,
            embeddings_generated=5,
            chunks_stored=5,
            processing_time=2.5,
            success=True,
            error_message=None,
            metadata={"pages": 3},
        )

        assert result.file_path == "test.pdf"
        assert result.document_id == "doc_1"
        assert result.chunks_processed == 5
        assert result.processing_time == 2.5
        assert result.success is True
        assert result.error_message is None
        assert result.metadata["pages"] == 3

    def test_chunks_per_second_property(self):
        """Test chunks_per_second property."""
        result = ProcessingResult(
            file_path="test.pdf",
            document_id="doc_1",
            chunks_processed=10,
            embeddings_generated=10,
            chunks_stored=10,
            processing_time=2.0,
            success=True,
        )

        assert result.chunks_per_second == 5.0

    def test_chunks_per_second_zero_time(self):
        """Test chunks_per_second with zero processing time."""
        result = ProcessingResult(
            file_path="test.pdf",
            document_id="doc_1",
            chunks_processed=5,
            embeddings_generated=5,
            chunks_stored=5,
            processing_time=0.0,
            success=True,
        )

        assert result.chunks_per_second == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ProcessingResult(
            file_path="test.pdf",
            document_id="doc_1",
            chunks_processed=3,
            embeddings_generated=3,
            chunks_stored=3,
            processing_time=1.5,
            success=True,
            metadata={"test": True},
        )

        result_dict = result.to_dict()

        assert result_dict["file_path"] == "test.pdf"
        assert result_dict["document_id"] == "doc_1"
        assert result_dict["chunks_processed"] == 3
        assert result_dict["processing_time"] == 1.5
        assert result_dict["success"] is True
        assert result_dict["chunks_per_second"] == 2.0
        assert result_dict["metadata"]["test"] is True


class TestPDFVectorPipeline:
    """Test PDFVectorPipeline class."""

    def test_initialization(self, test_config):
        """Test pipeline initialization."""
        with patch("pdf_vector_system.pipeline.PDFProcessor"):
            with patch("pdf_vector_system.pipeline.TextProcessor"):
                with patch("pdf_vector_system.pipeline.EmbeddingServiceFactory"):
                    with patch("pdf_vector_system.pipeline.ChromaDBClient"):
                        pipeline = PDFVectorPipeline(test_config)

                        assert pipeline.config == test_config
                        assert pipeline.pdf_processor is not None
                        assert pipeline.text_processor is not None
                        assert pipeline.embedding_service is not None
                        assert pipeline.vector_db is not None

    def test_process_pdf_success(self, test_config, temp_dir):
        """Test successful PDF processing."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF content")

        # Mock PDF extraction result
        mock_pdf_result = {
            "file_path": str(pdf_path),
            "file_name": "test.pdf",
            "page_count": 2,
            "text_content": {1: "Page 1 content", 2: "Page 2 content"},
            "total_characters": 30,
            "metadata": {"title": "Test Document"},
            "extraction_timestamp": time.time(),
        }

        # Mock text chunks
        mock_chunks = [
            Mock(content="Page 1 content", chunk_index=0, page_number=1),
            Mock(content="Page 2 content", chunk_index=1, page_number=2),
        ]

        # Mock document chunks with embeddings
        mock_doc_chunks = [
            DocumentChunk(
                id="test_doc_chunk_0",
                content="Page 1 content",
                embedding=[0.1, 0.2, 0.3],
                metadata={
                    "document_id": "test_doc",
                    "chunk_index": 0,
                    "page_number": 1,
                },
            ),
            DocumentChunk(
                id="test_doc_chunk_1",
                content="Page 2 content",
                embedding=[0.4, 0.5, 0.6],
                metadata={
                    "document_id": "test_doc",
                    "chunk_index": 1,
                    "page_number": 2,
                },
            ),
        ]

        with patch("pdf_vector_system.pipeline.PDFProcessor") as mock_pdf_class:
            with patch("pdf_vector_system.pipeline.TextProcessor") as mock_text_class:
                with patch(
                    "pdf_vector_system.pipeline.EmbeddingServiceFactory"
                ) as mock_factory:
                    with patch(
                        "pdf_vector_system.pipeline.ChromaDBClient"
                    ) as mock_chroma_class:
                        # Set up mocks
                        mock_pdf_processor = Mock()
                        mock_pdf_processor.extract_text.return_value = mock_pdf_result
                        mock_pdf_class.return_value = mock_pdf_processor

                        mock_text_processor = Mock()
                        mock_text_processor.clean_text.return_value = (
                            "cleaned text",
                            {},
                        )
                        mock_text_processor.chunk_text_with_metadata.return_value = (
                            mock_chunks
                        )
                        mock_text_class.return_value = mock_text_processor

                        mock_embedding_service = Mock()
                        mock_embedding_service.embed_texts.return_value = Mock(
                            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
                        )
                        mock_factory.create_service.return_value = (
                            mock_embedding_service
                        )

                        mock_vector_db = Mock()
                        mock_vector_db.add_documents.return_value = True
                        mock_chroma_class.return_value = mock_vector_db

                        # Test processing
                        pipeline = PDFVectorPipeline(test_config)

                        with patch.object(
                            pipeline, "_create_document_chunks"
                        ) as mock_create_chunks:
                            mock_create_chunks.return_value = mock_doc_chunks

                            result = pipeline.process_pdf(pdf_path)

                            assert isinstance(result, ProcessingResult)
                            assert result.success is True
                            assert result.file_path == str(pdf_path)
                            assert result.chunks_processed == 2
                            assert result.processing_time > 0
                            assert result.error_message is None

                            # Verify method calls
                            mock_pdf_processor.extract_text.assert_called_once_with(
                                pdf_path
                            )
                            mock_text_processor.process_pdf_content.assert_called_once()
                            mock_embedding_service.embed_texts.assert_called_once()
                            mock_vector_db.add_documents.assert_called_once()

    def test_process_pdf_file_not_found(self, test_config):
        """Test PDF processing with non-existent file."""
        with patch("pdf_vector_system.pipeline.PDFProcessor"):
            with patch("pdf_vector_system.pipeline.TextProcessor"):
                with patch("pdf_vector_system.pipeline.EmbeddingServiceFactory"):
                    with patch("pdf_vector_system.pipeline.ChromaDBClient"):
                        pipeline = PDFVectorPipeline(test_config)

                        non_existent_path = Path("non_existent.pdf")
                        result = pipeline.process_pdf(non_existent_path)

                        assert result.success is False
                        assert "does not exist" in result.error_message
                        assert result.chunks_processed == 0

    def test_process_pdf_extraction_error(self, test_config, temp_dir):
        """Test PDF processing with extraction error."""
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF content")

        with patch("pdf_vector_system.pipeline.PDFProcessor") as mock_pdf_class:
            with patch("pdf_vector_system.pipeline.TextProcessor"):
                with patch("pdf_vector_system.pipeline.EmbeddingServiceFactory"):
                    with patch("pdf_vector_system.pipeline.ChromaDBClient"):
                        # Mock PDF processor to raise error
                        mock_pdf_processor = Mock()
                        mock_pdf_processor.extract_text.side_effect = Exception(
                            "PDF extraction failed"
                        )
                        mock_pdf_class.return_value = mock_pdf_processor

                        pipeline = PDFVectorPipeline(test_config)
                        result = pipeline.process_pdf(pdf_path)

                        assert result.success is False
                        assert "PDF extraction failed" in result.error_message
                        assert result.chunks_created == 0

    def test_search_documents_success(self, test_config):
        """Test successful document search."""
        query = SearchQuery(query_text="test query", n_results=5)

        mock_search_results = [
            SearchResult(
                id="chunk_1",
                content="Test content 1",
                score=0.9,
                metadata={"document_id": "doc_1"},
            ),
            SearchResult(
                id="chunk_2",
                content="Test content 2",
                score=0.8,
                metadata={"document_id": "doc_2"},
            ),
        ]

        with patch("pdf_vector_system.pipeline.PDFProcessor"):
            with patch("pdf_vector_system.pipeline.TextProcessor"):
                with patch("pdf_vector_system.pipeline.EmbeddingServiceFactory"):
                    with patch(
                        "pdf_vector_system.pipeline.ChromaDBClient"
                    ) as mock_chroma_class:
                        mock_vector_db = Mock()
                        mock_vector_db.search_documents.return_value = (
                            mock_search_results
                        )
                        mock_chroma_class.return_value = mock_vector_db

                        pipeline = PDFVectorPipeline(test_config)
                        results = pipeline.search_documents(query)

                        assert len(results) == 2
                        assert all(isinstance(r, SearchResult) for r in results)
                        assert results[0].id == "chunk_1"
                        assert results[1].id == "chunk_2"

                        mock_vector_db.search_documents.assert_called_once_with(query)

    def test_search_documents_error(self, test_config):
        """Test document search with error."""
        query = SearchQuery(query_text="test query")

        with patch("pdf_vector_system.pipeline.PDFProcessor"):
            with patch("pdf_vector_system.pipeline.TextProcessor"):
                with patch("pdf_vector_system.pipeline.EmbeddingServiceFactory"):
                    with patch(
                        "pdf_vector_system.pipeline.ChromaDBClient"
                    ) as mock_chroma_class:
                        mock_vector_db = Mock()
                        mock_vector_db.search_documents.side_effect = Exception(
                            "Search failed"
                        )
                        mock_chroma_class.return_value = mock_vector_db

                        pipeline = PDFVectorPipeline(test_config)

                        with pytest.raises(
                            PipelineError, match="Error searching documents"
                        ):
                            pipeline.search_documents(query)

    def test_delete_document_success(self, test_config):
        """Test successful document deletion."""
        with patch("pdf_vector_system.pipeline.PDFProcessor"):
            with patch("pdf_vector_system.pipeline.TextProcessor"):
                with patch("pdf_vector_system.pipeline.EmbeddingServiceFactory"):
                    with patch(
                        "pdf_vector_system.pipeline.ChromaDBClient"
                    ) as mock_chroma_class:
                        mock_vector_db = Mock()
                        mock_vector_db.delete_documents_by_filter.return_value = True
                        mock_chroma_class.return_value = mock_vector_db

                        pipeline = PDFVectorPipeline(test_config)
                        result = pipeline.delete_document("doc_1")

                        assert result is True
                        mock_vector_db.delete_documents_by_filter.assert_called_once_with(
                            {"document_id": "doc_1"}
                        )

    def test_delete_document_error(self, test_config):
        """Test document deletion with error."""
        with patch("pdf_vector_system.pipeline.PDFProcessor"):
            with patch("pdf_vector_system.pipeline.TextProcessor"):
                with patch("pdf_vector_system.pipeline.EmbeddingServiceFactory"):
                    with patch(
                        "pdf_vector_system.pipeline.ChromaDBClient"
                    ) as mock_chroma_class:
                        mock_vector_db = Mock()
                        mock_vector_db.delete_documents_by_filter.side_effect = (
                            Exception("Delete failed")
                        )
                        mock_chroma_class.return_value = mock_vector_db

                        pipeline = PDFVectorPipeline(test_config)

                        with pytest.raises(
                            PipelineError, match="Error deleting document"
                        ):
                            pipeline.delete_document("doc_1")

    def test_get_document_info_success(self, test_config):
        """Test successful document info retrieval."""
        from pdf_vector_system.vector_db.models import DocumentInfo

        mock_doc_info = DocumentInfo(
            document_id="doc_1", chunk_count=5, total_characters=1000, page_count=3
        )

        with patch("pdf_vector_system.pipeline.PDFProcessor"):
            with patch("pdf_vector_system.pipeline.TextProcessor"):
                with patch("pdf_vector_system.pipeline.EmbeddingServiceFactory"):
                    with patch(
                        "pdf_vector_system.pipeline.ChromaDBClient"
                    ) as mock_chroma_class:
                        mock_vector_db = Mock()
                        mock_vector_db.get_document_info.return_value = mock_doc_info
                        mock_chroma_class.return_value = mock_vector_db

                        pipeline = PDFVectorPipeline(test_config)
                        info = pipeline.get_document_info("doc_1")

                        assert info == mock_doc_info
                        mock_vector_db.get_document_info.assert_called_once_with(
                            "doc_1"
                        )

    def test_list_documents(self, test_config):
        """Test listing documents."""
        mock_collections = [
            {"name": "collection_1", "metadata": {"type": "test"}},
            {"name": "collection_2", "metadata": {"type": "prod"}},
        ]

        with patch("pdf_vector_system.pipeline.PDFProcessor"):
            with patch("pdf_vector_system.pipeline.TextProcessor"):
                with patch("pdf_vector_system.pipeline.EmbeddingServiceFactory"):
                    with patch(
                        "pdf_vector_system.pipeline.ChromaDBClient"
                    ) as mock_chroma_class:
                        mock_vector_db = Mock()
                        mock_vector_db.list_collections.return_value = mock_collections
                        mock_chroma_class.return_value = mock_vector_db

                        pipeline = PDFVectorPipeline(test_config)
                        collections = pipeline.list_documents()

                        assert collections == mock_collections
                        mock_vector_db.list_collections.assert_called_once()

    def test_health_check_success(self, test_config):
        """Test successful health check."""
        with patch("pdf_vector_system.pipeline.PDFProcessor"):
            with patch("pdf_vector_system.pipeline.TextProcessor"):
                with patch(
                    "pdf_vector_system.pipeline.EmbeddingServiceFactory"
                ) as mock_factory:
                    with patch(
                        "pdf_vector_system.pipeline.ChromaDBClient"
                    ) as mock_chroma_class:
                        mock_embedding_service = Mock()
                        mock_embedding_service.health_check.return_value = True
                        mock_factory.create_service.return_value = (
                            mock_embedding_service
                        )

                        mock_vector_db = Mock()
                        mock_vector_db.health_check.return_value = True
                        mock_chroma_class.return_value = mock_vector_db

                        pipeline = PDFVectorPipeline(test_config)
                        result = pipeline.health_check()

                        assert result["status"] == "healthy"
                        assert result["embedding_service"] is True
                        assert result["vector_database"] is True

    def test_health_check_failure(self, test_config):
        """Test health check with failures."""
        with patch("pdf_vector_system.pipeline.PDFProcessor"):
            with patch("pdf_vector_system.pipeline.TextProcessor"):
                with patch(
                    "pdf_vector_system.pipeline.EmbeddingServiceFactory"
                ) as mock_factory:
                    with patch(
                        "pdf_vector_system.pipeline.ChromaDBClient"
                    ) as mock_chroma_class:
                        mock_embedding_service = Mock()
                        mock_embedding_service.health_check.return_value = False
                        mock_factory.create_service.return_value = (
                            mock_embedding_service
                        )

                        mock_vector_db = Mock()
                        mock_vector_db.health_check.return_value = True
                        mock_chroma_class.return_value = mock_vector_db

                        pipeline = PDFVectorPipeline(test_config)
                        result = pipeline.health_check()

                        assert result["status"] == "unhealthy"
                        assert result["embedding_service"] is False
                        assert result["vector_database"] is True

    def test_create_document_chunks(self, test_config):
        """Test document chunk creation."""
        mock_text_chunks = [
            Mock(
                content="Chunk 1",
                chunk_index=0,
                page_number=1,
                start_char=0,
                end_char=7,
            ),
            Mock(
                content="Chunk 2",
                chunk_index=1,
                page_number=1,
                start_char=8,
                end_char=15,
            ),
        ]

        mock_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        with patch("pdf_vector_system.pipeline.PDFProcessor"):
            with patch("pdf_vector_system.pipeline.TextProcessor"):
                with patch("pdf_vector_system.pipeline.EmbeddingServiceFactory"):
                    with patch("pdf_vector_system.pipeline.ChromaDBClient"):
                        pipeline = PDFVectorPipeline(test_config)

                        chunks = pipeline._create_document_chunks(
                            "test_doc",
                            mock_text_chunks,
                            mock_embeddings,
                            {"title": "Test Document"},
                        )

                        assert len(chunks) == 2
                        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
                        assert chunks[0].id == "test_doc_chunk_0"
                        assert chunks[0].content == "Chunk 1"
                        assert chunks[0].embedding == [0.1, 0.2, 0.3]
                        assert chunks[1].id == "test_doc_chunk_1"
                        assert chunks[1].content == "Chunk 2"
                        assert chunks[1].embedding == [0.4, 0.5, 0.6]
