"""Tests for PDFVectorPipeline class."""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from vectorflow.core.pipeline import PDFVectorPipeline, PipelineError, ProcessingResult
from vectorflow.core.vector_db.models import DocumentChunk, SearchQuery, SearchResult


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
        with patch("vectorflow.pipeline.PDFProcessor"):
            with patch("vectorflow.pipeline.TextProcessor"):
                with patch("vectorflow.pipeline.EmbeddingServiceFactory"):
                    with patch("vectorflow.pipeline.ChromaDBClient"):
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

        with patch("vectorflow.pipeline.PDFProcessor") as mock_pdf_class:
            with patch("vectorflow.pipeline.TextProcessor") as mock_text_class:
                with patch(
                    "vectorflow.pipeline.EmbeddingServiceFactory"
                ) as mock_factory:
                    with patch(
                        "vectorflow.pipeline.ChromaDBClient"
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
        with patch("vectorflow.pipeline.PDFProcessor"):
            with patch("vectorflow.pipeline.TextProcessor"):
                with patch("vectorflow.pipeline.EmbeddingServiceFactory"):
                    with patch("vectorflow.pipeline.ChromaDBClient"):
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

        with patch("vectorflow.pipeline.PDFProcessor") as mock_pdf_class:
            with patch("vectorflow.pipeline.TextProcessor"):
                with patch("vectorflow.pipeline.EmbeddingServiceFactory"):
                    with patch("vectorflow.pipeline.ChromaDBClient"):
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

        with patch("vectorflow.pipeline.PDFProcessor"):
            with patch("vectorflow.pipeline.TextProcessor"):
                with patch("vectorflow.pipeline.EmbeddingServiceFactory"):
                    with patch(
                        "vectorflow.pipeline.ChromaDBClient"
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

        with patch("vectorflow.pipeline.PDFProcessor"):
            with patch("vectorflow.pipeline.TextProcessor"):
                with patch("vectorflow.pipeline.EmbeddingServiceFactory"):
                    with patch(
                        "vectorflow.pipeline.ChromaDBClient"
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
        with patch("vectorflow.pipeline.PDFProcessor"):
            with patch("vectorflow.pipeline.TextProcessor"):
                with patch("vectorflow.pipeline.EmbeddingServiceFactory"):
                    with patch(
                        "vectorflow.pipeline.ChromaDBClient"
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
        with patch("vectorflow.pipeline.PDFProcessor"):
            with patch("vectorflow.pipeline.TextProcessor"):
                with patch("vectorflow.pipeline.EmbeddingServiceFactory"):
                    with patch(
                        "vectorflow.pipeline.ChromaDBClient"
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
        from vectorflow.core.vector_db.models import DocumentInfo

        mock_doc_info = DocumentInfo(
            document_id="doc_1", chunk_count=5, total_characters=1000, page_count=3
        )

        with patch("vectorflow.pipeline.PDFProcessor"):
            with patch("vectorflow.pipeline.TextProcessor"):
                with patch("vectorflow.pipeline.EmbeddingServiceFactory"):
                    with patch(
                        "vectorflow.pipeline.ChromaDBClient"
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

        with patch("vectorflow.pipeline.PDFProcessor"):
            with patch("vectorflow.pipeline.TextProcessor"):
                with patch("vectorflow.pipeline.EmbeddingServiceFactory"):
                    with patch(
                        "vectorflow.pipeline.ChromaDBClient"
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
        with patch("vectorflow.pipeline.PDFProcessor"):
            with patch("vectorflow.pipeline.TextProcessor"):
                with patch(
                    "vectorflow.pipeline.EmbeddingServiceFactory"
                ) as mock_factory:
                    with patch(
                        "vectorflow.pipeline.ChromaDBClient"
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
        with patch("vectorflow.pipeline.PDFProcessor"):
            with patch("vectorflow.pipeline.TextProcessor"):
                with patch(
                    "vectorflow.pipeline.EmbeddingServiceFactory"
                ) as mock_factory:
                    with patch(
                        "vectorflow.pipeline.ChromaDBClient"
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

        with patch("vectorflow.pipeline.PDFProcessor"):
            with patch("vectorflow.pipeline.TextProcessor"):
                with patch("vectorflow.pipeline.EmbeddingServiceFactory"):
                    with patch("vectorflow.pipeline.ChromaDBClient"):
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


class TestPDFVectorPipelineCurrentAPI:
    """Test PDFVectorPipeline with current API methods."""

    @pytest.fixture
    def pipeline(self, test_config):
        """Create a pipeline with mocked components."""
        with (
            patch("vectorflow.pipeline.PDFProcessor"),
            patch("vectorflow.pipeline.TextProcessor"),
            patch("vectorflow.pipeline.EmbeddingServiceFactory"),
            patch("vectorflow.pipeline.VectorDBFactory"),
        ):
            pipeline = PDFVectorPipeline(test_config)
            # Mock the vector DB
            pipeline.vector_db = Mock()
            pipeline.embedding_service = Mock()
            pipeline.batch_processor = Mock()
            yield pipeline

    def test_search_with_document_filter(self, pipeline):
        """Test search with document_id filter."""
        # Mock vector DB search
        expected_results = [
            SearchResult(
                id="test_1",
                content="Test content",
                score=0.95,
                metadata={"document_id": "doc1"},
            )
        ]
        pipeline.vector_db.search = Mock(return_value=expected_results)

        results = pipeline.search("test query", document_id="doc1")

        assert len(results) == 1
        # Verify search was called with correct query including where clause
        call_args = pipeline.vector_db.search.call_args[0][0]
        assert call_args.where == {"document_id": "doc1"}

    def test_search_with_page_filter(self, pipeline):
        """Test search with page_number filter."""

        # Mock vector DB search
        expected_results = [
            SearchResult(
                id="test_1",
                content="Test content",
                score=0.95,
                metadata={"page_number": 5},
            )
        ]
        pipeline.vector_db.search = Mock(return_value=expected_results)

        results = pipeline.search("test query", page_number=5)

        assert len(results) == 1
        call_args = pipeline.vector_db.search.call_args[0][0]
        assert call_args.where == {"page_number": 5}

    def test_search_with_multiple_filters(self, pipeline):
        """Test search with both document_id and page_number filters."""

        # Mock vector DB search
        expected_results = []
        pipeline.vector_db.search = Mock(return_value=expected_results)

        results = pipeline.search("test query", document_id="doc1", page_number=3)

        assert len(results) == 0
        call_args = pipeline.vector_db.search.call_args[0][0]
        assert call_args.where == {"document_id": "doc1", "page_number": 3}

    def test_search_custom_n_results(self, pipeline):
        """Test search with custom number of results."""

        # Mock vector DB search
        pipeline.vector_db.search = Mock(return_value=[])

        pipeline.search("test query", n_results=25)

        call_args = pipeline.vector_db.search.call_args[0][0]
        assert call_args.n_results == 25

    def test_get_collection_stats_success(self, pipeline):
        """Test successful collection statistics retrieval."""

        # Mock collection info and count
        mock_info = Mock()
        mock_info.name = "test_collection"
        mock_info.metadata = {"version": "1.0"}
        pipeline.vector_db.get_collection_info = Mock(return_value=mock_info)
        pipeline.vector_db.count_chunks = Mock(return_value=150)

        stats = pipeline.get_collection_stats()

        assert stats["collection_name"] == "test_collection"
        assert stats["total_chunks"] == 150
        assert stats["metadata"] == {"version": "1.0"}
        assert "error" not in stats

    def test_get_collection_stats_error(self, pipeline):
        """Test collection statistics retrieval with error."""

        # Mock error
        pipeline.vector_db.get_collection_info = Mock(
            side_effect=Exception("Connection failed")
        )

        stats = pipeline.get_collection_stats()

        assert stats["collection_name"] == "unknown"
        assert stats["total_chunks"] == 0
        assert stats["metadata"] == {}
        assert "error" in stats
        assert "Connection failed" in stats["error"]

    def test_get_vector_db_info(self, pipeline):
        """Test vector database backend info retrieval."""

        # Mock backend info
        backend_info = {
            "backend": "chromadb",
            "version": "0.4.0",
            "features": ["search", "filter"],
        }
        pipeline.vector_db.get_backend_info = Mock(return_value=backend_info)

        info = pipeline.get_vector_db_info()

        assert info == backend_info

    def test_get_collection_info_success(self, pipeline):
        """Test successful collection info retrieval."""

        # Mock collection info
        mock_info = Mock()
        mock_info.name = "test_collection"
        mock_info.count = 50
        mock_info.metadata = {"created": "2024-01-01"}
        pipeline.vector_db.get_collection_info = Mock(return_value=mock_info)

        info = pipeline.get_collection_info()

        assert info["name"] == "test_collection"
        assert info["count"] == 50
        assert info["metadata"] == {"created": "2024-01-01"}
        assert "error" not in info

    def test_get_collection_info_error(self, pipeline):
        """Test collection info retrieval with error."""

        # Mock error
        pipeline.vector_db.get_collection_info = Mock(
            side_effect=Exception("Database unavailable")
        )

        info = pipeline.get_collection_info()

        assert info["name"] == "unknown"
        assert info["count"] == 0
        assert info["metadata"] == {}
        assert "error" in info
        assert "Database unavailable" in info["error"]

    def test_get_documents_success(self, pipeline):
        """Test successful documents retrieval."""

        # Mock search results with multiple documents
        mock_results = [
            SearchResult(
                id="doc1_chunk_0",
                content="Content 1",
                score=1.0,
                metadata={
                    "document_id": "doc1",
                    "filename": "file1.pdf",
                    "created_at": "2024-01-01",
                },
            ),
            SearchResult(
                id="doc1_chunk_1",
                content="Content 2",
                score=1.0,
                metadata={
                    "document_id": "doc1",
                    "filename": "file1.pdf",
                    "created_at": "2024-01-01",
                },
            ),
            SearchResult(
                id="doc2_chunk_0",
                content="Content 3",
                score=1.0,
                metadata={
                    "document_id": "doc2",
                    "filename": "file2.pdf",
                    "created_at": "2024-01-02",
                },
            ),
        ]
        # Ensure the mock returns the results regardless of arguments
        pipeline.vector_db.search_by_metadata.return_value = mock_results

        documents = pipeline.get_documents()

        # Verify the mock was called
        pipeline.vector_db.search_by_metadata.assert_called_once_with({}, limit=10000)

        assert len(documents) == 2
        # Check doc1
        doc1 = next(d for d in documents if d["document_id"] == "doc1")
        assert doc1["chunks_count"] == 2
        assert doc1["filename"] == "file1.pdf"
        # Check doc2
        doc2 = next(d for d in documents if d["document_id"] == "doc2")
        assert doc2["chunks_count"] == 1
        assert doc2["filename"] == "file2.pdf"

    def test_get_documents_error(self, pipeline):
        """Test documents retrieval with error."""

        # Mock error
        pipeline.vector_db.search_by_metadata = Mock(
            side_effect=Exception("Query failed")
        )

        documents = pipeline.get_documents()

        assert documents == []

    def test_delete_document_current_api(self, pipeline):
        """Test delete_document with current API."""

        # Mock delete
        pipeline.vector_db.delete_document = Mock(return_value=5)

        deleted_count = pipeline.delete_document("doc1")

        assert deleted_count == 5
        pipeline.vector_db.delete_document.assert_called_once_with("doc1")

    def test_get_document_info_current_api(self, pipeline):
        """Test get_document_info with current API."""
        from vectorflow.core.vector_db.models import DocumentInfo

        # Mock document info
        mock_info = DocumentInfo(
            document_id="doc1",
            filename="test.pdf",
            chunk_count=10,
            total_characters=5000,
            created_at="2024-01-01",
        )
        pipeline.vector_db.get_document_info = Mock(return_value=mock_info)

        info = pipeline.get_document_info("doc1")

        assert isinstance(info, dict)
        assert info["document_id"] == "doc1"
        assert info["chunk_count"] == 10

    def test_health_check_current_api(self, pipeline):
        """Test health_check with current API (returns dict)."""

        # Mock health checks
        pipeline.embedding_service.health_check = Mock(return_value=True)
        pipeline.vector_db.health_check = Mock(return_value=True)

        result = pipeline.health_check()

        assert result["embedding_service"] is True
        assert result["vector_database"] is True
        assert result["pipeline"] is True

    def test_health_check_with_failures(self, pipeline):
        """Test health_check with component failures."""

        # Mock health check failures
        pipeline.embedding_service.health_check = Mock(return_value=False)
        pipeline.vector_db.health_check = Mock(return_value=False)

        result = pipeline.health_check()

        assert result["embedding_service"] is False
        assert result["vector_database"] is False
        assert result["pipeline"] is True

    def test_process_pdf_without_progress(self, pipeline, temp_dir):
        """Test PDF processing without progress bar."""

        # Create test PDF
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("dummy")

        # Mock components
        pipeline.pdf_processor.extract_text = Mock(
            return_value={
                "text_content": {1: "Test content"},
                "metadata": {},
                "page_count": 1,
                "total_characters": 12,
                "file_name": "test.pdf",
            }
        )

        mock_chunks = [
            Mock(
                content="Test content",
                chunk_index=0,
                start_char=0,
                end_char=12,
                length=12,
                source_info={"document_id": "test"},
            )
        ]
        pipeline.text_processor.chunk_text_with_metadata = Mock(
            return_value=mock_chunks
        )
        pipeline.text_processor.clean_text = Mock(return_value=("Test content", {}))

        mock_embedding_result = Mock(
            embeddings=[[0.1, 0.2, 0.3]], embedding_dimension=3
        )
        pipeline.batch_processor.process_texts = Mock(
            return_value=mock_embedding_result
        )

        pipeline.vector_db.add_chunks = Mock()

        # Process without progress
        result = pipeline.process_pdf(pdf_path, show_progress=False)

        assert result.success is True
        assert result.chunks_processed == 1

    def test_process_pdf_no_clean_text(self, pipeline, temp_dir):
        """Test PDF processing without text cleaning."""

        # Create test PDF
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("dummy")

        # Mock components
        pipeline.pdf_processor.extract_text = Mock(
            return_value={
                "text_content": {1: "Test   content  "},
                "metadata": {},
                "page_count": 1,
                "total_characters": 16,
                "file_name": "test.pdf",
            }
        )

        mock_chunks = [
            Mock(
                content="Test   content  ",
                chunk_index=0,
                start_char=0,
                end_char=16,
                length=16,
                source_info={"document_id": "test"},
            )
        ]
        pipeline.text_processor.chunk_text_with_metadata = Mock(
            return_value=mock_chunks
        )

        mock_embedding_result = Mock(
            embeddings=[[0.1, 0.2, 0.3]], embedding_dimension=3
        )
        pipeline.batch_processor.process_texts = Mock(
            return_value=mock_embedding_result
        )

        pipeline.vector_db.add_chunks = Mock()

        # Process without cleaning
        result = pipeline.process_pdf(pdf_path, clean_text=False)

        # Verify clean_text was NOT called
        pipeline.text_processor.clean_text.assert_not_called()
        assert result.success is True

    def test_process_pdf_custom_document_id(self, pipeline, temp_dir):
        """Test PDF processing with custom document ID."""

        # Create test PDF
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("dummy")

        # Mock components
        pipeline.pdf_processor.extract_text = Mock(
            return_value={
                "text_content": {1: "Test content"},
                "metadata": {},
                "page_count": 1,
                "total_characters": 12,
                "file_name": "test.pdf",
            }
        )

        mock_chunks = [
            Mock(
                content="Test content",
                chunk_index=0,
                start_char=0,
                end_char=12,
                length=12,
                source_info={"document_id": "custom_id"},
            )
        ]
        pipeline.text_processor.chunk_text_with_metadata = Mock(
            return_value=mock_chunks
        )
        pipeline.text_processor.clean_text = Mock(return_value=("Test content", {}))

        mock_embedding_result = Mock(
            embeddings=[[0.1, 0.2, 0.3]], embedding_dimension=3
        )
        pipeline.batch_processor.process_texts = Mock(
            return_value=mock_embedding_result
        )

        pipeline.vector_db.add_chunks = Mock()

        # Process with custom ID
        result = pipeline.process_pdf(pdf_path, document_id="custom_id")

        assert result.document_id == "custom_id"

    def test_process_pdf_empty_chunks(self, pipeline, temp_dir):
        """Test PDF processing when no chunks are generated."""

        # Create test PDF
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("dummy")

        # Mock components to return empty content
        pipeline.pdf_processor.extract_text = Mock(
            return_value={
                "text_content": {1: "   "},  # Only whitespace
                "metadata": {},
                "page_count": 1,
                "total_characters": 3,
                "file_name": "test.pdf",
            }
        )

        # Process should fail with no chunks
        result = pipeline.process_pdf(pdf_path)

        assert result.success is False
        assert "No text chunks were generated" in result.error_message

    def test_process_pdf_multiple_pages(self, pipeline, temp_dir):
        """Test PDF processing with multiple pages."""

        # Create test PDF
        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("dummy")

        # Mock components with multiple pages
        pipeline.pdf_processor.extract_text = Mock(
            return_value={
                "text_content": {1: "Page 1 content", 2: "Page 2 content"},
                "metadata": {},
                "page_count": 2,
                "total_characters": 30,
                "file_name": "test.pdf",
            }
        )

        mock_chunks_page1 = [
            Mock(
                content="Page 1 content",
                chunk_index=0,
                start_char=0,
                end_char=14,
                length=14,
                source_info={"document_id": "test", "page_number": 1},
            )
        ]
        mock_chunks_page2 = [
            Mock(
                content="Page 2 content",
                chunk_index=0,
                start_char=0,
                end_char=14,
                length=14,
                source_info={"document_id": "test", "page_number": 2},
            )
        ]

        # Mock chunk_text_with_metadata to return different chunks for different pages
        pipeline.text_processor.chunk_text_with_metadata = Mock(
            side_effect=[mock_chunks_page1, mock_chunks_page2]
        )
        pipeline.text_processor.clean_text = Mock(
            side_effect=[
                ("Page 1 content", {}),
                ("Page 2 content", {}),
            ]
        )

        mock_embedding_result = Mock(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], embedding_dimension=3
        )
        pipeline.batch_processor.process_texts = Mock(
            return_value=mock_embedding_result
        )

        pipeline.vector_db.add_chunks = Mock()

        # Process PDF
        result = pipeline.process_pdf(pdf_path)

        assert result.success is True
        assert result.chunks_processed == 2
        assert result.metadata["page_count"] == 2

    def test_create_document_chunks_mismatch(self, pipeline):
        """Test _create_document_chunks with mismatched lengths."""

        text_chunks = [
            Mock(
                content="Chunk 1",
                chunk_index=0,
                length=7,
                start_char=0,
                end_char=7,
                source_info={"document_id": "test"},
            )
        ]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]  # 2 embeddings for 1 chunk

        # Should raise ValueError
        with pytest.raises(ValueError, match="Mismatch between text chunks"):
            pipeline._create_document_chunks(text_chunks, embeddings)

    def test_pipeline_error_exception(self):
        """Test PipelineError can be raised and caught."""
        with pytest.raises(PipelineError):
            raise PipelineError("Test error")

    def test_processing_result_with_metadata(self):
        """Test ProcessingResult to_dict with metadata."""
        metadata = {"key": "value", "count": 42}
        result = ProcessingResult(
            document_id="test_doc",
            file_path="/path/to/file.pdf",
            success=True,
            chunks_processed=5,
            embeddings_generated=5,
            chunks_stored=5,
            processing_time=1.5,
            metadata=metadata,
        )

        result_dict = result.to_dict()
        assert result_dict["metadata"] == metadata
        assert result_dict["chunks_per_second"] == 5 / 1.5

    def test_processing_result_to_dict_no_metadata(self):
        """Test ProcessingResult to_dict without metadata."""
        result = ProcessingResult(
            document_id="test_doc",
            file_path="/path/to/file.pdf",
            success=False,
            chunks_processed=0,
            embeddings_generated=0,
            chunks_stored=0,
            processing_time=0.5,
            error_message="Test error",
        )

        result_dict = result.to_dict()
        assert result_dict["metadata"] == {}
        assert result_dict["error_message"] == "Test error"
