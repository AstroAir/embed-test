"""Integration tests for component interactions."""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tests.mocks.chromadb_mocks import MockChromaDBClient
from tests.mocks.embedding_mocks import MockEmbeddingService
from tests.mocks.pdf_mocks import create_mock_pdf_document
from vectorflow.core.config.settings import (
    ChromaDBConfig,
    Config,
    EmbeddingConfig,
    PDFConfig,
    TextProcessingConfig,
)
from vectorflow.core.pipeline import PDFVectorPipeline
from vectorflow.core.vector_db.models import SearchQuery


@pytest.mark.integration
class TestPDFToVectorIntegration:
    """Test integration from PDF processing to vector storage."""

    def test_pdf_to_chunks_integration(self, temp_dir):
        """Test PDF processing to text chunks integration."""
        # Create test configuration
        config = Config(
            pdf=PDFConfig(max_file_size_mb=10),
            text_processing=TextProcessingConfig(chunk_size=100, chunk_overlap=20),
            embedding=EmbeddingConfig(batch_size=2),
            chroma_db=ChromaDBConfig(persist_directory=temp_dir / "chroma_test"),
        )

        # Create a test PDF file
        pdf_path = temp_dir / "integration_test.pdf"
        pdf_path.write_text("Mock PDF content for integration testing")

        # Mock PDF extraction result
        {
            "file_path": str(pdf_path),
            "file_name": "integration_test.pdf",
            "page_count": 2,
            "text_content": {
                1: "This is the first page content with some meaningful text for testing.",
                2: "This is the second page content with additional text for comprehensive testing.",
            },
            "total_characters": 120,
            "metadata": {"title": "Integration Test Document"},
            "extraction_timestamp": time.time(),
        }

        with patch("vectorflow.pdf.processor.fitz.open") as mock_fitz:
            # Mock PDF document
            mock_doc = create_mock_pdf_document(
                page_count=2,
                content_per_page=[
                    "This is the first page content with some meaningful text for testing.",
                    "This is the second page content with additional text for comprehensive testing.",
                ],
            )
            mock_fitz.return_value = mock_doc

            with (
                patch(
                    "vectorflow.embeddings.factory.SentenceTransformersService"
                ) as mock_st_service,
                patch(
                    "vectorflow.vector_db.chromadb_client.chromadb.PersistentClient"
                ) as mock_chroma_client,
            ):
                # Set up embedding service mock
                mock_embedding_service = MockEmbeddingService(
                    "test-model", embedding_dim=384
                )
                mock_st_service.return_value = mock_embedding_service

                # Set up ChromaDB client mock
                mock_db_client = MockChromaDBClient()
                mock_chroma_client.return_value = mock_db_client._client

                # Create pipeline
                pipeline = PDFVectorPipeline(config)

                # Process PDF
                result = pipeline.process_pdf(pdf_path)

                # Verify processing result
                assert result.success is True
                assert result.chunks_created > 0
                assert result.processing_time > 0
                assert result.document_id is not None

                # Verify that chunks were created and stored
                assert mock_embedding_service.call_count > 0
                assert len(mock_embedding_service.last_texts) > 0

    def test_end_to_end_search_integration(self, temp_dir):
        """Test end-to-end integration from PDF processing to search."""
        # Create test configuration
        config = Config(
            pdf=PDFConfig(max_file_size_mb=10),
            text_processing=TextProcessingConfig(chunk_size=50, chunk_overlap=10),
            embedding=EmbeddingConfig(batch_size=2),
            chroma_db=ChromaDBConfig(persist_directory=temp_dir / "chroma_test"),
        )

        # Create test PDF files
        pdf1_path = temp_dir / "doc1.pdf"
        pdf2_path = temp_dir / "doc2.pdf"
        pdf1_path.write_text("Mock PDF 1")
        pdf2_path.write_text("Mock PDF 2")

        with patch("vectorflow.pdf.processor.fitz.open") as mock_fitz:
            with patch(
                "vectorflow.embeddings.factory.SentenceTransformersService"
            ) as mock_st_service:
                with patch(
                    "vectorflow.vector_db.chromadb_client.chromadb.PersistentClient"
                ) as mock_chroma_client:
                    # Set up mocks
                    def mock_fitz_side_effect(path):
                        if "doc1.pdf" in str(path):
                            return create_mock_pdf_document(
                                page_count=1,
                                content_per_page=[
                                    "Machine learning is a subset of artificial intelligence."
                                ],
                            )
                        return create_mock_pdf_document(
                            page_count=1,
                            content_per_page=[
                                "Deep learning uses neural networks for pattern recognition."
                            ],
                        )

                    mock_fitz.side_effect = mock_fitz_side_effect

                    mock_embedding_service = MockEmbeddingService(
                        "test-model", embedding_dim=384
                    )
                    mock_st_service.return_value = mock_embedding_service

                    mock_db_client = MockChromaDBClient()
                    mock_chroma_client.return_value = mock_db_client._client

                    # Create pipeline
                    pipeline = PDFVectorPipeline(config)

                    # Process both PDFs
                    result1 = pipeline.process_pdf(pdf1_path, document_id="doc1")
                    result2 = pipeline.process_pdf(pdf2_path, document_id="doc2")

                    assert result1.success is True
                    assert result2.success is True

                    # Perform search
                    query = SearchQuery(query_text="machine learning", n_results=5)
                    search_results = pipeline.search_documents(query)

                    # Verify search results
                    assert len(search_results) > 0
                    assert any(
                        "machine learning" in result.content.lower()
                        or "artificial intelligence" in result.content.lower()
                        for result in search_results
                    )


@pytest.mark.integration
class TestConfigurationIntegration:
    """Test configuration integration across components."""

    def test_config_propagation(self, temp_dir):
        """Test that configuration is properly propagated to all components."""
        # Create custom configuration
        config = Config(
            pdf=PDFConfig(max_file_size_mb=25, timeout_seconds=120),
            text_processing=TextProcessingConfig(chunk_size=200, chunk_overlap=40),
            embedding=EmbeddingConfig(batch_size=8),
            chroma_db=ChromaDBConfig(
                persist_directory=temp_dir / "custom_chroma",
                collection_name="custom_collection",
                max_results=25,
            ),
            debug=True,
            max_workers=2,
        )

        with (
            patch("vectorflow.pdf.processor.fitz.open"),
            patch(
                "vectorflow.embeddings.factory.SentenceTransformersService"
            ) as mock_st_service,
            patch(
                "vectorflow.vector_db.chromadb_client.chromadb.PersistentClient"
            ) as mock_chroma_client,
        ):
            mock_embedding_service = MockEmbeddingService("test-model")
            mock_st_service.return_value = mock_embedding_service

            mock_db_client = MockChromaDBClient()
            mock_chroma_client.return_value = mock_db_client._client

            # Create pipeline
            pipeline = PDFVectorPipeline(config)

            # Verify configuration propagation
            assert pipeline.config == config
            assert pipeline.pdf_processor.config.max_file_size_mb == 25
            assert pipeline.pdf_processor.config.timeout_seconds == 120
            assert pipeline.text_processor.config.chunk_size == 200
            assert pipeline.text_processor.config.chunk_overlap == 40
            assert pipeline.vector_db.config.collection_name == "custom_collection"
            assert pipeline.vector_db.config.max_results == 25

    def test_environment_variable_integration(self, temp_dir, env_vars):
        """Test environment variable integration."""
        # Environment variables are set in env_vars fixture
        config = Config()

        # Verify environment variables were loaded
        assert config.debug is True
        assert config.max_workers == 6
        assert config.embedding.openai_api_key == "test-api-key"


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling integration across components."""

    def test_pdf_processing_error_propagation(self, temp_dir):
        """Test error propagation from PDF processing."""
        config = Config(
            chroma_db=ChromaDBConfig(persist_directory=temp_dir / "chroma_test")
        )

        # Create a test file that will cause processing error
        pdf_path = temp_dir / "error_test.pdf"
        pdf_path.write_text("Mock PDF content")

        with patch("vectorflow.pdf.processor.fitz.open") as mock_fitz:
            with patch(
                "vectorflow.embeddings.factory.SentenceTransformersService"
            ) as mock_st_service:
                with patch(
                    "vectorflow.vector_db.chromadb_client.chromadb.PersistentClient"
                ) as mock_chroma_client:
                    # Mock PDF processor to raise error
                    mock_fitz.side_effect = Exception("PDF processing error")

                    mock_embedding_service = MockEmbeddingService("test-model")
                    mock_st_service.return_value = mock_embedding_service

                    mock_db_client = MockChromaDBClient()
                    mock_chroma_client.return_value = mock_db_client._client

                    # Create pipeline
                    pipeline = PDFVectorPipeline(config)

                    # Process PDF - should handle error gracefully
                    result = pipeline.process_pdf(pdf_path)

                    assert result.success is False
                    assert "PDF processing error" in result.error_message
                    assert result.chunks_created == 0

    def test_embedding_service_error_propagation(self, temp_dir):
        """Test error propagation from embedding service."""
        config = Config(
            chroma_db=ChromaDBConfig(persist_directory=temp_dir / "chroma_test")
        )

        pdf_path = temp_dir / "test.pdf"
        pdf_path.write_text("Mock PDF content")

        with patch("vectorflow.pdf.processor.fitz.open") as mock_fitz:
            with patch(
                "vectorflow.embeddings.factory.SentenceTransformersService"
            ) as mock_st_service:
                with patch(
                    "vectorflow.vector_db.chromadb_client.chromadb.PersistentClient"
                ) as mock_chroma_client:
                    # Set up PDF processing to succeed
                    mock_doc = create_mock_pdf_document(
                        page_count=1,
                        content_per_page=["Test content for embedding error test."],
                    )
                    mock_fitz.return_value = mock_doc

                    # Mock embedding service to raise error
                    mock_embedding_service = Mock()
                    mock_embedding_service.embed_texts.side_effect = Exception(
                        "Embedding service error"
                    )
                    mock_st_service.return_value = mock_embedding_service

                    mock_db_client = MockChromaDBClient()
                    mock_chroma_client.return_value = mock_db_client._client

                    # Create pipeline
                    pipeline = PDFVectorPipeline(config)

                    # Process PDF - should handle embedding error gracefully
                    result = pipeline.process_pdf(pdf_path)

                    assert result.success is False
                    assert "Embedding service error" in result.error_message
                    assert result.chunks_created == 0


@pytest.mark.integration
class TestPerformanceIntegration:
    """Test performance characteristics of integrated components."""

    def test_batch_processing_performance(self, temp_dir):
        """Test batch processing performance integration."""
        config = Config(
            text_processing=TextProcessingConfig(chunk_size=100, chunk_overlap=20),
            embedding=EmbeddingConfig(batch_size=4),
            chroma_db=ChromaDBConfig(persist_directory=temp_dir / "chroma_test"),
        )

        pdf_path = temp_dir / "large_doc.pdf"
        pdf_path.write_text("Mock large PDF content")

        # Create content that will generate multiple chunks
        large_content = " ".join(
            [f"This is sentence {i} in a large document." for i in range(50)]
        )

        with patch("vectorflow.pdf.processor.fitz.open") as mock_fitz:
            with patch(
                "vectorflow.embeddings.factory.SentenceTransformersService"
            ) as mock_st_service:
                with patch(
                    "vectorflow.vector_db.chromadb_client.chromadb.PersistentClient"
                ) as mock_chroma_client:
                    # Set up PDF processing
                    mock_doc = create_mock_pdf_document(
                        page_count=1, content_per_page=[large_content]
                    )
                    mock_fitz.return_value = mock_doc

                    # Set up embedding service with batch processing
                    mock_embedding_service = MockEmbeddingService(
                        "test-model", embedding_dim=384
                    )
                    mock_st_service.return_value = mock_embedding_service

                    mock_db_client = MockChromaDBClient()
                    mock_chroma_client.return_value = mock_db_client._client

                    # Create pipeline
                    pipeline = PDFVectorPipeline(config)

                    # Process PDF and measure performance
                    start_time = time.time()
                    result = pipeline.process_pdf(pdf_path)
                    end_time = time.time()

                    # Verify successful processing
                    assert result.success is True
                    assert result.chunks_created > 5  # Should create multiple chunks
                    assert result.processing_time > 0

                    # Verify batch processing was used
                    assert mock_embedding_service.call_count > 0

                    # Performance should be reasonable
                    total_time = end_time - start_time
                    assert total_time < 10.0  # Should complete within 10 seconds

    def test_concurrent_processing_integration(self, temp_dir):
        """Test concurrent processing capabilities."""
        config = Config(
            max_workers=2,
            chroma_db=ChromaDBConfig(persist_directory=temp_dir / "chroma_test"),
        )

        # Create multiple test files
        pdf_paths = []
        for i in range(3):
            pdf_path = temp_dir / f"concurrent_test_{i}.pdf"
            pdf_path.write_text(f"Mock PDF content {i}")
            pdf_paths.append(pdf_path)

        with patch("vectorflow.pdf.processor.fitz.open") as mock_fitz:
            with patch(
                "vectorflow.embeddings.factory.SentenceTransformersService"
            ) as mock_st_service:
                with patch(
                    "vectorflow.vector_db.chromadb_client.chromadb.PersistentClient"
                ) as mock_chroma_client:
                    # Set up mocks
                    def mock_fitz_side_effect(path):
                        return create_mock_pdf_document(
                            page_count=1,
                            content_per_page=[f"Content for {Path(path).name}"],
                        )

                    mock_fitz.side_effect = mock_fitz_side_effect

                    mock_embedding_service = MockEmbeddingService(
                        "test-model", embedding_dim=384
                    )
                    mock_st_service.return_value = mock_embedding_service

                    mock_db_client = MockChromaDBClient()
                    mock_chroma_client.return_value = mock_db_client._client

                    # Create pipeline
                    pipeline = PDFVectorPipeline(config)

                    # Process multiple files
                    results = []
                    for i, pdf_path in enumerate(pdf_paths):
                        result = pipeline.process_pdf(pdf_path, document_id=f"doc_{i}")
                        results.append(result)

                    # Verify all processing succeeded
                    assert all(result.success for result in results)
                    assert len(results) == 3

                    # Verify different documents were created
                    document_ids = [result.document_id for result in results]
                    assert len(set(document_ids)) == 3  # All unique


@pytest.mark.integration
class TestHealthCheckIntegration:
    """Test health check integration across all components."""

    def test_system_health_check(self, temp_dir):
        """Test comprehensive system health check."""
        config = Config(
            chroma_db=ChromaDBConfig(persist_directory=temp_dir / "chroma_test")
        )

        with (
            patch(
                "vectorflow.embeddings.factory.SentenceTransformersService"
            ) as mock_st_service,
            patch(
                "vectorflow.vector_db.chromadb_client.chromadb.PersistentClient"
            ) as mock_chroma_client,
        ):
            # Set up healthy mocks
            mock_embedding_service = MockEmbeddingService("test-model")
            mock_st_service.return_value = mock_embedding_service

            mock_db_client = MockChromaDBClient()
            mock_chroma_client.return_value = mock_db_client._client

            # Create pipeline
            pipeline = PDFVectorPipeline(config)

            # Perform health check
            health_result = pipeline.health_check()

            # Verify health check results
            assert health_result["status"] == "healthy"
            assert health_result["embedding_service"] is True
            assert health_result["vector_database"] is True
            assert "timestamp" in health_result

    def test_unhealthy_system_detection(self, temp_dir):
        """Test detection of unhealthy system components."""
        config = Config(
            chroma_db=ChromaDBConfig(persist_directory=temp_dir / "chroma_test")
        )

        with (
            patch(
                "vectorflow.embeddings.factory.SentenceTransformersService"
            ) as mock_st_service,
            patch(
                "vectorflow.vector_db.chromadb_client.chromadb.PersistentClient"
            ) as mock_chroma_client,
        ):
            # Set up unhealthy embedding service
            mock_embedding_service = Mock()
            mock_embedding_service.health_check.return_value = False
            mock_st_service.return_value = mock_embedding_service

            # Set up healthy database
            mock_db_client = MockChromaDBClient()
            mock_chroma_client.return_value = mock_db_client._client

            # Create pipeline
            pipeline = PDFVectorPipeline(config)

            # Perform health check
            health_result = pipeline.health_check()

            # Verify unhealthy status is detected
            assert health_result["status"] == "unhealthy"
            assert health_result["embedding_service"] is False
            assert health_result["vector_database"] is True
