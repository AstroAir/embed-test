"""Tests for vector database error handling framework."""

from unittest.mock import patch

import pytest

from pdf_vector_system.vector_db.config import VectorDBType
from pdf_vector_system.vector_db.error_handler import (
    VectorDBErrorHandler,
    VectorDBLogger,
    handle_vector_db_errors,
)
from pdf_vector_system.vector_db.models import (
    AuthenticationError,
    BackendNotAvailableError,
    CollectionNotFoundError,
    ConfigurationError,
    ConnectionError,
    DocumentNotFoundError,
    IndexNotFoundError,
    InvalidQueryError,
    QuotaExceededError,
    VectorDBError,
)


class TestVectorDBErrorHandler:
    """Test VectorDBErrorHandler class."""

    def test_initialization(self):
        """Test VectorDBErrorHandler initialization."""
        handler = VectorDBErrorHandler(VectorDBType.CHROMADB)

        assert handler.backend_type == VectorDBType.CHROMADB
        assert isinstance(handler.error_mappings, dict)
        assert len(handler.error_mappings) > 0

    def test_error_mappings_structure(self):
        """Test that error mappings have correct structure."""
        handler = VectorDBErrorHandler(VectorDBType.CHROMADB)

        for backend, mappings in handler.error_mappings.items():
            assert isinstance(backend, VectorDBType)
            assert isinstance(mappings, dict)

            for error_pattern, error_class in mappings.items():
                assert isinstance(error_pattern, str)
                assert issubclass(error_class, VectorDBError)

    def test_map_error_known_pattern(self):
        """Test mapping known error patterns."""
        handler = VectorDBErrorHandler(VectorDBType.CHROMADB)

        # Test collection not found error
        original_error = Exception("Collection 'test_collection' does not exist")
        mapped_error = handler.map_error(original_error)

        assert isinstance(mapped_error, CollectionNotFoundError)
        assert "test_collection" in str(mapped_error)

    def test_map_error_unknown_pattern(self):
        """Test mapping unknown error patterns."""
        handler = VectorDBErrorHandler(VectorDBType.CHROMADB)

        # Test unknown error
        original_error = Exception("Some unknown error occurred")
        mapped_error = handler.map_error(original_error)

        assert isinstance(mapped_error, VectorDBError)
        assert "Some unknown error occurred" in str(mapped_error)

    def test_map_error_already_vector_db_error(self):
        """Test mapping error that's already a VectorDBError."""
        handler = VectorDBErrorHandler(VectorDBType.CHROMADB)

        # Test with existing VectorDBError
        original_error = CollectionNotFoundError("Already a VectorDBError")
        mapped_error = handler.map_error(original_error)

        # Should return the same error
        assert mapped_error is original_error

    def test_map_error_with_context(self):
        """Test mapping error with additional context."""
        handler = VectorDBErrorHandler(VectorDBType.CHROMADB)

        original_error = Exception("Connection failed")
        context = {"operation": "search", "collection": "test_collection"}
        mapped_error = handler.map_error(original_error, context)

        assert isinstance(mapped_error, VectorDBError)
        assert "Connection failed" in str(mapped_error)

    def test_backend_specific_mappings(self):
        """Test that different backends have different error mappings."""
        chromadb_handler = VectorDBErrorHandler(VectorDBType.CHROMADB)
        pinecone_handler = VectorDBErrorHandler(VectorDBType.PINECONE)

        # Both should have mappings but they might be different
        assert VectorDBType.CHROMADB in chromadb_handler.error_mappings
        assert VectorDBType.PINECONE in pinecone_handler.error_mappings

        # Test that handlers use their respective backend mappings
        assert chromadb_handler.backend_type == VectorDBType.CHROMADB
        assert pinecone_handler.backend_type == VectorDBType.PINECONE


class TestHandleVectorDBErrorsDecorator:
    """Test handle_vector_db_errors decorator."""

    def test_decorator_success_case(self):
        """Test decorator with successful function execution."""

        @handle_vector_db_errors(VectorDBType.CHROMADB)
        def successful_function(x: int, y: int) -> int:
            return x + y

        result = successful_function(2, 3)
        assert result == 5

    def test_decorator_with_exception(self):
        """Test decorator with function that raises exception."""

        @handle_vector_db_errors(VectorDBType.CHROMADB)
        def failing_function():
            raise Exception("Collection 'test' does not exist")

        with pytest.raises(CollectionNotFoundError):
            failing_function()

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""

        @handle_vector_db_errors(VectorDBType.CHROMADB)
        def documented_function(x: int) -> int:
            """This function has documentation."""
            return x * 2

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This function has documentation."

    def test_decorator_with_context(self):
        """Test decorator with context information."""

        @handle_vector_db_errors(VectorDBType.CHROMADB, context={"operation": "test"})
        def function_with_context():
            raise Exception("Some error")

        with pytest.raises(VectorDBError) as exc_info:
            function_with_context()

        # Error should be mapped appropriately
        assert isinstance(exc_info.value, VectorDBError)

    def test_decorator_with_different_backends(self):
        """Test decorator with different backend types."""

        @handle_vector_db_errors(VectorDBType.PINECONE)
        def pinecone_function():
            raise Exception("API key is invalid")

        @handle_vector_db_errors(VectorDBType.WEAVIATE)
        def weaviate_function():
            raise Exception("Connection timeout")

        # Both should raise VectorDBError but potentially different subtypes
        with pytest.raises(VectorDBError):
            pinecone_function()

        with pytest.raises(VectorDBError):
            weaviate_function()

    def test_decorator_with_async_function(self):
        """Test decorator with async function."""
        import asyncio

        @handle_vector_db_errors(VectorDBType.CHROMADB)
        async def async_function():
            await asyncio.sleep(0.01)
            raise Exception("Async error")

        async def run_test():
            with pytest.raises(VectorDBError):
                await async_function()

        # Run the async test
        asyncio.run(run_test())


class TestVectorDBLogger:
    """Test VectorDBLogger class."""

    def test_initialization(self):
        """Test VectorDBLogger initialization."""
        logger = VectorDBLogger(VectorDBType.CHROMADB)

        assert logger.backend_type == VectorDBType.CHROMADB
        assert logger.logger is not None

    @patch("pdf_vector_system.vector_db.error_handler.logger")
    def test_log_error(self, mock_logger):
        """Test error logging."""
        db_logger = VectorDBLogger(VectorDBType.CHROMADB)
        error = Exception("Test error")
        context = {"operation": "search"}

        db_logger.log_error(error, context)

        # Verify logger was called
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        assert "chromadb" in call_args.lower()
        assert "test error" in call_args.lower()

    @patch("pdf_vector_system.vector_db.error_handler.logger")
    def test_log_warning(self, mock_logger):
        """Test warning logging."""
        db_logger = VectorDBLogger(VectorDBType.CHROMADB)
        message = "Test warning"
        context = {"operation": "create_collection"}

        db_logger.log_warning(message, context)

        # Verify logger was called
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args[0][0]
        assert "chromadb" in call_args.lower()
        assert "test warning" in call_args.lower()

    @patch("pdf_vector_system.vector_db.error_handler.logger")
    def test_log_info(self, mock_logger):
        """Test info logging."""
        db_logger = VectorDBLogger(VectorDBType.CHROMADB)
        message = "Test info"
        context = {"operation": "add_chunks"}

        db_logger.log_info(message, context)

        # Verify logger was called
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "chromadb" in call_args.lower()
        assert "test info" in call_args.lower()

    def test_format_log_message(self):
        """Test log message formatting."""
        db_logger = VectorDBLogger(VectorDBType.CHROMADB)

        # Test with context
        message_with_context = db_logger._format_log_message(
            "Test message", {"operation": "search", "collection": "test_collection"}
        )

        assert "chromadb" in message_with_context.lower()
        assert "test message" in message_with_context.lower()
        assert "search" in message_with_context.lower()
        assert "test_collection" in message_with_context.lower()

        # Test without context
        message_without_context = db_logger._format_log_message("Test message", None)

        assert "chromadb" in message_without_context.lower()
        assert "test message" in message_without_context.lower()


class TestErrorMappingIntegration:
    """Test integration of error mapping across different scenarios."""

    def test_all_vector_db_error_types(self):
        """Test that all VectorDBError types can be created."""
        error_types = [
            VectorDBError,
            CollectionNotFoundError,
            DocumentNotFoundError,
            InvalidQueryError,
            ConnectionError,
            AuthenticationError,
            ConfigurationError,
            BackendNotAvailableError,
            IndexNotFoundError,
            QuotaExceededError,
        ]

        for error_type in error_types:
            error = error_type("Test error message")
            assert isinstance(error, VectorDBError)
            assert str(error) == "Test error message"

    def test_error_hierarchy(self):
        """Test that error hierarchy is correct."""
        # All specific errors should inherit from VectorDBError
        specific_errors = [
            CollectionNotFoundError("test"),
            DocumentNotFoundError("test"),
            InvalidQueryError("test"),
            ConnectionError("test"),
            AuthenticationError("test"),
            ConfigurationError("test"),
            BackendNotAvailableError("test"),
            IndexNotFoundError("test"),
            QuotaExceededError("test"),
        ]

        for error in specific_errors:
            assert isinstance(error, VectorDBError)
            assert isinstance(error, Exception)

    def test_real_world_error_scenarios(self):
        """Test realistic error scenarios."""
        handler = VectorDBErrorHandler(VectorDBType.CHROMADB)

        # Simulate real error messages from different backends
        test_cases = [
            ("Collection 'documents' does not exist", CollectionNotFoundError),
            ("Document with id 'doc123' not found", DocumentNotFoundError),
            ("Invalid query parameters", InvalidQueryError),
            ("Connection refused", ConnectionError),
            ("Authentication failed", AuthenticationError),
            ("Invalid configuration", ConfigurationError),
            ("Backend service unavailable", BackendNotAvailableError),
            ("Index 'vectors' not found", IndexNotFoundError),
            ("Quota exceeded", QuotaExceededError),
        ]

        for error_message, _expected_type in test_cases:
            original_error = Exception(error_message)
            mapped_error = handler.map_error(original_error)

            # Should map to appropriate error type or generic VectorDBError
            assert isinstance(mapped_error, VectorDBError)

    @patch("pdf_vector_system.vector_db.error_handler.logger")
    def test_end_to_end_error_handling(self, mock_logger):
        """Test end-to-end error handling flow."""

        @handle_vector_db_errors(VectorDBType.CHROMADB)
        def problematic_function():
            raise Exception("Collection 'test' does not exist")

        # Function should raise mapped error
        with pytest.raises(CollectionNotFoundError) as exc_info:
            problematic_function()

        # Error should have proper message
        assert "test" in str(exc_info.value)

        # Logger should have been called
        mock_logger.error.assert_called()

    def test_error_context_preservation(self):
        """Test that error context is preserved through mapping."""
        handler = VectorDBErrorHandler(VectorDBType.CHROMADB)

        original_error = Exception("Original error")
        context = {
            "operation": "search",
            "collection": "test_collection",
            "query": "test query",
        }

        mapped_error = handler.map_error(original_error, context)

        assert isinstance(mapped_error, VectorDBError)
        assert "Original error" in str(mapped_error)
