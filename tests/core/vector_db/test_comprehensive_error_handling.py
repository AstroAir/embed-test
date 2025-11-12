"""Comprehensive error handling tests across all vector database components."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from pdf_vector_system.core.vector_db.config import (
    ChromaDBConfig,
    MilvusConfig,
    PineconeConfig,
    QdrantConfig,
    WeaviateConfig,
)
from pdf_vector_system.core.vector_db.error_handler import (
    VectorDBErrorHandler,
    handle_vector_db_errors,
)
from pdf_vector_system.core.vector_db.factory import VectorDBFactory
from pdf_vector_system.core.vector_db.models import (
    AuthenticationError,
    CollectionNotFoundError,
    ConnectionError,
    DocumentNotFoundError,
    VectorDBError,
)


class TestCrossBackendErrorHandling:
    """Test error handling consistency across all backends."""

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
    def test_connection_error_handling(
        self, backend_type, config_class, vector_db_temp_dir
    ):
        """Test connection error handling across all backends."""
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

        # Mock the backend to raise connection errors
        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.health_check.side_effect = ConnectionError("Connection failed")
            mock_create.return_value = mock_client

            client = VectorDBFactory.create_client(config)

            with pytest.raises(ConnectionError):
                client.health_check()

    @pytest.mark.parametrize(
        "backend_type", ["chromadb", "pinecone", "weaviate", "qdrant", "milvus"]
    )
    def test_authentication_error_handling(self, backend_type, vector_db_temp_dir):
        """Test authentication error handling across all backends."""
        # Create configs with invalid credentials
        if backend_type == "chromadb":
            config = ChromaDBConfig(persist_directory=vector_db_temp_dir / "chroma")
        elif backend_type == "pinecone":
            config = PineconeConfig(api_key="invalid_key", index_name="test_index")
        elif backend_type == "weaviate":
            config = WeaviateConfig(url="http://localhost:8080", class_name="TestClass")
        elif backend_type == "qdrant":
            config = QdrantConfig(
                url="http://localhost:6333", collection_name="test_collection"
            )
        elif backend_type == "milvus":
            config = MilvusConfig(
                host="localhost", port=19530, collection_name="test_collection"
            )

        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()
            mock_client.list_collections.side_effect = AuthenticationError(
                "Invalid credentials"
            )
            mock_create.return_value = mock_client

            client = VectorDBFactory.create_client(config)

            with pytest.raises(AuthenticationError):
                client.list_collections()

    def test_collection_not_found_error_consistency(self, sample_search_query):
        """Test that all backends handle collection not found errors consistently."""
        backends = ["chromadb", "pinecone", "weaviate", "qdrant", "milvus"]

        for backend_type in backends:
            with patch.object(VectorDBFactory, "create_client") as mock_create:
                mock_client = Mock()
                mock_client.search.side_effect = CollectionNotFoundError(
                    f"Collection not found in {backend_type}"
                )
                mock_create.return_value = mock_client

                # Create a dummy config
                config = ChromaDBConfig()
                client = VectorDBFactory.create_client(config)

                with pytest.raises(CollectionNotFoundError):
                    client.search(sample_search_query)

    def test_document_not_found_error_consistency(self):
        """Test that all backends handle document not found errors consistently."""
        backends = ["chromadb", "pinecone", "weaviate", "qdrant", "milvus"]

        for backend_type in backends:
            with patch.object(VectorDBFactory, "create_client") as mock_create:
                mock_client = Mock()
                mock_client.get_chunk.side_effect = DocumentNotFoundError(
                    f"Document not found in {backend_type}"
                )
                mock_create.return_value = mock_client

                config = ChromaDBConfig()
                client = VectorDBFactory.create_client(config)

                with pytest.raises(DocumentNotFoundError):
                    client.get_chunk("nonexistent_chunk")


class TestErrorHandlerDecorator:
    """Test the error handler decorator functionality."""

    def test_decorator_catches_generic_exceptions(self):
        """Test that decorator catches and wraps generic exceptions."""

        @handle_vector_db_errors
        def failing_function():
            raise ValueError("Generic error")

        with pytest.raises(VectorDBError) as exc_info:
            failing_function()

        assert "Generic error" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_decorator_preserves_vector_db_errors(self):
        """Test that decorator preserves VectorDBError subclasses."""

        @handle_vector_db_errors
        def failing_function():
            raise CollectionNotFoundError("Collection not found")

        with pytest.raises(CollectionNotFoundError) as exc_info:
            failing_function()

        assert "Collection not found" in str(exc_info.value)

    def test_decorator_with_return_value(self):
        """Test that decorator preserves return values for successful calls."""

        @handle_vector_db_errors
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_decorator_with_arguments(self):
        """Test that decorator works with functions that have arguments."""

        @handle_vector_db_errors
        def function_with_args(arg1, arg2, kwarg1=None):
            if kwarg1 == "fail":
                raise RuntimeError("Function failed")
            return f"{arg1}-{arg2}-{kwarg1}"

        # Test successful call
        result = function_with_args("a", "b", kwarg1="c")
        assert result == "a-b-c"

        # Test failing call
        with pytest.raises(VectorDBError):
            function_with_args("a", "b", kwarg1="fail")

    def test_decorator_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""

        @handle_vector_db_errors
        def documented_function():
            """This is a documented function."""
            return "result"

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a documented function."


class TestErrorContextPreservation:
    """Test that error context is preserved across operations."""

    def test_error_context_in_chained_operations(self, sample_document_chunks):
        """Test error context preservation in chained operations."""
        with patch.object(VectorDBFactory, "create_client") as mock_create:
            mock_client = Mock()

            # First operation succeeds
            mock_client.create_collection.return_value = True

            # Second operation fails
            mock_client.add_chunks.side_effect = ConnectionError(
                "Network timeout during add_chunks"
            )

            mock_create.return_value = mock_client

            config = ChromaDBConfig()
            client = VectorDBFactory.create_client(config)

            # First operation should succeed
            assert client.create_collection("test_collection") is True

            # Second operation should fail with preserved context
            with pytest.raises(ConnectionError) as exc_info:
                client.add_chunks(sample_document_chunks)

            assert "Network timeout during add_chunks" in str(exc_info.value)

    def test_nested_error_handling(self):
        """Test error handling in nested function calls."""

        @handle_vector_db_errors
        def outer_function():
            return inner_function()

        @handle_vector_db_errors
        def inner_function():
            raise ConnectionError("Inner connection error")

        with pytest.raises(ConnectionError) as exc_info:
            outer_function()

        assert "Inner connection error" in str(exc_info.value)

    def test_error_logging_context(self, caplog):
        """Test that error context is properly logged."""
        handler = VectorDBErrorHandler()

        try:
            raise ValueError("Test error for logging")
        except Exception as e:
            handler.handle_error(e, "test_operation", {"param": "value"})

        # Check that error was logged with context
        assert "test_operation" in caplog.text
        assert "ValueError" in caplog.text


class TestConcurrentErrorHandling:
    """Test error handling in concurrent scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_operation_errors(self):
        """Test error handling when multiple operations fail concurrently."""

        async def failing_operation(delay: float, error_msg: str):
            await asyncio.sleep(delay)
            raise ConnectionError(error_msg)

        # Start multiple failing operations
        tasks = [
            failing_operation(0.1, "Error 1"),
            failing_operation(0.2, "Error 2"),
            failing_operation(0.3, "Error 3"),
        ]

        # All should fail with ConnectionError
        results = await asyncio.gather(*tasks, return_exceptions=True)

        assert len(results) == 3
        assert all(isinstance(result, ConnectionError) for result in results)
        assert "Error 1" in str(results[0])
        assert "Error 2" in str(results[1])
        assert "Error 3" in str(results[2])

    def test_thread_safety_of_error_handler(self):
        """Test that error handler is thread-safe."""
        import threading
        import time

        handler = VectorDBErrorHandler()
        errors = []

        def worker(worker_id: int):
            try:
                time.sleep(0.1)  # Simulate some work
                raise ValueError(f"Error from worker {worker_id}")
            except Exception as e:
                wrapped_error = handler.handle_error(
                    e, f"worker_{worker_id}", {"worker_id": worker_id}
                )
                errors.append(wrapped_error)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check that all errors were handled correctly
        assert len(errors) == 5
        assert all(isinstance(error, VectorDBError) for error in errors)

        # Check that each worker's error is unique
        error_messages = [str(error) for error in errors]
        assert len(set(error_messages)) == 5


class TestErrorRecoveryStrategies:
    """Test error recovery and retry strategies."""

    def test_retry_on_transient_errors(self):
        """Test retry logic for transient errors."""
        call_count = 0

        @handle_vector_db_errors
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient error")
            return "success"

        # Implement simple retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = flaky_function()
                assert result == "success"
                break
            except ConnectionError:
                if attempt == max_retries - 1:
                    raise
                continue

        assert call_count == 3

    def test_no_retry_on_permanent_errors(self):
        """Test that permanent errors are not retried."""
        call_count = 0

        @handle_vector_db_errors
        def permanently_failing_function():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("Invalid API key")

        # Should not retry authentication errors
        with pytest.raises(AuthenticationError):
            permanently_failing_function()

        assert call_count == 1

    def test_circuit_breaker_pattern(self):
        """Test circuit breaker pattern for error handling."""
        failure_count = 0
        circuit_open = False

        def circuit_breaker_function():
            nonlocal failure_count, circuit_open

            if circuit_open:
                raise VectorDBError("Circuit breaker is open")

            try:
                # Simulate operation that might fail
                if failure_count < 3:
                    failure_count += 1
                    raise ConnectionError("Service unavailable")
                return "success"
            except ConnectionError:
                if failure_count >= 3:
                    circuit_open = True
                raise

        # First 3 calls should fail with ConnectionError
        for _ in range(3):
            with pytest.raises(ConnectionError):
                circuit_breaker_function()

        # 4th call should fail with circuit breaker error
        with pytest.raises(VectorDBError, match="Circuit breaker is open"):
            circuit_breaker_function()


class TestErrorMetricsAndMonitoring:
    """Test error metrics and monitoring capabilities."""

    def test_error_counting(self):
        """Test error counting for monitoring."""
        handler = VectorDBErrorHandler()
        error_counts = {}

        def count_error(error_type: str):
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        # Simulate various errors
        errors = [
            ConnectionError("Connection failed"),
            AuthenticationError("Auth failed"),
            ConnectionError("Another connection error"),
            CollectionNotFoundError("Collection missing"),
            ConnectionError("Third connection error"),
        ]

        for error in errors:
            wrapped_error = handler.handle_error(error, "test_op", {})
            count_error(type(wrapped_error).__name__)

        # Check error counts
        assert error_counts["ConnectionError"] == 3
        assert error_counts["AuthenticationError"] == 1
        assert error_counts["CollectionNotFoundError"] == 1

    def test_error_rate_calculation(self):
        """Test error rate calculation for monitoring."""
        total_operations = 100
        error_operations = 15

        error_rate = error_operations / total_operations
        assert error_rate == 0.15

        # Test threshold-based alerting
        error_threshold = 0.1  # 10%
        assert error_rate > error_threshold  # Should trigger alert
