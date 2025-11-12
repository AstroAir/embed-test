"""Universal error handling and logging framework for vector databases."""

import functools
import traceback
from typing import Any, Callable, Optional, TypeVar

from loguru import logger

from pdf_vector_system.core.vector_db.config import VectorDBType
from pdf_vector_system.core.vector_db.models import (
    AuthenticationError,
    CollectionNotFoundError,
    ConfigurationError,
    ConnectionError,
    DocumentNotFoundError,
    IndexNotFoundError,
    InvalidQueryError,
    QuotaExceededError,
    VectorDBError,
)

F = TypeVar("F", bound=Callable[..., Any])


class VectorDBErrorHandler:
    """Universal error handler for vector database operations."""

    def __init__(self, backend_type: Optional[VectorDBType] = None) -> None:
        """
        Initialize the error handler.

        Args:
            backend_type: The vector database backend type
        """
        self.backend_type = backend_type
        self.error_mappings = self.ERROR_MAPPINGS

    # Mapping of common error patterns to our exception types
    ERROR_MAPPINGS: dict[VectorDBType, dict[str, type[VectorDBError]]] = {
        VectorDBType.CHROMADB: {
            "collection": CollectionNotFoundError,
            "not found": DocumentNotFoundError,
            "invalid": InvalidQueryError,
            "connection": ConnectionError,
            "auth": AuthenticationError,
        },
        VectorDBType.PINECONE: {
            "not found": IndexNotFoundError,
            "unauthorized": AuthenticationError,
            "forbidden": AuthenticationError,
            "quota": QuotaExceededError,
            "rate limit": QuotaExceededError,
            "connection": ConnectionError,
            "invalid": InvalidQueryError,
        },
        VectorDBType.WEAVIATE: {
            "class not found": CollectionNotFoundError,
            "object not found": DocumentNotFoundError,
            "unauthorized": AuthenticationError,
            "connection": ConnectionError,
            "invalid": InvalidQueryError,
        },
        VectorDBType.QDRANT: {
            "collection not found": CollectionNotFoundError,
            "point not found": DocumentNotFoundError,
            "unauthorized": AuthenticationError,
            "connection": ConnectionError,
            "invalid": InvalidQueryError,
        },
        VectorDBType.MILVUS: {
            "collection not exist": CollectionNotFoundError,
            "entity not found": DocumentNotFoundError,
            "connection": ConnectionError,
            "invalid": InvalidQueryError,
        },
    }

    def map_error(
        self,
        error: Exception,
        operation: str = "unknown",
        context: Optional[dict[str, Any]] = None,
    ) -> VectorDBError:
        """
        Map a backend-specific error to a VectorDBError.

        Args:
            error: Original exception
            operation: Name of the operation that failed
            context: Additional context information

        Returns:
            Appropriate VectorDBError subclass
        """
        if isinstance(error, VectorDBError):
            return error

        return self.handle_error(
            error,
            self.backend_type or VectorDBType.CHROMADB,
            operation,
            context,
        )

    @classmethod
    def handle_error(
        cls,
        error: Exception,
        backend_type: VectorDBType,
        operation: str,
        context: Optional[dict[str, Any]] = None,
    ) -> VectorDBError:
        """
        Convert backend-specific errors to universal VectorDBError types.

        Args:
            error: Original exception
            backend_type: Type of vector database backend
            operation: Name of the operation that failed
            context: Additional context information

        Returns:
            Appropriate VectorDBError subclass
        """
        error_message = str(error).lower()
        # Handle both enum and string backend types
        if isinstance(backend_type, str):
            backend_name = backend_type
            # Try to convert string to enum for mappings
            try:
                backend_enum = VectorDBType(backend_type)
            except (ValueError, KeyError):
                backend_enum = None
        else:
            backend_name = backend_type.value
            backend_enum = backend_type

        # Get error mappings for this backend
        mappings = cls.ERROR_MAPPINGS.get(backend_enum, {}) if backend_enum else {}

        # Find matching error type
        for pattern, error_class in mappings.items():
            if pattern in error_message:
                message = f"Operation '{operation}' failed: {error!s}"
                if context:
                    message += f" (Context: {context})"
                return error_class(message, backend_name, error)

        # Default to generic VectorDBError
        message = f"Operation '{operation}' failed: {error!s}"
        if context:
            message += f" (Context: {context})"
        return VectorDBError(message, backend_name, error)

    @classmethod
    def log_error(
        cls,
        error: VectorDBError,
        operation: str,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log error with appropriate level and context.

        Args:
            error: VectorDBError to log
            operation: Name of the operation that failed
            context: Additional context information
        """
        log_context = {
            "operation": operation,
            "backend": error.backend,
            "error_type": type(error).__name__,
        }

        if context:
            log_context.update(context)

        if error.original_error:
            log_context["original_error"] = str(error.original_error)
            log_context["traceback"] = traceback.format_exc()

        # Log with appropriate level based on error type
        if isinstance(error, (AuthenticationError, ConfigurationError)):
            logger.error(f"Critical error in {operation}: {error}", **log_context)
        elif isinstance(
            error, (CollectionNotFoundError, DocumentNotFoundError, IndexNotFoundError)
        ):
            logger.warning(f"Resource not found in {operation}: {error}", **log_context)
        elif isinstance(error, (QuotaExceededError, ConnectionError)):
            logger.warning(f"Service issue in {operation}: {error}", **log_context)
        else:
            logger.error(f"Error in {operation}: {error}", **log_context)


def handle_vector_db_errors(
    backend_type: Optional[VectorDBType] = None,
    operation: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    reraise: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to handle vector database errors uniformly.

    Can be used with or without arguments:
    - @handle_vector_db_errors  # Uses defaults
    - @handle_vector_db_errors(backend_type=VectorDBType.CHROMADB)

    Args:
        backend_type: Type of vector database backend (defaults to CHROMADB)
        operation: Name of the operation (auto-detected if None)
        context: Additional context information
        reraise: Whether to reraise the converted error

    Returns:
        Decorated function
    """
    # Support usage without parentheses: @handle_vector_db_errors
    if callable(backend_type):
        func = backend_type
        backend_type = VectorDBType.CHROMADB

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            op_name = operation or func.__name__
            try:
                return func(*args, **kwargs)
            except VectorDBError:
                # Already a VectorDBError, just reraise
                raise
            except Exception as e:
                # Convert to VectorDBError
                vector_error = VectorDBErrorHandler.handle_error(
                    e, backend_type, op_name, context
                )
                VectorDBErrorHandler.log_error(vector_error, op_name, context)

                if reraise:
                    raise vector_error from e
                return None

        return wrapper

    # Normal usage with parentheses: @handle_vector_db_errors(...)
    backend = backend_type or VectorDBType.CHROMADB

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            op_name = operation or func.__name__
            try:
                return func(*args, **kwargs)
            except VectorDBError:
                # Already a VectorDBError, just reraise
                raise
            except Exception as e:
                # Convert to VectorDBError
                vector_error = VectorDBErrorHandler.handle_error(
                    e, backend, op_name, context
                )
                VectorDBErrorHandler.log_error(vector_error, op_name, context)

                if reraise:
                    raise vector_error from e
                return None

        return wrapper  # type: ignore

    return decorator


class VectorDBLogger:
    """Enhanced logger for vector database operations."""

    def __init__(self, backend_type: Optional[VectorDBType] = None) -> None:
        """
        Initialize the logger.

        Args:
            backend_type: The vector database backend type
        """
        self.backend_type = backend_type
        self.logger = logger

    def log_error(
        self, error: Exception, context: Optional[dict[str, Any]] = None
    ) -> None:
        """Log an error with context."""
        message = self._format_log_message(str(error), context)
        self.logger.error(message)

    def log_warning(
        self, message: str, context: Optional[dict[str, Any]] = None
    ) -> None:
        """Log a warning with context."""
        formatted_message = self._format_log_message(message, context)
        self.logger.warning(formatted_message)

    def log_info(self, message: str, context: Optional[dict[str, Any]] = None) -> None:
        """Log an info message with context."""
        formatted_message = self._format_log_message(message, context)
        self.logger.info(formatted_message)

    def _format_log_message(
        self, message: str, context: Optional[dict[str, Any]] = None
    ) -> str:
        """Format a log message with backend and context information."""
        backend_name = self.backend_type.value if self.backend_type else "unknown"
        formatted = f"[{backend_name}] {message}"

        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            formatted += f" ({context_str})"

        return formatted

    @staticmethod
    def log_operation_start(
        operation: str, backend: str, context: Optional[dict[str, Any]] = None
    ) -> None:
        """Log the start of a vector database operation."""
        log_context = {"backend": backend, "operation": operation}
        if context:
            log_context.update(context)

        logger.debug(f"Starting {operation}", **log_context)

    @staticmethod
    def log_operation_success(
        operation: str,
        backend: str,
        duration: Optional[float] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log successful completion of a vector database operation."""
        log_context: dict[str, Any] = {"backend": backend, "operation": operation}
        if duration is not None:
            log_context["duration_ms"] = round(duration * 1000, 2)
        if context:
            log_context.update(context)

        message = f"Completed {operation}"
        if duration is not None:
            message += f" in {duration:.3f}s"

        logger.info(message, **log_context)

    @staticmethod
    def log_performance_warning(
        operation: str,
        backend: str,
        duration: float,
        threshold: float = 5.0,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log performance warning for slow operations."""
        if duration > threshold:
            log_context = {
                "backend": backend,
                "operation": operation,
                "duration_ms": round(duration * 1000, 2),
                "threshold_ms": round(threshold * 1000, 2),
            }
            if context:
                log_context.update(context)

            logger.warning(
                f"Slow operation: {operation} took {duration:.3f}s (threshold: {threshold:.3f}s)",
                **log_context,
            )

    @staticmethod
    def log_backend_info(backend: str, info: dict[str, Any]) -> None:
        """Log backend information and capabilities."""
        logger.info(f"Backend info for {backend}", backend=backend, **info)

    @staticmethod
    def log_health_check(
        backend: str, healthy: bool, details: Optional[dict[str, Any]] = None
    ) -> None:
        """Log health check results."""
        log_context = {"backend": backend, "healthy": healthy}
        if details:
            log_context.update(details)

        if healthy:
            logger.info(f"Health check passed for {backend}", **log_context)
        else:
            logger.warning(f"Health check failed for {backend}", **log_context)


def with_error_handling(backend_type: VectorDBType):
    """
    Class decorator to add error handling to all methods of a vector database client.

    Args:
        backend_type: Type of vector database backend

    Returns:
        Decorated class
    """

    def class_decorator(cls: type) -> type:
        # Get all methods that don't start with underscore
        for attr_name in dir(cls):
            if not attr_name.startswith("_"):
                attr = getattr(cls, attr_name)
                if callable(attr):
                    # Apply error handling decorator
                    decorated_method = handle_vector_db_errors(
                        backend_type=backend_type,
                        operation=f"{cls.__name__}.{attr_name}",
                    )(attr)
                    setattr(cls, attr_name, decorated_method)

        return cls

    return class_decorator


# Convenience functions for common error scenarios
def raise_collection_not_found(collection_name: str, backend: str) -> None:
    """Raise CollectionNotFoundError with standard message."""
    raise CollectionNotFoundError(
        f"Collection '{collection_name}' not found", backend=backend
    )


def raise_document_not_found(document_id: str, backend: str) -> None:
    """Raise DocumentNotFoundError with standard message."""
    raise DocumentNotFoundError(f"Document '{document_id}' not found", backend=backend)


def raise_invalid_query(reason: str, backend: str) -> None:
    """Raise InvalidQueryError with standard message."""
    raise InvalidQueryError(f"Invalid query: {reason}", backend=backend)


def raise_connection_error(backend: str, details: str) -> None:
    """Raise ConnectionError with standard message."""
    raise ConnectionError(f"Failed to connect to {backend}: {details}", backend=backend)


def raise_authentication_error(backend: str, details: str) -> None:
    """Raise AuthenticationError with standard message."""
    raise AuthenticationError(
        f"Authentication failed for {backend}: {details}", backend=backend
    )
