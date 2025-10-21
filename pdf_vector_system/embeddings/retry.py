"""Enhanced retry mechanisms for embedding services."""

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from pdf_vector_system.utils.logging import LoggerMixin


class RetryStrategy(str, Enum):
    """Available retry strategies."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


class FailureType(str, Enum):
    """Types of failures that can trigger retries."""

    RATE_LIMIT = "rate_limit"
    NETWORK_ERROR = "network_error"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    QUOTA_EXCEEDED = "quota_exceeded"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    jitter_range: float = 0.1
    backoff_multiplier: float = 2.0

    # Failure-specific configurations
    failure_strategies: dict[FailureType, "RetryConfig"] = field(default_factory=dict)

    # Circuit breaker settings
    circuit_breaker_enabled: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout: float = 30.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.base_delay <= 0:
            raise ValueError("base_delay must be positive")
        if self.max_delay <= 0:
            raise ValueError("max_delay must be positive")
        if self.backoff_multiplier <= 1:
            raise ValueError("backoff_multiplier must be greater than 1")
        if not 0 <= self.jitter_range <= 1:
            raise ValueError("jitter_range must be between 0 and 1")


@dataclass
class RetryAttempt:
    """Information about a retry attempt."""

    attempt_number: int
    delay: float
    failure_type: FailureType
    exception: Exception
    timestamp: float = field(default_factory=time.time)


class CircuitBreakerState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""

    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    total_requests: int = 0
    total_failures: int = 0


class RetryableError(Exception):
    """Base class for retryable errors."""

    def __init__(self, message: str, failure_type: FailureType = FailureType.UNKNOWN):
        super().__init__(message)
        self.failure_type = failure_type


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""


T = TypeVar("T")


class RetryHandler(LoggerMixin):
    """Enhanced retry handler with circuit breaker and failure-specific strategies."""

    def __init__(self, config: RetryConfig, name: str = "RetryHandler"):
        """
        Initialize retry handler.

        Args:
            config: Retry configuration
            name: Name for logging purposes
        """
        self.config = config
        self.name = name
        self.circuit_breaker_stats = CircuitBreakerStats()
        self._fibonacci_cache: list[int] = [1, 1]

        self.logger.info(f"Initialized {name} with {config.max_retries} max retries")

    def execute(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            Exception: If all retries are exhausted
        """
        if self._is_circuit_breaker_open():
            raise CircuitBreakerOpenError(
                f"Circuit breaker is open for {self.name}. "
                f"Last failure: {self.circuit_breaker_stats.last_failure_time}"
            )

        last_exception: Optional[Exception] = None
        retry_attempts: list[RetryAttempt] = []

        for attempt in range(self.config.max_retries + 1):
            try:
                self.circuit_breaker_stats.total_requests += 1
                result = func(*args, **kwargs)

                # Success - reset circuit breaker
                self._record_success()

                if retry_attempts:
                    self.logger.info(
                        f"{self.name} succeeded after {len(retry_attempts)} retries"
                    )

                return result

            except Exception as e:
                last_exception = e
                failure_type = self._classify_failure(e)

                # Record failure for circuit breaker
                self._record_failure()

                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt, failure_type)

                    retry_attempt = RetryAttempt(
                        attempt_number=attempt + 1,
                        delay=delay,
                        failure_type=failure_type,
                        exception=e,
                    )
                    retry_attempts.append(retry_attempt)

                    self.logger.warning(
                        f"{self.name} attempt {attempt + 1} failed ({failure_type}): {e!s}. "
                        f"Retrying in {delay:.2f}s"
                    )

                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"{self.name} failed after {self.config.max_retries + 1} attempts. "
                        f"Final error: {e!s}"
                    )

        # All retries exhausted
        if last_exception:
            raise last_exception
        raise RuntimeError(f"{self.name} failed with unknown error")

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self.config.circuit_breaker_enabled:
            return False

        stats = self.circuit_breaker_stats

        if stats.state == CircuitBreakerState.CLOSED:
            return False
        if stats.state == CircuitBreakerState.OPEN:
            # Check if recovery timeout has passed
            if (
                stats.last_failure_time
                and time.time() - stats.last_failure_time
                >= self.config.circuit_breaker_recovery_timeout
            ):
                stats.state = CircuitBreakerState.HALF_OPEN
                self.logger.info(f"Circuit breaker for {self.name} moved to HALF_OPEN")
                return False
            return True
        # HALF_OPEN
        return False

    def _record_success(self) -> None:
        """Record successful execution."""
        stats = self.circuit_breaker_stats
        stats.last_success_time = time.time()

        if stats.state == CircuitBreakerState.HALF_OPEN:
            stats.state = CircuitBreakerState.CLOSED
            stats.failure_count = 0
            self.logger.info(f"Circuit breaker for {self.name} moved to CLOSED")

    def _record_failure(self) -> None:
        """Record failed execution."""
        stats = self.circuit_breaker_stats
        stats.failure_count += 1
        stats.total_failures += 1
        stats.last_failure_time = time.time()

        if (
            stats.state == CircuitBreakerState.CLOSED
            and stats.failure_count >= self.config.circuit_breaker_failure_threshold
        ):
            stats.state = CircuitBreakerState.OPEN
            self.logger.warning(
                f"Circuit breaker for {self.name} moved to OPEN after "
                f"{stats.failure_count} failures"
            )

    def _classify_failure(self, exception: Exception) -> FailureType:
        """
        Classify the type of failure.

        Args:
            exception: The exception that occurred

        Returns:
            FailureType classification
        """
        if isinstance(exception, RetryableError):
            return exception.failure_type

        # Common classification patterns
        exception_str = str(exception).lower()
        exception_type = type(exception).__name__.lower()

        if "rate limit" in exception_str or "429" in exception_str:
            return FailureType.RATE_LIMIT
        if "timeout" in exception_str or "timeout" in exception_type:
            return FailureType.TIMEOUT
        if "network" in exception_str or "connection" in exception_str:
            return FailureType.NETWORK_ERROR
        if "401" in exception_str or "403" in exception_str or "auth" in exception_str:
            return FailureType.AUTHENTICATION
        if "quota" in exception_str or "limit" in exception_str:
            return FailureType.QUOTA_EXCEEDED
        if any(code in exception_str for code in ["500", "502", "503", "504"]):
            return FailureType.SERVER_ERROR
        return FailureType.UNKNOWN

    def _calculate_delay(self, attempt: int, failure_type: FailureType) -> float:
        """
        Calculate delay for retry attempt.

        Args:
            attempt: Current attempt number (0-based)
            failure_type: Type of failure

        Returns:
            Delay in seconds
        """
        # Check for failure-specific configuration
        if failure_type in self.config.failure_strategies:
            config = self.config.failure_strategies[failure_type]
        else:
            config = self.config

        if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.backoff_multiplier**attempt)
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * (attempt + 1)
        elif config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay
        elif config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = config.base_delay * self._get_fibonacci(attempt + 1)
        else:
            # Fallback to fixed delay for unknown strategies
            delay = config.base_delay  # type: ignore[unreachable]

        # Apply maximum delay limit
        delay = min(delay, config.max_delay)

        # Apply jitter if enabled
        if config.jitter:
            jitter_amount = delay * config.jitter_range
            delay += random.uniform(-jitter_amount, jitter_amount)
            delay = max(0, delay)  # Ensure non-negative

        return delay

    def _get_fibonacci(self, n: int) -> int:
        """Get nth Fibonacci number (cached)."""
        while len(self._fibonacci_cache) <= n:
            next_fib = self._fibonacci_cache[-1] + self._fibonacci_cache[-2]
            self._fibonacci_cache.append(next_fib)
        return self._fibonacci_cache[n]

    def get_stats(self) -> dict[str, Any]:
        """Get retry handler statistics."""
        return {
            "name": self.name,
            "circuit_breaker_state": self.circuit_breaker_stats.state,
            "total_requests": self.circuit_breaker_stats.total_requests,
            "total_failures": self.circuit_breaker_stats.total_failures,
            "failure_rate": (
                self.circuit_breaker_stats.total_failures
                / max(1, self.circuit_breaker_stats.total_requests)
            ),
            "last_failure_time": self.circuit_breaker_stats.last_failure_time,
            "last_success_time": self.circuit_breaker_stats.last_success_time,
        }

    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker to closed state."""
        self.circuit_breaker_stats = CircuitBreakerStats()
        self.logger.info(f"Circuit breaker for {self.name} manually reset")


def with_retry(
    config: RetryConfig, name: Optional[str] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for adding retry logic to functions.

    Args:
        config: Retry configuration
        name: Optional name for the retry handler

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        handler_name = name or f"{func.__module__}.{func.__name__}"
        retry_handler = RetryHandler(config, handler_name)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry_handler.execute(func, *args, **kwargs)

        # Attach retry handler for access to stats
        wrapper.retry_handler = retry_handler  # type: ignore

        return wrapper

    return decorator


class RetryMonitor(LoggerMixin):
    """Monitor and aggregate retry statistics across multiple handlers."""

    def __init__(self) -> None:
        """Initialize retry monitor."""
        self.handlers: dict[str, RetryHandler] = {}
        self.global_stats = {
            "total_requests": 0,
            "total_failures": 0,
            "total_retries": 0,
            "circuit_breakers_open": 0,
        }

    def register_handler(self, handler: RetryHandler) -> None:
        """Register a retry handler for monitoring."""
        self.handlers[handler.name] = handler
        self.logger.debug(f"Registered retry handler: {handler.name}")

    def get_global_stats(self) -> dict[str, Any]:
        """Get aggregated statistics across all handlers."""
        stats: dict[str, Any] = self.global_stats.copy()

        # Aggregate from all handlers
        for handler in self.handlers.values():
            handler_stats = handler.get_stats()
            stats["total_requests"] += handler_stats["total_requests"]
            stats["total_failures"] += handler_stats["total_failures"]

            if handler_stats["circuit_breaker_state"] == CircuitBreakerState.OPEN:
                stats["circuit_breakers_open"] += 1

        # Calculate derived metrics
        if stats["total_requests"] > 0:
            stats["global_failure_rate"] = float(stats["total_failures"]) / float(
                stats["total_requests"]
            )
        else:
            stats["global_failure_rate"] = 0.0

        stats["active_handlers"] = len(self.handlers)

        return stats

    def get_handler_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all registered handlers."""
        return {name: handler.get_stats() for name, handler in self.handlers.items()}

    def reset_all_circuit_breakers(self) -> None:
        """Reset all circuit breakers to closed state."""
        for handler in self.handlers.values():
            handler.reset_circuit_breaker()
        self.logger.info("Reset all circuit breakers")


# Global retry monitor instance
retry_monitor = RetryMonitor()


class ProviderCircuitBreaker(LoggerMixin):
    """Circuit breaker specifically designed for embedding providers."""

    def __init__(
        self,
        provider_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
    ):
        """
        Initialize provider circuit breaker.

        Args:
            provider_name: Name of the provider
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            success_threshold: Number of successes needed to close circuit
        """
        self.provider_name = provider_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_success_time: Optional[float] = None

        self.logger.info(
            f"Initialized circuit breaker for {provider_name} "
            f"(failure_threshold={failure_threshold}, recovery_timeout={recovery_timeout}s)"
        )

    def can_execute(self) -> bool:
        """Check if requests can be executed through this provider."""
        current_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            return True
        if self.state == CircuitBreakerState.OPEN:
            if (
                self.last_failure_time
                and current_time - self.last_failure_time >= self.recovery_timeout
            ):
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.logger.info(
                    f"Circuit breaker for {self.provider_name} moved to HALF_OPEN"
                )
                return True
            return False
        # HALF_OPEN
        return True

    def record_success(self) -> None:
        """Record a successful operation."""
        self.last_success_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.logger.info(
                    f"Circuit breaker for {self.provider_name} moved to CLOSED"
                )
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.last_failure_time = time.time()
        self.failure_count += 1

        if self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(
                    f"Circuit breaker for {self.provider_name} moved to OPEN "
                    f"after {self.failure_count} failures"
                )
        elif self.state == CircuitBreakerState.HALF_OPEN:
            # Failed during recovery, go back to OPEN
            self.state = CircuitBreakerState.OPEN
            self.success_count = 0
            self.logger.warning(
                f"Circuit breaker for {self.provider_name} moved back to OPEN "
                f"(failed during recovery)"
            )

    def force_open(self) -> None:
        """Manually force circuit breaker to OPEN state."""
        self.state = CircuitBreakerState.OPEN
        self.last_failure_time = time.time()
        self.logger.warning(
            f"Circuit breaker for {self.provider_name} manually forced to OPEN"
        )

    def force_close(self) -> None:
        """Manually force circuit breaker to CLOSED state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.logger.info(
            f"Circuit breaker for {self.provider_name} manually forced to CLOSED"
        )

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "provider_name": self.provider_name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "recovery_timeout": self.recovery_timeout,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
            "time_since_last_failure": (
                time.time() - self.last_failure_time if self.last_failure_time else None
            ),
            "can_execute": self.can_execute(),
        }
