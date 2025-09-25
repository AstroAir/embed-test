"""Health check system for embedding providers."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

from loguru import logger

from .base import EmbeddingService
from .provider_configs import ProviderHealthCheckConfigs
from ..utils.logging import LoggerMixin


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    provider_name: str
    status: HealthStatus
    response_time: float
    timestamp: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        """Check if the provider is healthy."""
        return self.status == HealthStatus.HEALTHY
    
    @property
    def age_seconds(self) -> float:
        """Get age of this health check result in seconds."""
        return time.time() - self.timestamp


class ProviderHealthChecker(LoggerMixin):
    """Health checker for a specific embedding provider."""
    
    def __init__(
        self,
        provider_name: str,
        embedding_service: EmbeddingService,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize provider health checker.
        
        Args:
            provider_name: Name of the provider
            embedding_service: The embedding service to check
            config: Health check configuration
        """
        self.provider_name = provider_name
        self.embedding_service = embedding_service
        self.config = config or ProviderHealthCheckConfigs.get_health_check_config(provider_name)
        
        self.last_check_result: Optional[HealthCheckResult] = None
        self.check_history: List[HealthCheckResult] = []
        self.consecutive_failures = 0
        
        self.logger.info(f"Initialized health checker for {provider_name}")
    
    def check_health(self, force: bool = False) -> HealthCheckResult:
        """
        Perform health check on the provider.
        
        Args:
            force: Force health check even if recent check exists
            
        Returns:
            HealthCheckResult
        """
        # Check if we need to perform a new health check
        if not force and self.last_check_result:
            age = self.last_check_result.age_seconds
            if age < self.config["check_interval"]:
                self.logger.debug(
                    f"Using cached health check for {self.provider_name} "
                    f"(age: {age:.1f}s)"
                )
                return self.last_check_result
        
        start_time = time.time()
        
        try:
            # Perform the actual health check
            self.logger.debug(f"Performing health check for {self.provider_name}")
            
            test_text = self.config["test_text"]
            embedding = self.embedding_service.embed_single(test_text)
            
            response_time = time.time() - start_time
            
            # Validate the embedding
            if not embedding:
                raise ValueError("Empty embedding returned")
            
            expected_dim = self.config.get("expected_dimension")
            if expected_dim and len(embedding) != expected_dim:
                raise ValueError(
                    f"Expected dimension {expected_dim}, got {len(embedding)}"
                )
            
            # Check for valid numbers
            if not all(isinstance(x, (int, float)) for x in embedding):
                raise ValueError("Embedding contains invalid values")
            
            # Success
            result = HealthCheckResult(
                provider_name=self.provider_name,
                status=HealthStatus.HEALTHY,
                response_time=response_time,
                metadata={
                    "embedding_dimension": len(embedding),
                    "test_text": test_text,
                    "consecutive_failures": self.consecutive_failures
                }
            )
            
            self.consecutive_failures = 0
            self.logger.debug(
                f"Health check passed for {self.provider_name} "
                f"(response_time: {response_time:.2f}s)"
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            self.consecutive_failures += 1
            
            # Determine status based on failure count
            if self.consecutive_failures >= self.config["failure_threshold"]:
                status = HealthStatus.UNHEALTHY
            else:
                status = HealthStatus.DEGRADED
            
            result = HealthCheckResult(
                provider_name=self.provider_name,
                status=status,
                response_time=response_time,
                error_message=str(e),
                metadata={
                    "consecutive_failures": self.consecutive_failures,
                    "failure_threshold": self.config["failure_threshold"],
                    "test_text": self.config["test_text"]
                }
            )
            
            self.logger.warning(
                f"Health check failed for {self.provider_name} "
                f"(failures: {self.consecutive_failures}/{self.config['failure_threshold']}): {str(e)}"
            )
        
        # Store result
        self.last_check_result = result
        self.check_history.append(result)
        
        # Limit history size
        if len(self.check_history) > 100:
            self.check_history = self.check_history[-50:]
        
        return result
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for this provider."""
        if not self.last_check_result:
            return {
                "provider_name": self.provider_name,
                "status": HealthStatus.UNKNOWN,
                "last_check": None,
                "consecutive_failures": self.consecutive_failures,
                "total_checks": len(self.check_history)
            }
        
        # Calculate success rate from recent history
        recent_checks = [
            check for check in self.check_history[-20:]  # Last 20 checks
            if check.age_seconds < 3600  # Within last hour
        ]
        
        success_rate = 0.0
        if recent_checks:
            successful = sum(1 for check in recent_checks if check.is_healthy)
            success_rate = successful / len(recent_checks)
        
        return {
            "provider_name": self.provider_name,
            "status": self.last_check_result.status,
            "last_check": self.last_check_result.timestamp,
            "last_response_time": self.last_check_result.response_time,
            "consecutive_failures": self.consecutive_failures,
            "total_checks": len(self.check_history),
            "success_rate": success_rate,
            "is_healthy": self.last_check_result.is_healthy,
            "age_seconds": self.last_check_result.age_seconds
        }


class HealthCheckManager(LoggerMixin):
    """Manager for health checks across all embedding providers."""
    
    def __init__(self) -> None:
        """Initialize health check manager."""
        self.checkers: Dict[str, ProviderHealthChecker] = {}
        self.logger.info("Initialized health check manager")
    
    def register_provider(
        self,
        provider_name: str,
        embedding_service: EmbeddingService,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a provider for health checking.
        
        Args:
            provider_name: Name of the provider
            embedding_service: The embedding service
            config: Optional health check configuration
        """
        checker = ProviderHealthChecker(provider_name, embedding_service, config)
        self.checkers[provider_name] = checker
        self.logger.info(f"Registered provider for health checking: {provider_name}")
    
    def check_provider_health(self, provider_name: str, force: bool = False) -> HealthCheckResult:
        """
        Check health of a specific provider.
        
        Args:
            provider_name: Name of the provider
            force: Force health check
            
        Returns:
            HealthCheckResult
            
        Raises:
            KeyError: If provider is not registered
        """
        if provider_name not in self.checkers:
            raise KeyError(f"Provider {provider_name} not registered")
        
        return self.checkers[provider_name].check_health(force=force)
    
    def check_all_providers(self, force: bool = False) -> Dict[str, HealthCheckResult]:
        """
        Check health of all registered providers.
        
        Args:
            force: Force health checks
            
        Returns:
            Dictionary mapping provider names to health check results
        """
        results = {}
        for provider_name, checker in self.checkers.items():
            try:
                results[provider_name] = checker.check_health(force=force)
            except Exception as e:
                self.logger.error(f"Health check failed for {provider_name}: {str(e)}")
                results[provider_name] = HealthCheckResult(
                    provider_name=provider_name,
                    status=HealthStatus.UNHEALTHY,
                    response_time=0.0,
                    error_message=f"Health check exception: {str(e)}"
                )
        
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall health status across all providers."""
        if not self.checkers:
            return {
                "overall_status": HealthStatus.UNKNOWN,
                "healthy_providers": 0,
                "total_providers": 0,
                "providers": {}
            }
        
        provider_summaries = {}
        healthy_count = 0
        
        for provider_name, checker in self.checkers.items():
            summary = checker.get_health_summary()
            provider_summaries[provider_name] = summary
            
            if summary["is_healthy"]:
                healthy_count += 1
        
        total_providers = len(self.checkers)
        health_ratio = healthy_count / total_providers if total_providers > 0 else 0
        
        # Determine overall status
        if health_ratio >= 1.0:
            overall_status = HealthStatus.HEALTHY
        elif health_ratio >= 0.5:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.UNHEALTHY
        
        return {
            "overall_status": overall_status,
            "healthy_providers": healthy_count,
            "total_providers": total_providers,
            "health_ratio": health_ratio,
            "providers": provider_summaries
        }
    
    def get_unhealthy_providers(self) -> List[str]:
        """Get list of unhealthy provider names."""
        unhealthy = []
        for provider_name, checker in self.checkers.items():
            if checker.last_check_result and not checker.last_check_result.is_healthy:
                unhealthy.append(provider_name)
        return unhealthy


# Global health check manager instance
health_check_manager: HealthCheckManager = HealthCheckManager()
