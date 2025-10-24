"""Advanced caching and performance optimization for embedding systems.

This module provides intelligent caching strategies including:
- Memory-based caching with LRU eviction
- Persistent disk caching with compression
- Distributed caching support
- Cache warming and preloading
- Performance monitoring and optimization
"""

import hashlib
import importlib.util
import pickle
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

REDIS_AVAILABLE = importlib.util.find_spec("redis") is not None

try:
    import lz4.frame

    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False

from pdf_vector_system.embeddings.base import EmbeddingResult
from pdf_vector_system.utils.logging import LoggerMixin


class CacheStrategy(Enum):
    """Cache strategies."""

    MEMORY_ONLY = "memory_only"
    DISK_ONLY = "disk_only"
    MEMORY_DISK = "memory_disk"
    DISTRIBUTED = "distributed"


@dataclass
class CacheConfig:
    """Configuration for caching."""

    strategy: CacheStrategy = CacheStrategy.MEMORY_DISK
    max_memory_size: int = 1000  # Maximum number of items in memory
    max_disk_size: int = 10000  # Maximum number of items on disk
    cache_dir: Optional[Path] = None
    compression_enabled: bool = True
    ttl_seconds: Optional[int] = None  # Time to live

    # Redis configuration (for distributed caching)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    # Performance settings
    enable_cache_warming: bool = True
    preload_popular_embeddings: bool = True
    cache_hit_threshold: int = 3  # Cache items accessed this many times


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: EmbeddingResult
    created_at: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0

    def update_access(self) -> None:
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1

    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return time.time() - self.created_at

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired (if TTL is set)."""
        return False  # Implement TTL logic if needed


class BaseCacheBackend(ABC, LoggerMixin):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[EmbeddingResult]:
        """Get value from cache."""

    @abstractmethod
    def put(self, key: str, value: EmbeddingResult) -> None:
        """Put value in cache."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""

    @abstractmethod
    def size(self) -> int:
        """Get cache size."""

    @abstractmethod
    def keys(self) -> list[str]:
        """Get all cache keys."""


class MemoryCacheBackend(BaseCacheBackend):
    """Memory-based cache with LRU eviction."""

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize memory cache."""
        self.max_size = max_size
        self.cache: dict[str, CacheEntry] = {}
        self.access_order: list[str] = []
        self.lock = threading.RLock()

        self.logger.info(f"Initialized memory cache with max size: {max_size}")

    def get(self, key: str) -> Optional[EmbeddingResult]:
        """Get value from memory cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.update_access()

                # Update LRU order
                if key in self.access_order:
                    self.access_order.remove(key)
                self.access_order.append(key)

                return entry.value
            return None

    def put(self, key: str, value: EmbeddingResult) -> None:
        """Put value in memory cache."""
        with self.lock:
            # Calculate size
            size_bytes = self._calculate_size(value)

            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                size_bytes=size_bytes,
            )

            # Add to cache
            self.cache[key] = entry

            # Update access order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

            # Evict if necessary
            self._evict_if_needed()

    def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                return True
            return False

    def clear(self) -> None:
        """Clear memory cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)

    def keys(self) -> list[str]:
        """Get all cache keys."""
        return list(self.cache.keys())

    def _evict_if_needed(self) -> None:
        """Evict least recently used items if cache is full."""
        while len(self.cache) > self.max_size:
            if self.access_order:
                lru_key = self.access_order.pop(0)
                if lru_key in self.cache:
                    del self.cache[lru_key]

    def _calculate_size(self, value: EmbeddingResult) -> int:
        """Calculate approximate size of embedding result."""
        # Rough estimation
        embeddings_size = (
            len(value.embeddings) * value.embedding_dimension * 4
        )  # 4 bytes per float
        metadata_size = len(str(value.metadata)) * 2  # Rough estimate
        return embeddings_size + metadata_size


class DiskCacheBackend(BaseCacheBackend):
    """Disk-based cache with compression."""

    def __init__(
        self, cache_dir: Path, max_size: int = 10000, compression_enabled: bool = True
    ) -> None:
        """Initialize disk cache."""
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.compression_enabled = compression_enabled and LZ4_AVAILABLE

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Index file for metadata
        self.index_file = self.cache_dir / "cache_index.pkl"
        self.index: dict[str, dict[str, Any]] = self._load_index()

        self.logger.info(
            f"Initialized disk cache at {cache_dir} "
            f"(max_size: {max_size}, compression: {self.compression_enabled})"
        )

    def get(self, key: str) -> Optional[EmbeddingResult]:
        """Get value from disk cache."""
        if key not in self.index:
            return None

        try:
            file_path = self.cache_dir / f"{key}.cache"
            if not file_path.exists():
                # Remove from index if file doesn't exist
                del self.index[key]
                self._save_index()
                return None

            # Load data
            with file_path.open("rb") as f:
                data = f.read()

            # Decompress if needed
            if self.compression_enabled:
                data = lz4.frame.decompress(data)

            # Deserialize
            result: EmbeddingResult = pickle.loads(data)

            # Update access statistics
            self.index[key]["last_accessed"] = time.time()
            self.index[key]["access_count"] += 1
            self._save_index()

            return result

        except Exception as e:
            self.logger.error(f"Error loading from disk cache: {e!s}")
            return None

    def put(self, key: str, value: EmbeddingResult) -> None:
        """Put value in disk cache."""
        try:
            # Serialize
            data = pickle.dumps(value)

            # Compress if enabled
            if self.compression_enabled:
                data = lz4.frame.compress(data)

            # Save to file
            file_path = self.cache_dir / f"{key}.cache"
            with file_path.open("wb") as f:
                f.write(data)

            # Update index
            self.index[key] = {
                "created_at": time.time(),
                "last_accessed": time.time(),
                "access_count": 1,
                "size_bytes": len(data),
                "file_path": str(file_path),
            }

            # Evict if needed
            self._evict_if_needed()

            # Save index
            self._save_index()

        except Exception as e:
            self.logger.error(f"Error saving to disk cache: {e!s}")

    def delete(self, key: str) -> bool:
        """Delete value from disk cache."""
        if key not in self.index:
            return False

        try:
            file_path = Path(self.index[key]["file_path"])
            if file_path.exists():
                file_path.unlink()

            del self.index[key]
            self._save_index()
            return True

        except Exception as e:
            self.logger.error(f"Error deleting from disk cache: {e!s}")
            return False

    def clear(self) -> None:
        """Clear disk cache."""
        try:
            for key in list(self.index.keys()):
                self.delete(key)
        except Exception as e:
            self.logger.error(f"Error clearing disk cache: {e!s}")

    def size(self) -> int:
        """Get cache size."""
        return len(self.index)

    def keys(self) -> list[str]:
        """Get all cache keys."""
        return list(self.index.keys())

    def _load_index(self) -> dict[str, dict[str, Any]]:
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with self.index_file.open("rb") as f:
                    loaded_index: dict[str, dict[str, Any]] = pickle.load(f)
                    return loaded_index
            except Exception as e:
                self.logger.warning(f"Error loading cache index: {e!s}")
        return {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        try:
            with self.index_file.open("wb") as f:
                pickle.dump(self.index, f)
        except Exception as e:
            self.logger.error(f"Error saving cache index: {e!s}")

    def _evict_if_needed(self) -> None:
        """Evict least recently used items if cache is full."""
        while len(self.index) > self.max_size:
            # Find LRU item
            lru_key = min(
                self.index.keys(), key=lambda k: self.index[k]["last_accessed"]
            )
            self.delete(lru_key)


class EmbeddingCache(LoggerMixin):
    """High-level embedding cache with multiple backends."""

    def __init__(self, config: CacheConfig) -> None:
        """Initialize embedding cache."""
        self.config = config
        self.backends: list[BaseCacheBackend] = []

        # Initialize backends based on strategy
        if config.strategy in [CacheStrategy.MEMORY_ONLY, CacheStrategy.MEMORY_DISK]:
            memory_backend = MemoryCacheBackend(config.max_memory_size)
            self.backends.append(memory_backend)

        if config.strategy in [CacheStrategy.DISK_ONLY, CacheStrategy.MEMORY_DISK]:
            if config.cache_dir:
                disk_backend = DiskCacheBackend(
                    config.cache_dir, config.max_disk_size, config.compression_enabled
                )
                self.backends.append(disk_backend)

        # Performance tracking
        self.stats = {"hits": 0, "misses": 0, "puts": 0, "evictions": 0}

        self.logger.info(
            f"Initialized embedding cache with {len(self.backends)} backends"
        )

    def get_cache_key(self, texts: list[str], model_name: str, **kwargs: Any) -> str:
        """Generate cache key for texts and parameters."""
        # Create a hash of the input
        content = {"texts": texts, "model_name": model_name, **kwargs}

        content_str = str(sorted(content.items()))
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def get(self, cache_key: str) -> Optional[EmbeddingResult]:
        """Get embedding result from cache."""
        for backend in self.backends:
            result = backend.get(cache_key)
            if result is not None:
                self.stats["hits"] += 1
                self.logger.debug(f"Cache hit for key: {cache_key}")
                return result

        self.stats["misses"] += 1
        self.logger.debug(f"Cache miss for key: {cache_key}")
        return None

    def put(self, cache_key: str, result: EmbeddingResult) -> None:
        """Put embedding result in cache."""
        for backend in self.backends:
            backend.put(cache_key, result)

        self.stats["puts"] += 1
        self.logger.debug(f"Cached result for key: {cache_key}")

    def clear(self) -> None:
        """Clear all cache backends."""
        for backend in self.backends:
            backend.clear()

        self.stats = {"hits": 0, "misses": 0, "puts": 0, "evictions": 0}
        self.logger.info("Cleared all cache backends")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0

        backend_stats = []
        for _i, backend in enumerate(self.backends):
            backend_stats.append(
                {
                    "backend_type": type(backend).__name__,
                    "size": backend.size(),
                    "keys_count": len(backend.keys()),
                }
            )

        return {
            "hit_rate": hit_rate,
            "total_hits": self.stats["hits"],
            "total_misses": self.stats["misses"],
            "total_puts": self.stats["puts"],
            "backends": backend_stats,
        }
