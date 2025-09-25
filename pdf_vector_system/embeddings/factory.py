"""Factory for creating embedding services and batch processing utilities."""

from typing import Dict, Any, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import psutil
import gc

from loguru import logger

from .base import EmbeddingService, EmbeddingBatch, EmbeddingResult
from .sentence_transformers_service import SentenceTransformersService
from .openai_service import OpenAIEmbeddingService
from .provider_configs import ProviderBatchConfigs
from ..config.settings import EmbeddingConfig, EmbeddingModelType
from ..utils.logging import LoggerMixin
from ..utils.progress import ProgressTracker


class EmbeddingServiceFactory:
    """Factory for creating embedding services."""
    
    @staticmethod
    def create_service(config: EmbeddingConfig) -> EmbeddingService:
        """
        Create an embedding service based on configuration.

        Args:
            config: Embedding configuration

        Returns:
            Configured embedding service
        """
        if config.model_type == EmbeddingModelType.SENTENCE_TRANSFORMERS:
            return SentenceTransformersService(
                model_name=config.model_name,
                device=config.huggingface_device,  # Allow device specification
                cache_folder=config.huggingface_cache_dir,
                trust_remote_code=config.huggingface_trust_remote_code
            )

        elif config.model_type == EmbeddingModelType.OPENAI:
            return OpenAIEmbeddingService(
                model_name=config.model_name,
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
                max_retries=config.max_retries,
                timeout=config.timeout_seconds,
                batch_size=config.batch_size
            )

        elif config.model_type == EmbeddingModelType.COHERE:
            # Import here to avoid circular imports and optional dependency
            try:
                from .cohere_service import CohereEmbeddingService
                return CohereEmbeddingService(
                    model_name=config.model_name,
                    api_key=config.cohere_api_key,
                    base_url=config.cohere_base_url,
                    max_retries=config.max_retries,
                    timeout=config.timeout_seconds,
                    batch_size=config.batch_size
                )
            except ImportError as e:
                raise ValueError(
                    f"Cohere provider requires 'cohere' package. Install with: pip install cohere"
                ) from e

        elif config.model_type == EmbeddingModelType.HUGGINGFACE:
            try:
                from .huggingface_service import HuggingFaceEmbeddingService
                return HuggingFaceEmbeddingService(
                    model_name=config.model_name,
                    device=config.huggingface_device,
                    cache_dir=config.huggingface_cache_dir,
                    trust_remote_code=config.huggingface_trust_remote_code,
                    batch_size=config.batch_size
                )
            except ImportError as e:
                raise ValueError(
                    f"Hugging Face provider requires 'transformers' package. "
                    f"Install with: pip install transformers torch"
                ) from e

        elif config.model_type == EmbeddingModelType.GOOGLE_USE:
            try:
                from .google_use_service import GoogleUSEService
                return GoogleUSEService(
                    model_name=config.model_name,
                    cache_dir=config.google_use_cache_dir,
                    version=config.google_use_version,
                    batch_size=config.batch_size
                )
            except ImportError as e:
                raise ValueError(
                    f"Google USE provider requires 'tensorflow' and 'tensorflow-hub' packages. "
                    f"Install with: pip install tensorflow tensorflow-hub"
                ) from e

        elif config.model_type == EmbeddingModelType.GOOGLE_GEMINI:
            try:
                from .gemini_service import GeminiEmbeddingService
                return GeminiEmbeddingService(
                    model_name=config.model_name,
                    api_key=config.google_gemini_api_key,
                    base_url=config.google_gemini_base_url,
                    project_id=config.google_gemini_project_id,
                    location=config.google_gemini_location,
                    use_vertex_ai=bool(config.google_gemini_project_id),  # Use Vertex AI if project_id is provided
                    max_retries=config.max_retries,
                    timeout=config.timeout_seconds,
                    batch_size=config.batch_size
                )
            except ImportError as e:
                raise ValueError(
                    f"Google Gemini provider requires 'requests' package. "
                    f"Install with: pip install requests"
                ) from e

        elif config.model_type == EmbeddingModelType.AZURE_OPENAI:
            try:
                from .azure_openai_service import AzureOpenAIEmbeddingService
                return AzureOpenAIEmbeddingService(
                    model_name=config.model_name,
                    api_key=config.azure_openai_api_key,
                    endpoint=config.azure_openai_endpoint,
                    api_version=config.azure_openai_api_version,
                    deployment_name=config.azure_openai_deployment_name,
                    max_retries=config.max_retries,
                    timeout=config.timeout_seconds,
                    batch_size=config.batch_size
                )
            except ImportError as e:
                raise ValueError(
                    f"Azure OpenAI provider requires 'openai' package. "
                    f"Install with: pip install openai"
                ) from e

        else:
            raise ValueError(f"Unsupported embedding model type: {config.model_type}")
    
    @staticmethod
    def get_available_models() -> Dict[str, List[str]]:
        """
        Get available models for each service type.

        Returns:
            Dictionary mapping service types to available models
        """
        return {
            "sentence-transformers": [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "all-distilroberta-v1",
                "paraphrase-multilingual-MiniLM-L12-v2",
                "multi-qa-MiniLM-L6-cos-v1",
                "all-MiniLM-L12-v2",
                "paraphrase-MiniLM-L6-v2"
            ],
            "openai": [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002"
            ],
            "cohere": [
                "embed-english-v3.0",
                "embed-multilingual-v3.0",
                "embed-english-light-v3.0",
                "embed-multilingual-light-v3.0",
                "embed-english-v2.0",
                "embed-multilingual-v2.0"
            ],
            "huggingface": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "microsoft/DialoGPT-medium",
                "distilbert-base-uncased",
                "bert-base-uncased",
                "roberta-base"
            ],
            "google_use": [
                "universal-sentence-encoder",
                "universal-sentence-encoder-large",
                "universal-sentence-encoder-multilingual",
                "universal-sentence-encoder-multilingual-large"
            ],
            "google_gemini": [
                "gemini-embedding-001",
                "gemini-embedding-exp-03-07",
                "text-embedding-004"
            ],
            "azure_openai": [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002"
            ]
        }


class BatchEmbeddingProcessor(LoggerMixin):
    """Processor for handling batch embedding generation with optimization."""
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        max_workers: int = 4,
        batch_size: Optional[int] = None
    ) -> None:
        """
        Initialize batch embedding processor.

        Args:
            embedding_service: Embedding service to use
            max_workers: Maximum number of worker threads
            batch_size: Batch size for processing (uses service default if None)
        """
        self.embedding_service = embedding_service
        self.max_workers = max_workers
        self.batch_size: int = batch_size if batch_size is not None else getattr(embedding_service, 'batch_size', 32)
        
        self.logger.info(
            f"Initialized BatchEmbeddingProcessor with {max_workers} workers, "
            f"batch size: {self.batch_size}"
        )
    
    def process_texts(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = True
    ) -> EmbeddingResult:
        """
        Process a list of texts to generate embeddings efficiently.
        
        Args:
            texts: List of texts to embed
            metadata: Optional metadata for each text
            show_progress: Whether to show progress bar
            
        Returns:
            Combined EmbeddingResult
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        start_time = time.time()
        
        # Create batches
        batches = self._create_batches(texts, metadata)
        
        self.logger.info(f"Processing {len(texts)} texts in {len(batches)} batches")
        
        all_embeddings = []
        total_tokens = 0
        batch_metadata = []
        
        if show_progress:
            with ProgressTracker() as progress:
                task_name = progress.add_task(
                    "embedding_batch",
                    f"Generating embeddings",
                    total=len(batches)
                )
                
                for i, batch in enumerate(batches):
                    result = self.embedding_service.embed_batch(batch)
                    all_embeddings.extend(result.embeddings)
                    
                    if result.token_count:
                        total_tokens += result.token_count
                    
                    if result.metadata:
                        batch_metadata.append(result.metadata)
                    
                    progress.update_task(task_name, advance=1)
        else:
            for batch in batches:
                result = self.embedding_service.embed_batch(batch)
                all_embeddings.extend(result.embeddings)
                
                if result.token_count:
                    total_tokens += result.token_count
                
                if result.metadata:
                    batch_metadata.append(result.metadata)
        
        processing_time = time.time() - start_time
        
        # Create combined result
        combined_result = EmbeddingResult(
            embeddings=all_embeddings,
            model_name=self.embedding_service.model_name,
            embedding_dimension=self.embedding_service.embedding_dimension,
            processing_time=processing_time,
            token_count=total_tokens if total_tokens > 0 else None,
            metadata={
                "batch_processor": True,
                "total_batches": len(batches),
                "batch_size": self.batch_size,
                "max_workers": self.max_workers,
                "texts_per_second": len(texts) / processing_time,
                "batch_metadata": batch_metadata
            }
        )
        
        self.logger.info(
            f"Processed {len(texts)} texts in {processing_time:.2f}s "
            f"({len(texts) / processing_time:.1f} texts/sec)"
        )
        
        return combined_result
    
    def process_texts_parallel(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = True
    ) -> EmbeddingResult:
        """
        Process texts in parallel using multiple workers.
        
        Args:
            texts: List of texts to embed
            metadata: Optional metadata for each text
            show_progress: Whether to show progress bar
            
        Returns:
            Combined EmbeddingResult
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        start_time = time.time()
        
        # Create batches
        batches = self._create_batches(texts, metadata)
        
        self.logger.info(
            f"Processing {len(texts)} texts in {len(batches)} batches "
            f"using {self.max_workers} workers"
        )
        
        all_embeddings: List[List[float]] = [[] for _ in range(len(texts))]  # Preserve order
        total_tokens = 0
        batch_metadata = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.embedding_service.embed_batch, batch): (i, batch)
                for i, batch in enumerate(batches)
            }
            
            if show_progress:
                with ProgressTracker() as progress:
                    task_name = progress.add_task(
                        "parallel_embedding",
                        f"Generating embeddings (parallel)",
                        total=len(batches)
                    )
                    
                    for future in as_completed(future_to_batch):
                        batch_index, batch = future_to_batch[future]
                        
                        try:
                            result = future.result()
                            
                            # Place embeddings in correct positions
                            start_idx = batch_index * self.batch_size
                            for j, embedding in enumerate(result.embeddings):
                                all_embeddings[start_idx + j] = embedding
                            
                            if result.token_count:
                                total_tokens += result.token_count
                            
                            if result.metadata:
                                batch_metadata.append(result.metadata)
                            
                            progress.update_task(task_name, advance=1)
                            
                        except Exception as e:
                            self.logger.error(f"Batch {batch_index} failed: {str(e)}")
                            raise
            else:
                for future in as_completed(future_to_batch):
                    batch_index, batch = future_to_batch[future]
                    
                    try:
                        result = future.result()
                        
                        # Place embeddings in correct positions
                        start_idx = batch_index * self.batch_size
                        for j, embedding in enumerate(result.embeddings):
                            all_embeddings[start_idx + j] = embedding
                        
                        if result.token_count:
                            total_tokens += result.token_count
                        
                        if result.metadata:
                            batch_metadata.append(result.metadata)
                        
                    except Exception as e:
                        self.logger.error(f"Batch {batch_index} failed: {str(e)}")
                        raise
        
        # Remove None values (shouldn't happen, but safety check)
        all_embeddings = [emb for emb in all_embeddings if emb is not None]
        
        processing_time = time.time() - start_time
        
        # Create combined result
        combined_result = EmbeddingResult(
            embeddings=all_embeddings,
            model_name=self.embedding_service.model_name,
            embedding_dimension=self.embedding_service.embedding_dimension,
            processing_time=processing_time,
            token_count=total_tokens if total_tokens > 0 else None,
            metadata={
                "batch_processor": True,
                "parallel_processing": True,
                "total_batches": len(batches),
                "batch_size": self.batch_size,
                "max_workers": self.max_workers,
                "texts_per_second": len(texts) / processing_time,
                "batch_metadata": batch_metadata
            }
        )
        
        self.logger.info(
            f"Processed {len(texts)} texts in parallel in {processing_time:.2f}s "
            f"({len(texts) / processing_time:.1f} texts/sec)"
        )
        
        return combined_result
    
    def _create_batches(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[EmbeddingBatch]:
        """
        Create batches from texts and metadata.
        
        Args:
            texts: List of texts
            metadata: Optional metadata list
            
        Returns:
            List of EmbeddingBatch objects
        """
        batches = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_metadata = None
            
            if metadata:
                batch_metadata = metadata[i:i + self.batch_size]
            
            batch = EmbeddingBatch(
                texts=batch_texts,
                batch_id=f"batch_{i // self.batch_size}",
                metadata=batch_metadata
            )
            
            batches.append(batch)
        
        return batches


class EnhancedBatchEmbeddingProcessor(LoggerMixin):
    """Enhanced batch embedding processor with memory monitoring and adaptive sizing."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        max_workers: int = 4,
        batch_size: Optional[int] = None,
        memory_limit_mb: Optional[int] = None,
        adaptive_batch_sizing: bool = True
    ) -> None:
        """
        Initialize enhanced batch embedding processor.

        Args:
            embedding_service: Embedding service to use
            max_workers: Maximum number of worker threads
            batch_size: Initial batch size (uses provider default if None)
            memory_limit_mb: Memory limit in MB (uses provider default if None)
            adaptive_batch_sizing: Enable adaptive batch sizing
        """
        self.embedding_service = embedding_service
        self.max_workers = max_workers
        self.adaptive_batch_sizing = adaptive_batch_sizing

        # Get provider-specific configuration
        provider_name = self._get_provider_name()
        provider_config = ProviderBatchConfigs.get_batch_config(provider_name)

        self.initial_batch_size = batch_size or provider_config.get("optimal_batch_size", 32)
        self.current_batch_size = self.initial_batch_size
        self.max_batch_size = provider_config.get("max_batch_size", 100)
        self.memory_limit_mb = memory_limit_mb or provider_config.get("memory_limit_mb", 512)

        # Performance tracking
        self.batch_performance_history: List[Dict[str, Any]] = []
        self.memory_usage_history: List[float] = []

        self.logger.info(
            f"Initialized EnhancedBatchEmbeddingProcessor for {provider_name} "
            f"(batch_size: {self.current_batch_size}, memory_limit: {self.memory_limit_mb}MB, "
            f"adaptive: {self.adaptive_batch_sizing})"
        )

    def _get_provider_name(self) -> str:
        """Get provider name from embedding service."""
        service_class = self.embedding_service.__class__.__name__
        if "OpenAI" in service_class:
            if "Azure" in service_class:
                return "azure_openai"
            return "openai"
        elif "Cohere" in service_class:
            return "cohere"
        elif "HuggingFace" in service_class:
            return "huggingface"
        elif "GoogleUSE" in service_class:
            return "google_use"
        elif "SentenceTransformers" in service_class:
            return "sentence_transformers"
        else:
            return "unknown"

    def process_texts_adaptive(
        self,
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = True
    ) -> EmbeddingResult:
        """
        Process texts with adaptive batch sizing and memory monitoring.

        Args:
            texts: List of texts to embed
            metadata: Optional metadata for each text
            show_progress: Whether to show progress bar

        Returns:
            Combined EmbeddingResult
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        start_time = time.time()
        initial_memory = psutil.virtual_memory().percent

        all_embeddings = []
        total_tokens = 0
        batch_metadata = []
        processed_count = 0

        self.logger.info(
            f"Processing {len(texts)} texts with adaptive batch sizing "
            f"(initial batch size: {self.current_batch_size})"
        )

        if show_progress:
            with ProgressTracker() as progress:
                task_name = progress.add_task(
                    "adaptive_embedding",
                    f"Generating embeddings (adaptive)",
                    total=len(texts)
                )

                while processed_count < len(texts):
                    # Determine batch size
                    remaining_texts = len(texts) - processed_count
                    actual_batch_size = min(self.current_batch_size, remaining_texts)

                    # Create batch
                    batch_texts = texts[processed_count:processed_count + actual_batch_size]
                    batch_meta = None
                    if metadata:
                        batch_meta = metadata[processed_count:processed_count + actual_batch_size]

                    batch = EmbeddingBatch(
                        texts=batch_texts,
                        batch_id=f"adaptive_batch_{processed_count // actual_batch_size}",
                        metadata=batch_meta
                    )

                    # Process batch with monitoring
                    batch_start_time = time.time()
                    memory_before = psutil.virtual_memory().percent

                    try:
                        result = self.embedding_service.embed_batch(batch)
                        all_embeddings.extend(result.embeddings)

                        if result.token_count:
                            total_tokens += result.token_count

                        if result.metadata:
                            batch_metadata.append(result.metadata)

                        # Record performance
                        batch_time = time.time() - batch_start_time
                        memory_after = psutil.virtual_memory().percent

                        self._record_batch_performance(
                            batch_size=actual_batch_size,
                            processing_time=batch_time,
                            memory_before=memory_before,
                            memory_after=memory_after,
                            success=True
                        )

                        processed_count += actual_batch_size
                        progress.update_task(task_name, advance=actual_batch_size)

                        # Adaptive batch sizing
                        if self.adaptive_batch_sizing:
                            self._adjust_batch_size(batch_time, memory_after - memory_before)

                        # Memory management
                        if memory_after > initial_memory + 20:  # 20% increase
                            self.logger.debug("High memory usage detected, running garbage collection")
                            gc.collect()

                    except Exception as e:
                        self.logger.error(f"Batch processing failed: {str(e)}")

                        # Record failure and reduce batch size
                        self._record_batch_performance(
                            batch_size=actual_batch_size,
                            processing_time=time.time() - batch_start_time,
                            memory_before=memory_before,
                            memory_after=psutil.virtual_memory().percent,
                            success=False
                        )

                        if self.adaptive_batch_sizing and self.current_batch_size > 1:
                            self.current_batch_size = max(1, self.current_batch_size // 2)
                            self.logger.warning(f"Reduced batch size to {self.current_batch_size} due to failure")

                        raise
        else:
            # Process without progress tracking (similar logic)
            while processed_count < len(texts):
                remaining_texts = len(texts) - processed_count
                actual_batch_size = min(self.current_batch_size, remaining_texts)

                batch_texts = texts[processed_count:processed_count + actual_batch_size]
                batch_meta = None
                if metadata:
                    batch_meta = metadata[processed_count:processed_count + actual_batch_size]

                batch = EmbeddingBatch(
                    texts=batch_texts,
                    batch_id=f"adaptive_batch_{processed_count // actual_batch_size}",
                    metadata=batch_meta
                )

                batch_start_time = time.time()
                memory_before = psutil.virtual_memory().percent

                result = self.embedding_service.embed_batch(batch)
                all_embeddings.extend(result.embeddings)

                if result.token_count:
                    total_tokens += result.token_count

                if result.metadata:
                    batch_metadata.append(result.metadata)

                batch_time = time.time() - batch_start_time
                memory_after = psutil.virtual_memory().percent

                self._record_batch_performance(
                    batch_size=actual_batch_size,
                    processing_time=batch_time,
                    memory_before=memory_before,
                    memory_after=memory_after,
                    success=True
                )

                processed_count += actual_batch_size

                if self.adaptive_batch_sizing:
                    self._adjust_batch_size(batch_time, memory_after - memory_before)

        processing_time = time.time() - start_time

        # Create combined result
        combined_result = EmbeddingResult(
            embeddings=all_embeddings,
            model_name=self.embedding_service.model_name,
            embedding_dimension=self.embedding_service.embedding_dimension,
            processing_time=processing_time,
            token_count=total_tokens if total_tokens > 0 else None,
            metadata={
                "enhanced_batch_processor": True,
                "adaptive_batch_sizing": self.adaptive_batch_sizing,
                "initial_batch_size": self.initial_batch_size,
                "final_batch_size": self.current_batch_size,
                "total_batches": len(self.batch_performance_history),
                "max_workers": self.max_workers,
                "texts_per_second": len(texts) / processing_time if processing_time > 0 else 0,
                "memory_limit_mb": self.memory_limit_mb,
                "batch_metadata": batch_metadata,
                "performance_stats": self._get_performance_stats()
            }
        )

        texts_per_sec = len(texts) / processing_time if processing_time > 0 else 0
        self.logger.info(
            f"Processed {len(texts)} texts in {processing_time:.2f}s "
            f"({texts_per_sec:.1f} texts/sec) "
            f"with final batch size: {self.current_batch_size}"
        )

        return combined_result

    def _record_batch_performance(
        self,
        batch_size: int,
        processing_time: float,
        memory_before: float,
        memory_after: float,
        success: bool
    ) -> None:
        """Record performance metrics for a batch."""
        performance_record = {
            "batch_size": batch_size,
            "processing_time": processing_time,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_delta": memory_after - memory_before,
            "texts_per_second": batch_size / processing_time if processing_time > 0 else 0,
            "success": success,
            "timestamp": time.time()
        }

        self.batch_performance_history.append(performance_record)
        self.memory_usage_history.append(memory_after)

        # Limit history size
        if len(self.batch_performance_history) > 100:
            self.batch_performance_history = self.batch_performance_history[-50:]
        if len(self.memory_usage_history) > 100:
            self.memory_usage_history = self.memory_usage_history[-50:]

    def _adjust_batch_size(self, processing_time: float, memory_delta: float) -> None:
        """Adjust batch size based on performance metrics."""
        if len(self.batch_performance_history) < 3:
            return  # Need some history to make decisions

        recent_performance = self.batch_performance_history[-3:]
        avg_time_per_text = sum(p["processing_time"] / p["batch_size"] for p in recent_performance) / len(recent_performance)
        avg_memory_delta = sum(p["memory_delta"] for p in recent_performance) / len(recent_performance)

        # Decision logic
        should_increase = (
            avg_time_per_text < 0.1 and  # Fast processing
            avg_memory_delta < 5.0 and   # Low memory usage
            self.current_batch_size < self.max_batch_size
        )

        should_decrease = (
            avg_time_per_text > 1.0 or   # Slow processing
            avg_memory_delta > 10.0 or   # High memory usage
            psutil.virtual_memory().percent > 80  # System memory pressure
        )

        if should_increase:
            new_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.5))
            if new_batch_size != self.current_batch_size:
                self.logger.debug(f"Increasing batch size from {self.current_batch_size} to {new_batch_size}")
                self.current_batch_size = new_batch_size
        elif should_decrease:
            new_batch_size = max(1, int(self.current_batch_size * 0.7))
            if new_batch_size != self.current_batch_size:
                self.logger.debug(f"Decreasing batch size from {self.current_batch_size} to {new_batch_size}")
                self.current_batch_size = new_batch_size

    def _get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.batch_performance_history:
            return {}

        successful_batches = [p for p in self.batch_performance_history if p["success"]]

        if not successful_batches:
            return {"success_rate": 0.0}

        avg_processing_time = sum(p["processing_time"] for p in successful_batches) / len(successful_batches)
        avg_texts_per_second = sum(p["texts_per_second"] for p in successful_batches) / len(successful_batches)
        avg_memory_delta = sum(p["memory_delta"] for p in successful_batches) / len(successful_batches)

        return {
            "success_rate": len(successful_batches) / len(self.batch_performance_history),
            "avg_processing_time": avg_processing_time,
            "avg_texts_per_second": avg_texts_per_second,
            "avg_memory_delta": avg_memory_delta,
            "total_batches": len(self.batch_performance_history),
            "batch_size_range": {
                "min": min(p["batch_size"] for p in self.batch_performance_history),
                "max": max(p["batch_size"] for p in self.batch_performance_history),
                "current": self.current_batch_size
            }
        }

    def reset_performance_history(self) -> None:
        """Reset performance tracking history."""
        self.batch_performance_history.clear()
        self.memory_usage_history.clear()
        self.current_batch_size = self.initial_batch_size
        self.logger.info("Reset performance history and batch size")
