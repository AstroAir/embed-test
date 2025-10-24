"""Factory for creating embedding services and batch processing utilities."""

import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import psutil

from pdf_vector_system.config.settings import EmbeddingConfig, EmbeddingModelType
from pdf_vector_system.embeddings.base import (
    EmbeddingBatch,
    EmbeddingResult,
    EmbeddingService,
)
from pdf_vector_system.embeddings.openai_service import OpenAIEmbeddingService
from pdf_vector_system.embeddings.provider_configs import ProviderBatchConfigs
from pdf_vector_system.embeddings.sentence_transformers_service import (
    SentenceTransformersService,
)
from pdf_vector_system.utils.logging import LoggerMixin
from pdf_vector_system.utils.progress import ProgressTracker


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
                trust_remote_code=config.huggingface_trust_remote_code,
            )

        if config.model_type == EmbeddingModelType.OPENAI:
            return OpenAIEmbeddingService(
                model_name=config.model_name,
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
                max_retries=config.max_retries,
                timeout=config.timeout_seconds,
                batch_size=config.batch_size,
            )

        if config.model_type == EmbeddingModelType.COHERE:
            # Import here to avoid circular imports and optional dependency
            try:
                from pdf_vector_system.embeddings.cohere_service import (
                    CohereEmbeddingService,
                )

                return CohereEmbeddingService(
                    model_name=config.model_name,
                    api_key=config.cohere_api_key,
                    base_url=config.cohere_base_url,
                    max_retries=config.max_retries,
                    timeout=config.timeout_seconds,
                    batch_size=config.batch_size,
                )
            except ImportError as e:
                raise ValueError(
                    "Cohere provider requires 'cohere' package. Install with: pip install cohere"
                ) from e

        elif config.model_type == EmbeddingModelType.HUGGINGFACE:
            try:
                from pdf_vector_system.embeddings.huggingface_service import (
                    HuggingFaceEmbeddingService,
                )

                return HuggingFaceEmbeddingService(
                    model_name=config.model_name,
                    device=config.huggingface_device,
                    cache_dir=config.huggingface_cache_dir,
                    trust_remote_code=config.huggingface_trust_remote_code,
                    batch_size=config.batch_size,
                )
            except ImportError as e:
                raise ValueError(
                    "Hugging Face provider requires 'transformers' package. "
                    "Install with: pip install transformers torch"
                ) from e

        elif config.model_type == EmbeddingModelType.GOOGLE_USE:
            try:
                from pdf_vector_system.embeddings.google_use_service import (
                    GoogleUSEService,
                )

                return GoogleUSEService(
                    model_name=config.model_name,
                    cache_dir=config.google_use_cache_dir,
                    version=config.google_use_version,
                    batch_size=config.batch_size,
                )
            except ImportError as e:
                raise ValueError(
                    "Google USE provider requires 'tensorflow' and 'tensorflow-hub' packages. "
                    "Install with: pip install tensorflow tensorflow-hub"
                ) from e

        elif config.model_type == EmbeddingModelType.GOOGLE_GEMINI:
            try:
                from pdf_vector_system.embeddings.gemini_service import (
                    GeminiEmbeddingService,
                )

                return GeminiEmbeddingService(
                    model_name=config.model_name,
                    api_key=config.google_gemini_api_key,
                    base_url=config.google_gemini_base_url,
                    project_id=config.google_gemini_project_id,
                    location=config.google_gemini_location,
                    use_vertex_ai=bool(
                        config.google_gemini_project_id
                    ),  # Use Vertex AI if project_id is provided
                    max_retries=config.max_retries,
                    timeout=config.timeout_seconds,
                    batch_size=config.batch_size,
                )
            except ImportError as e:
                raise ValueError(
                    "Google Gemini provider requires 'requests' package. "
                    "Install with: pip install requests"
                ) from e

        elif config.model_type == EmbeddingModelType.AZURE_OPENAI:
            try:
                from pdf_vector_system.embeddings.azure_openai_service import (
                    AzureOpenAIEmbeddingService,
                )

                return AzureOpenAIEmbeddingService(
                    model_name=config.model_name,
                    api_key=config.azure_openai_api_key,
                    endpoint=config.azure_openai_endpoint,
                    api_version=config.azure_openai_api_version,
                    deployment_name=config.azure_openai_deployment_name,
                    max_retries=config.max_retries,
                    timeout=config.timeout_seconds,
                    batch_size=config.batch_size,
                )
            except ImportError as e:
                raise ValueError(
                    "Azure OpenAI provider requires 'openai' package. "
                    "Install with: pip install openai"
                ) from e

        else:
            raise ValueError(f"Unsupported embedding model type: {config.model_type}")

    @staticmethod
    def get_supported_models() -> dict[EmbeddingModelType, list[str]]:
        """
        Get supported models for each embedding service type.

        Returns:
            Dictionary mapping model types to lists of supported models
        """
        return {
            EmbeddingModelType.SENTENCE_TRANSFORMERS: [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "all-distilroberta-v1",
                "paraphrase-MiniLM-L6-v2",
                "paraphrase-multilingual-MiniLM-L12-v2",
                "multi-qa-MiniLM-L6-cos-v1",
            ],
            EmbeddingModelType.OPENAI: [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
            EmbeddingModelType.COHERE: [
                "embed-english-v3.0",
                "embed-multilingual-v3.0",
                "embed-english-light-v3.0",
                "embed-multilingual-light-v3.0",
                "embed-english-v2.0",
                "embed-multilingual-v2.0",
            ],
            EmbeddingModelType.HUGGINGFACE: [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "distilbert-base-uncased",
                "bert-base-uncased",
            ],
            EmbeddingModelType.GOOGLE_USE: [
                "universal-sentence-encoder",
                "universal-sentence-encoder-large",
                "universal-sentence-encoder-multilingual",
                "universal-sentence-encoder-multilingual-large",
            ],
            EmbeddingModelType.GOOGLE_GEMINI: [
                "gemini-embedding-001",
                "gemini-embedding-exp-03-07",
                "text-embedding-004",
            ],
            EmbeddingModelType.AZURE_OPENAI: [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
        }

    @staticmethod
    def get_available_models() -> dict[str, list[str]]:
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
                "paraphrase-MiniLM-L6-v2",
            ],
            "openai": [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
            "cohere": [
                "embed-english-v3.0",
                "embed-multilingual-v3.0",
                "embed-english-light-v3.0",
                "embed-multilingual-light-v3.0",
                "embed-english-v2.0",
                "embed-multilingual-v2.0",
            ],
            "huggingface": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "microsoft/DialoGPT-medium",
                "distilbert-base-uncased",
                "bert-base-uncased",
                "roberta-base",
            ],
            "google_use": [
                "universal-sentence-encoder",
                "universal-sentence-encoder-large",
                "universal-sentence-encoder-multilingual",
                "universal-sentence-encoder-multilingual-large",
            ],
            "google_gemini": [
                "gemini-embedding-001",
                "gemini-embedding-exp-03-07",
                "text-embedding-004",
            ],
            "azure_openai": [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ],
        }


class BatchEmbeddingProcessor(LoggerMixin):
    """Processor for handling batch embedding generation with optimization."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        max_workers: int = 4,
        batch_size: Optional[int] = None,
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
        self.batch_size: int = (
            batch_size
            if batch_size is not None
            else getattr(embedding_service, "batch_size", 32)
        )
        self.processed_batches: list[list[str]] = (
            []
        )  # Track processed batches for testing

        self.logger.info(
            f"Initialized BatchEmbeddingProcessor with {max_workers} workers, "
            f"batch size: {self.batch_size}"
        )

    def process_texts(
        self,
        texts: list[str],
        metadata: Optional[list[dict[str, Any]]] = None,
        show_progress: bool = True,
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
            # Return empty result for empty input
            return EmbeddingResult(
                embeddings=[],
                model_name=self.embedding_service.model_name,
                embedding_dimension=self.embedding_service.embedding_dimension,
                processing_time=0.0,
                token_count=0,
                metadata={"batch_processor": True, "texts_per_second": 0.0},
            )

        start_time = time.time()

        # Create batches
        batches = self._create_batches(texts, metadata)

        # Reset processed batches tracking
        self.processed_batches = []

        self.logger.info(f"Processing {len(texts)} texts in {len(batches)} batches")

        all_embeddings = []
        total_tokens = 0
        batch_metadata = []

        if show_progress:
            with ProgressTracker() as progress:
                task_name = progress.add_task(
                    "embedding_batch", "Generating embeddings", total=len(batches)
                )

                for _i, batch in enumerate(batches):
                    result = self.embedding_service.embed_batch(batch)
                    all_embeddings.extend(result.embeddings)

                    # Track processed batch
                    self.processed_batches.append(batch.texts)

                    if result.token_count:
                        total_tokens += result.token_count

                    if result.metadata:
                        batch_metadata.append(result.metadata)

                    progress.update_task(task_name, advance=1)
        else:
            for batch in batches:
                result = self.embedding_service.embed_batch(batch)
                all_embeddings.extend(result.embeddings)

                # Track processed batch
                self.processed_batches.append(batch.texts)

                if result.token_count:
                    total_tokens += result.token_count

                if result.metadata:
                    batch_metadata.append(result.metadata)

        processing_time = max(
            time.time() - start_time, 0.001
        )  # Ensure minimum processing time

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
                "texts_per_second": (
                    len(texts) / processing_time if processing_time > 0 else 0.0
                ),
                "batch_metadata": batch_metadata,
            },
        )

        texts_per_sec = len(texts) / processing_time if processing_time > 0 else 0.0
        self.logger.info(
            f"Processed {len(texts)} texts in {processing_time:.2f}s "
            f"({texts_per_sec:.1f} texts/sec)"
        )

        return combined_result

    def process_texts_parallel(
        self,
        texts: list[str],
        metadata: Optional[list[dict[str, Any]]] = None,
        show_progress: bool = True,
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

        all_embeddings: list[list[float]] = [
            [] for _ in range(len(texts))
        ]  # Preserve order
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
                        "Generating embeddings (parallel)",
                        total=len(batches),
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
                            self.logger.error(f"Batch {batch_index} failed: {e!s}")
                            raise
            else:
                for future in as_completed(future_to_batch):
                    batch_index, _batch = future_to_batch[future]

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
                        self.logger.error(f"Batch {batch_index} failed: {e!s}")
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
                "batch_metadata": batch_metadata,
            },
        )

        self.logger.info(
            f"Processed {len(texts)} texts in parallel in {processing_time:.2f}s "
            f"({len(texts) / processing_time:.1f} texts/sec)"
        )

        return combined_result

    def _create_batches(
        self, texts: list[str], metadata: Optional[list[dict[str, Any]]] = None
    ) -> list[EmbeddingBatch]:
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
            batch_texts = texts[i : i + self.batch_size]
            batch_metadata = {}

            if metadata:
                if isinstance(metadata, list):
                    # Store the list of metadata for this batch
                    batch_metadata["text_metadata"] = metadata[i : i + self.batch_size]
                elif isinstance(metadata, dict):
                    # Use the same metadata dict for all texts in the batch
                    batch_metadata.update(metadata)

            batch = EmbeddingBatch(
                texts=batch_texts,
                batch_id=f"batch_{i // self.batch_size}",
                metadata=batch_metadata,
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
        adaptive_batch_sizing: bool = True,
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

        self.initial_batch_size = batch_size or provider_config.get(
            "optimal_batch_size", 32
        )
        self.current_batch_size = self.initial_batch_size
        self.max_batch_size = provider_config.get("max_batch_size", 100)
        self.memory_limit_mb = memory_limit_mb or provider_config.get(
            "memory_limit_mb", 512
        )

        # Performance tracking
        self.batch_performance_history: list[dict[str, Any]] = []
        self.memory_usage_history: list[float] = []

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
        if "Cohere" in service_class:
            return "cohere"
        if "HuggingFace" in service_class:
            return "huggingface"
        if "GoogleUSE" in service_class:
            return "google_use"
        if "SentenceTransformers" in service_class:
            return "sentence_transformers"
        return "unknown"

    def process_texts_adaptive(
        self,
        texts: list[str],
        metadata: Optional[list[dict[str, Any]]] = None,
        show_progress: bool = True,
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
                    "Generating embeddings (adaptive)",
                    total=len(texts),
                )

                while processed_count < len(texts):
                    # Determine batch size
                    remaining_texts = len(texts) - processed_count
                    actual_batch_size = min(self.current_batch_size, remaining_texts)

                    # Create batch
                    batch_texts = texts[
                        processed_count : processed_count + actual_batch_size
                    ]
                    batch_meta = None
                    if metadata:
                        batch_meta = metadata[
                            processed_count : processed_count + actual_batch_size
                        ]

                    batch = EmbeddingBatch(
                        texts=batch_texts,
                        batch_id=f"adaptive_batch_{processed_count // actual_batch_size}",
                        metadata=batch_meta,
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
                            success=True,
                        )

                        processed_count += actual_batch_size
                        progress.update_task(task_name, advance=actual_batch_size)

                        # Adaptive batch sizing
                        if self.adaptive_batch_sizing:
                            self._adjust_batch_size(
                                batch_time, memory_after - memory_before
                            )

                        # Memory management
                        if memory_after > initial_memory + 20:  # 20% increase
                            self.logger.debug(
                                "High memory usage detected, running garbage collection"
                            )
                            gc.collect()

                    except Exception as e:
                        self.logger.error(f"Batch processing failed: {e!s}")

                        # Record failure and reduce batch size
                        self._record_batch_performance(
                            batch_size=actual_batch_size,
                            processing_time=time.time() - batch_start_time,
                            memory_before=memory_before,
                            memory_after=psutil.virtual_memory().percent,
                            success=False,
                        )

                        if self.adaptive_batch_sizing and self.current_batch_size > 1:
                            self.current_batch_size = max(
                                1, self.current_batch_size // 2
                            )
                            self.logger.warning(
                                f"Reduced batch size to {self.current_batch_size} due to failure"
                            )

                        raise
        else:
            # Process without progress tracking (similar logic)
            while processed_count < len(texts):
                remaining_texts = len(texts) - processed_count
                actual_batch_size = min(self.current_batch_size, remaining_texts)

                batch_texts = texts[
                    processed_count : processed_count + actual_batch_size
                ]
                batch_meta = None
                if metadata:
                    batch_meta = metadata[
                        processed_count : processed_count + actual_batch_size
                    ]

                batch = EmbeddingBatch(
                    texts=batch_texts,
                    batch_id=f"adaptive_batch_{processed_count // actual_batch_size}",
                    metadata=batch_meta,
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
                    success=True,
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
                "texts_per_second": (
                    len(texts) / processing_time if processing_time > 0 else 0
                ),
                "memory_limit_mb": self.memory_limit_mb,
                "batch_metadata": batch_metadata,
                "performance_stats": self._get_performance_stats(),
            },
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
        success: bool,
    ) -> None:
        """Record performance metrics for a batch."""
        performance_record = {
            "batch_size": batch_size,
            "processing_time": processing_time,
            "memory_before": memory_before,
            "memory_after": memory_after,
            "memory_delta": memory_after - memory_before,
            "texts_per_second": (
                batch_size / processing_time if processing_time > 0 else 0
            ),
            "success": success,
            "timestamp": time.time(),
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
        avg_time_per_text = sum(
            p["processing_time"] / p["batch_size"] for p in recent_performance
        ) / len(recent_performance)
        avg_memory_delta = sum(p["memory_delta"] for p in recent_performance) / len(
            recent_performance
        )

        # Decision logic
        should_increase = (
            avg_time_per_text < 0.1  # Fast processing
            and avg_memory_delta < 5.0  # Low memory usage
            and self.current_batch_size < self.max_batch_size
        )

        should_decrease = (
            avg_time_per_text > 1.0  # Slow processing
            or avg_memory_delta > 10.0  # High memory usage
            or psutil.virtual_memory().percent > 80  # System memory pressure
        )

        if should_increase:
            new_batch_size = min(
                self.max_batch_size, int(self.current_batch_size * 1.5)
            )
            if new_batch_size != self.current_batch_size:
                self.logger.debug(
                    f"Increasing batch size from {self.current_batch_size} to {new_batch_size}"
                )
                self.current_batch_size = new_batch_size
        elif should_decrease:
            new_batch_size = max(1, int(self.current_batch_size * 0.7))
            if new_batch_size != self.current_batch_size:
                self.logger.debug(
                    f"Decreasing batch size from {self.current_batch_size} to {new_batch_size}"
                )
                self.current_batch_size = new_batch_size

    def _get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        if not self.batch_performance_history:
            return {}

        successful_batches = [p for p in self.batch_performance_history if p["success"]]

        if not successful_batches:
            return {"success_rate": 0.0}

        avg_processing_time = sum(
            p["processing_time"] for p in successful_batches
        ) / len(successful_batches)
        avg_texts_per_second = sum(
            p["texts_per_second"] for p in successful_batches
        ) / len(successful_batches)
        avg_memory_delta = sum(p["memory_delta"] for p in successful_batches) / len(
            successful_batches
        )

        return {
            "success_rate": len(successful_batches)
            / len(self.batch_performance_history),
            "avg_processing_time": avg_processing_time,
            "avg_texts_per_second": avg_texts_per_second,
            "avg_memory_delta": avg_memory_delta,
            "total_batches": len(self.batch_performance_history),
            "batch_size_range": {
                "min": min(p["batch_size"] for p in self.batch_performance_history),
                "max": max(p["batch_size"] for p in self.batch_performance_history),
                "current": self.current_batch_size,
            },
        }

    def reset_performance_history(self) -> None:
        """Reset performance tracking history."""
        self.batch_performance_history.clear()
        self.memory_usage_history.clear()
        self.current_batch_size = self.initial_batch_size
        self.logger.info("Reset performance history and batch size")


# Additional factory methods for provider validation and introspection
class EmbeddingProviderRegistry:
    """Registry for embedding provider information and validation."""

    @staticmethod
    def get_provider_info(provider: EmbeddingModelType) -> dict[str, Any]:
        """
        Get comprehensive information about a specific embedding provider.

        Args:
            provider: The embedding provider type

        Returns:
            Dictionary containing provider information including supported models,
            required configuration, and capabilities
        """
        provider_info = {
            EmbeddingModelType.SENTENCE_TRANSFORMERS: {
                "name": "Sentence Transformers",
                "type": "local",
                "requires_api_key": False,
                "supports_batch": True,
                "supports_streaming": False,
                "default_model": "all-MiniLM-L6-v2",
                "supported_models": [
                    "all-MiniLM-L6-v2",
                    "all-mpnet-base-v2",
                    "all-distilroberta-v1",
                    "paraphrase-MiniLM-L6-v2",
                    "paraphrase-multilingual-MiniLM-L12-v2",
                    "multi-qa-MiniLM-L6-cos-v1",
                ],
                "required_packages": ["sentence-transformers", "torch"],
                "install_command": "pip install sentence-transformers torch",
                "config_params": [
                    "model_name",
                    "device",
                    "cache_folder",
                    "trust_remote_code",
                ],
                "typical_dimensions": [384, 768, 1024],
                "max_sequence_length": 512,
            },
            EmbeddingModelType.OPENAI: {
                "name": "OpenAI",
                "type": "api",
                "requires_api_key": True,
                "supports_batch": True,
                "supports_streaming": False,
                "default_model": "text-embedding-3-small",
                "supported_models": [
                    "text-embedding-3-small",
                    "text-embedding-3-large",
                    "text-embedding-ada-002",
                ],
                "required_packages": ["openai"],
                "install_command": "pip install openai",
                "config_params": [
                    "api_key",
                    "model_name",
                    "base_url",
                    "max_retries",
                    "timeout",
                ],
                "env_vars": ["OPENAI_API_KEY"],
                "typical_dimensions": [1536, 3072],
                "max_sequence_length": 8191,
                "rate_limits": "Varies by tier",
            },
            EmbeddingModelType.COHERE: {
                "name": "Cohere",
                "type": "api",
                "requires_api_key": True,
                "supports_batch": True,
                "supports_streaming": False,
                "default_model": "embed-english-v3.0",
                "supported_models": [
                    "embed-english-v3.0",
                    "embed-multilingual-v3.0",
                    "embed-english-light-v3.0",
                    "embed-multilingual-light-v3.0",
                    "embed-english-v2.0",
                    "embed-multilingual-v2.0",
                ],
                "required_packages": ["cohere"],
                "install_command": "pip install cohere>=4.0.0",
                "config_params": ["api_key", "model_name", "base_url", "input_type"],
                "env_vars": ["COHERE_API_KEY"],
                "typical_dimensions": [384, 768, 1024, 4096],
                "max_sequence_length": 512,
                "rate_limits": "Varies by tier",
            },
            EmbeddingModelType.HUGGINGFACE: {
                "name": "Hugging Face Transformers",
                "type": "local",
                "requires_api_key": False,
                "supports_batch": True,
                "supports_streaming": False,
                "default_model": "sentence-transformers/all-MiniLM-L6-v2",
                "supported_models": [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                    "distilbert-base-uncased",
                    "bert-base-uncased",
                    "roberta-base",
                ],
                "required_packages": ["transformers", "torch"],
                "install_command": "pip install transformers torch",
                "config_params": [
                    "model_name",
                    "device",
                    "cache_dir",
                    "trust_remote_code",
                    "max_length",
                ],
                "typical_dimensions": [384, 768, 1024],
                "max_sequence_length": 512,
            },
            EmbeddingModelType.GOOGLE_USE: {
                "name": "Google Universal Sentence Encoder",
                "type": "local",
                "requires_api_key": False,
                "supports_batch": True,
                "supports_streaming": False,
                "default_model": "universal-sentence-encoder",
                "supported_models": [
                    "universal-sentence-encoder",
                    "universal-sentence-encoder-large",
                    "universal-sentence-encoder-multilingual",
                    "universal-sentence-encoder-multilingual-large",
                ],
                "required_packages": ["tensorflow", "tensorflow-hub"],
                "install_command": "pip install tensorflow tensorflow-hub",
                "config_params": ["model_name", "cache_dir", "version"],
                "typical_dimensions": [512],
                "max_sequence_length": "Variable",
            },
            EmbeddingModelType.GOOGLE_GEMINI: {
                "name": "Google Gemini",
                "type": "api",
                "requires_api_key": True,
                "supports_batch": True,
                "supports_streaming": False,
                "default_model": "gemini-embedding-001",
                "supported_models": [
                    "gemini-embedding-001",
                    "gemini-embedding-exp-03-07",
                    "text-embedding-004",
                ],
                "required_packages": ["requests"],
                "install_command": "pip install requests",
                "config_params": [
                    "api_key",
                    "model_name",
                    "project_id",
                    "location",
                    "use_vertex_ai",
                ],
                "env_vars": ["GOOGLE_GEMINI_API_KEY", "GOOGLE_AI_API_KEY"],
                "typical_dimensions": [768],
                "max_sequence_length": 2048,
                "rate_limits": "Varies by tier",
            },
            EmbeddingModelType.AZURE_OPENAI: {
                "name": "Azure OpenAI",
                "type": "api",
                "requires_api_key": True,
                "supports_batch": True,
                "supports_streaming": False,
                "default_model": "text-embedding-3-small",
                "supported_models": [
                    "text-embedding-3-small",
                    "text-embedding-3-large",
                    "text-embedding-ada-002",
                ],
                "required_packages": ["openai"],
                "install_command": "pip install openai",
                "config_params": [
                    "api_key",
                    "endpoint",
                    "api_version",
                    "deployment_name",
                ],
                "env_vars": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
                "typical_dimensions": [1536, 3072],
                "max_sequence_length": 8191,
                "rate_limits": "Varies by deployment",
            },
        }

        return provider_info.get(provider, {})

    @staticmethod
    def validate_provider_config(config: EmbeddingConfig) -> tuple[bool, list[str]]:
        """
        Validate embedding configuration for a specific provider.

        Args:
            config: Embedding configuration to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Get provider info
        provider_info = EmbeddingProviderRegistry.get_provider_info(config.model_type)
        if not provider_info:
            errors.append(f"Unknown provider type: {config.model_type}")
            return False, errors

        # Check API key requirements
        if provider_info.get("requires_api_key", False):
            if config.model_type == EmbeddingModelType.OPENAI:
                if not config.openai_api_key:
                    errors.append("OpenAI API key is required but not provided")
            elif config.model_type == EmbeddingModelType.COHERE:
                if not config.cohere_api_key:
                    errors.append("Cohere API key is required but not provided")
            elif config.model_type == EmbeddingModelType.AZURE_OPENAI:
                if not config.azure_openai_api_key:
                    errors.append("Azure OpenAI API key is required but not provided")
                if not config.azure_openai_endpoint:
                    errors.append("Azure OpenAI endpoint is required but not provided")
            elif config.model_type == EmbeddingModelType.GOOGLE_GEMINI:
                if not config.google_gemini_api_key:
                    errors.append("Google Gemini API key is required but not provided")

        # Validate model name
        supported_models = provider_info.get("supported_models", [])
        if supported_models and config.model_name not in supported_models:
            errors.append(
                f"Model '{config.model_name}' is not in the list of known supported models for {provider_info['name']}. "
                f"Supported models: {', '.join(supported_models)}"
            )

        # Validate batch size
        if config.batch_size <= 0:
            errors.append(f"Invalid batch size: {config.batch_size}")

        return len(errors) == 0, errors

    @staticmethod
    def list_all_providers() -> list[dict[str, Any]]:
        """
        List all available embedding providers with their basic info.

        Returns:
            List of dictionaries containing provider information
        """
        providers = []
        for provider_type in EmbeddingModelType:
            info = EmbeddingProviderRegistry.get_provider_info(provider_type)
            if info:
                providers.append(
                    {
                        "type": provider_type.value,
                        "name": info.get("name"),
                        "is_local": info.get("type") == "local",
                        "requires_api_key": info.get("requires_api_key"),
                        "default_model": info.get("default_model"),
                        "model_count": len(info.get("supported_models", [])),
                    }
                )
        return providers

    @staticmethod
    def get_recommended_config(
        use_case: str = "general",
        prefer_local: bool = False,
        max_cost: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Get recommended embedding configuration based on use case.

        Args:
            use_case: The use case (general, semantic_search, clustering, qa, etc.)
            prefer_local: Whether to prefer local models over API-based
            max_cost: Maximum acceptable cost level (free, low, medium, high)

        Returns:
            Dictionary with recommended configuration
        """
        recommendations = {
            "general": {
                "local": {
                    "model_type": EmbeddingModelType.SENTENCE_TRANSFORMERS,
                    "model_name": "all-MiniLM-L6-v2",
                    "reason": "Fast, efficient, good general-purpose embeddings",
                },
                "api": {
                    "model_type": EmbeddingModelType.OPENAI,
                    "model_name": "text-embedding-3-small",
                    "reason": "High quality, cost-effective API-based embeddings",
                },
            },
            "semantic_search": {
                "local": {
                    "model_type": EmbeddingModelType.SENTENCE_TRANSFORMERS,
                    "model_name": "all-mpnet-base-v2",
                    "reason": "Excellent for semantic similarity tasks",
                },
                "api": {
                    "model_type": EmbeddingModelType.COHERE,
                    "model_name": "embed-english-v3.0",
                    "reason": "Optimized for search and retrieval",
                },
            },
            "multilingual": {
                "local": {
                    "model_type": EmbeddingModelType.GOOGLE_USE,
                    "model_name": "universal-sentence-encoder-multilingual",
                    "reason": "Strong multilingual support",
                },
                "api": {
                    "model_type": EmbeddingModelType.COHERE,
                    "model_name": "embed-multilingual-v3.0",
                    "reason": "Excellent multilingual API-based embeddings",
                },
            },
            "high_quality": {
                "local": {
                    "model_type": EmbeddingModelType.SENTENCE_TRANSFORMERS,
                    "model_name": "all-mpnet-base-v2",
                    "reason": "Best quality local model",
                },
                "api": {
                    "model_type": EmbeddingModelType.OPENAI,
                    "model_name": "text-embedding-3-large",
                    "reason": "Highest quality embeddings available",
                },
            },
        }

        use_case_config = recommendations.get(use_case, recommendations["general"])
        preference = "local" if prefer_local else "api"

        return use_case_config.get(preference, use_case_config["local"])
