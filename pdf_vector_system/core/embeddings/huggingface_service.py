"""Hugging Face transformers embedding service implementation.

This module provides an EmbeddingService implementation that uses
Hugging Face (transformers) models to generate sentence / text embeddings.
It supports:
- Lazy model/tokenizer loading
- Batch embedding generation with retry logic
- Optional L2 normalization
- Device (CPU / CUDA) auto selection
- Memory usage monitoring and cleanup
"""

import contextlib
import gc
import time
import warnings
from typing import TYPE_CHECKING, Any, Optional

import psutil

from pdf_vector_system.core.embeddings.base import (
    EmbeddingResult,
    EmbeddingService,
    EmbeddingServiceError,
    ModelNotFoundError,
)
from pdf_vector_system.core.embeddings.provider_configs import ProviderRetryConfigs
from pdf_vector_system.core.embeddings.retry import (
    FailureType,
    RetryableError,
    RetryHandler,
)
from pdf_vector_system.core.utils.progress import PerformanceTimer

if TYPE_CHECKING:  # pragma: no cover
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


class HuggingFaceEmbeddingService(EmbeddingService):
    """Embedding service using Hugging Face transformers library.

    Parameters:
        model_name: Hugging Face model identifier.
        device: Optional explicit device (e.g. "cpu", "cuda", "cuda:0").
        cache_dir: Optional directory for model/tokenizer caching.
        trust_remote_code: Allow custom model code (HF param).
        batch_size: Number of texts per forward pass.
        max_length: Max token length (truncation applied).
        normalize_embeddings: Whether to L2-normalize output vectors.
        **kwargs: Extra arguments passed to base EmbeddingService.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        batch_size: int = 16,
        max_length: int = 512,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name, **kwargs)

        self.device = device
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        # Lazy-loaded members
        self._model: Optional[PreTrainedModel] = None
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._device_name: Optional[str] = None

        # Suppress noisy future warnings from transformers
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

        retry_config = ProviderRetryConfigs.get_local_model_config()
        self.retry_handler = RetryHandler(
            retry_config, f"HuggingFaceService-{model_name}"
        )

        self.logger.info(
            f"Configured HuggingFaceEmbeddingService with model: {model_name}"
        )

    @property
    def model(self) -> "PreTrainedModel":
        """Return the loaded transformers model (load lazily if needed)."""
        if self._model is None:
            self._load_model()
        if self._model is None:
            raise ModelNotFoundError(f"Model {self.model_name} failed to load")
        return self._model

    @property
    def tokenizer(self) -> "PreTrainedTokenizerBase":
        """Return the loaded tokenizer (load lazily if needed)."""
        if self._tokenizer is None:
            self._load_model()
        if self._tokenizer is None:
            raise ModelNotFoundError(f"Tokenizer for {self.model_name} failed to load")
        return self._tokenizer

    def _load_model(self) -> None:
        """Load the Hugging Face model and tokenizer if not already loaded.

        Handles device resolution and switches model to eval mode.
        Raises:
            ModelNotFoundError: If transformers/torch missing or load fails.
        """
        if self._model is not None and self._tokenizer is not None:
            return  # Already loaded
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            with PerformanceTimer(f"Loading HuggingFace model {self.model_name}"):
                # Resolve device automatically if not set
                if self.device:
                    self._device_name = self.device
                else:
                    self._device_name = "cuda" if torch.cuda.is_available() else "cpu"

                self.logger.info(f"Loading model on device: {self._device_name}")

                # Load tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=self.trust_remote_code,
                )

                # Load model
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=self.trust_remote_code,
                )

                if self._model is None:
                    raise ModelNotFoundError(
                        f"AutoModel returned None for {self.model_name}"
                    )

                # Move to device & set eval
                if self._device_name:
                    self._model.to(self._device_name)
                self._model.eval()

                self.logger.info(
                    f"Loaded HuggingFace model {self.model_name} on device: {self._device_name}"
                )

        except ImportError as e:
            error_msg = (
                "Hugging Face transformers and torch are required. "
                "Install with: pip install transformers torch"
            )
            self.logger.error(error_msg)
            raise ModelNotFoundError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load model {self.model_name}: {e!s}"
            self.logger.error(error_msg)
            raise ModelNotFoundError(error_msg) from e

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """Generate embeddings for a list of input texts.

        Parameters:
            texts: Raw input strings (non-empty).

        Returns:
            EmbeddingResult containing vectors and metadata.

        Raises:
            ValueError: If input list is empty.
            EmbeddingServiceError: On processing failure.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        validated_texts = self.validate_texts(texts)

        start_time = time.time()
        all_embeddings: list[list[float]] = []

        try:
            initial_memory = psutil.virtual_memory().percent

            with PerformanceTimer(
                f"Generating HuggingFace embeddings for {len(texts)} texts",
                log_result=False,
            ):
                for i in range(0, len(validated_texts), self.batch_size):
                    batch_texts = validated_texts[i : i + self.batch_size]

                    self.logger.debug(
                        f"Processing batch {i // self.batch_size + 1} with {len(batch_texts)} texts"
                    )

                    # Execute with retry logic
                    batch_embeddings = self.retry_handler.execute(
                        self._generate_batch_embeddings, batch_texts
                    )
                    all_embeddings.extend(batch_embeddings)

                    # Basic memory pressure mitigation
                    current_memory = psutil.virtual_memory().percent
                    if current_memory - initial_memory > 20:
                        self.logger.debug(
                            "High memory usage detected, running garbage collection"
                        )
                        gc.collect()
                        if self._device_name == "cuda":
                            try:
                                import torch

                                torch.cuda.empty_cache()
                            except ImportError:
                                pass

            processing_time = time.time() - start_time
            embedding_dim = len(all_embeddings[0]) if all_embeddings else 0

            result = EmbeddingResult(
                embeddings=all_embeddings,
                model_name=self.model_name,
                embedding_dimension=embedding_dim,
                processing_time=processing_time,
                metadata={
                    "device": self._device_name,
                    "model_type": "huggingface",
                    "normalized": self.normalize_embeddings,
                    "batch_size": self.batch_size,
                    "max_length": self.max_length,
                    "batch_count": (len(texts) + self.batch_size - 1)
                    // self.batch_size,
                },
            )

            self.logger.debug(
                f"Generated {len(all_embeddings)} HuggingFace embeddings "
                f"(dim: {embedding_dim}) in {processing_time:.2f}s"
            )
            return result

        except Exception as e:
            error_msg = f"Failed to generate HuggingFace embeddings: {e!s}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e

    def embed_single(self, text: str) -> list[float]:
        """Generate an embedding for a single text string.

        Parameters:
            text: Non-empty input string.

        Returns:
            Single embedding vector.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        try:
            embeddings = self.retry_handler.execute(
                self._generate_batch_embeddings, [text.strip()]
            )
            return embeddings[0]
        except Exception as e:
            error_msg = (
                f"Failed to generate HuggingFace embedding for single text: {e!s}"
            )
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e

    def get_embedding_dimension(self) -> int:
        """Infer embedding dimension directly from model config or a test embedding."""
        try:
            config = self.model.config
            if hasattr(config, "hidden_size"):
                return int(config.hidden_size)
            if hasattr(config, "d_model"):
                return int(config.d_model)
            test_embedding = self.embed_single("test")
            return len(test_embedding)
        except Exception as e:
            error_msg = f"Failed to determine embedding dimension: {e!s}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e

    def _generate_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Internal batch embedding generation (retry target).

        Parameters:
            texts: Batch of prevalidated strings.

        Returns:
            List of embedding vectors.

        Raises:
            RetryableError: Wrapped transient failures (OOM, timeout, etc.)
        """
        try:
            import torch

            if self._model is None or self._tokenizer is None:
                self._load_model()
            if self._model is None or self._tokenizer is None:
                raise EmbeddingServiceError("Model or tokenizer not loaded")

            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            assert self._device_name is not None, "Device not set"
            inputs = {k: v.to(self._device_name) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                embeddings_list: list[list[float]] = embeddings.cpu().numpy().tolist()
            return embeddings_list
        except Exception as e:
            # Map generic failures to retryable types
            error_str = str(e).lower()
            if "out of memory" in error_str:
                raise RetryableError(str(e), FailureType.UNKNOWN) from e
            if "timeout" in error_str:
                raise RetryableError(str(e), FailureType.TIMEOUT) from e
            raise RetryableError(str(e), FailureType.UNKNOWN) from e

    def _mean_pooling(self, model_output: Any, attention_mask: Any) -> Any:
        """Mean pooling over token embeddings using attention mask."""
        import torch

        token_embeddings = model_output[0]  # (batch, seq_len, hidden)
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def get_model_info(self) -> dict[str, Any]:
        """Return extended model metadata (safe fallback on failure)."""
        base_info = super().get_model_info()
        try:
            model_info = {
                **base_info,
                "device": self._device_name,
                "model_type": "huggingface",
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "normalize_embeddings": self.normalize_embeddings,
                "cache_dir": self.cache_dir,
                "trust_remote_code": self.trust_remote_code,
                "model_loaded": self._model is not None,
            }
            if self._model is not None:
                config = self.model.config
                model_info["model_config"] = {
                    "hidden_size": getattr(config, "hidden_size", None),
                    "num_attention_heads": getattr(config, "num_attention_heads", None),
                    "num_hidden_layers": getattr(config, "num_hidden_layers", None),
                    "vocab_size": getattr(config, "vocab_size", None),
                }
            return model_info
        except Exception as e:
            self.logger.warning(f"Error getting extended model info: {e!s}")
            return base_info

    def unload_model(self) -> None:
        """Explicitly free model & tokenizer resources and clear CUDA cache if needed."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if self._device_name == "cuda":
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
        gc.collect()
        self.logger.info(f"Unloaded HuggingFace model {self.model_name}")

    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        with contextlib.suppress(Exception):
            self.unload_model()
