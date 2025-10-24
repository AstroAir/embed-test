"""Enhanced Sentence Transformers embedding service implementation with advanced features."""

import time
import warnings
from typing import Any, Optional, cast

import numpy as np
from sentence_transformers import SentenceTransformer

from pdf_vector_system.embeddings.base import (
    EmbeddingResult,
    EmbeddingService,
    EmbeddingServiceError,
    ModelNotFoundError,
)
from pdf_vector_system.embeddings.chunking import (
    ChunkerFactory,
    ChunkingConfig,
    ChunkingStrategy,
)
from pdf_vector_system.embeddings.preprocessing import (
    AdvancedTextPreprocessor,
    PreprocessingLevel,
    PreprocessorFactory,
    TextType,
)
from pdf_vector_system.embeddings.quality import EmbeddingQualityValidator
from pdf_vector_system.embeddings.tokenization import BaseTokenizer, TokenizerFactory
from pdf_vector_system.utils.progress import PerformanceTimer


class SentenceTransformersService(EmbeddingService):
    """Enhanced embedding service using sentence-transformers library with advanced features."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        trust_remote_code: bool = False,
        enable_advanced_preprocessing: bool = True,
        enable_quality_validation: bool = True,
        preprocessing_level: PreprocessingLevel = PreprocessingLevel.STANDARD,
        text_type: TextType = TextType.GENERAL,
        **kwargs: Any,
    ) -> None:
        """
        Initialize enhanced sentence-transformers embedding service.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run the model on ('cpu', 'cuda', etc.)
            cache_folder: Folder to cache downloaded models
            trust_remote_code: Whether to trust remote code in models
            enable_advanced_preprocessing: Enable advanced text preprocessing
            enable_quality_validation: Enable embedding quality validation
            preprocessing_level: Level of text preprocessing
            text_type: Type of text content for optimized processing
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)

        self.device = device
        self.cache_folder = cache_folder
        self.trust_remote_code = trust_remote_code
        self.enable_advanced_preprocessing = enable_advanced_preprocessing
        self.enable_quality_validation = enable_quality_validation
        self._model: Optional[SentenceTransformer] = None

        # Initialize advanced components
        self.preprocessor: Optional[AdvancedTextPreprocessor]
        if self.enable_advanced_preprocessing:
            self.preprocessor = PreprocessorFactory.create_preprocessor(
                level=preprocessing_level, text_type=text_type
            )
            self.logger.info(
                f"Initialized advanced preprocessor: {preprocessing_level.value}"
            )
        else:
            self.preprocessor = None

        self.quality_validator: Optional[EmbeddingQualityValidator]
        if self.enable_quality_validation:
            self.quality_validator = EmbeddingQualityValidator()
            self.logger.info("Initialized embedding quality validator")
        else:
            self.quality_validator = None

        # Initialize tokenizer for advanced features
        self.tokenizer: Optional[BaseTokenizer]
        try:
            tokenizer_config = TokenizerFactory.get_recommended_config(model_name)
            self.tokenizer = TokenizerFactory.create_tokenizer(tokenizer_config)
            self.logger.info(f"Initialized tokenizer: {tokenizer_config.method.value}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize tokenizer: {e!s}")
            self.tokenizer = None

        # Suppress some warnings from sentence-transformers
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

        self.logger.info(
            f"Configured Enhanced SentenceTransformersService with model: {model_name}"
        )

    @property
    def model(self) -> SentenceTransformer:
        """Get or load the sentence transformer model."""
        if self._model is None:
            self._load_model()
        if self._model is None:
            raise RuntimeError(
                "Failed to load model - model is still None after loading"
            )
        return self._model

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            with PerformanceTimer(f"Loading model {self.model_name}"):
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=self.cache_folder,
                    trust_remote_code=self.trust_remote_code,
                )

                self.logger.info(
                    f"Loaded model {self.model_name} on device: {self._model.device}"
                )

        except Exception as e:
            error_msg = f"Failed to load model {self.model_name}: {e!s}"
            self.logger.error(error_msg)
            raise ModelNotFoundError(error_msg) from e

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts with advanced preprocessing and quality validation.

        Args:
            texts: List of texts to embed

        Returns:
            EmbeddingResult containing the generated embeddings with quality metrics
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        # Advanced preprocessing if enabled
        processed_texts = texts
        preprocessing_metadata = {}

        if self.enable_advanced_preprocessing and self.preprocessor:
            processed_texts = []
            preprocessing_stats: list[dict[str, Any]] = []

            for text in texts:
                result = self.preprocessor.preprocess(text)
                processed_texts.append(result.processed_text)
                preprocessing_stats.append(
                    {
                        "compression_ratio": result.compression_ratio,
                        "quality_score": result.quality_score,
                        "language": result.language,
                        "transformations": result.transformations,
                    }
                )

            preprocessing_metadata = {
                "preprocessing_enabled": True,
                "preprocessing_level": self.preprocessor.config.level.value,
                "avg_compression_ratio": float(
                    np.mean(
                        [float(s["compression_ratio"]) for s in preprocessing_stats]
                    )
                ),
                "avg_quality_score": float(
                    np.mean(
                        [float(s["quality_score"] or 0.0) for s in preprocessing_stats]
                    )
                ),
                "preprocessing_stats": preprocessing_stats,
            }

            self.logger.debug(
                f"Advanced preprocessing completed for {len(texts)} texts"
            )

        # Validate and clean texts
        validated_texts = self.validate_texts(processed_texts)

        start_time = time.time()

        try:
            # Generate embeddings with enhanced parameters
            with PerformanceTimer(
                f"Generating embeddings for {len(texts)} texts", log_result=False
            ):
                embeddings = self.model.encode(
                    validated_texts,
                    convert_to_numpy=True,
                    show_progress_bar=len(texts) > 10,
                    batch_size=32,
                    normalize_embeddings=True,
                    convert_to_tensor=False,
                )

            processing_time = time.time() - start_time

            # Ensure numpy array for type checking
            embeddings_array = cast("np.ndarray", embeddings)
            embeddings_list = embeddings_array.tolist()

            # Get model information
            if embeddings_array.ndim != 2:
                raise EmbeddingServiceError(
                    f"Unexpected embeddings shape (expected 2D): {embeddings_array.shape}"
                )
            embedding_dim = int(embeddings_array.shape[1])

            # Create base result
            embedding_result = EmbeddingResult(
                embeddings=embeddings_list,
                model_name=self.model_name,
                embedding_dimension=embedding_dim,
                processing_time=processing_time,
                metadata={
                    "device": str(self.model.device),
                    "model_type": "sentence-transformers-enhanced",
                    "normalized": True,
                    "batch_size": len(texts),
                    **preprocessing_metadata,
                },
            )

            # Quality validation if enabled
            if self.enable_quality_validation and self.quality_validator:
                try:
                    quality_report = self.quality_validator.validate_embeddings(
                        embedding_result, texts=validated_texts
                    )
                    if embedding_result.metadata is not None:
                        embedding_result.metadata["quality_validation"] = (
                            quality_report.to_dict()
                        )
                    self.logger.debug(
                        f"Quality validation completed. Overall score: {quality_report.overall_score:.3f}"
                    )
                except Exception as e:
                    self.logger.warning(f"Quality validation failed: {e!s}")

            self.logger.debug(
                f"Generated {len(embeddings_list)} enhanced embeddings "
                f"(dim: {embedding_dim}) in {processing_time:.2f}s"
            )

            return embedding_result

        except Exception as e:
            error_msg = f"Failed to generate embeddings: {e!s}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e

    def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            # Generate embedding for single text
            embedding = self.model.encode(
                [text.strip()], convert_to_numpy=True, normalize_embeddings=True
            )

            embedding_array = cast("np.ndarray", embedding)
            if embedding_array.ndim != 2 or embedding_array.shape[0] < 1:
                raise EmbeddingServiceError("Unexpected single embedding shape")
            return embedding_array[0].tolist()

        except Exception as e:
            error_msg = f"Failed to generate embedding for single text: {e!s}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service.

        Returns:
            Embedding dimension
        """
        try:
            # Get dimension from model
            dimension = self.model.get_sentence_embedding_dimension()
            if dimension is None:
                raise ValueError("Model returned None for embedding dimension")
            return int(dimension)

        except Exception as e:
            # Fallback: generate a test embedding to get dimension
            try:
                test_embedding = self.embed_single("test")
                return len(test_embedding)
            except Exception:
                error_msg = f"Failed to determine embedding dimension: {e!s}"
                self.logger.error(error_msg)
                raise EmbeddingServiceError(error_msg) from e

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode texts in batches with more control over the process.

        Args:
            texts: List of texts to encode
            batch_size: Size of each batch
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings

        Returns:
            Numpy array of embeddings
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        validated_texts = self.validate_texts(texts)

        try:
            embeddings = self.model.encode(
                validated_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            )

            embeddings_array = cast("np.ndarray", embeddings)
            self.logger.debug(
                f"Encoded {len(texts)} texts in batches of {batch_size} "
                f"(shape: {embeddings_array.shape})"
            )

            return np.asarray(embeddings_array)

        except Exception as e:
            error_msg = f"Failed to encode batch: {e!s}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e

    def similarity(
        self, embeddings1: np.ndarray, embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.
        """
        try:
            from sentence_transformers.util import cos_sim

            # Convert to torch tensors if necessary
            try:
                import torch

                e1 = (
                    embeddings1
                    if isinstance(embeddings1, torch.Tensor)
                    else torch.from_numpy(embeddings1)
                )
                e2 = (
                    embeddings2
                    if isinstance(embeddings2, torch.Tensor)
                    else torch.from_numpy(embeddings2)
                )
            except Exception as conv_err:
                raise EmbeddingServiceError(
                    f"Torch conversion failed: {conv_err}"
                ) from conv_err

            similarity_tensor = cos_sim(e1, e2)  # expects torch.Tensor
            similarity_matrix = similarity_tensor.cpu().numpy()
            return np.asarray(similarity_matrix)

        except Exception as e:
            error_msg = f"Failed to compute similarity: {e!s}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e

    def embed_with_chunking(
        self,
        text: str,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_BASED,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> EmbeddingResult:
        """
        Embed text with advanced chunking strategies.

        Args:
            text: Text to embed
            chunking_strategy: Strategy for chunking text
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks

        Returns:
            EmbeddingResult with chunked embeddings
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")

        # Create chunking configuration
        chunking_config = ChunkingConfig(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Create chunker
        chunker = ChunkerFactory.create_chunker(chunking_config, self.tokenizer)

        # Chunk the text
        chunks = chunker.chunk_text(text)

        if not chunks:
            raise ValueError("No valid chunks generated from text")

        # Extract chunk contents
        chunk_texts = [chunk.content for chunk in chunks]

        # Generate embeddings for chunks
        result = self.embed_texts(chunk_texts)

        # Add chunking metadata
        if result.metadata is not None:
            result.metadata["chunking"] = {
                "strategy": chunking_strategy.value,
                "chunk_count": len(chunks),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunks_metadata": [chunk.metadata.__dict__ for chunk in chunks],
            }

        self.logger.debug(
            f"Generated embeddings for {len(chunks)} chunks using {chunking_strategy.value}"
        )

        return result

    def get_model_info(self) -> dict[str, Any]:
        """
        Get comprehensive information about the enhanced model.

        Returns:
            Dictionary containing model information
        """
        base_info = super().get_model_info()

        try:
            model_info = {
                **base_info,
                "device": str(self.model.device) if self._model else self.device,
                "max_seq_length": getattr(self.model, "max_seq_length", None),
                "model_loaded": self._model is not None,
                "cache_folder": self.cache_folder,
                "trust_remote_code": self.trust_remote_code,
                "enhanced_features": {
                    "advanced_preprocessing": self.enable_advanced_preprocessing,
                    "quality_validation": self.enable_quality_validation,
                    "tokenizer_available": self.tokenizer is not None,
                    "preprocessor_available": self.preprocessor is not None,
                    "quality_validator_available": self.quality_validator is not None,
                },
            }

            # Add tokenizer info if available
            if self._model and hasattr(self.model, "tokenizer"):
                tokenizer_obj = getattr(self.model, "tokenizer", None)
                if tokenizer_obj is not None and hasattr(tokenizer_obj, "__len__"):
                    try:
                        model_info["tokenizer_vocab_size"] = len(tokenizer_obj)
                    except Exception:
                        model_info["tokenizer_vocab_size"] = None

            # Add advanced component info
            if self.tokenizer:
                model_info["tokenizer_info"] = {
                    "method": self.tokenizer.config.method.value,
                    "vocab_size": self.tokenizer.get_vocab_size(),
                }

            if self.preprocessor:
                model_info["preprocessor_info"] = {
                    "level": self.preprocessor.config.level.value,
                    "text_type": self.preprocessor.config.text_type.value,
                }

            return model_info

        except Exception as e:
            self.logger.warning(f"Error getting extended model info: {e!s}")
            return base_info

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self.logger.info(f"Unloaded model {self.model_name}")

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        try:
            self.unload_model()
        except Exception:
            pass  # Ignore errors during cleanup
