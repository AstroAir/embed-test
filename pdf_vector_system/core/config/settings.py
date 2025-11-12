"""Configuration settings for PDF Vector System using Pydantic models."""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingModelType(str, Enum):
    """Supported embedding model types."""

    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OPENAI = "openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    GOOGLE_USE = "google_use"
    GOOGLE_GEMINI = "google_gemini"
    AZURE_OPENAI = "azure_openai"


class LogLevel(str, Enum):
    """Supported log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class PDFConfig(BaseModel):
    """Configuration for PDF processing."""

    max_file_size_mb: int = Field(
        default=100, description="Maximum PDF file size in MB"
    )
    timeout_seconds: int = Field(
        default=300, description="PDF processing timeout in seconds"
    )
    extract_images: bool = Field(
        default=False, description="Whether to extract images from PDFs"
    )

    @field_validator("max_file_size_mb")
    @classmethod
    def validate_file_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_file_size_mb must be positive")
        return v

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("timeout_seconds must be positive")
        return v


class TextProcessingConfig(BaseModel):
    """Configuration for text processing and chunking."""

    chunk_size: int = Field(
        default=1000, description="Size of text chunks in characters"
    )
    chunk_overlap: int = Field(
        default=100, description="Overlap between chunks in characters"
    )
    separators: list[str] = Field(
        default=["\n\n", "\n", " ", ""],
        description="Text separators for chunking in order of preference",
    )
    min_chunk_size: int = Field(
        default=50, description="Minimum chunk size in characters"
    )

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chunk_size must be positive")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info: Any) -> int:
        if v < 0:
            raise ValueError("chunk_overlap cannot be negative")
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError("chunk_overlap cannot be larger than chunk_size")
        return v

    @field_validator("min_chunk_size")
    @classmethod
    def validate_min_chunk_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("min_chunk_size must be positive")
        return v


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    model_config = {"protected_namespaces": ()}

    model_type: EmbeddingModelType = Field(
        default=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        description="Type of embedding model to use",
    )
    model_name: str = Field(
        default="all-MiniLM-L6-v2", description="Name of the embedding model"
    )
    batch_size: int = Field(
        default=32, description="Batch size for embedding generation"
    )
    max_retries: int = Field(
        default=3, description="Maximum number of retries for API calls"
    )
    timeout_seconds: int = Field(
        default=60, description="Timeout for embedding generation"
    )

    # OpenAI specific settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_base_url: Optional[str] = Field(
        default=None, description="OpenAI API base URL"
    )

    # Cohere specific settings
    cohere_api_key: Optional[str] = Field(default=None, description="Cohere API key")
    cohere_base_url: Optional[str] = Field(
        default=None, description="Cohere API base URL"
    )

    # Azure OpenAI specific settings
    azure_openai_api_key: Optional[str] = Field(
        default=None, description="Azure OpenAI API key"
    )
    azure_openai_endpoint: Optional[str] = Field(
        default=None, description="Azure OpenAI endpoint"
    )
    azure_openai_api_version: str = Field(
        default="2024-02-01", description="Azure OpenAI API version"
    )
    azure_openai_deployment_name: Optional[str] = Field(
        default=None, description="Azure OpenAI deployment name"
    )

    # Hugging Face specific settings
    huggingface_cache_dir: Optional[str] = Field(
        default=None, description="Hugging Face cache directory"
    )
    huggingface_device: Optional[str] = Field(
        default=None, description="Device for Hugging Face models (cpu, cuda, etc.)"
    )
    huggingface_trust_remote_code: bool = Field(
        default=False, description="Trust remote code for Hugging Face models"
    )

    # Google USE specific settings
    google_use_cache_dir: Optional[str] = Field(
        default=None, description="Google USE model cache directory"
    )
    google_use_version: str = Field(default="4", description="Google USE model version")

    # Google Gemini specific settings
    google_gemini_api_key: Optional[str] = Field(
        default=None, description="Google Gemini API key"
    )
    google_gemini_base_url: Optional[str] = Field(
        default=None, description="Google Gemini API base URL"
    )
    google_gemini_project_id: Optional[str] = Field(
        default=None, description="Google Cloud project ID for Vertex AI"
    )
    google_gemini_location: str = Field(
        default="us-central1", description="Google Cloud location for Vertex AI"
    )

    # Advanced batch processing settings
    adaptive_batch_sizing: bool = Field(
        default=True, description="Enable adaptive batch sizing"
    )
    memory_limit_mb: Optional[int] = Field(
        default=None, description="Memory limit for batch processing in MB"
    )
    parallel_batches: Optional[int] = Field(
        default=None, description="Number of parallel batches"
    )

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("model_name cannot be empty")
        return v.strip()

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("batch_size must be positive")
        return v

    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_retries must be non-negative")
        return v

    @field_validator("memory_limit_mb")
    @classmethod
    def validate_memory_limit(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("memory_limit_mb must be positive")
        return v

    @field_validator("parallel_batches")
    @classmethod
    def validate_parallel_batches(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("parallel_batches must be positive")
        return v

    @field_validator("azure_openai_api_version")
    @classmethod
    def validate_azure_api_version(cls, v: str) -> str:
        # Basic validation for Azure API version format
        if not v or len(v) < 10:  # e.g., "2024-02-01"
            raise ValueError("azure_openai_api_version must be a valid API version")
        return v

    @field_validator("google_use_version")
    @classmethod
    def validate_google_use_version(cls, v: str) -> str:
        valid_versions = ["4", "5"]
        if v not in valid_versions:
            raise ValueError(f"google_use_version must be one of {valid_versions}")
        return v

    @field_validator("google_gemini_location")
    @classmethod
    def validate_google_gemini_location(cls, v: str) -> str:
        # Basic validation for Google Cloud location format
        if not v or len(v) < 3:
            raise ValueError(
                "google_gemini_location must be a valid Google Cloud location"
            )
        return v


class ChromaDBConfig(BaseModel):
    """Configuration for ChromaDB vector database."""

    persist_directory: Path = Field(
        default=Path("./chroma_db"), description="Directory to persist ChromaDB data"
    )
    collection_name: str = Field(
        default="pdf_documents",
        description="Default collection name for storing embeddings",
    )
    distance_metric: str = Field(
        default="cosine", description="Distance metric for similarity search"
    )
    max_results: int = Field(
        default=100, description="Maximum number of search results"
    )

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("collection_name cannot be empty")
        # ChromaDB collection names must be alphanumeric with underscores/hyphens
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "collection_name must be alphanumeric with underscores or hyphens"
            )
        return v

    @field_validator("distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        valid_metrics = ["cosine", "l2", "ip"]
        if v not in valid_metrics:
            raise ValueError(f"distance_metric must be one of {valid_metrics}")
        return v

    @field_validator("max_results")
    @classmethod
    def validate_max_results(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_results must be positive")
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: LogLevel = Field(default=LogLevel.INFO, description="Log level")
    format: str = Field(
        default="{time} | {level} | {name} | {message}",
        description="Log format string",
    )
    file_path: Optional[Path] = Field(default=None, description="Log file path")
    rotation: str = Field(default="10 MB", description="Log file rotation size")
    retention: str = Field(default="30 days", description="Log file retention period")


class Config(BaseSettings):
    """Main configuration class for PDF Vector System."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # Sub-configurations
    pdf: PDFConfig = Field(default_factory=PDFConfig)
    text_processing: TextProcessingConfig = Field(default_factory=TextProcessingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chroma_db: ChromaDBConfig = Field(
        default_factory=ChromaDBConfig
    )  # Backward compatibility
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # New vector database configuration (optional for backward compatibility)
    vector_db: Optional[Union[ChromaDBConfig, Any]] = Field(
        default=None,
        description="Vector database configuration (new multi-backend system)",
    )

    # Global settings
    debug: bool = Field(default=False, description="Enable debug mode")
    max_workers: int = Field(default=4, description="Maximum number of worker threads")

    @field_validator("max_workers")
    @classmethod
    def validate_max_workers(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_workers must be positive")
        return v

    def get_vector_db_config(self) -> ChromaDBConfig:
        """
        Get the effective vector database configuration.

        Returns the new vector_db config if available, otherwise falls back to chroma_db
        for backward compatibility.

        Returns:
            Vector database configuration
        """
        if self.vector_db is not None:
            return self.vector_db
        # Convert old ChromaDBConfig to new format for backward compatibility
        from pdf_vector_system.core.vector_db.config import (
            ChromaDBConfig as NewChromaDBConfig,
        )

        return NewChromaDBConfig(
            persist_directory=self.chroma_db.persist_directory,
            collection_name=self.chroma_db.collection_name,
            distance_metric=self.chroma_db.distance_metric,
            max_results=self.chroma_db.max_results,
        )

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization setup."""
        # Set API keys from environment if not provided
        if self.embedding.model_type == EmbeddingModelType.OPENAI:
            if not self.embedding.openai_api_key:
                self.embedding.openai_api_key = os.getenv("OPENAI_API_KEY")
        elif self.embedding.model_type == EmbeddingModelType.COHERE:
            if not self.embedding.cohere_api_key:
                self.embedding.cohere_api_key = os.getenv("COHERE_API_KEY")
        elif self.embedding.model_type == EmbeddingModelType.AZURE_OPENAI:
            if not self.embedding.azure_openai_api_key:
                self.embedding.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
            if not self.embedding.azure_openai_endpoint:
                self.embedding.azure_openai_endpoint = os.getenv(
                    "AZURE_OPENAI_ENDPOINT"
                )

        # Create persist directory if it doesn't exist
        self.chroma_db.persist_directory.mkdir(parents=True, exist_ok=True)

        # Adjust logging level if debug mode is enabled
        if self.debug:
            self.logging.level = LogLevel.DEBUG


# Global configuration instance
config = Config()
