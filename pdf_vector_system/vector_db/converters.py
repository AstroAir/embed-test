"""Universal conversion utilities for different vector database backends."""

from typing import Any, Union

from pdf_vector_system.vector_db.config import VectorDBType
from pdf_vector_system.vector_db.models import DocumentChunk, SearchQuery


class VectorDBConverter:
    """Universal converter for different vector database formats."""

    def __init__(self) -> None:
        """Initialize the converter."""

    # Instance methods for backward compatibility with tests
    def to_chromadb_format(self, chunks: list[DocumentChunk]) -> dict[str, list[Any]]:
        """Convert chunks to ChromaDB format (instance method)."""
        return self.chunks_to_chromadb_format(chunks)

    def to_pinecone_format(self, chunks: list[DocumentChunk]) -> list[dict[str, Any]]:
        """Convert chunks to Pinecone format (instance method)."""
        return self.chunks_to_pinecone_format(chunks)

    def to_weaviate_format(
        self, chunks: list[DocumentChunk], class_name: str = "Document"
    ) -> list[dict[str, Any]]:
        """Convert chunks to Weaviate format (instance method)."""
        return self.chunks_to_weaviate_format(chunks, class_name)

    def to_qdrant_format(self, chunks: list[DocumentChunk]) -> list[dict[str, Any]]:
        """Convert chunks to Qdrant format (instance method)."""
        return self.chunks_to_qdrant_format(chunks)

    def to_milvus_format(self, chunks: list[DocumentChunk]) -> dict[str, list[Any]]:
        """Convert chunks to Milvus format (instance method)."""
        return self.chunks_to_milvus_format(chunks)

    def convert_search_query(self, query: SearchQuery, backend: str) -> dict[str, Any]:
        """Convert search query to backend-specific format (instance method)."""
        backend_type = VectorDBType(backend.lower())
        return self.convert_query_for_backend(query, backend_type)

    @staticmethod
    def chunks_to_chromadb_format(chunks: list[DocumentChunk]) -> dict[str, list[Any]]:
        """
        Convert DocumentChunk list to ChromaDB format.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            Dictionary in ChromaDB batch format
        """
        return {
            "ids": [chunk.id for chunk in chunks],
            "documents": [chunk.content for chunk in chunks],
            "embeddings": [chunk.embedding for chunk in chunks],
            "metadatas": [chunk.metadata for chunk in chunks],
        }

    @staticmethod
    def query_to_chromadb_params(query: SearchQuery) -> dict[str, Any]:
        """
        Convert SearchQuery to ChromaDB query parameters.

        Args:
            query: SearchQuery object

        Returns:
            Dictionary of ChromaDB query parameters
        """
        include = []
        if query.include_distances:
            include.append("distances")
        if query.include_metadata:
            include.append("metadatas")
        if query.include_documents:
            include.append("documents")

        params = {"n_results": query.n_results, "include": include}

        if query.where:
            params["where"] = query.where
        if query.where_document:
            params["where_document"] = query.where_document

        return params

    @staticmethod
    def chunks_to_pinecone_format(chunks: list[DocumentChunk]) -> list[dict[str, Any]]:
        """
        Convert DocumentChunk list to Pinecone format.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            List of dictionaries in Pinecone upsert format
        """
        vectors = []
        for chunk in chunks:
            vector = {
                "id": chunk.id,
                "values": chunk.embedding,
                "metadata": {
                    **chunk.metadata,
                    "content": chunk.content,  # Pinecone stores content in metadata
                },
            }
            vectors.append(vector)
        return vectors

    @staticmethod
    def query_to_pinecone_params(query: SearchQuery) -> dict[str, Any]:
        """
        Convert SearchQuery to Pinecone query parameters.

        Args:
            query: SearchQuery object

        Returns:
            Dictionary of Pinecone query parameters
        """
        params: dict[str, Any] = {
            "top_k": query.n_results,
            "include_metadata": query.include_metadata,
            # Include embeddings if distances needed
            "include_values": query.include_distances,
        }

        if query.where:
            params["filter"] = query.where

        return params

    @staticmethod
    def chunks_to_weaviate_format(
        chunks: list[DocumentChunk], class_name: str
    ) -> list[dict[str, Any]]:
        """
        Convert DocumentChunk list to Weaviate format.

        Args:
            chunks: List of DocumentChunk objects
            class_name: Weaviate class name

        Returns:
            List of dictionaries in Weaviate batch format
        """
        objects = []
        for chunk in chunks:
            obj = {
                "class": class_name,
                "id": chunk.id,
                "properties": {"content": chunk.content, **chunk.metadata},
                "vector": chunk.embedding,
            }
            objects.append(obj)
        return objects

    @staticmethod
    def query_to_weaviate_params(query: SearchQuery) -> dict[str, Any]:
        """
        Convert SearchQuery to Weaviate query parameters.

        Args:
            query: SearchQuery object

        Returns:
            Dictionary of Weaviate query parameters
        """
        params: dict[str, Any] = {"limit": query.n_results}

        # Weaviate uses different field names for includes
        additional: list[str] = []
        if query.include_distances:
            additional.append("distance")
        if query.include_metadata:
            additional.append("id")

        if additional:
            params["additional"] = additional

        if query.where:
            params["where"] = query.where

        return params

    @staticmethod
    def chunks_to_qdrant_format(chunks: list[DocumentChunk]) -> list[dict[str, Any]]:
        """
        Convert DocumentChunk list to Qdrant format.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            List of dictionaries in Qdrant upsert format
        """
        points = []
        for chunk in chunks:
            point = {
                "id": chunk.id,
                "vector": chunk.embedding,
                "payload": {"content": chunk.content, **chunk.metadata},
            }
            points.append(point)
        return points

    @staticmethod
    def query_to_qdrant_params(query: SearchQuery) -> dict[str, Any]:
        """
        Convert SearchQuery to Qdrant query parameters.

        Args:
            query: SearchQuery object

        Returns:
            Dictionary of Qdrant query parameters
        """
        params: dict[str, Any] = {
            "limit": query.n_results,
            "with_payload": query.include_metadata,
            "with_vectors": query.include_distances,  # Include vectors if distances needed
        }

        if query.where:
            params["query_filter"] = query.where

        return params

    @staticmethod
    def chunks_to_milvus_format(chunks: list[DocumentChunk]) -> dict[str, Any]:
        """
        Convert DocumentChunk list to Milvus format.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            Dictionary in Milvus insert format
        """
        # Milvus requires separate lists for each field
        data = {
            "id": [chunk.id for chunk in chunks],
            "content": [chunk.content for chunk in chunks],
            "embedding": [chunk.embedding for chunk in chunks],
        }

        # Add metadata fields (Milvus requires flattened structure)
        if chunks:
            # Get all unique metadata keys
            metadata_keys: set[str] = set()
            for chunk in chunks:
                metadata_keys.update(chunk.metadata.keys())

            # Create separate lists for each metadata field
            for key in metadata_keys:
                data[f"metadata_{key}"] = [chunk.metadata.get(key) for chunk in chunks]

        return data

    @staticmethod
    def query_to_milvus_params(query: SearchQuery) -> dict[str, Any]:
        """
        Convert SearchQuery to Milvus query parameters.

        Args:
            query: SearchQuery object

        Returns:
            Dictionary of Milvus query parameters
        """
        params: dict[str, Any] = {
            "limit": query.n_results,
        }

        additional: list[str] = []
        if query.include_metadata:
            additional = ["*"]
        else:
            additional = ["id", "content"]

        params["output_fields"] = additional

        if query.where:
            # Convert where clause to Milvus expression format
            # This is a simplified conversion - real implementation would need more sophisticated parsing
            expr_parts = []
            for key, value in query.where.items():
                if isinstance(value, str):
                    expr_parts.append(f'{key} == "{value}"')
                else:
                    expr_parts.append(f"{key} == {value}")

            if expr_parts:
                params["expr"] = " and ".join(expr_parts)

        return params

    @staticmethod
    def convert_chunks_for_backend(
        chunks: list[DocumentChunk], backend_type: VectorDBType, **kwargs: Any
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        """
        Convert chunks to the appropriate format for a specific backend.

        Args:
            chunks: List of DocumentChunk objects
            backend_type: Target backend type
            **kwargs: Additional backend-specific parameters

        Returns:
            Converted data in backend-specific format
        """
        if backend_type == VectorDBType.CHROMADB:
            return VectorDBConverter.chunks_to_chromadb_format(chunks)
        if backend_type == VectorDBType.PINECONE:
            return VectorDBConverter.chunks_to_pinecone_format(chunks)
        if backend_type == VectorDBType.WEAVIATE:
            class_name = kwargs.get("class_name", "Document")
            return VectorDBConverter.chunks_to_weaviate_format(chunks, class_name)
        if backend_type == VectorDBType.QDRANT:
            return VectorDBConverter.chunks_to_qdrant_format(chunks)
        if backend_type == VectorDBType.MILVUS:
            return VectorDBConverter.chunks_to_milvus_format(chunks)
        raise ValueError(f"Unsupported backend type: {backend_type}")

    @staticmethod
    def convert_query_for_backend(
        query: SearchQuery, backend_type: VectorDBType
    ) -> dict[str, Any]:
        """
        Convert query to the appropriate format for a specific backend.

        Args:
            query: SearchQuery object
            backend_type: Target backend type

        Returns:
            Converted query parameters in backend-specific format
        """
        if backend_type == VectorDBType.CHROMADB:
            return VectorDBConverter.query_to_chromadb_params(query)
        if backend_type == VectorDBType.PINECONE:
            return VectorDBConverter.query_to_pinecone_params(query)
        if backend_type == VectorDBType.WEAVIATE:
            return VectorDBConverter.query_to_weaviate_params(query)
        if backend_type == VectorDBType.QDRANT:
            return VectorDBConverter.query_to_qdrant_params(query)
        if backend_type == VectorDBType.MILVUS:
            return VectorDBConverter.query_to_milvus_params(query)
        raise ValueError(f"Unsupported backend type: {backend_type}")
