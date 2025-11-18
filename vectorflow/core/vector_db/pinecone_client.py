"""Pinecone client for vector database operations."""

import time
from typing import TYPE_CHECKING, Any, Optional

try:
    import pinecone
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    pinecone = None
    Pinecone = None
    ServerlessSpec = None

from vectorflow.core.utils.logging import LoggerMixin
from vectorflow.core.vector_db.config import PineconeConfig, VectorDBType
from vectorflow.core.vector_db.converters import VectorDBConverter
from vectorflow.core.vector_db.error_handler import handle_vector_db_errors
from vectorflow.core.vector_db.interface import VectorDBInterface
from vectorflow.core.vector_db.models import (
    CollectionInfo,
    CollectionNotFoundError,
    DocumentChunk,
    DocumentInfo,
    DocumentNotFoundError,
    SearchQuery,
    SearchResult,
    VectorDBError,
)

if TYPE_CHECKING:
    from pinecone import Index as PineconeIndexType
    from pinecone import Pinecone as PineconeClientType
else:
    PineconeIndexType = Any  # type: ignore
    PineconeClientType = Any  # type: ignore


class PineconeClient(VectorDBInterface, LoggerMixin):
    """Pinecone client for vector database operations."""

    def __init__(self, config: PineconeConfig):
        """
        Initialize Pinecone client.

        Args:
            config: Pinecone configuration
        """
        self.config = config
        self._client: Optional[PineconeClientType] = None
        self._pinecone: Optional[PineconeClientType] = None
        self._index: Optional[PineconeIndexType] = None

        self.logger.info(f"Initialized PineconeClient with config: {config}")

    @property
    def client(self) -> PineconeClientType:
        """Get or create Pinecone client."""
        if self._client is None:
            self._create_client()
        return self._client

    @property
    def pinecone(self) -> PineconeClientType:
        """Get or create Pinecone client (alias for client)."""
        return self.client

    @property
    def index(self) -> PineconeIndexType:
        """Get or create Pinecone index for the configured collection."""
        if self._index is None:
            self._create_index()
        return self._index

    def _create_client(self) -> None:
        """Create Pinecone client with proper configuration."""
        try:
            if pinecone is None:
                raise VectorDBError(
                    "Pinecone client not installed. Install with: pip install pinecone",
                    backend=VectorDBType.PINECONE.value,
                )

            # Only pass api_key - environment is deprecated in newer Pinecone versions
            init_kwargs: dict[str, Any] = {"api_key": self.config.api_key}

            self._client = pinecone.Pinecone(**init_kwargs)
            self._pinecone = self._client
            self.logger.info("Successfully initialized Pinecone client")

        except Exception as exc:
            raise VectorDBError(
                f"Failed to initialize Pinecone client: {exc!s}",
                backend=VectorDBType.PINECONE.value,
                original_error=exc,
            ) from exc

    def _create_index(self) -> None:
        """Create or connect to Pinecone index."""
        index_name = self.config.index_name

        try:
            existing_indexes = self._normalize_index_names(self.pinecone.list_indexes())
            if index_name not in existing_indexes:
                create_kwargs = self._prepare_index_creation_kwargs(
                    index_name, getattr(self.config, "index_options", None)
                )
                self.pinecone.create_index(**create_kwargs)

                while True:
                    description = self.pinecone.describe_index(index_name)
                    if self._is_index_ready(description):
                        break
                    time.sleep(1)

                self.logger.info(f"Created new Pinecone index: {index_name}")

            self._index = self.pinecone.Index(index_name)
            self.logger.info(f"Connected to Pinecone index: {index_name}")

        except Exception as exc:
            raise VectorDBError(
                f"Failed to create/connect to index: {exc!s}",
                backend=VectorDBType.PINECONE.value,
                original_error=exc,
            ) from exc

    def _prepare_index_creation_kwargs(
        self,
        name: str,
        overrides: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        overrides = overrides or {}
        dimension = overrides.get("dimension", self.config.dimension)
        metric = overrides.get("metric", self.config.metric)

        kwargs: dict[str, Any] = {
            "name": name,
            "dimension": dimension,
            "metric": metric,
        }

        spec = overrides.get("spec") or getattr(self.config, "spec", None)
        if spec is not None:
            kwargs["spec"] = spec
        else:
            cloud = overrides.get("cloud") or getattr(self.config, "cloud", None)
            region = overrides.get("region") or getattr(self.config, "region", None)
            if cloud and region:
                try:
                    if ServerlessSpec is None:
                        raise ImportError("ServerlessSpec not available")

                    kwargs["spec"] = ServerlessSpec(cloud=cloud, region=region)
                except Exception:
                    pass

        pod_type = overrides.get("pod_type") or getattr(self.config, "pod_type", None)
        if pod_type and "spec" not in kwargs:
            kwargs["pod_type"] = pod_type

        return kwargs

    def _get_index(self, collection_name: Optional[str] = None) -> PineconeIndexType:
        target_name = collection_name or self.config.index_name
        if target_name == self.config.index_name:
            return self.index
        return self.pinecone.Index(target_name)

    @staticmethod
    def _normalize_index_names(indexes: Any) -> list[str]:
        names: list[str] = []
        if isinstance(indexes, dict) and "indexes" in indexes:
            indexes = indexes["indexes"]

        iterable = indexes if isinstance(indexes, (list, tuple, set)) else [indexes]

        for item in iterable:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict):
                name = item.get("name")
                if name:
                    names.append(name)
            else:
                name = getattr(item, "name", None)
                if name:
                    names.append(name)
        return names

    @staticmethod
    def _is_index_ready(description: Any) -> bool:
        status = (
            description.get("status")
            if isinstance(description, dict)
            else getattr(description, "status", None)
        )
        if isinstance(status, dict):
            return bool(status.get("ready"))
        return bool(getattr(status, "ready", False)) if status is not None else False

    @staticmethod
    def _get_attr(source: Any, name: str, default: Any = None) -> Any:
        if isinstance(source, dict):
            return source.get(name, default)
        return getattr(source, name, default)

    @staticmethod
    def _convert_chunks_to_vectors(chunks: list[DocumentChunk]) -> list[dict[str, Any]]:
        vectors: list[dict[str, Any]] = []
        for chunk in chunks:
            metadata: dict[str, Any] = dict(chunk.metadata or {})
            if "content" not in metadata:
                metadata["content"] = chunk.content
            document_id = getattr(chunk, "document_id", None)
            if document_id and "document_id" not in metadata:
                metadata["document_id"] = document_id

            vectors.append(
                {
                    "id": chunk.id,
                    "values": list(chunk.embedding),
                    "metadata": metadata,
                }
            )
        return vectors

    @handle_vector_db_errors(backend_type=VectorDBType.PINECONE, operation="add_chunks")
    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        collection_name: Optional[str] = None,
    ) -> None:
        """Add document chunks to Pinecone index."""
        if not chunks:
            return

        index = self._get_index(collection_name)

        try:
            vectors = self._convert_chunks_to_vectors(chunks)

            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i : i + batch_size]
                # Pass namespace parameter if collection_name is provided
                if collection_name:
                    index.upsert(vectors=batch, namespace=collection_name)
                else:
                    index.upsert(vectors=batch)

            self.logger.info(
                f"Added {len(chunks)} chunks to Pinecone index {collection_name or self.config.index_name}"
            )

        except Exception as exc:
            raise VectorDBError(
                f"Failed to add chunks to Pinecone: {exc!s}",
                backend=VectorDBType.PINECONE.value,
                original_error=exc,
            ) from exc

    @handle_vector_db_errors(backend_type=VectorDBType.PINECONE, operation="search")
    def search(
        self,
        query: SearchQuery,
        query_embedding: Optional[list[float]] = None,
        collection_name: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search for similar documents in Pinecone."""
        index = self._get_index(collection_name)

        try:
            if not query_embedding:
                raise VectorDBError(
                    "query_embedding is required for search",
                    backend=VectorDBType.PINECONE.value,
                ) from None

            search_params = VectorDBConverter.query_to_pinecone_params(query)

            # Build query arguments, merging namespace if provided
            query_kwargs = {"vector": query_embedding, **search_params}

            if collection_name:
                query_kwargs["namespace"] = collection_name

            results = index.query(**query_kwargs)

            matches = (
                getattr(results, "matches", None) or results.get("matches", [])
                if isinstance(results, dict)
                else []
            )
            search_results: list[SearchResult] = []
            for match in matches:
                metadata = (
                    getattr(match, "metadata", None) or match.get("metadata", {})
                    if isinstance(match, dict)
                    else {}
                )
                content = metadata.get("content", "")
                match_id = getattr(match, "id", None) or match.get("id")
                score = getattr(match, "score", None) or match.get("score")

                if match_id is None:
                    continue

                search_results.append(
                    SearchResult(
                        id=str(match_id),
                        content=content,
                        score=score,
                        metadata=metadata,
                    )
                )

            self.logger.debug(
                f"Found {len(search_results)} results for query in index {collection_name or self.config.index_name}"
            )
            return search_results

        except Exception as exc:
            raise VectorDBError(
                f"Search failed in Pinecone: {exc!s}",
                backend=VectorDBType.PINECONE.value,
                original_error=exc,
            ) from exc

    @handle_vector_db_errors(backend_type=VectorDBType.PINECONE, operation="get_chunks")
    def get_chunks(
        self,
        chunk_ids: list[str],
        collection_name: Optional[str] = None,
        include_embeddings: bool = False,
    ) -> list[DocumentChunk]:
        """Retrieve specific chunks by their IDs."""
        index = self._get_index(collection_name)

        try:
            response = index.fetch(ids=chunk_ids)

            vectors = (
                getattr(response, "vectors", None) or response.get("vectors", {})
                if isinstance(response, dict)
                else {}
            )
            chunks: list[DocumentChunk] = []

            for chunk_id, vector_data in vectors.items():
                metadata = (
                    getattr(vector_data, "metadata", None)
                    or vector_data.get("metadata", {})
                    if isinstance(vector_data, dict)
                    else {}
                )
                values = (
                    getattr(vector_data, "values", None)
                    or vector_data.get("values", [])
                    if isinstance(vector_data, dict)
                    else []
                )

                chunk = DocumentChunk(
                    id=str(chunk_id),
                    content=metadata.get("content", ""),
                    embedding=list(values) if include_embeddings else [],
                    metadata=metadata,
                )
                chunks.append(chunk)

            return chunks

        except Exception as exc:
            raise VectorDBError(
                f"Failed to retrieve chunks from Pinecone: {exc!s}",
                backend=VectorDBType.PINECONE.value,
                original_error=exc,
            ) from exc

    def get_chunk(
        self,
        chunk_id: str,
        collection_name: Optional[str] = None,
        include_embeddings: bool = False,
    ) -> DocumentChunk:
        """
        Retrieve a specific chunk by its ID.

        Args:
            chunk_id: ID of the chunk to retrieve
            collection_name: Source collection name (uses default if None)
            include_embeddings: Whether to include embedding vectors

        Returns:
            DocumentChunk object

        Raises:
            DocumentNotFoundError: If chunk is not found
            VectorDBError: If retrieval fails
        """
        chunks = self.get_chunks([chunk_id], collection_name, include_embeddings)
        if not chunks:
            raise DocumentNotFoundError(f"Chunk '{chunk_id}' not found")
        return chunks[0]

    @handle_vector_db_errors(
        backend_type=VectorDBType.PINECONE, operation="update_chunks"
    )
    def update_chunks(
        self,
        chunks: list[DocumentChunk],
        collection_name: Optional[str] = None,
    ) -> None:
        """Update existing document chunks."""
        self.add_chunks(chunks, collection_name)

    def update_chunk(
        self, chunk: DocumentChunk, collection_name: Optional[str] = None
    ) -> None:
        """
        Update a single chunk (convenience method).
        """
        self.update_chunks([chunk], collection_name)

    @handle_vector_db_errors(
        backend_type=VectorDBType.PINECONE, operation="delete_chunks"
    )
    def delete_chunks(
        self,
        chunk_ids: list[str],
        collection_name: Optional[str] = None,
    ) -> None:
        """Delete specific chunks by their IDs."""
        index = self._get_index(collection_name)

        try:
            # Pass namespace parameter if collection_name is provided
            if collection_name:
                index.delete(ids=chunk_ids, namespace=collection_name)
            else:
                index.delete(ids=chunk_ids)
            self.logger.info(
                f"Deleted {len(chunk_ids)} chunks from Pinecone index {collection_name or self.config.index_name}"
            )

        except Exception as exc:
            raise VectorDBError(
                f"Failed to delete chunks from Pinecone: {exc!s}",
                backend=VectorDBType.PINECONE.value,
                original_error=exc,
            ) from exc

    @handle_vector_db_errors(
        backend_type=VectorDBType.PINECONE, operation="delete_document"
    )
    def delete_document(
        self, document_id: str, collection_name: Optional[str] = None
    ) -> int:
        """Delete all chunks belonging to a document."""
        index = self._get_index(collection_name)

        try:
            # Query for all chunks with this document_id
            query_response = index.query(
                vector=[0.0] * self.config.dimension,  # Dummy vector
                filter={"document_id": {"$eq": document_id}},
                top_k=10000,  # Large number to get all chunks
                include_metadata=True,
            )

            matches = (
                getattr(query_response, "matches", None)
                or query_response.get("matches", [])
                if isinstance(query_response, dict)
                else []
            )

            if not matches:
                return 0

            # Extract chunk IDs
            chunk_ids = [
                match.id if hasattr(match, "id") else match.get("id")
                for match in matches
            ]
            # Filter out None values
            chunk_ids = [cid for cid in chunk_ids if cid]

            if chunk_ids:
                # Delete all chunks for this document
                index.delete(ids=chunk_ids)
                self.logger.info(
                    f"Deleted {len(chunk_ids)} chunks for document {document_id}"
                )
                return len(chunk_ids)

            return 0

        except Exception as exc:
            raise VectorDBError(
                f"Failed to delete document from Pinecone: {exc!s}",
                backend=VectorDBType.PINECONE.value,
                original_error=exc,
            ) from exc

    def get_collection(self, name: Optional[str] = None) -> Any:
        """Get an existing collection (index in Pinecone)."""
        return self._get_index(name)

    def collection_exists(self, name: Optional[str] = None) -> bool:
        """
        Check if a collection (index) exists.

        Args:
            name: Collection name (uses default if None)

        Returns:
            True if collection exists, False otherwise

        Raises:
            VectorDBError: If check operation fails
        """
        index_name = name or self.config.index_name
        try:
            indexes = self._normalize_index_names(self.pinecone.list_indexes())
            return index_name in indexes
        except Exception as e:
            error_msg = f"Failed to check if index '{index_name}' exists: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    @handle_vector_db_errors(
        backend_type=VectorDBType.PINECONE, operation="search_by_metadata"
    )
    def search_by_metadata(
        self,
        metadata_filter: dict[str, Any],
        collection_name: Optional[str] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search for documents by metadata filters only."""
        index = self._get_index(collection_name)

        try:
            zero_vector = [0.0] * self.config.dimension

            results = index.query(
                vector=zero_vector,
                top_k=limit,
                include_metadata=True,
                filter=metadata_filter,
            )

            matches = (
                getattr(results, "matches", None) or results.get("matches", [])
                if isinstance(results, dict)
                else []
            )
            search_results: list[SearchResult] = []

            for match in matches:
                metadata = (
                    getattr(match, "metadata", None) or match.get("metadata", {})
                    if isinstance(match, dict)
                    else {}
                )
                content = metadata.get("content", "")
                match_id = getattr(match, "id", None) or match.get("id")

                if match_id is None:
                    continue

                search_results.append(
                    SearchResult(
                        id=str(match_id),
                        content=content,
                        score=1.0,
                        metadata=metadata,
                    )
                )

            return search_results

        except Exception as exc:
            raise VectorDBError(
                f"Metadata search failed in Pinecone: {exc!s}",
                backend=VectorDBType.PINECONE.value,
                original_error=exc,
            ) from exc

    def get_document_info(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> DocumentInfo:
        """
        Get information about a document.

        Args:
            document_id: Document ID
            collection_name: Source collection name (uses default if None)

        Returns:
            DocumentInfo object

        """
        index = self._get_index(collection_name)

        try:
            # Query using metadata filter only; use a zero vector as query anchor
            zero_vector = [0.0] * self.config.dimension
            response = index.query(
                vector=zero_vector,
                top_k=10000,
                include_metadata=True,
                filter={"document_id": {"$eq": document_id}},
            )

            matches = (
                getattr(response, "matches", None) or response.get("matches", [])
                if isinstance(response, dict)
                else []
            )

            if not matches:
                raise DocumentNotFoundError(f"Document {document_id} not found")

            chunk_count = 0
            total_characters = 0
            page_numbers: set[int] = set()
            created_times: list[float] = []
            filename: Optional[str] = None

            for match in matches:
                chunk_count += 1
                metadata = (
                    getattr(match, "metadata", None) or match.get("metadata", {})
                    if isinstance(match, dict)
                    else {}
                )
                content = metadata.get("content", "")
                if isinstance(content, str) and content:
                    total_characters += len(content)
                else:
                    try:
                        total_characters += int(metadata.get("content_length", 0) or 0)
                    except Exception:
                        pass

                if "page_number" in metadata:
                    try:
                        page_numbers.add(int(metadata["page_number"]))
                    except Exception:
                        pass

                if "created_at" in metadata:
                    try:
                        created_times.append(float(metadata["created_at"]))
                    except Exception:
                        pass

                if filename is None:
                    filename = metadata.get("filename") or metadata.get("file_name")

            created_at_str = str(int(min(created_times))) if created_times else None

            return DocumentInfo(
                document_id=document_id,
                chunk_count=chunk_count,
                total_characters=total_characters,
                filename=filename,
                page_count=len(page_numbers) if page_numbers else None,
                created_at=created_at_str,
            )

        except DocumentNotFoundError:
            raise
        except Exception as exc:
            raise VectorDBError(
                f"Failed to get document info: {exc!s}",
                backend=VectorDBType.PINECONE.value,
                original_error=exc,
            ) from exc

    def find_similar_chunks(
        self,
        chunk_id: str,
        collection_name: Optional[str] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Find chunks similar to a given chunk."""
        index = self._get_index(collection_name)

        try:
            chunks = self.get_chunks(
                [chunk_id],
                collection_name=collection_name,
                include_embeddings=True,
            )
            if not chunks or not chunks[0].embedding:
                raise DocumentNotFoundError(
                    f"Chunk {chunk_id} not found or has no embedding"
                )

            reference_embedding = chunks[0].embedding

            results = index.query(
                vector=reference_embedding,
                top_k=limit + 1,
                include_metadata=True,
            )

            matches = (
                getattr(results, "matches", None) or results.get("matches", [])
                if isinstance(results, dict)
                else []
            )
            search_results: list[SearchResult] = []

            for match in matches:
                match_id = getattr(match, "id", None) or match.get("id")
                if match_id is None or str(match_id) == chunk_id:
                    continue

                metadata = (
                    getattr(match, "metadata", None) or match.get("metadata", {})
                    if isinstance(match, dict)
                    else {}
                )
                content = metadata.get("content", "")
                score = getattr(match, "score", None) or match.get("score")

                search_results.append(
                    SearchResult(
                        id=str(match_id),
                        content=content,
                        score=score,
                        metadata=metadata,
                    )
                )

            return search_results[:limit]

        except DocumentNotFoundError:
            raise
        except Exception as exc:
            raise VectorDBError(
                f"Failed to find similar chunks: {exc!s}",
                backend=VectorDBType.PINECONE.value,
                original_error=exc,
            ) from exc

    def create_collection(
        self,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        get_or_create: bool = False,
        **kwargs: Any,
    ) -> bool:
        """Create a new collection (index in Pinecone).

        Args:
            name: Index name
            metadata: Metadata for the index
            get_or_create: If True, return existing index if it exists
            **kwargs: Additional Pinecone-specific parameters (dimension, metric, spec, cloud, region)

        Returns:
            True if index was created or already exists
        """
        index_name = name or self.config.index_name
        metadata = metadata or {}

        # Merge kwargs into metadata for index creation
        overrides = {**metadata, **kwargs}

        try:
            existing_indexes = self._normalize_index_names(self.pinecone.list_indexes())
            if index_name in existing_indexes:
                if not get_or_create:
                    raise VectorDBError(
                        f"Index {index_name} already exists",
                        backend=VectorDBType.PINECONE.value,
                    ) from None
                return True

            create_kwargs = self._prepare_index_creation_kwargs(index_name, overrides)
            self.pinecone.create_index(**create_kwargs)

            while True:
                description = self.pinecone.describe_index(index_name)
                if self._is_index_ready(description):
                    break
                time.sleep(1)

            if index_name == self.config.index_name:
                self._index = self.pinecone.Index(index_name)

            self.logger.info(f"Created Pinecone index: {index_name}")
            return True

        except Exception as exc:
            raise VectorDBError(
                f"Failed to create index: {exc!s}",
                backend=VectorDBType.PINECONE.value,
                original_error=exc,
            ) from exc

    def delete_collection(self, name: Optional[str] = None) -> None:
        """Delete a collection (index in Pinecone).

        Raises:
            CollectionNotFoundError: If index doesn't exist
            VectorDBError: If deletion fails
        """
        index_name = name or self.config.index_name

        try:
            self.pinecone.delete_index(index_name)
            if index_name == self.config.index_name:
                self._index = None

            self.logger.info(f"Deleted Pinecone index: {index_name}")

        except Exception as exc:
            # Check if it's a "not found" error
            error_msg = str(exc).lower()
            if "not found" in error_msg or "does not exist" in error_msg:
                raise CollectionNotFoundError(
                    f"Index {index_name} does not exist"
                ) from exc
            raise VectorDBError(
                f"Failed to delete index: {exc!s}",
                backend=VectorDBType.PINECONE.value,
                original_error=exc,
            ) from exc

    def list_collections(self) -> list[CollectionInfo]:
        """List all collections (indexes in Pinecone)."""
        try:
            index_names = self._normalize_index_names(self.pinecone.list_indexes())
            return [CollectionInfo(name=name) for name in index_names]

        except Exception as exc:
            raise VectorDBError(
                f"Failed to list indexes: {exc!s}",
                backend=VectorDBType.PINECONE.value,
                original_error=exc,
            ) from exc

    def get_collection_info(self, name: Optional[str] = None) -> CollectionInfo:
        """Get information about a collection (index in Pinecone)."""
        index_name = name or self.config.index_name
        index = self._get_index(index_name)

        try:
            index_stats = index.describe_index_stats()

            # Try to get index description, but handle gracefully if not available
            try:
                index_description = self.pinecone.describe_index(index_name)
            except Exception:
                # If describe_index fails, try to get info from stats or use defaults
                index_description = {}

            total_vector_count = self._get_attr(index_stats, "total_vector_count", 0)
            namespaces = self._get_attr(index_stats, "namespaces", {})

            # Get dimension from various sources, with fallback to config
            dimension = (
                self._get_attr(index_stats, "dimension")
                or self._get_attr(index_description, "dimension")
                or self.config.dimension
            )

            metric = self._get_attr(index_description, "metric", self.config.metric)
            status = self._get_attr(index_description, "status", {})

            return CollectionInfo(
                name=index_name,
                count=int(total_vector_count or 0),
                metadata={
                    "dimension": dimension,
                    "metric": metric,
                    "status": status,
                    "namespaces": namespaces,
                },
            )

        except Exception as exc:
            raise VectorDBError(
                f"Failed to get index info: {exc!s}",
                backend=VectorDBType.PINECONE.value,
                original_error=exc,
            ) from exc

    def count_chunks(self, collection_name: Optional[str] = None) -> int:
        """Count total number of chunks in the index."""
        index = self._get_index(collection_name)

        try:
            stats = index.describe_index_stats()
            total_vector_count = self._get_attr(stats, "total_vector_count", 0)
            return int(total_vector_count or 0)
        except Exception as exc:
            raise VectorDBError(
                f"Failed to count chunks: {exc!s}",
                backend=VectorDBType.PINECONE.value,
                original_error=exc,
            ) from exc

    def clear_collection(self, collection_name: Optional[str] = None) -> None:
        """Remove all chunks from the index."""
        index = self._get_index(collection_name)

        try:
            index.delete(delete_all=True)
            self.logger.info(
                f"Cleared all chunks from Pinecone index {collection_name or self.config.index_name}"
            )

        except Exception as exc:
            raise VectorDBError(
                f"Failed to clear index: {exc!s}",
                backend=VectorDBType.PINECONE.value,
                original_error=exc,
            ) from exc

    def get_backend_info(self) -> dict[str, Any]:
        """Get information about the Pinecone backend."""
        try:
            return {
                "backend": VectorDBType.PINECONE.value,
                "type": "cloud",
                "environment": getattr(self.config, "environment", None),
                "index_name": self.config.index_name,
                "dimension": self.config.dimension,
                "metric": self.config.metric,
                "capabilities": [
                    "cloud_storage",
                    "high_performance",
                    "metadata_filtering",
                    "similarity_search",
                    "real_time_updates",
                    "horizontal_scaling",
                ],
            }
        except Exception as exc:
            return {
                "backend": VectorDBType.PINECONE.value,
                "error": str(exc),
            }

    def health_check(self) -> bool:
        """Check if Pinecone is healthy and accessible."""
        try:
            self.index.describe_index_stats()
            return True
        except Exception as exc:
            self.logger.warning(f"Pinecone health check failed: {exc!s}")
            return False
