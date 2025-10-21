"""Milvus client for vector database operations."""

from typing import Any, Optional

try:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )
except ImportError:
    connections = None
    utility = None
    Collection = None
    CollectionSchema = None
    DataType = None
    FieldSchema = None

from pdf_vector_system.utils.logging import LoggerMixin
from pdf_vector_system.vector_db.config import MilvusConfig
from pdf_vector_system.vector_db.converters import VectorDBConverter
from pdf_vector_system.vector_db.error_handler import handle_vector_db_errors
from pdf_vector_system.vector_db.interface import VectorDBInterface
from pdf_vector_system.vector_db.models import (
    CollectionInfo,
    DocumentChunk,
    DocumentInfo,
    DocumentNotFoundError,
    SearchQuery,
    SearchResult,
    VectorDBError,
)


class MilvusClient(VectorDBInterface, LoggerMixin):
    """Milvus client for vector database operations."""

    def __init__(self, config: MilvusConfig):
        """
        Initialize Milvus client.

        Args:
            config: Milvus configuration
        """
        self.config = config
        self._client = None
        self._connections = None

        self.logger.info(f"Initialized MilvusClient with config: {config}")

    @property
    def client(self):
        """Get or create Milvus client (connections)."""
        if self._client is None:
            self._create_connection()
        return self._client

    @property
    def connections(self):
        """Get or create Milvus connections (alias for client)."""
        return self.client

    def _create_connection(self) -> None:
        """Create Milvus connection with proper configuration."""
        try:
            if connections is None:
                raise VectorDBError(
                    "Milvus client not installed. Install with: pip install pymilvus",
                    backend="milvus",
                )

            # Connect to Milvus
            connections.connect(
                alias="default",
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
            )

            self._client = connections
            self._connections = connections
            self.logger.info("Successfully connected to Milvus")

        except Exception as e:
            raise VectorDBError(
                f"Failed to connect to Milvus: {e!s}",
                backend="milvus",
                original_error=e,
            ) from e

    def _ensure_collection_exists(
        self, collection_name: str, vector_dim: Optional[int] = None
    ) -> None:
        """Ensure Milvus collection exists, create if not."""
        try:
            if Collection is None or CollectionSchema is None:
                raise VectorDBError(
                    "Milvus client not installed. Install with: pip install pymilvus",
                    backend="milvus",
                )

            if utility is None:
                raise VectorDBError(
                    "Milvus utility not available",
                    backend="milvus",
                )

            # Use provided vector_dim or fall back to config
            dimension = vector_dim or self.config.dimension

            # Check if collection exists
            try:
                collection_exists = utility.has_collection(collection_name)
            except Exception:
                # If check fails (e.g., no connection in tests), assume collection doesn't exist
                collection_exists = False

            if not collection_exists:
                # Define collection schema
                fields = [
                    FieldSchema(
                        name="id",
                        dtype=DataType.VARCHAR,
                        max_length=255,
                        is_primary=True,
                    ),
                    FieldSchema(
                        name="content", dtype=DataType.VARCHAR, max_length=65535
                    ),
                    FieldSchema(
                        name="embedding",
                        dtype=DataType.FLOAT_VECTOR,
                        dim=dimension,
                    ),
                    FieldSchema(
                        name="document_id", dtype=DataType.VARCHAR, max_length=255
                    ),
                    FieldSchema(name="page_number", dtype=DataType.INT64),
                    FieldSchema(name="chunk_index", dtype=DataType.INT64),
                    FieldSchema(name="created_at", dtype=DataType.DOUBLE),
                ]

                schema = CollectionSchema(
                    fields=fields, description=f"Document chunks for {collection_name}"
                )

                # Create collection
                collection = Collection(name=collection_name, schema=schema)

                # Create index for vector field
                index_params = {
                    "metric_type": self.config.metric_type,
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024},
                }
                collection.create_index(
                    field_name="embedding", index_params=index_params
                )

                self.logger.info(f"Created Milvus collection: {collection_name}")

        except Exception as e:
            raise VectorDBError(
                f"Failed to ensure collection exists: {e!s}",
                backend="milvus",
                original_error=e,
            ) from e

    @handle_vector_db_errors(backend_type="milvus", operation="add_chunks")
    def add_chunks(
        self, chunks: list[DocumentChunk], collection_name: Optional[str] = None
    ) -> None:
        """Add document chunks to Milvus collection."""
        if not chunks:
            return

        try:
            if Collection is None:
                raise VectorDBError(
                    "Milvus Collection not available",
                    backend="milvus",
                )

            collection_name = collection_name or self.config.collection_name
            self._ensure_collection_exists(collection_name)

            # Convert chunks to Milvus format
            data = VectorDBConverter.chunks_to_milvus_format(chunks)

            # Insert data
            collection = Collection(collection_name)
            collection.insert(data)
            collection.flush()

            self.logger.info(
                f"Added {len(chunks)} chunks to Milvus collection {collection_name}"
            )

        except Exception as e:
            raise VectorDBError(
                f"Failed to add chunks to Milvus: {e!s}",
                backend="milvus",
                original_error=e,
            )

    @handle_vector_db_errors(backend_type="milvus", operation="search")
    def search(
        self,
        query: SearchQuery,
        query_embedding: Optional[list[float]] = None,
        collection_name: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search for similar documents in Milvus."""
        try:
            if Collection is None:
                raise VectorDBError(
                    "Milvus Collection not available",
                    backend="milvus",
                )

            if not query_embedding:
                raise VectorDBError(
                    "query_embedding is required for Milvus search", backend="milvus"
                )

            collection_name = collection_name or self.config.collection_name
            collection = Collection(collection_name)

            # Load collection into memory
            collection.load()

            # Build search parameters
            search_params = VectorDBConverter.query_to_milvus_params(query)

            # Build expression filter if metadata filter is provided
            expr = None
            if query.where:
                # Convert metadata filter to Milvus expression
                # For simple equality filters like {"document_id": "doc1"}
                expr_parts = []
                for key, value in query.where.items():
                    if isinstance(value, str):
                        expr_parts.append(f'{key} == "{value}"')
                    else:
                        expr_parts.append(f"{key} == {value}")
                expr = " and ".join(expr_parts)

            # Perform search
            search_kwargs = {
                "data": [query_embedding],
                "anns_field": "embedding",
                "param": search_params,
                "limit": query.n_results,
                "output_fields": [
                    "id",
                    "content",
                    "document_id",
                    "page_number",
                    "chunk_index",
                    "created_at",
                ],
            }

            if expr:
                search_kwargs["expr"] = expr

            results = collection.search(**search_kwargs)

            # Convert results to SearchResult objects
            search_results = []
            for hits in results:
                for hit in hits:
                    entity = hit.entity

                    # Get ID from hit object or entity
                    chunk_id = getattr(hit, "id", entity.get("id", ""))

                    # Convert distance to score (1 - distance for similarity)
                    # Milvus returns distance, we need to convert to similarity score
                    if hasattr(hit, "distance"):
                        score = 1.0 - hit.distance
                    elif hasattr(hit, "score"):
                        score = hit.score
                    else:
                        score = 0.0

                    search_result = SearchResult(
                        id=chunk_id,
                        content=entity.get("content", ""),
                        score=score,
                        metadata={
                            "document_id": entity.get("document_id"),
                            "page_number": entity.get("page_number"),
                            "chunk_index": entity.get("chunk_index"),
                            "created_at": entity.get("created_at"),
                        },
                    )
                    search_results.append(search_result)

            self.logger.debug(f"Found {len(search_results)} results for query")
            return search_results

        except Exception as e:
            raise VectorDBError(
                f"Search failed in Milvus: {e!s}", backend="milvus", original_error=e
            )

    @handle_vector_db_errors(backend_type="milvus", operation="get_chunks")
    def get_chunks(
        self,
        chunk_ids: list[str],
        collection_name: Optional[str] = None,
        include_embeddings: bool = False,
    ) -> list[DocumentChunk]:
        """Retrieve specific chunks by their IDs."""
        try:
            if Collection is None:
                raise VectorDBError(
                    "Milvus Collection not available",
                    backend="milvus",
                )

            collection_name = collection_name or self.config.collection_name
            collection = Collection(collection_name)
            collection.load()

            # Query by IDs
            output_fields = [
                "id",
                "content",
                "document_id",
                "page_number",
                "chunk_index",
                "created_at",
            ]
            if include_embeddings:
                output_fields.append("embedding")

            expr = f"id in {chunk_ids}"
            results = collection.query(expr=expr, output_fields=output_fields)

            chunks = []
            for result in results:
                embedding = result.get("embedding", []) if include_embeddings else []

                chunk = DocumentChunk(
                    id=result.get("id", ""),
                    content=result.get("content", ""),
                    embedding=embedding,
                    metadata={
                        "document_id": result.get("document_id"),
                        "page_number": result.get("page_number"),
                        "chunk_index": result.get("chunk_index"),
                        "created_at": result.get("created_at"),
                    },
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            raise VectorDBError(
                f"Failed to retrieve chunks from Milvus: {e!s}",
                backend="milvus",
                original_error=e,
            )

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
            VectorDBError: If retrieval fails
            DocumentNotFoundError: If chunk is not found
        """
        chunks = self.get_chunks([chunk_id], collection_name, include_embeddings)
        if not chunks:
            raise DocumentNotFoundError(f"Chunk '{chunk_id}' not found")
        return chunks[0]

    @handle_vector_db_errors(backend_type="milvus", operation="update_chunks")
    def update_chunks(
        self, chunks: list[DocumentChunk], collection_name: Optional[str] = None
    ) -> None:
        """Update existing document chunks."""
        try:
            if Collection is None:
                raise VectorDBError(
                    "Milvus Collection not available",
                    backend="milvus",
                )

            collection_name = collection_name or self.config.collection_name
            collection = Collection(collection_name)

            # Delete existing chunks
            chunk_ids = [chunk.id for chunk in chunks]
            expr = f"id in {chunk_ids}"
            collection.delete(expr)

            # Insert updated chunks
            self.add_chunks(chunks, collection_name)

            self.logger.info(
                f"Updated {len(chunks)} chunks in Milvus collection {collection_name}"
            )

        except Exception as e:
            raise VectorDBError(
                f"Failed to update chunks in Milvus: {e!s}",
                backend="milvus",
                original_error=e,
            )

    def update_chunk(
        self, chunk: DocumentChunk, collection_name: Optional[str] = None
    ) -> None:
        """
        Update a single chunk (convenience method).
        """
        self.update_chunks([chunk], collection_name)

    @handle_vector_db_errors(backend_type="milvus", operation="delete_chunks")
    def delete_chunks(
        self, chunk_ids: list[str], collection_name: Optional[str] = None
    ) -> None:
        """Delete specific chunks by their IDs."""
        try:
            if Collection is None:
                raise VectorDBError(
                    "Milvus Collection not available",
                    backend="milvus",
                )

            collection_name = collection_name or self.config.collection_name
            collection = Collection(collection_name)

            expr = f"id in {chunk_ids}"
            collection.delete(expr)

            self.logger.info(
                f"Deleted {len(chunk_ids)} chunks from Milvus collection {collection_name}"
            )

        except Exception as e:
            raise VectorDBError(
                f"Failed to delete chunks from Milvus: {e!s}",
                backend="milvus",
                original_error=e,
            )

    @handle_vector_db_errors(backend_type="milvus", operation="delete_document")
    def delete_document(
        self, document_id: str, collection_name: Optional[str] = None
    ) -> int:
        """Delete all chunks belonging to a document."""
        try:
            if Collection is None:
                raise VectorDBError(
                    "Milvus Collection not available",
                    backend="milvus",
                )

            collection_name = collection_name or self.config.collection_name
            collection = Collection(collection_name)
            collection.load()

            # Query for all chunks with this document_id
            expr = f'document_id == "{document_id}"'
            results = collection.query(
                expr=expr,
                output_fields=["id"],
                limit=10000,  # Large number to get all chunks
            )

            if not results:
                return 0

            # Extract chunk IDs
            chunk_ids = [str(result["id"]) for result in results]

            if chunk_ids:
                # Delete all chunks for this document
                delete_expr = f"id in {chunk_ids}"
                collection.delete(delete_expr)

                self.logger.info(
                    f"Deleted {len(chunk_ids)} chunks for document {document_id}"
                )
                return len(chunk_ids)

            return 0

        except Exception as e:
            raise VectorDBError(
                f"Failed to delete document from Milvus: {e!s}",
                backend="milvus",
                original_error=e,
            )

    def get_collection(self, name: Optional[str] = None) -> Any:
        """Get an existing collection in Milvus."""
        if Collection is None:
            raise VectorDBError(
                "Milvus Collection not available",
                backend="milvus",
            )

        collection_name = name or self.config.collection_name
        return Collection(collection_name)

    def collection_exists(self, name: Optional[str] = None) -> bool:
        """
        Check if a collection exists.

        Args:
            name: Collection name (uses default if None)

        Returns:
            True if collection exists, False otherwise

        Raises:
            VectorDBError: If check operation fails
        """
        collection_name = name or self.config.collection_name
        try:
            if utility is None:
                raise VectorDBError(
                    "Milvus utility not available",
                    backend="milvus",
                )
            return utility.has_collection(collection_name)
        except Exception as e:
            error_msg = (
                f"Failed to check if collection '{collection_name}' exists: {e!s}"
            )
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    @handle_vector_db_errors(backend_type="milvus", operation="search_by_metadata")
    def search_by_metadata(
        self,
        metadata_filter: dict[str, Any],
        collection_name: Optional[str] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search for documents by metadata filters only."""
        try:
            if Collection is None:
                raise VectorDBError(
                    "Milvus Collection not available",
                    backend="milvus",
                )

            collection_name = collection_name or self.config.collection_name
            collection = Collection(collection_name)
            collection.load()

            # Build filter expression
            conditions = []
            for key, value in metadata_filter.items():
                if isinstance(value, str):
                    conditions.append(f'{key} == "{value}"')
                else:
                    conditions.append(f"{key} == {value}")

            expr = " and ".join(conditions) if conditions else ""

            # Query with filter
            results = collection.query(
                expr=expr,
                output_fields=[
                    "id",
                    "content",
                    "document_id",
                    "page_number",
                    "chunk_index",
                    "created_at",
                ],
                limit=limit,
            )

            search_results = []
            for result in results:
                search_result = SearchResult(
                    id=result.get("id", ""),
                    content=result.get("content", ""),
                    score=1.0,  # No similarity score for metadata-only search
                    metadata={
                        "document_id": result.get("document_id"),
                        "page_number": result.get("page_number"),
                        "chunk_index": result.get("chunk_index"),
                        "created_at": result.get("created_at"),
                    },
                )
                search_results.append(search_result)

            return search_results

        except Exception as e:
            raise VectorDBError(
                f"Metadata search failed in Milvus: {e!s}",
                backend="milvus",
                original_error=e,
            )

    def get_document_info(
        self, document_id: str, collection_name: Optional[str] = None
    ) -> DocumentInfo:
        """Get information about a document."""
        try:
            # Search for chunks with this document_id
            results = self.search_by_metadata(
                {"document_id": document_id},
                collection_name=collection_name,
                limit=1000,  # Get all chunks for this document
            )

            if not results:
                raise DocumentNotFoundError(f"Document {document_id} not found")

            chunk_count = len(results)
            total_characters = sum(len(r.content) for r in results)

            # Extract metadata information
            page_numbers = set()
            created_times = []

            for result in results:
                metadata = result.metadata
                if metadata.get("page_number") is not None:
                    page_numbers.add(metadata["page_number"])
                if metadata.get("created_at") is not None:
                    created_times.append(float(metadata["created_at"]))

            return DocumentInfo(
                document_id=document_id,
                chunk_count=chunk_count,
                total_characters=total_characters,
                page_numbers=sorted(page_numbers),
                created_at=min(created_times) if created_times else None,
                updated_at=max(created_times) if created_times else None,
            )

        except DocumentNotFoundError:
            raise
        except Exception as e:
            raise VectorDBError(
                f"Failed to get document info: {e!s}",
                backend="milvus",
                original_error=e,
            )

    def find_similar_chunks(
        self, chunk_id: str, collection_name: Optional[str] = None, limit: int = 10
    ) -> list[SearchResult]:
        """Find chunks similar to a given chunk."""
        try:
            # Get the reference chunk with embedding
            chunks = self.get_chunks(
                [chunk_id], collection_name, include_embeddings=True
            )
            if not chunks or not chunks[0].embedding:
                raise DocumentNotFoundError(
                    f"Chunk {chunk_id} not found or has no embedding"
                )

            reference_embedding = chunks[0].embedding
            collection_name = collection_name or self.config.collection_name

            # Create a temporary query to search for similar chunks
            query = SearchQuery(
                query_text="",  # Not used for embedding search
                n_results=limit + 1,  # +1 to exclude the reference chunk
            )

            # Search for similar chunks
            results = self.search(query, reference_embedding, collection_name)

            # Filter out the reference chunk
            filtered_results = [r for r in results if r.id != chunk_id]
            return filtered_results[:limit]

        except DocumentNotFoundError:
            raise
        except Exception as e:
            raise VectorDBError(
                f"Failed to find similar chunks: {e!s}",
                backend="milvus",
                original_error=e,
            )

    def create_collection(
        self,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        get_or_create: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Create a new collection in Milvus.

        Args:
            name: Collection name (uses default if None)
            metadata: Optional metadata (not used in Milvus)
            get_or_create: If True, get existing collection or create new one (default: False)
            **kwargs: Backend-specific parameters (e.g., vector_dim for Milvus)

        Returns:
            True if collection was created or retrieved successfully
        """
        collection_name = name or self.config.collection_name
        vector_dim = kwargs.get("vector_dim")

        try:
            if utility is None:
                raise VectorDBError(
                    "Milvus utility not available",
                    backend="milvus",
                )

            # Check if collection already exists
            if utility.has_collection(collection_name):
                if get_or_create:
                    self.logger.debug(f"Collection {collection_name} already exists")
                    return True
                raise VectorDBError(
                    f"Collection {collection_name} already exists",
                    backend="milvus",
                )

            # Create collection using the ensure method
            self._ensure_collection_exists(collection_name, vector_dim=vector_dim)
            self.logger.info(f"Created Milvus collection: {collection_name}")
            return True

        except Exception as e:
            raise VectorDBError(
                f"Failed to create collection: {e!s}",
                backend="milvus",
                original_error=e,
            )

    def delete_collection(self, name: Optional[str] = None) -> bool:
        """Delete a collection in Milvus.

        Returns:
            True if collection was deleted successfully
        """
        collection_name = name or self.config.collection_name
        try:
            if utility is None:
                raise VectorDBError(
                    "Milvus utility not available",
                    backend="milvus",
                )

            utility.drop_collection(collection_name)
            self.logger.info(f"Deleted Milvus collection: {collection_name}")
            return True

        except Exception as e:
            raise VectorDBError(
                f"Failed to delete collection: {e!s}",
                backend="milvus",
                original_error=e,
            )

    def list_collections(self) -> list[str]:
        """List all collections in Milvus.

        Returns:
            List of collection names
        """
        try:
            if utility is None:
                raise VectorDBError(
                    "Milvus utility not available",
                    backend="milvus",
                )

            return utility.list_collections()
        except Exception as e:
            raise VectorDBError(
                f"Failed to list collections: {e!s}", backend="milvus", original_error=e
            )

    def _get_collection(self, name: str):
        """Get a collection object."""
        if Collection is None:
            raise VectorDBError(
                "Milvus Collection not available",
                backend="milvus",
            )

        return Collection(name)

    def get_collection_info(self, name: Optional[str] = None) -> CollectionInfo:
        """Get information about a collection in Milvus."""
        collection_name = name or self.config.collection_name

        try:
            if Collection is None:
                raise VectorDBError(
                    "Milvus Collection not available",
                    backend="milvus",
                )

            collection = Collection(collection_name)

            return CollectionInfo(
                name=collection_name,
                count=collection.num_entities,
                metadata={
                    "description": collection.description,
                    "schema": str(collection.schema),
                    "is_empty": collection.is_empty,
                    "primary_field": (
                        collection.primary_field.name
                        if collection.primary_field
                        else None
                    ),
                },
            )

        except Exception as e:
            raise VectorDBError(
                f"Failed to get collection info: {e!s}",
                backend="milvus",
                original_error=e,
            )

    def count_chunks(self, collection_name: Optional[str] = None) -> int:
        """Count total number of chunks in a collection."""
        collection_name = collection_name or self.config.collection_name

        try:
            if Collection is None:
                raise VectorDBError(
                    "Milvus Collection not available",
                    backend="milvus",
                )

            collection = Collection(collection_name)
            return collection.num_entities
        except Exception as e:
            raise VectorDBError(
                f"Failed to count chunks: {e!s}", backend="milvus", original_error=e
            )

    def clear_collection(self, collection_name: Optional[str] = None) -> None:
        """Remove all chunks from a collection."""
        collection_name = collection_name or self.config.collection_name

        try:
            if Collection is None:
                raise VectorDBError(
                    "Milvus Collection not available",
                    backend="milvus",
                )

            collection = Collection(collection_name)

            # Delete all entities using a filter that matches everything
            collection.delete("id != ''")  # Simple expression that matches all

            self.logger.info(
                f"Cleared all chunks from Milvus collection {collection_name}"
            )

        except Exception as e:
            raise VectorDBError(
                f"Failed to clear collection: {e!s}", backend="milvus", original_error=e
            )

    def get_backend_info(self) -> dict[str, Any]:
        """Get information about the Milvus backend."""
        try:
            return {
                "backend": "milvus",
                "type": "self-hosted",
                "host": self.config.host,
                "port": self.config.port,
                "collection_name": self.config.collection_name,
                "dimension": self.config.dimension,
                "metric_type": self.config.metric_type,
                "capabilities": [
                    "high_performance",
                    "horizontal_scaling",
                    "metadata_filtering",
                    "similarity_search",
                    "batch_operations",
                    "index_management",
                    "distributed_computing",
                ],
            }
        except Exception as e:
            return {"backend": "milvus", "error": str(e)}

    def health_check(self) -> bool:
        """Check if Milvus is healthy and accessible."""
        try:
            if utility is None:
                return False

            # Try to list collections
            utility.list_collections()
            return True
        except Exception as e:
            self.logger.warning(f"Milvus health check failed: {e!s}")
            return False
