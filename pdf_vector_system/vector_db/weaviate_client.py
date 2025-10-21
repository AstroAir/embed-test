"""Weaviate client for vector database operations."""

from typing import Any, Optional

try:
    import weaviate
    from weaviate import WeaviateClient as WeaviateClientLib
except ImportError:
    weaviate = None
    WeaviateClientLib = None

from pdf_vector_system.utils.logging import LoggerMixin
from pdf_vector_system.vector_db.config import WeaviateConfig
from pdf_vector_system.vector_db.converters import VectorDBConverter
from pdf_vector_system.vector_db.error_handler import handle_vector_db_errors
from pdf_vector_system.vector_db.interface import VectorDBInterface
from pdf_vector_system.vector_db.models import (
    CollectionInfo,
    CollectionNotFoundError,
    DocumentChunk,
    DocumentInfo,
    DocumentNotFoundError,
    SearchQuery,
    SearchResult,
    VectorDBError,
)


class WeaviateClient(VectorDBInterface, LoggerMixin):
    """Weaviate client for vector database operations."""

    def __init__(self, config: WeaviateConfig):
        """
        Initialize Weaviate client.

        Args:
            config: Weaviate configuration
        """
        self.config = config
        self._client = None

        self.logger.info(f"Initialized WeaviateClient with config: {config}")

    @property
    def client(self):
        """Get or create Weaviate client."""
        if self._client is None:
            self._create_client()
        return self._client

    def _create_client(self) -> None:
        """Create Weaviate client with proper configuration (v4 API)."""
        try:
            import weaviate
            from weaviate.auth import AuthApiKey

            # Prepare authentication for v4 API
            auth_credentials = None
            if self.config.api_key:
                auth_credentials = AuthApiKey(self.config.api_key)

            # Create client using v4 API
            if "weaviate.io" in self.config.url or "wcs" in self.config.url:
                # Weaviate Cloud Services
                self._client = weaviate.connect_to_wcs(
                    cluster_url=self.config.url,
                    auth_credentials=auth_credentials,
                    timeout=(5, 60),  # (connection, read) timeout
                )
            else:
                # Local or custom Weaviate instance
                http_host = (
                    self.config.url.replace("http://", "")
                    .replace("https://", "")
                    .split(":")[0]
                )
                http_port = (
                    int(self.config.url.split(":")[-1])
                    if ":" in self.config.url.split("//")[-1]
                    else 8080
                )
                http_secure = "https" in self.config.url

                # For gRPC, use same host but port 50051 by default
                grpc_host = http_host
                grpc_port = 50051
                grpc_secure = http_secure

                self._client = weaviate.connect_to_custom(
                    http_host=http_host,
                    http_port=http_port,
                    http_secure=http_secure,
                    grpc_host=grpc_host,
                    grpc_port=grpc_port,
                    grpc_secure=grpc_secure,
                    auth_credentials=auth_credentials,
                )

            self.logger.info("Successfully initialized Weaviate client (v4 API)")

        except ImportError:
            raise VectorDBError(
                "Weaviate client not installed. Install with: pip install weaviate-client",
                backend="weaviate",
            ) from None
        except Exception as e:
            raise VectorDBError(
                f"Failed to initialize Weaviate client: {e!s}",
                backend="weaviate",
                original_error=e,
            ) from e

    def _get_class_name(self, collection_name: Optional[str] = None) -> str:
        """Get Weaviate class name from collection name."""
        class_name = collection_name or self.config.class_name
        # Weaviate class names must start with uppercase (preserve rest of the case)
        return class_name[0].upper() + class_name[1:] if class_name else class_name

    @handle_vector_db_errors(backend_type="weaviate", operation="add_chunks")
    def add_chunks(
        self, chunks: list[DocumentChunk], collection_name: Optional[str] = None
    ) -> None:
        """Add document chunks to Weaviate."""
        if not chunks:
            return

        try:
            class_name = self._get_class_name(collection_name)

            # Ensure class exists
            self._ensure_class_exists(class_name)

            # Convert chunks to Weaviate format
            objects = VectorDBConverter.chunks_to_weaviate_format(chunks, class_name)

            # Batch insert objects using v4 API
            collection = self.client.collections.get(class_name)

            # Prepare data for batch insert
            data_objects = []
            for obj in objects:
                data_objects.append(
                    {
                        "properties": obj["properties"],
                        "uuid": obj["id"],
                        "vector": obj["vector"],
                    }
                )

            # Insert in batches
            collection.data.insert_many(data_objects)

            self.logger.info(
                f"Added {len(chunks)} chunks to Weaviate class {class_name}"
            )

        except Exception as e:
            raise VectorDBError(
                f"Failed to add chunks to Weaviate: {e!s}",
                backend="weaviate",
                original_error=e,
            ) from e

    def _ensure_class_exists(self, class_name: str) -> None:
        """Ensure Weaviate collection exists, create if not (v3 schema API)."""
        try:
            # Check if collection exists
            if not self.collection_exists(class_name):
                # Create class using v3 schema API
                class_obj = {
                    "class": class_name,
                    "description": f"Document chunks for {class_name}",
                    "vectorizer": "none",  # Manual vectors
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "Document content",
                        },
                        {
                            "name": "document_id",
                            "dataType": ["text"],
                            "description": "Source document ID",
                        },
                        {
                            "name": "page_number",
                            "dataType": ["int"],
                            "description": "Page number in document",
                        },
                        {
                            "name": "chunk_index",
                            "dataType": ["int"],
                            "description": "Chunk index in document",
                        },
                        {
                            "name": "created_at",
                            "dataType": ["number"],
                            "description": "Creation timestamp",
                        },
                    ],
                }

                self.client.schema.create_class(class_obj)
                self.logger.info(f"Created Weaviate collection: {class_name}")

        except Exception as e:
            raise VectorDBError(
                f"Failed to ensure class exists: {e!s}",
                backend="weaviate",
                original_error=e,
            ) from e

    @handle_vector_db_errors(backend_type="weaviate", operation="search")
    def search(
        self,
        query: SearchQuery,
        query_embedding: Optional[list[float]] = None,
        collection_name: Optional[str] = None,
    ) -> list[SearchResult]:
        """Search for similar documents in Weaviate."""
        try:
            class_name = self._get_class_name(collection_name)

            if not query_embedding:
                raise VectorDBError("query_embedding is required", backend="weaviate")

            # Build query using v4 API
            collection = self.client.collections.get(class_name)

            # Prepare query parameters
            query_params = {
                "limit": query.n_results,
                "return_metadata": ["certainty", "distance"],
                "return_properties": [
                    "content",
                    "document_id",
                    "page_number",
                    "chunk_index",
                    "created_at",
                ],
            }

            # Add where filter if provided
            if query.where:
                query_params["where"] = self._build_where_filter(query.where)

            # Execute vector search using v4 API
            result = collection.query.near_vector(
                near_vector=query_embedding, **query_params
            )

            # Convert results to SearchResult objects (v4 API format)
            search_results = []

            for obj in result.objects:
                # v4 API provides objects with properties and metadata
                properties = obj.properties
                metadata = obj.metadata

                search_result = SearchResult(
                    id=str(obj.uuid),
                    content=properties.get("content", ""),
                    score=metadata.certainty if hasattr(metadata, "certainty") else 0.0,
                    metadata={
                        "document_id": properties.get("document_id"),
                        "page_number": properties.get("page_number"),
                        "chunk_index": properties.get("chunk_index"),
                        "created_at": properties.get("created_at"),
                    },
                )
                search_results.append(search_result)

            self.logger.debug(f"Found {len(search_results)} results for query")
            return search_results

        except Exception as e:
            raise VectorDBError(
                f"Search failed in Weaviate: {e!s}",
                backend="weaviate",
                original_error=e,
            ) from e

    def _build_where_filter(self, where_dict: dict[str, Any]) -> dict[str, Any]:
        """Build Weaviate where filter from dictionary."""
        # Simple implementation - can be extended for complex filters
        conditions = []

        for key, value in where_dict.items():
            condition = {
                "path": [key],
                "operator": "Equal",
                "valueString" if isinstance(value, str) else "valueNumber": value,
            }
            conditions.append(condition)

        if len(conditions) == 1:
            return conditions[0]
        return {"operator": "And", "operands": conditions}

    @handle_vector_db_errors(backend_type="weaviate", operation="get_chunks")
    def get_chunks(
        self,
        chunk_ids: list[str],
        collection_name: Optional[str] = None,
        include_embeddings: bool = False,
    ) -> list[DocumentChunk]:
        """Retrieve specific chunks by their IDs."""
        try:
            class_name = self._get_class_name(collection_name)

            chunks = []
            for chunk_id in chunk_ids:
                try:
                    # Get object by UUID
                    obj = self.client.data_object.get_by_id(
                        uuid=chunk_id,
                        class_name=class_name,
                        with_vector=include_embeddings,
                    )

                    properties = obj.get("properties", {})
                    vector = obj.get("vector", []) if include_embeddings else []

                    chunk = DocumentChunk(
                        id=chunk_id,
                        content=properties.get("content", ""),
                        embedding=vector,
                        metadata={
                            "document_id": properties.get("document_id"),
                            "page_number": properties.get("page_number"),
                            "chunk_index": properties.get("chunk_index"),
                            "created_at": properties.get("created_at"),
                        },
                    )
                    chunks.append(chunk)

                except Exception:
                    # Object not found, skip
                    continue

            return chunks

        except Exception as e:
            raise VectorDBError(
                f"Failed to retrieve chunks from Weaviate: {e!s}",
                backend="weaviate",
                original_error=e,
            ) from e

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

    @handle_vector_db_errors(backend_type="weaviate", operation="update_chunks")
    def update_chunks(
        self, chunks: list[DocumentChunk], collection_name: Optional[str] = None
    ) -> None:
        """Update existing document chunks."""
        try:
            class_name = self._get_class_name(collection_name)

            for chunk in chunks:
                # Update object
                properties = {"content": chunk.content, **chunk.metadata}

                self.client.data_object.update(
                    uuid=chunk.id,
                    class_name=class_name,
                    data_object=properties,
                    vector=chunk.embedding,
                )

            self.logger.info(
                f"Updated {len(chunks)} chunks in Weaviate class {class_name}"
            )

        except Exception as e:
            raise VectorDBError(
                f"Failed to update chunks in Weaviate: {e!s}",
                backend="weaviate",
                original_error=e,
            ) from e

    @handle_vector_db_errors(backend_type="weaviate", operation="update_chunk")
    def update_chunk(
        self, chunk: DocumentChunk, collection_name: Optional[str] = None
    ) -> None:
        """Update an existing document chunk."""
        self.update_chunks([chunk], collection_name)

    @handle_vector_db_errors(backend_type="weaviate", operation="delete_chunks")
    def delete_chunks(
        self, chunk_ids: list[str], collection_name: Optional[str] = None
    ) -> None:
        """Delete specific chunks by their IDs."""
        try:
            class_name = self._get_class_name(collection_name)

            for chunk_id in chunk_ids:
                try:
                    self.client.data_object.delete(uuid=chunk_id, class_name=class_name)
                except Exception:
                    # Object might not exist, continue
                    continue

            self.logger.info(
                f"Deleted {len(chunk_ids)} chunks from Weaviate class {class_name}"
            )

        except Exception as e:
            raise VectorDBError(
                f"Failed to delete chunks from Weaviate: {e!s}",
                backend="weaviate",
                original_error=e,
            ) from e

    @handle_vector_db_errors(backend_type="weaviate", operation="delete_document")
    def delete_document(
        self, document_id: str, collection_name: Optional[str] = None
    ) -> int:
        """Delete all chunks belonging to a document."""
        try:
            class_name = self._get_class_name(collection_name)
            collection = self.client.collections.get(class_name)

            # Query for all chunks with this document_id
            response = collection.query.fetch_objects(
                where={
                    "path": ["document_id"],
                    "operator": "Equal",
                    "valueText": document_id,
                },
                limit=10000,  # Large number to get all chunks
            )

            objects = getattr(response, "objects", [])
            if not objects:
                return 0

            # Extract UUIDs
            uuids = [obj.uuid for obj in objects if hasattr(obj, "uuid")]

            if uuids:
                # Delete all chunks for this document
                for uuid in uuids:
                    collection.data.delete_by_id(uuid)

                self.logger.info(
                    f"Deleted {len(uuids)} chunks for document {document_id}"
                )
                return len(uuids)

            return 0

        except Exception as e:
            raise VectorDBError(
                f"Failed to delete document from Weaviate: {e!s}",
                backend="weaviate",
                original_error=e,
            ) from e

    def get_collection(self, name: Optional[str] = None) -> Any:
        """Get an existing collection (class in Weaviate)."""
        class_name = self._get_class_name(name)
        return self.client.collections.get(class_name)

    def collection_exists(self, name: Optional[str] = None) -> bool:
        """
        Check if a collection (class) exists.

        Args:
            name: Collection name (uses default if None)

        Returns:
            True if collection exists, False otherwise

        Raises:
            VectorDBError: If check operation fails
        """
        class_name = self._get_class_name(name)
        try:
            # Check schema directly to avoid issues with list_collections
            schema = self.client.schema.get()
            classes = schema.get("classes", [])
            # Extract class names from schema, handling both real and mock objects
            class_names = []
            for cls in classes:
                if hasattr(cls, "get"):
                    # Real dict-like object
                    class_names.append(cls.get("class"))
                elif hasattr(cls, "__getitem__"):
                    # Mock object that supports item access
                    try:
                        class_names.append(cls["class"])
                    except (KeyError, TypeError):
                        # Fallback: convert to string and check
                        class_names.append(str(cls))
                else:
                    # Convert to string as fallback
                    class_names.append(str(cls))
            return class_name in class_names
        except Exception as e:
            error_msg = f"Failed to check if class '{class_name}' exists: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    @handle_vector_db_errors(backend_type="weaviate", operation="search_by_metadata")
    def search_by_metadata(
        self,
        metadata_filter: dict[str, Any],
        collection_name: Optional[str] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Search for documents by metadata filters only."""
        try:
            class_name = self._get_class_name(collection_name)

            # Build query with where filter using v4 API
            collection = self.client.collections.get(class_name)
            where_filter = self._build_where_filter(metadata_filter)

            result = collection.query.fetch_objects(
                where=where_filter,
                limit=limit,
                return_properties=[
                    "content",
                    "document_id",
                    "page_number",
                    "chunk_index",
                    "created_at",
                ],
            )

            # Convert results
            search_results = []

            for obj in result.objects:
                properties = obj.properties

                search_result = SearchResult(
                    id=str(obj.uuid),
                    content=properties.get("content", ""),
                    score=1.0,  # No similarity score for metadata-only search
                    metadata={
                        "document_id": properties.get("document_id"),
                        "page_number": properties.get("page_number"),
                        "chunk_index": properties.get("chunk_index"),
                        "created_at": properties.get("created_at"),
                    },
                )
                search_results.append(search_result)

            return search_results

        except Exception as e:
            raise VectorDBError(
                f"Metadata search failed in Weaviate: {e!s}",
                backend="weaviate",
                original_error=e,
            ) from e

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
                backend="weaviate",
                original_error=e,
            ) from e

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

            # Search for similar chunks
            class_name = self._get_class_name(collection_name)

            # Use v4 API for similarity search
            collection = self.client.collections.get(class_name)

            result = collection.query.near_vector(
                near_vector=reference_embedding,
                limit=limit + 1,  # +1 to exclude the reference chunk
                return_metadata=["certainty", "distance"],
                return_properties=[
                    "content",
                    "document_id",
                    "page_number",
                    "chunk_index",
                    "created_at",
                ],
            )

            # Convert and filter results
            search_results = []

            for obj in result.objects:
                obj_id = str(obj.uuid)
                properties = obj.properties
                metadata = obj.metadata

                if obj_id != chunk_id:  # Exclude the reference chunk
                    search_result = SearchResult(
                        id=obj_id,
                        content=properties.get("content", ""),
                        score=(
                            metadata.certainty
                            if hasattr(metadata, "certainty")
                            else 0.0
                        ),
                        metadata={
                            "document_id": properties.get("document_id"),
                            "page_number": properties.get("page_number"),
                            "chunk_index": properties.get("chunk_index"),
                            "created_at": properties.get("created_at"),
                        },
                    )
                    search_results.append(search_result)

            return search_results[:limit]

        except DocumentNotFoundError:
            raise
        except Exception as e:
            raise VectorDBError(
                f"Failed to find similar chunks: {e!s}",
                backend="weaviate",
                original_error=e,
            ) from e

    def create_collection(
        self,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        get_or_create: bool = True,
    ) -> Any:
        """Create a new collection (class in Weaviate)."""
        try:
            class_name = self._get_class_name(name)

            # Check if class already exists
            if self.collection_exists(class_name):
                if get_or_create:
                    return self.get_collection(class_name)
                raise VectorDBError(
                    f"Collection {class_name} already exists",
                    backend="weaviate",
                ) from None

            # Create class
            self._ensure_class_exists(class_name)
            self.logger.info(f"Created Weaviate class: {class_name}")
            return self.get_collection(class_name)

        except Exception as e:
            raise VectorDBError(
                f"Failed to create class: {e!s}", backend="weaviate", original_error=e
            ) from e

    def delete_collection(self, name: Optional[str] = None) -> None:
        """Delete a collection (class in Weaviate)."""
        try:
            class_name = self._get_class_name(name)

            # Check if class exists
            if not self.collection_exists(class_name):
                raise CollectionNotFoundError(
                    f"Collection {class_name} not found", backend="weaviate"
                ) from None

            self.client.schema.delete_class(class_name)
            self.logger.info(f"Deleted Weaviate class: {class_name}")

        except Exception as e:
            raise VectorDBError(
                f"Failed to delete class: {e!s}", backend="weaviate", original_error=e
            ) from e

    def list_collections(self) -> list[CollectionInfo]:
        """List all collections (classes in Weaviate)."""
        try:
            schema = self.client.schema.get()
            collections = []

            for cls in schema.get("classes", []):
                class_name = cls["class"]
                try:
                    # Get count for each collection
                    result = (
                        self.client.query.aggregate(class_name).with_meta_count().do()
                    )
                    count = (
                        result.get("data", {})
                        .get("Aggregate", {})
                        .get(class_name, [{}])[0]
                        .get("meta", {})
                        .get("count", 0)
                    )

                    collections.append(
                        CollectionInfo(
                            name=class_name,
                            count=count,
                            metadata={
                                "description": cls.get("description", ""),
                                "vectorizer": cls.get("vectorizer", "none"),
                            },
                        )
                    )
                except Exception:
                    # If count fails, add with count 0
                    collections.append(
                        CollectionInfo(
                            name=class_name,
                            count=0,
                            metadata={"description": cls.get("description", "")},
                        )
                    )

            return collections
        except Exception as e:
            raise VectorDBError(
                f"Failed to list classes: {e!s}", backend="weaviate", original_error=e
            ) from e

    def get_collection_info(self, name: Optional[str] = None) -> CollectionInfo:
        """Get information about a collection (class in Weaviate)."""
        class_name = self._get_class_name(name)

        try:
            # Get class schema
            schema = self.client.schema.get(class_name)

            # Count objects in class
            result = self.client.query.aggregate(class_name).with_meta_count().do()

            count = (
                result.get("data", {})
                .get("Aggregate", {})
                .get(class_name, [{}])[0]
                .get("meta", {})
                .get("count", 0)
            )

            return CollectionInfo(
                name=class_name,
                count=count,
                metadata={
                    "description": schema.get("description", ""),
                    "properties": [
                        prop["name"] for prop in schema.get("properties", [])
                    ],
                    "vectorizer": schema.get("vectorizer", "none"),
                },
            )

        except Exception as e:
            raise VectorDBError(
                f"Failed to get class info: {e!s}", backend="weaviate", original_error=e
            ) from e

    def count_chunks(self, collection_name: Optional[str] = None) -> int:
        """Count total number of chunks in a class."""
        class_name = self._get_class_name(collection_name)

        try:
            result = self.client.query.aggregate(class_name).with_meta_count().do()

            return (
                result.get("data", {})
                .get("Aggregate", {})
                .get(class_name, [{}])[0]
                .get("meta", {})
                .get("count", 0)
            )

        except Exception as e:
            raise VectorDBError(
                f"Failed to count chunks: {e!s}", backend="weaviate", original_error=e
            ) from e

    def clear_collection(self, collection_name: Optional[str] = None) -> None:
        """Remove all chunks from a class."""
        class_name = self._get_class_name(collection_name)

        try:
            # Get all object IDs
            result = (
                self.client.query.get(class_name)
                .with_additional(["id"])
                .with_limit(10000)  # Weaviate limit
                .do()
            )

            objects = result.get("data", {}).get("Get", {}).get(class_name, [])

            # Delete all objects
            for obj in objects:
                obj_id = obj.get("_additional", {}).get("id")
                if obj_id:
                    try:
                        self.client.data_object.delete(
                            uuid=obj_id, class_name=class_name
                        )
                    except Exception:
                        continue

            self.logger.info(f"Cleared all chunks from Weaviate class {class_name}")

        except Exception as e:
            raise VectorDBError(
                f"Failed to clear class: {e!s}", backend="weaviate", original_error=e
            ) from e

    def close(self) -> None:
        """Close Weaviate client connection (v4 API requirement)."""
        if self._client is not None:
            try:
                self._client.close()
                self.logger.info("Closed Weaviate client connection")
            except Exception as e:
                self.logger.warning(f"Error closing Weaviate client: {e!s}")
            finally:
                self._client = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()

    def get_backend_info(self) -> dict[str, Any]:
        """Get information about the Weaviate backend."""
        try:
            # Get Weaviate meta information (v4 API)
            meta = self.client.get_meta()

            return {
                "backend": "weaviate",
                "type": "cloud" if "weaviate.io" in self.config.url else "self-hosted",
                "url": self.config.url,
                "class_name": self.config.class_name,
                "version": meta.get("version", "unknown"),
                "capabilities": [
                    "graphql_api",
                    "semantic_search",
                    "metadata_filtering",
                    "similarity_search",
                    "real_time_updates",
                    "schema_management",
                    "multi_tenancy",
                ],
            }
        except Exception as e:
            return {"backend": "weaviate", "error": str(e)}

    def health_check(self) -> bool:
        """Check if Weaviate is healthy and accessible."""
        try:
            # Try to get meta information
            self.client.get_meta()
            return True
        except Exception as e:
            self.logger.warning(f"Weaviate health check failed: {e!s}")
            return False
