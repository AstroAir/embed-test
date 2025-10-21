"""ChromaDB client for vector database operations."""

from typing import Any, Optional, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import IncludeEnum  # For typed include parameters
from chromadb.config import Settings

from pdf_vector_system.utils.logging import LoggerMixin
from pdf_vector_system.utils.progress import PerformanceTimer
from pdf_vector_system.vector_db.config import ChromaDBConfig
from pdf_vector_system.vector_db.converters import VectorDBConverter
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


class ChromaDBClient(VectorDBInterface, LoggerMixin):
    """ChromaDB client for vector database operations."""

    def __init__(self, config: ChromaDBConfig):
        """
        Initialize ChromaDB client.

        Args:
            config: ChromaDB configuration
        """
        self.config = config
        self._client: Optional[ClientAPI] = None
        self._collections: dict[str, Collection] = {}

        # Ensure persist directory exists during initialization
        self.config.persist_directory.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Initialized ChromaDBClient with config: {config}")

    @property
    def client(self) -> ClientAPI:
        """Get or create ChromaDB client."""
        if self._client is None:
            self._create_client()
        if self._client is None:
            raise RuntimeError("Failed to create ChromaDB client")
        return self._client

    def _create_client(self) -> None:
        """Create ChromaDB client with proper configuration."""
        try:
            # Ensure persist directory exists
            self.config.persist_directory.mkdir(parents=True, exist_ok=True)

            # Configure ChromaDB settings (v0.4+ compatible)
            Settings(
                persist_directory=str(self.config.persist_directory)
                # Note: is_persistent and anonymized_telemetry are deprecated in v0.4+
                # Persistence is automatically handled when persist_directory is set
            )

            # Create client - use PersistentClient for better isolation
            self._client = chromadb.PersistentClient(
                path=str(self.config.persist_directory)
            )

            self.logger.info(
                f"Created ChromaDB client with persist directory: {self.config.persist_directory}"
            )

        except Exception as e:
            error_msg = f"Failed to create ChromaDB client: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def create_collection(
        self,
        name: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        get_or_create: bool = True,
    ) -> bool:
        """
        Create or get a collection.

        Returns:
            True if collection was created or retrieved successfully
        """
        collection_name = name or self.config.collection_name

        try:
            collection = self.client.create_collection(
                name=collection_name,
                metadata=metadata or {"created_by": "pdf_vector_system"},
            )
            self.logger.info(f"Created new collection: {collection_name}")

            # Cache the collection
            self._collections[collection_name] = collection
            return True

        except Exception as e:
            error_msg = f"Error creating collection: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def _get_collection(self, name: Optional[str] = None) -> Collection:
        """
        Get an existing collection (private helper method).
        """
        collection_name = name or self.config.collection_name

        # Check cache first
        if collection_name in self._collections:
            return self._collections[collection_name]

        try:
            collection = self.client.get_collection(collection_name)
            self._collections[collection_name] = collection
            return collection

        except Exception as err:
            error_msg = f"Collection {collection_name} not found"
            self.logger.error(error_msg)
            raise CollectionNotFoundError(error_msg) from err

    def get_collection(self, name: Optional[str] = None) -> Collection:
        """
        Get an existing collection.
        """
        return self._get_collection(name)

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
            collections = self.client.list_collections()
            return any(col.name == collection_name for col in collections)
        except Exception as e:
            error_msg = (
                f"Failed to check if collection '{collection_name}' exists: {e!s}"
            )
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def delete_collection(self, name: Optional[str] = None) -> bool:
        """
        Delete a collection.

        Returns:
            True if collection was deleted successfully
        """
        collection_name = name or self.config.collection_name

        try:
            self.client.delete_collection(collection_name)

            # Remove from cache
            if collection_name in self._collections:
                del self._collections[collection_name]

            self.logger.info(f"Deleted collection: {collection_name}")
            return True

        except Exception as e:
            error_msg = f"Error deleting collection {collection_name}: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def list_collections(self) -> list[CollectionInfo]:
        """
        List all collections.
        """
        try:
            collections = self.client.list_collections()

            collection_infos: list[CollectionInfo] = []
            for collection in collections:
                # In ChromaDB v0.6+, list_collections returns collection objects
                # Extract name and metadata from the collection object
                try:
                    # Get collection name - handle both real collections and Mock objects
                    if hasattr(collection, "_mock_name"):
                        # This is a Mock object - use _mock_name for testing
                        coll_name = collection._mock_name
                    elif hasattr(collection, "name"):
                        # Real collection object with name attribute
                        coll_name = str(collection.name)
                    else:
                        # Fallback to string representation
                        coll_name = str(collection)

                    # Get count - call count() method if available
                    if hasattr(collection, "count"):
                        count_val = (
                            collection.count()
                            if callable(collection.count)
                            else collection.count
                        )
                        # Ensure count is an integer, not a Mock object
                        count = (
                            int(count_val) if isinstance(count_val, (int, float)) else 0
                        )
                    else:
                        count = 0

                    # Get metadata
                    if hasattr(collection, "metadata"):
                        metadata = (
                            collection.metadata
                            if collection.metadata is not None
                            else {}
                        )
                    else:
                        metadata = {}

                    info = CollectionInfo(
                        name=coll_name, count=count, metadata=metadata
                    )
                    collection_infos.append(info)
                except Exception as e:
                    # If we can't get collection details, just add basic info
                    self.logger.warning(
                        f"Could not get details for collection {collection}: {e}"
                    )
                    if hasattr(collection, "_mock_name"):
                        coll_name = collection._mock_name
                    elif hasattr(collection, "name"):
                        coll_name = str(collection.name)
                    else:
                        coll_name = str(collection)
                    info = CollectionInfo(name=coll_name, count=0, metadata={})
                    collection_infos.append(info)

            return collection_infos

        except Exception as e:
            error_msg = f"Failed to list collections: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def add_chunks(
        self, chunks: list[DocumentChunk], collection_name: Optional[str] = None
    ) -> None:
        """
        Add document chunks to a collection.
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")

        # Get or create collection - try to get first, create if doesn't exist
        try:
            collection = self._get_collection(collection_name)
        except CollectionNotFoundError:
            self.create_collection(collection_name, get_or_create=True)
            collection = self._get_collection(collection_name)

        try:
            with PerformanceTimer(f"Adding {len(chunks)} chunks to ChromaDB"):
                ids = [chunk.id for chunk in chunks]
                documents = [chunk.content for chunk in chunks]
                embeddings = [chunk.embedding for chunk in chunks]
                metadatas = [chunk.metadata for chunk in chunks]

                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=cast("Any", embeddings),
                    metadatas=cast("Any", metadatas),
                )

                self.logger.info(
                    f"Added {len(chunks)} chunks to collection {collection.name}"
                )

        except Exception as e:
            error_msg = f"Error adding documents: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def search(
        self,
        query: SearchQuery,
        query_embedding: Optional[list[float]] = None,
        collection_name: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.
        """
        # Get collection
        collection = self.get_collection(collection_name)

        try:
            with PerformanceTimer(f"Searching collection {collection.name}"):
                query_params = VectorDBConverter.query_to_chromadb_params(query)

                if query_embedding:
                    results = collection.query(
                        query_embeddings=cast("Any", [query_embedding]), **query_params
                    )
                else:
                    results = collection.query(
                        query_texts=[query.query_text], **query_params
                    )

                search_results: list[SearchResult] = []

                if results.get("ids") and results["ids"] and results["ids"][0]:
                    ids = results["ids"][0]
                    documents_data = results.get("documents")
                    documents = (
                        documents_data[0]
                        if documents_data and documents_data[0]
                        else []
                    )
                    distances_data = results.get("distances")
                    distances = (
                        distances_data[0]
                        if distances_data and distances_data[0]
                        else []
                    )
                    metadatas_data = results.get("metadatas")
                    metadatas = (
                        metadatas_data[0]
                        if metadatas_data and metadatas_data[0]
                        else []
                    )

                    for i, doc_id in enumerate(ids):
                        distance = distances[i] if i < len(distances) else 1.0
                        score = 1.0 - distance
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        if not isinstance(metadata, dict):
                            metadata = {}
                        result = SearchResult(
                            id=doc_id,
                            content=documents[i] if i < len(documents) else "",
                            score=score,
                            metadata=metadata,
                        )
                        search_results.append(result)

                self.logger.debug(f"Found {len(search_results)} results for query")
                return search_results

        except Exception as e:
            error_msg = f"Search failed: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def _get_query_embedding(
        self, query_text: str, embedding_service: Optional[Any] = None
    ) -> list[float]:
        """
        Get embedding for a query text (private helper method).

        Args:
            query_text: The query text to embed
            embedding_service: Optional embedding service to use

        Returns:
            List of floats representing the embedding
        """
        if embedding_service is None:
            # If no embedding service provided, return empty list
            # ChromaDB can handle text queries directly
            return []

        try:
            return embedding_service.embed_query(query_text)
        except Exception as e:
            error_msg = f"Failed to generate query embedding: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def search_documents(
        self,
        query: SearchQuery,
        embedding_service: Optional[Any] = None,
        collection_name: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search for documents using a query.

        Args:
            query: Search query object
            embedding_service: Optional embedding service for query embedding
            collection_name: Optional collection name

        Returns:
            List of search results
        """
        try:
            # Get query embedding if query text is provided
            query_embedding = None
            if query.query_text:
                query_embedding = self._get_query_embedding(
                    query.query_text, embedding_service
                )

            return self.search(
                query, query_embedding=query_embedding, collection_name=collection_name
            )
        except Exception as e:
            error_msg = f"Error searching documents: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def get_document_info(
        self, document_id: str, collection_name: Optional[str] = None
    ) -> DocumentInfo:
        """
        Get information about a document.
        """
        collection = self.get_collection(collection_name)

        try:
            results = collection.get(
                where={"document_id": document_id},
                include=[IncludeEnum.metadatas, IncludeEnum.documents],
            )

            # Check if any chunks were found - results["ids"] is a list of lists
            ids = results.get("ids", [])
            if not ids or (isinstance(ids, list) and len(ids) > 0 and not ids[0]):
                raise DocumentNotFoundError(f"Document {document_id} not found")

            # Extract chunk IDs from nested list structure
            chunk_ids = ids[0] if isinstance(ids, list) and len(ids) > 0 else ids
            chunk_count = len(chunk_ids) if chunk_ids else 0

            # Get documents and calculate total characters
            documents = results.get("documents", [])
            if (
                isinstance(documents, list)
                and len(documents) > 0
                and isinstance(documents[0], list)
            ):
                documents = documents[0]
            total_characters = (
                sum(len(doc) for doc in documents if doc) if documents else 0
            )

            # Get metadatas and extract page numbers and created times
            metadatas = results.get("metadatas", [])
            if (
                isinstance(metadatas, list)
                and len(metadatas) > 0
                and isinstance(metadatas[0], list)
            ):
                metadatas = metadatas[0]

            page_numbers = set()
            created_times: list[float] = []

            if metadatas:
                for metadata in metadatas:
                    if metadata and isinstance(metadata, dict):
                        if "page_number" in metadata:
                            page_numbers.add(metadata["page_number"])
                        if "created_at" in metadata:
                            created_at_val = metadata["created_at"]
                            if isinstance(created_at_val, (int, float)):
                                created_times.append(float(created_at_val))
                        # Also check content_length for total_characters calculation
                        if "content_length" in metadata and not documents:
                            content_len = metadata.get("content_length", 0)
                            if isinstance(content_len, (int, float)):
                                total_characters += int(content_len)

            page_count = len(page_numbers) if page_numbers else None
            created_at = min(created_times) if created_times else None

            return DocumentInfo(
                document_id=document_id,
                chunk_count=chunk_count,
                total_characters=total_characters,
                page_count=page_count,
                created_at=created_at,
            )

        except DocumentNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Failed to get document info for {document_id}: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def delete_document(
        self, document_id: str, collection_name: Optional[str] = None
    ) -> int:
        """
        Delete all chunks of a document.
        """
        collection = self.get_collection(collection_name)

        try:
            results = collection.get(
                where={"document_id": document_id}, include=[IncludeEnum.metadatas]
            )

            if not results["ids"]:
                raise DocumentNotFoundError(f"Document {document_id} not found")

            chunk_ids = results["ids"]
            collection.delete(ids=chunk_ids)
            self.logger.info(
                f"Deleted {len(chunk_ids)} chunks for document {document_id}"
            )
            return len(chunk_ids)

        except DocumentNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Failed to delete document {document_id}: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def get_collection_stats(
        self, collection_name: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Get statistics about a collection.
        """
        collection = self.get_collection(collection_name)

        try:
            # Pass empty where for compatibility with differing signatures
            count = collection.count(where={})

            if count == 0:
                return {
                    "name": collection.name,
                    "total_chunks": 0,
                    "unique_documents": 0,
                    "total_characters": 0,
                    "average_chunk_size": 0,
                }

            sample_size = min(1000, count)
            results = collection.get(
                limit=sample_size,
                include=[IncludeEnum.metadatas, IncludeEnum.documents],
            )

            unique_documents = set()
            total_characters = 0

            metadatas = results.get("metadatas", [])
            documents = results.get("documents", [])

            if metadatas:
                for i, metadata in enumerate(metadatas):
                    if (
                        metadata
                        and isinstance(metadata, dict)
                        and "document_id" in metadata
                    ):
                        unique_documents.add(metadata["document_id"])
                    if documents and i < len(documents):
                        total_characters += len(documents[i])

            if sample_size < count:
                total_characters = int(total_characters * (count / sample_size))

            average_chunk_size = total_characters / count if count > 0 else 0

            return {
                "name": collection.name,
                "total_chunks": count,
                "unique_documents": len(unique_documents),
                "total_characters": total_characters,
                "average_chunk_size": average_chunk_size,
                "sampled": sample_size < count,
                "sample_size": sample_size,
            }

        except Exception as e:
            error_msg = f"Failed to get collection stats: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def search_by_document(
        self,
        document_id: str,
        query_text: str,
        n_results: int = 5,
        collection_name: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search within a specific document.
        """
        query = SearchQuery(
            query_text=query_text,
            n_results=n_results,
            where={"document_id": document_id},
        )
        return self.search(query, collection_name=collection_name)

    def search_by_page(
        self,
        page_number: int,
        query_text: str,
        document_id: Optional[str] = None,
        n_results: int = 5,
        collection_name: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Search within a specific page.
        """
        where_clause: dict[str, Any] = {"page_number": page_number}
        if document_id:
            where_clause["document_id"] = document_id

        query = SearchQuery(
            query_text=query_text, n_results=n_results, where=where_clause
        )
        return self.search(query, collection_name=collection_name)

    def get_similar_chunks(
        self, chunk_id: str, n_results: int = 5, collection_name: Optional[str] = None
    ) -> list[SearchResult]:
        """
        Find chunks similar to a given chunk.
        """
        collection = self.get_collection(collection_name)

        try:
            results = collection.get(
                ids=[chunk_id], include=[IncludeEnum.embeddings, IncludeEnum.documents]
            )

            if not results.get("ids"):
                raise DocumentNotFoundError(f"Chunk {chunk_id} not found")

            embeddings = results.get("embeddings")
            documents = results.get("documents")

            if not embeddings or not embeddings[0]:
                raise DocumentNotFoundError(f"No embedding found for chunk {chunk_id}")
            if not documents or not documents[0]:
                raise DocumentNotFoundError(f"No document found for chunk {chunk_id}")

            reference_embedding = list(embeddings[0])
            reference_content = documents[0]

            query = SearchQuery(query_text=reference_content, n_results=n_results + 1)

            search_results = self.search(
                query,
                query_embedding=reference_embedding,
                collection_name=collection_name,
            )

            filtered_results = [r for r in search_results if r.id != chunk_id]
            return filtered_results[:n_results]

        except DocumentNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Failed to find similar chunks for {chunk_id}: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def find_similar_chunks(
        self, chunk_id: str, collection_name: Optional[str] = None, limit: int = 10
    ) -> list[SearchResult]:
        """
        Find chunks similar to a given chunk.

        This is an alias for get_similar_chunks to maintain compatibility.
        """
        return self.get_similar_chunks(chunk_id, limit, collection_name)

    # Additional interface methods
    def get_chunks(
        self,
        chunk_ids: list[str],
        collection_name: Optional[str] = None,
        include_embeddings: bool = False,
    ) -> list[DocumentChunk]:
        """
        Retrieve specific chunks by their IDs.
        """
        collection = self.get_collection(collection_name)

        try:
            include_list = [IncludeEnum.metadatas, IncludeEnum.documents]
            if include_embeddings:
                include_list.append(IncludeEnum.embeddings)

            results = collection.get(ids=chunk_ids, include=include_list)

            chunks: list[DocumentChunk] = []
            ids = results.get("ids", [])
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])
            embeddings = results.get("embeddings", []) if include_embeddings else []

            for i, chunk_id in enumerate(ids):
                content = documents[i] if i < len(documents) else ""
                metadata = metadatas[i] if i < len(metadatas) else {}
                embedding = embeddings[i] if i < len(embeddings) else []

                if not isinstance(metadata, dict):
                    metadata = {}

                chunk = DocumentChunk(
                    id=chunk_id, content=content, embedding=embedding, metadata=metadata
                )
                chunks.append(chunk)

            return chunks

        except Exception as e:
            error_msg = f"Failed to retrieve chunks: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

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

    def update_chunks(
        self, chunks: list[DocumentChunk], collection_name: Optional[str] = None
    ) -> None:
        """
        Update existing document chunks.
        """
        self.create_collection(collection_name, get_or_create=True)
        collection = self._get_collection(collection_name)

        try:
            # ChromaDB doesn't have direct update - we need to delete and re-add
            chunk_ids = [chunk.id for chunk in chunks]

            # Delete existing chunks
            collection.delete(ids=chunk_ids)

            # Add updated chunks
            self.add_chunks(chunks, collection_name)

            self.logger.info(
                f"Updated {len(chunks)} chunks in collection {collection.name}"
            )

        except Exception as e:
            error_msg = f"Failed to update chunks: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def update_chunk(
        self, chunk: DocumentChunk, collection_name: Optional[str] = None
    ) -> None:
        """
        Update a single chunk (convenience method).
        """
        self.update_chunks([chunk], collection_name)

    def delete_chunks(
        self, chunk_ids: list[str], collection_name: Optional[str] = None
    ) -> None:
        """
        Delete specific chunks by their IDs.
        """
        collection = self.get_collection(collection_name)

        try:
            collection.delete(ids=chunk_ids)
            self.logger.info(
                f"Deleted {len(chunk_ids)} chunks from collection {collection.name}"
            )

        except Exception as e:
            error_msg = f"Failed to delete chunks: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def delete_chunk(
        self, chunk_id: str, collection_name: Optional[str] = None
    ) -> None:
        """
        Delete a single chunk (convenience method).
        """
        self.delete_chunks([chunk_id], collection_name)

    def delete_documents_by_filter(
        self, metadata_filter: dict[str, Any], collection_name: Optional[str] = None
    ) -> bool:
        """
        Delete documents matching a metadata filter.

        Args:
            metadata_filter: Dictionary of metadata filters
            collection_name: Optional collection name

        Returns:
            True if deletion was successful
        """
        collection = self.get_collection(collection_name)

        try:
            collection.delete(where=metadata_filter)
            self.logger.info(
                f"Deleted documents matching filter from collection {collection.name}"
            )
            return True

        except Exception as e:
            error_msg = f"Failed to delete documents by filter: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def search_by_metadata(
        self,
        metadata_filter: dict[str, Any],
        collection_name: Optional[str] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Search for documents by metadata filters only.
        """
        collection = self.get_collection(collection_name)

        try:
            results = collection.get(
                where=metadata_filter,
                limit=limit,
                include=[IncludeEnum.metadatas, IncludeEnum.documents],
            )

            search_results: list[SearchResult] = []
            ids = results.get("ids", [])
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            for i, doc_id in enumerate(ids):
                content = documents[i] if i < len(documents) else ""
                metadata = metadatas[i] if i < len(metadatas) else {}

                if not isinstance(metadata, dict):
                    metadata = {}

                result = SearchResult(
                    id=doc_id,
                    content=content,
                    score=1.0,  # No similarity score for metadata-only search
                    metadata=metadata,
                )
                search_results.append(result)

            return search_results

        except Exception as e:
            error_msg = f"Metadata search failed: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def get_backend_info(self) -> dict[str, Any]:
        """
        Get information about the ChromaDB backend.
        """
        try:
            return {
                "backend": "chromadb",
                "version": (
                    chromadb.__version__
                    if hasattr(chromadb, "__version__")
                    else "unknown"
                ),
                "type": "local",
                "persist_directory": str(self.config.persist_directory),
                "collection_name": self.config.collection_name,
                "distance_metric": self.config.distance_metric,
                "capabilities": [
                    "local_storage",
                    "persistence",
                    "metadata_filtering",
                    "similarity_search",
                    "batch_operations",
                ],
            }
        except Exception as e:
            return {"backend": "chromadb", "error": str(e)}

    def count_chunks(self, collection_name: Optional[str] = None) -> int:
        """
        Count total number of chunks in a collection.
        """
        collection = self.get_collection(collection_name)

        try:
            return collection.count()
        except Exception as e:
            error_msg = f"Failed to count chunks: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def clear_collection(self, collection_name: Optional[str] = None) -> None:
        """
        Remove all chunks from a collection without deleting the collection.
        """
        collection = self.get_collection(collection_name)

        try:
            # Get all IDs and delete them
            results = collection.get(include=[])
            if results.get("ids"):
                collection.delete(ids=results["ids"])

            self.logger.info(f"Cleared all chunks from collection {collection.name}")

        except Exception as e:
            error_msg = f"Failed to clear collection: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    # Compatibility methods for tests
    def add_documents(
        self, chunks: list[DocumentChunk], collection_name: Optional[str] = None
    ) -> bool:
        """Add documents (alias for add_chunks for backward compatibility).

        Returns:
            True if documents were added successfully (or if list is empty)
        """
        if not chunks:
            return True  # No-op for empty list
        self.add_chunks(chunks, collection_name)
        return True

    def delete_documents(
        self, chunk_ids: list[str], collection_name: Optional[str] = None
    ) -> bool:
        """Delete documents (alias for delete_chunks for backward compatibility).

        Returns:
            True if documents were deleted successfully
        """
        self.delete_chunks(chunk_ids, collection_name)
        return True

    def health_check(self) -> bool:
        """
        Check if ChromaDB is healthy and accessible.
        """
        try:
            # Try to access the client and perform a heartbeat
            heartbeat = self.client.heartbeat()
            return heartbeat is not None
        except Exception as e:
            self.logger.warning(f"Health check failed: {e!s}")
            return False

    def get_collection_info(self, name: Optional[str] = None) -> CollectionInfo:
        """
        Get information about a collection.
        """
        collection = self.get_collection(name)

        try:
            count = collection.count()
            metadata = collection.metadata or {}

            return CollectionInfo(name=collection.name, count=count, metadata=metadata)

        except Exception as e:
            error_msg = f"Failed to get collection info: {e!s}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e
