"""ChromaDB client for vector database operations."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, cast
import time

import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from chromadb.api import ClientAPI
from loguru import logger

from .models import (
    DocumentChunk, SearchResult, SearchQuery, CollectionInfo, DocumentInfo,
    VectorDBError, CollectionNotFoundError, DocumentNotFoundError, InvalidQueryError
)
from ..config.settings import ChromaDBConfig
from ..utils.logging import LoggerMixin
from ..utils.progress import PerformanceTimer


class ChromaDBClient(LoggerMixin):
    """ChromaDB client for vector database operations."""
    
    def __init__(self, config: ChromaDBConfig):
        """
        Initialize ChromaDB client.
        
        Args:
            config: ChromaDB configuration
        """
        self.config = config
        self._client: Optional[ClientAPI] = None
        self._collections: Dict[str, Collection] = {}
        
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
            
            # Configure ChromaDB settings
            settings = Settings(
                persist_directory=str(self.config.persist_directory),
                is_persistent=True,
                anonymized_telemetry=False
            )
            
            # Create client
            self._client = chromadb.Client(settings)
            
            self.logger.info(f"Created ChromaDB client with persist directory: {self.config.persist_directory}")
            
        except Exception as e:
            error_msg = f"Failed to create ChromaDB client: {str(e)}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e
    
    def create_collection(
        self,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        get_or_create: bool = True
    ) -> Collection:
        """
        Create or get a collection.
        
        Args:
            name: Collection name (uses default if None)
            metadata: Collection metadata
            get_or_create: Whether to get existing collection if it exists
            
        Returns:
            ChromaDB Collection object
        """
        collection_name = name or self.config.collection_name
        
        try:
            if get_or_create:
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata=metadata or {"created_by": "pdf_vector_system"}
                )
                self.logger.debug(f"Got or created collection: {collection_name}")
            else:
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata=metadata or {"created_by": "pdf_vector_system"}
                )
                self.logger.info(f"Created new collection: {collection_name}")
            
            # Cache the collection
            self._collections[collection_name] = collection
            
            return collection
            
        except Exception as e:
            error_msg = f"Failed to create collection {collection_name}: {str(e)}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e
    
    def get_collection(self, name: Optional[str] = None) -> Collection:
        """
        Get an existing collection.
        
        Args:
            name: Collection name (uses default if None)
            
        Returns:
            ChromaDB Collection object
        """
        collection_name = name or self.config.collection_name
        
        # Check cache first
        if collection_name in self._collections:
            return self._collections[collection_name]
        
        try:
            collection = self.client.get_collection(name=collection_name)
            self._collections[collection_name] = collection
            return collection
            
        except Exception as e:
            error_msg = f"Collection {collection_name} not found"
            self.logger.error(error_msg)
            raise CollectionNotFoundError(error_msg) from e
    
    def delete_collection(self, name: Optional[str] = None) -> None:
        """
        Delete a collection.
        
        Args:
            name: Collection name (uses default if None)
        """
        collection_name = name or self.config.collection_name
        
        try:
            self.client.delete_collection(name=collection_name)
            
            # Remove from cache
            if collection_name in self._collections:
                del self._collections[collection_name]
            
            self.logger.info(f"Deleted collection: {collection_name}")
            
        except Exception as e:
            error_msg = f"Failed to delete collection {collection_name}: {str(e)}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e
    
    def list_collections(self) -> List[CollectionInfo]:
        """
        List all collections.
        
        Returns:
            List of CollectionInfo objects
        """
        try:
            collections = self.client.list_collections()
            
            collection_infos = []
            for collection in collections:
                count = collection.count()
                metadata = collection.metadata or {}
                
                info = CollectionInfo(
                    name=collection.name,
                    count=count,
                    metadata=metadata
                )
                collection_infos.append(info)
            
            return collection_infos
            
        except Exception as e:
            error_msg = f"Failed to list collections: {str(e)}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e
    
    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        collection_name: Optional[str] = None
    ) -> None:
        """
        Add document chunks to a collection.
        
        Args:
            chunks: List of DocumentChunk objects
            collection_name: Collection name (uses default if None)
        """
        if not chunks:
            raise ValueError("Chunks list cannot be empty")
        
        collection = self.get_collection(collection_name)
        
        try:
            with PerformanceTimer(f"Adding {len(chunks)} chunks to ChromaDB"):
                # Prepare data for ChromaDB
                ids = [chunk.id for chunk in chunks]
                documents = [chunk.content for chunk in chunks]
                embeddings = [chunk.embedding for chunk in chunks]
                metadatas = [chunk.metadata for chunk in chunks]

                # Add to collection
                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=cast(Any, embeddings),  # ChromaDB type compatibility
                    metadatas=cast(Any, metadatas)     # ChromaDB type compatibility
                )
                
                self.logger.info(f"Added {len(chunks)} chunks to collection {collection.name}")
                
        except Exception as e:
            error_msg = f"Failed to add chunks to collection: {str(e)}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e
    
    def search(
        self,
        query: SearchQuery,
        query_embedding: Optional[List[float]] = None,
        collection_name: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: SearchQuery object
            query_embedding: Pre-computed query embedding (if None, will use query_text)
            collection_name: Collection name (uses default if None)
            
        Returns:
            List of SearchResult objects
        """
        collection = self.get_collection(collection_name)
        
        try:
            with PerformanceTimer(f"Searching collection {collection.name}"):
                # Prepare query parameters
                query_params = query.to_chroma_params()
                
                if query_embedding:
                    # Use provided embedding
                    results = collection.query(
                        query_embeddings=cast(Any, [query_embedding]),
                        **query_params
                    )
                else:
                    # Use text query (ChromaDB will generate embedding)
                    results = collection.query(
                        query_texts=[query.query_text],
                        **query_params
                    )
                
                # Convert results to SearchResult objects
                search_results = []
                
                if results.get("ids") and results["ids"] and results["ids"][0]:  # Check if we have results
                    ids = results["ids"][0]
                    documents_data = results.get("documents")
                    documents = documents_data[0] if documents_data and documents_data[0] else []
                    distances_data = results.get("distances")
                    distances = distances_data[0] if distances_data and distances_data[0] else []
                    metadatas_data = results.get("metadatas")
                    metadatas = metadatas_data[0] if metadatas_data and metadatas_data[0] else []

                    for i, doc_id in enumerate(ids):
                        # Convert distance to similarity score (1 - distance for cosine)
                        distance = distances[i] if i < len(distances) else 1.0
                        score = 1.0 - distance  # Assuming cosine distance

                        # Safely get metadata
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        # Ensure metadata is a dict
                        if not isinstance(metadata, dict):
                            metadata = {}

                        result = SearchResult(
                            id=doc_id,
                            content=documents[i] if i < len(documents) else "",
                            score=score,
                            metadata=metadata
                        )
                        search_results.append(result)
                
                self.logger.debug(f"Found {len(search_results)} results for query")
                
                return search_results
                
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e
    
    def get_document_info(
        self,
        document_id: str,
        collection_name: Optional[str] = None
    ) -> DocumentInfo:
        """
        Get information about a document.
        
        Args:
            document_id: Document ID
            collection_name: Collection name (uses default if None)
            
        Returns:
            DocumentInfo object
        """
        collection = self.get_collection(collection_name)
        
        try:
            # Query for all chunks of this document
            results = collection.get(
                where={"document_id": document_id},
                include=["metadatas", "documents"]
            )
            
            if not results["ids"]:
                raise DocumentNotFoundError(f"Document {document_id} not found")
            
            # Calculate statistics
            chunk_count = len(results["ids"]) if results["ids"] else 0
            documents = results.get("documents", [])
            total_characters = sum(len(doc) for doc in documents) if documents else 0

            # Extract metadata
            metadatas = results.get("metadatas", [])
            page_numbers = set()
            created_times = []

            if metadatas:
                for metadata in metadatas:
                    if metadata and isinstance(metadata, dict):
                        if "page_number" in metadata:
                            page_numbers.add(metadata["page_number"])
                        if "created_at" in metadata:
                            created_at_val = metadata["created_at"]
                            if isinstance(created_at_val, (int, float)):
                                created_times.append(float(created_at_val))

            page_count = len(page_numbers) if page_numbers else None
            created_at = min(created_times) if created_times else None
            
            return DocumentInfo(
                document_id=document_id,
                chunk_count=chunk_count,
                total_characters=total_characters,
                page_count=page_count,
                created_at=created_at
            )
            
        except DocumentNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Failed to get document info for {document_id}: {str(e)}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e
    
    def delete_document(
        self,
        document_id: str,
        collection_name: Optional[str] = None
    ) -> int:
        """
        Delete all chunks of a document.
        
        Args:
            document_id: Document ID
            collection_name: Collection name (uses default if None)
            
        Returns:
            Number of chunks deleted
        """
        collection = self.get_collection(collection_name)
        
        try:
            # Get all chunk IDs for this document
            results = collection.get(
                where={"document_id": document_id},
                include=["metadatas"]
            )
            
            if not results["ids"]:
                raise DocumentNotFoundError(f"Document {document_id} not found")
            
            chunk_ids = results["ids"]
            
            # Delete all chunks
            collection.delete(ids=chunk_ids)
            
            self.logger.info(f"Deleted {len(chunk_ids)} chunks for document {document_id}")
            
            return len(chunk_ids)
            
        except DocumentNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Failed to delete document {document_id}: {str(e)}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e
    
    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about a collection.
        
        Args:
            collection_name: Collection name (uses default if None)
            
        Returns:
            Dictionary containing collection statistics
        """
        collection = self.get_collection(collection_name)
        
        try:
            count = collection.count()
            
            if count == 0:
                return {
                    "name": collection.name,
                    "total_chunks": 0,
                    "unique_documents": 0,
                    "total_characters": 0,
                    "average_chunk_size": 0
                }
            
            # Get sample of documents to calculate statistics
            sample_size = min(1000, count)
            results = collection.get(
                limit=sample_size,
                include=["metadatas", "documents"]
            )
            
            # Calculate statistics
            unique_documents = set()
            total_characters = 0

            metadatas = results.get("metadatas", [])
            documents = results.get("documents", [])

            if metadatas:
                for i, metadata in enumerate(metadatas):
                    if metadata and isinstance(metadata, dict) and "document_id" in metadata:
                        unique_documents.add(metadata["document_id"])

                    if documents and i < len(documents):
                        total_characters += len(documents[i])
            
            # Extrapolate if we sampled
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
                "sample_size": sample_size
            }
            
        except Exception as e:
            error_msg = f"Failed to get collection stats: {str(e)}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e

    def search_by_document(
        self,
        document_id: str,
        query_text: str,
        n_results: int = 5,
        collection_name: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search within a specific document.

        Args:
            document_id: Document ID to search within
            query_text: Query text
            n_results: Number of results to return
            collection_name: Collection name (uses default if None)

        Returns:
            List of SearchResult objects
        """
        query = SearchQuery(
            query_text=query_text,
            n_results=n_results,
            where={"document_id": document_id}
        )

        return self.search(query, collection_name=collection_name)

    def search_by_page(
        self,
        page_number: int,
        query_text: str,
        document_id: Optional[str] = None,
        n_results: int = 5,
        collection_name: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Search within a specific page.

        Args:
            page_number: Page number to search within
            query_text: Query text
            document_id: Optional document ID to further filter
            n_results: Number of results to return
            collection_name: Collection name (uses default if None)

        Returns:
            List of SearchResult objects
        """
        where_clause: Dict[str, Any] = {"page_number": page_number}
        if document_id:
            where_clause["document_id"] = document_id

        query = SearchQuery(
            query_text=query_text,
            n_results=n_results,
            where=where_clause
        )

        return self.search(query, collection_name=collection_name)

    def get_similar_chunks(
        self,
        chunk_id: str,
        n_results: int = 5,
        collection_name: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Find chunks similar to a given chunk.

        Args:
            chunk_id: ID of the reference chunk
            n_results: Number of results to return
            collection_name: Collection name (uses default if None)

        Returns:
            List of SearchResult objects
        """
        collection = self.get_collection(collection_name)

        try:
            # Get the reference chunk
            results = collection.get(
                ids=[chunk_id],
                include=["embeddings", "documents"]
            )

            if not results.get("ids"):
                raise DocumentNotFoundError(f"Chunk {chunk_id} not found")

            # Use the chunk's embedding for similarity search
            embeddings = results.get("embeddings")
            documents = results.get("documents")

            if not embeddings or not embeddings[0]:
                raise DocumentNotFoundError(f"No embedding found for chunk {chunk_id}")
            if not documents or not documents[0]:
                raise DocumentNotFoundError(f"No document found for chunk {chunk_id}")

            reference_embedding = list(embeddings[0])  # Convert to list for type compatibility
            reference_content = documents[0]

            query = SearchQuery(
                query_text=reference_content,  # Fallback text
                n_results=n_results + 1  # +1 to exclude the reference chunk itself
            )

            search_results = self.search(
                query,
                query_embedding=reference_embedding,
                collection_name=collection_name
            )

            # Filter out the reference chunk itself
            filtered_results = [r for r in search_results if r.id != chunk_id]

            return filtered_results[:n_results]

        except DocumentNotFoundError:
            raise
        except Exception as e:
            error_msg = f"Failed to find similar chunks for {chunk_id}: {str(e)}"
            self.logger.error(error_msg)
            raise VectorDBError(error_msg) from e
