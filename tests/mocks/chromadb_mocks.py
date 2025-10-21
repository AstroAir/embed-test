"""Mock implementations for ChromaDB components."""

from typing import Any, Optional


class MockCollection:
    """Mock ChromaDB Collection for testing."""

    def __init__(
        self, name: str = "test_collection", metadata: Optional[dict[str, Any]] = None
    ):
        self.name = name
        self.metadata = metadata or {"created_by": "pdf_vector_system"}
        self._documents: dict[str, dict[str, Any]] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._metadatas: dict[str, dict[str, Any]] = {}

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add documents to the collection."""
        for id_, doc, emb, meta in zip(ids, documents, embeddings, metadatas):
            self._documents[id_] = doc
            self._embeddings[id_] = emb
            self._metadatas[id_] = meta

    def query(
        self,
        query_texts: Optional[list[str]] = None,
        query_embeddings: Optional[list[list[float]]] = None,
        n_results: int = 10,
        where: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
        include: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Query the collection."""
        include = include or ["documents", "metadatas", "distances"]

        # Simple mock implementation - return first n_results
        ids = list(self._documents.keys())[:n_results]

        result = {"ids": [ids]}

        if "documents" in include:
            result["documents"] = [[self._documents[id_] for id_ in ids]]

        if "metadatas" in include:
            result["metadatas"] = [[self._metadatas[id_] for id_ in ids]]

        if "distances" in include:
            # Mock distances - closer to 0 means more similar
            result["distances"] = [[0.1 + i * 0.1 for i in range(len(ids))]]

        if "embeddings" in include:
            result["embeddings"] = [[self._embeddings[id_] for id_ in ids]]

        return result

    def get(
        self,
        ids: Optional[list[str]] = None,
        where: Optional[dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Get documents from the collection."""
        include = include or ["documents", "metadatas"]

        if ids:
            filtered_ids = [id_ for id_ in ids if id_ in self._documents]
        else:
            filtered_ids = list(self._documents.keys())

        if where:
            # Simple filtering by document_id
            if "document_id" in where:
                filtered_ids = [
                    id_
                    for id_ in filtered_ids
                    if self._metadatas.get(id_, {}).get("document_id")
                    == where["document_id"]
                ]

        if limit:
            filtered_ids = filtered_ids[:limit]

        result = {"ids": filtered_ids}

        if "documents" in include:
            result["documents"] = [self._documents[id_] for id_ in filtered_ids]

        if "metadatas" in include:
            result["metadatas"] = [self._metadatas[id_] for id_ in filtered_ids]

        if "embeddings" in include:
            result["embeddings"] = [self._embeddings[id_] for id_ in filtered_ids]

        return result

    def delete(
        self, ids: Optional[list[str]] = None, where: Optional[dict[str, Any]] = None
    ) -> None:
        """Delete documents from the collection."""
        if ids:
            for id_ in ids:
                self._documents.pop(id_, None)
                self._embeddings.pop(id_, None)
                self._metadatas.pop(id_, None)
        elif where:
            # Simple filtering by document_id
            if "document_id" in where:
                ids_to_delete = [
                    id_
                    for id_, meta in self._metadatas.items()
                    if meta.get("document_id") == where["document_id"]
                ]
                for id_ in ids_to_delete:
                    self._documents.pop(id_, None)
                    self._embeddings.pop(id_, None)
                    self._metadatas.pop(id_, None)

    def count(self) -> int:
        """Get the number of documents in the collection."""
        return len(self._documents)

    def peek(self, limit: int = 10) -> dict[str, Any]:
        """Peek at the first few documents in the collection."""
        ids = list(self._documents.keys())[:limit]
        return {
            "ids": ids,
            "documents": [self._documents[id_] for id_ in ids],
            "metadatas": [self._metadatas[id_] for id_ in ids],
            "embeddings": [self._embeddings[id_] for id_ in ids],
        }


class MockChromaDBClient:
    """Mock ChromaDB Client for testing."""

    def __init__(self):
        self._collections: dict[str, MockCollection] = {}
        self.heartbeat_count = 0

    def heartbeat(self) -> int:
        """Mock heartbeat method."""
        self.heartbeat_count += 1
        return self.heartbeat_count

    def create_collection(
        self,
        name: str,
        metadata: Optional[dict[str, Any]] = None,
        embedding_function: Optional[Any] = None,
        get_or_create: bool = False,
    ) -> MockCollection:
        """Create a new collection."""
        if name in self._collections and not get_or_create:
            raise ValueError(f"Collection {name} already exists")

        if name not in self._collections:
            self._collections[name] = MockCollection(name, metadata)

        return self._collections[name]

    def get_or_create_collection(
        self,
        name: str,
        metadata: Optional[dict[str, Any]] = None,
        embedding_function: Optional[Any] = None,
    ) -> MockCollection:
        """Get or create a collection."""
        return self.create_collection(
            name, metadata, embedding_function, get_or_create=True
        )

    def get_collection(
        self, name: str, embedding_function: Optional[Any] = None
    ) -> MockCollection:
        """Get an existing collection."""
        if name not in self._collections:
            raise ValueError(f"Collection {name} does not exist")
        return self._collections[name]

    def list_collections(self) -> list[MockCollection]:
        """List all collections."""
        return list(self._collections.values())

    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        if name in self._collections:
            del self._collections[name]

    def reset(self) -> bool:
        """Reset the client (clear all collections)."""
        self._collections.clear()
        return True


def create_mock_chromadb_client() -> MockChromaDBClient:
    """Factory function to create a mock ChromaDB client."""
    return MockChromaDBClient()


def create_mock_collection(
    name: str = "test_collection",
    documents: Optional[list[str]] = None,
    embeddings: Optional[list[list[float]]] = None,
    metadatas: Optional[list[dict[str, Any]]] = None,
) -> MockCollection:
    """Factory function to create a mock collection with test data."""
    collection = MockCollection(name)

    if documents and embeddings and metadatas:
        ids = [f"doc_{i}" for i in range(len(documents))]
        collection.add(ids, documents, embeddings, metadatas)

    return collection
