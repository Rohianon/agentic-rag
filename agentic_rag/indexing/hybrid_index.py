"""Hybrid index combining vector search with metadata filtering."""

import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from .chunker import Chunk, ChunkType


@dataclass
class IndexedDocument:
    """Tracks indexed document metadata."""

    filename: str
    chunk_count: int
    table_count: int


class HybridIndex:
    """
    Hybrid RAG index using ChromaDB for vector storage and metadata filtering.

    Architecture:
    1. Vector Store: Semantic embeddings for similarity search
    2. Metadata Store: Structured data (tables as JSON) for precise lookup
    3. Hybrid queries: Combine semantic + metadata filters

    Why ChromaDB:
    - Supports metadata filtering alongside vector search
    - Persistent storage for production use
    - Simple local setup, easy migration to cloud
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_dir: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        openai_client: OpenAI | None = None,
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.openai_client = openai_client or OpenAI()

        # Initialize ChromaDB
        if persist_dir:
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Separate storage for structured table data
        self._table_store: dict[str, dict] = {}

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding from OpenAI."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )
        return response.data[0].embedding

    def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """
        Add chunks to the index.

        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []
        embeddings = []

        # Batch embed all chunks
        texts = [chunk.content for chunk in chunks]
        all_embeddings = self._get_embeddings_batch(texts)

        for chunk, embedding in zip(chunks, all_embeddings):
            ids.append(chunk.id)
            documents.append(chunk.content)
            embeddings.append(embedding)

            # Prepare metadata (ChromaDB only supports primitive types)
            meta = {
                "chunk_type": chunk.chunk_type.value,
                "source_file": chunk.source_file,
                "page_num": chunk.page_num,
                "token_count": chunk.token_count,
            }
            meta.update({k: v for k, v in chunk.metadata.items()
                        if isinstance(v, (str, int, float, bool))})
            metadatas.append(meta)

            # Store structured data separately for tables
            if chunk.chunk_type == ChunkType.TABLE and chunk.structured_data:
                self._table_store[chunk.id] = chunk.structured_data

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return len(chunks)

    def search(
        self,
        query: str,
        n_results: int = 5,
        chunk_type: ChunkType | None = None,
        source_file: str | None = None,
        include_tables: bool = True,
    ) -> list[dict]:
        """
        Hybrid search combining semantic similarity with metadata filters.

        Returns list of results with:
        - content: The chunk text
        - metadata: Source info, page number, etc.
        - distance: Similarity score
        - structured_data: For tables, the full JSON data
        """
        # Build metadata filter
        where_filter = {}
        if chunk_type:
            where_filter["chunk_type"] = chunk_type.value
        if source_file:
            where_filter["source_file"] = source_file

        # Get query embedding
        query_embedding = self._get_embedding(query)

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            result = {
                "id": chunk_id,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "relevance_score": 1 - results["distances"][0][i],  # Convert distance to similarity
            }

            # Add structured data for tables
            if chunk_id in self._table_store:
                result["structured_data"] = self._table_store[chunk_id]

            formatted.append(result)

        return formatted

    def search_tables_only(self, query: str, n_results: int = 3) -> list[dict]:
        """Search only table chunks."""
        return self.search(
            query=query,
            n_results=n_results,
            chunk_type=ChunkType.TABLE,
        )

    def get_all_tables(self) -> dict[str, dict]:
        """Return all stored table data."""
        return self._table_store.copy()

    def get_stats(self) -> dict:
        """Get index statistics."""
        count = self.collection.count()

        # Count by type
        text_results = self.collection.get(
            where={"chunk_type": "text"},
            include=[],
        )
        table_results = self.collection.get(
            where={"chunk_type": "table"},
            include=[],
        )

        return {
            "total_chunks": count,
            "text_chunks": len(text_results["ids"]),
            "table_chunks": len(table_results["ids"]),
            "tables_with_data": len(self._table_store),
        }

    def clear(self):
        """Clear all data from the index."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._table_store.clear()

    def save_table_store(self, path: str | Path):
        """Persist table store to disk."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self._table_store, f)

    def load_table_store(self, path: str | Path):
        """Load table store from disk."""
        path = Path(path)
        if path.exists():
            with open(path) as f:
                self._table_store = json.load(f)
