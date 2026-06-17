import chromadb
from typing import List, Dict, Optional
from pathlib import Path


class VectorStore:

    def __init__(
        self,
        collection_name: str = "service_manual_specs",
        persist_dir: str | Path = "chroma_db",
    ):
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        print(f"[VectorStore] Initializing ChromaDB at: {persist_dir}")
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection_name = collection_name
        print(f"[VectorStore] Collection '{collection_name}' has "
              f"{self.collection.count()} documents.")

    def add_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        if not chunks:
            print("[VectorStore] No chunks to add.")
            return

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "page": c["page"],
                "section": c["section"],
                "has_specs": str(c["has_specs"]),
            }
            for c in chunks
        ]

        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                embeddings=embeddings[i:end],
                metadatas=metadatas[i:end],
            )
            print(f"[VectorStore] Added batch {i//batch_size + 1} "
                  f"({end - i} chunks)")

        print(f"[VectorStore] Total documents: {self.collection.count()}")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 8,
        filter_specs: bool = False,
    ) -> List[Dict]:
        where_filter = None
        if filter_specs:
            where_filter = {"has_specs": "True"}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                distance = results["distances"][0][i]
                formatted.append({
                    "text": doc,
                    "page": meta.get("page", "?"),
                    "section": meta.get("section", "Unknown"),
                    "score": round(1 - distance, 4),
                })

        return formatted

    def is_indexed(self) -> bool:
        return self.collection.count() > 0

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print("[VectorStore] Collection cleared.")


if __name__ == "__main__":
    store = VectorStore()
    print(f"Collection count: {store.collection.count()}")
    print(f"Is indexed: {store.is_indexed()}")
