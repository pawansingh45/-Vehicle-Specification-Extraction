from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[Embeddings] Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_embedding_dimension()
        print(f"[Embeddings] Model loaded. Dimension: {self.dimension}")

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        print(f"[Embeddings] Embedding {len(texts)} texts (batch_size={batch_size})...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        print(f"[Embeddings] Done. Generated {len(embeddings)} embeddings.")
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding[0].tolist()


if __name__ == "__main__":
    engine = EmbeddingEngine()

    test_texts = [
        "Tighten the brake caliper bolt to 35 Nm",
        "The engine oil capacity is 6.0 quarts",
        "Front suspension description and operation",
    ]

    embeddings = engine.embed_texts(test_texts)
    print(f"\nEmbedding shapes: {len(embeddings)} x {len(embeddings[0])}")

    query_emb = engine.embed_query("brake caliper torque")
    print(f"Query embedding shape: {len(query_emb)}")
