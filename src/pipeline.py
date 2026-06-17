import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Optional

from .config import (
    PDF_PATH, CHROMA_DB_DIR, OUTPUT_DIR,
    GROQ_API_KEY, MODEL_NAME, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, COLLECTION_NAME,
)
from .pdf_parser import extract_text_from_pdf
from .chunker import chunk_pages
from .embeddings import EmbeddingEngine
from .vector_store import VectorStore
from .llm_extractor import LLMExtractor


class SpecExtractionPipeline:

    def __init__(self):
        print("=" * 60)
        print("  Vehicle Specification Extraction Pipeline")
        print("=" * 60)

        self.embedding_engine = EmbeddingEngine(EMBEDDING_MODEL)
        self.vector_store = VectorStore(COLLECTION_NAME, CHROMA_DB_DIR)
        self.llm_extractor = LLMExtractor(GROQ_API_KEY, MODEL_NAME)

        print(f"\n[Pipeline] Initialized successfully.")
        print(f"  PDF: {PDF_PATH.name}")
        print(f"  LLM: {MODEL_NAME}")
        print(f"  Embeddings: {EMBEDDING_MODEL}")
        print(f"  Vector DB: {CHROMA_DB_DIR}")
        print()

    def index_pdf(self, force_reindex: bool = False) -> int:
        if self.vector_store.is_indexed() and not force_reindex:
            count = self.vector_store.collection.count()
            print(f"[Pipeline] Index already exists with {count} chunks. "
                  f"Use --reindex to rebuild.")
            return count

        if force_reindex:
            print("[Pipeline] Clearing existing index...")
            self.vector_store.clear()

        start_time = time.time()

        print("\n[Step 1/3] Parsing PDF...")
        pages = extract_text_from_pdf(PDF_PATH)

        print("\n[Step 2/3] Chunking text...")
        chunks = chunk_pages(pages, CHUNK_SIZE, CHUNK_OVERLAP)

        print("\n[Step 3/3] Embedding and storing chunks...")
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_engine.embed_texts(texts)
        self.vector_store.add_chunks(chunks, embeddings)

        elapsed = time.time() - start_time
        print(f"\n[Pipeline] Indexing complete! "
              f"{len(chunks)} chunks in {elapsed:.1f}s")

        return len(chunks)

    def query(self, query_text: str, top_k: Optional[int] = None) -> Dict:
        if not self.vector_store.is_indexed():
            print("[Pipeline] No index found. Run indexing first.")
            return {"query": query_text, "specs": [], "sources": []}

        k = top_k or TOP_K
        print(f"\n{'-' * 50}")
        print(f"Query: {query_text}")
        print(f"{'-' * 50}")

        query_embedding = self.embedding_engine.embed_query(query_text)

        results = self.vector_store.search(query_embedding, k, filter_specs=True)
        if len(results) < 3:
            results = self.vector_store.search(query_embedding, k, filter_specs=False)

        print(f"[Pipeline] Retrieved {len(results)} relevant chunks.")
        for i, r in enumerate(results[:3]):
            print(f"  [{i+1}] Page {r['page']} | Score: {r['score']} | "
                  f"{r['text'][:80]}...")

        specs = self.llm_extractor.extract_specs(query_text, results)

        return {
            "query": query_text,
            "specs": specs,
            "sources": results,
        }

    def batch_query(self, queries: List[str]) -> List[Dict]:
        all_results = []
        for i, q in enumerate(queries):
            print(f"\n[Batch {i+1}/{len(queries)}]")
            result = self.query(q)
            all_results.append(result)
        return all_results

    def export_json(self, results: List[Dict], output_path: Optional[str] = None) -> str:
        if output_path is None:
            output_path = str(OUTPUT_DIR / "extracted_specs.json")

        all_specs = []
        for result in results:
            query = result.get("query", "")
            for spec in result.get("specs", []):
                spec["query"] = query
                all_specs.append(spec)

        output = {
            "total_specs": len(all_specs),
            "results": results,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n[Pipeline] Exported {len(all_specs)} specs to {output_path}")
        return output_path

    def export_csv(self, results: List[Dict], output_path: Optional[str] = None) -> str:
        if output_path is None:
            output_path = str(OUTPUT_DIR / "extracted_specs.csv")

        all_specs = []
        for result in results:
            query = result.get("query", "")
            for spec in result.get("specs", []):
                spec["query"] = query
                all_specs.append(spec)

        if not all_specs:
            print("[Pipeline] No specs to export.")
            return output_path

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["query", "component", "spec_type", "value", "unit",
                       "context", "source_pages"]

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_specs)

        print(f"\n[Pipeline] Exported {len(all_specs)} specs to {output_path}")
        return output_path
