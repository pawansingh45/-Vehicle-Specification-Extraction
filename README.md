🔧 Vehicle Specification Extraction — RAG Pipeline

A **Retrieval-Augmented Generation (RAG)** system that extracts vehicle specifications (torque values, fluid capacities, part numbers) from automotive service manual PDFs using LLMs and semantic retrieval.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LLM](https://img.shields.io/badge/LLM-Llama%203.3%2070B-orange)
![Vector DB](https://img.shields.io/badge/VectorDB-ChromaDB-green)

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Design Decisions](#design-decisions)
- [Project Structure](#project-structure)
- [Output Format](#output-format)
- [Ideas for Improvement](#ideas-for-improvement)

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  PDF Manual  │────▶│  PyMuPDF     │────▶│  Smart Chunker   │
│  (852 pages) │     │  Text Extrac │     │  Section-aware   │
└──────────────┘     └──────────────┘     └────────┬─────────┘
                                                    │
                                          ┌─────────▼─────────┐
                                          │  Sentence-BERT    │
                                          │  all-MiniLM-L6-v2 │
                                          │  (Local Embeddings)│
                                          └─────────┬─────────┘
                                                    │
                                          ┌─────────▼─────────┐
                                          │    ChromaDB        │
                                          │  (Vector Store)    │
                                          └─────────┬─────────┘
                                                    │
┌──────────────┐     ┌──────────────┐     ┌─────────▼─────────┐
│  User Query  │────▶│  Embed Query │────▶│  Similarity Search│
└──────────────┘     └──────────────┘     └─────────┬─────────┘
                                                    │
                                          ┌─────────▼─────────┐
                                          │   Groq LLM        │
                                          │  Llama 3.3 70B    │
                                          │  (Structured Ext.)│
                                          └─────────┬─────────┘
                                                    │
                                          ┌─────────▼─────────┐
                                          │  Structured Output │
                                          │  JSON / CSV        │
                                          └───────────────────┘
```

---

## 🚀 Setup

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com) (sign up → API Keys → Create)

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API key (already set in .env, or update it)
#    Edit .env and set your GROQ_API_KEY

# 3. Index the PDF (first run only — takes ~2-3 minutes)
python main.py --index

# 4. Run a query
python main.py --query "Torque for wheel nut"
```

---

## 💻 Usage

### CLI Interface

```bash
# Index the service manual PDF
python main.py --index

# Single query
python main.py --query "What is the torque for suspension bolts?"

# Run demo queries (batch mode)
python main.py --batch

# Export results to JSON and CSV
python main.py --batch --export both

# Force re-index and run full pipeline
python main.py --reindex --batch --export json
```

### Streamlit UI (Bonus)

```bash
streamlit run app.py
```

Opens a professional dark-themed dashboard where you can:
- Type queries and see extracted specs as formatted cards
- View source chunks with relevance scores
- Download results as JSON or CSV
- See raw LLM output for debugging

---

## 🧠 Design Decisions

### 1. PDF Parsing: PyMuPDF over pdfminer

| Factor | PyMuPDF | pdfminer |
|--------|---------|----------|
| **Speed** | ~3x faster | Slower |
| **Text quality** | Excellent | Good |
| **Memory** | Lower | Higher |
| **Maintenance** | Actively maintained | Less active |

PyMuPDF (`fitz`) provides faster extraction with better text quality. The service manual has minimal complex layouts, making text-mode extraction sufficient.

### 2. Embeddings: Local Sentence-Transformers over OpenAI

- **Cost**: $0 — runs entirely on local CPU
- **Privacy**: No data leaves the machine
- **Speed**: Fast enough for 852 pages (~2 min to embed all chunks)
- **Quality**: `all-MiniLM-L6-v2` achieves strong retrieval performance for this domain

### 3. Vector Store: ChromaDB over FAISS

- **Metadata filtering**: ChromaDB lets us filter by `has_specs=True` to prioritize spec-containing chunks
- **Persistence**: Built-in persistent storage — no need to re-index on restart
- **Simplicity**: Single Python package, no C++ dependencies

### 4. LLM: Groq/Llama 3.3 70B over GPT-4

- **Cost**: Free tier with generous limits
- **Speed**: Groq's LPU delivers ~500 tokens/sec — near-instant responses
- **Quality**: Llama 3.3 70B is competitive with GPT-4 for structured extraction
- **JSON mode**: Native `response_format={"type": "json_object"}` support

### 5. Chunking Strategy: Section-Aware

Instead of naive fixed-size chunking, the pipeline:
1. Cleans PDF artifacts (watermarks, URLs, headers)
2. Splits on section headers (SECTION, REMOVAL AND INSTALLATION, etc.)
3. Recursively splits large sections on paragraph → sentence → word boundaries
4. Tags chunks containing spec keywords (Nm, lb-ft, torque, capacity) for priority retrieval

---

## 📁 Project Structure

```
Assignment/
├── .env                    # API key configuration
├── .gitignore
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── src/                    # Core pipeline modules
│   ├── __init__.py
│   ├── config.py           # Centralized settings
│   ├── pdf_parser.py       # PDF text extraction (PyMuPDF)
│   ├── chunker.py          # Section-aware text chunking
│   ├── embeddings.py       # Sentence-Transformers engine
│   ├── vector_store.py     # ChromaDB vector storage
│   ├── llm_extractor.py    # Groq LLM structured extraction
│   └── pipeline.py         # End-to-end orchestration
│
├── main.py                 # CLI entry point
├── app.py                  # Streamlit UI (Bonus)
│
├── output/                 # Generated output files
│   ├── extracted_specs.json
│   └── extracted_specs.csv
│
├── chroma_db/              # Persistent vector database
│
└── sample-service-manual 1.pdf  # Input PDF
```

---

## 📊 Output Format

### JSON Output

```json
[
  {
    "component": "Wheel Nut",
    "spec_type": "Torque",
    "value": "204",
    "unit": "Nm",
    "context": "Tighten the wheel nuts in a star pattern",
    "source_pages": "41, 42"
  }
]
```

### CSV Output

| component | spec_type | value | unit | context | source_pages |
|-----------|-----------|-------|------|---------|-------------|
| Wheel Nut | Torque | 204 | Nm | Tighten in star pattern | 41, 42 |

---

## 🛠️ Tools Used

| Component | Tool | Purpose |
|-----------|------|---------|
| PDF Parsing | PyMuPDF (fitz) | Text extraction from PDF |
| Embeddings | Sentence-Transformers (all-MiniLM-L6-v2) | Local text embeddings |
| Vector Store | ChromaDB | Persistent similarity search |
| LLM | Groq (Llama 3.3 70B) | Structured spec extraction |
| UI | Streamlit | Interactive dashboard |
| Export | pandas | CSV generation |

---

## 📝 License

This project was built as an assignment submission.
](https://github.com/pawansingh45/-Vehicle-Specification-Extraction)
