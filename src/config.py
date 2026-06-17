import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
PDF_PATH = PROJECT_ROOT / "sample-service-manual 1.pdf"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K = int(os.getenv("TOP_K", "8"))
COLLECTION_NAME = "service_manual_specs"

SPEC_KEYWORDS = [
    "nm", "lb-ft", "lb-in", "torque", "tighten",
    "capacity", "fluid", "quart", "liter", "gallon",
    "part number", "part no", "p/n",
    "specification", "spec",
    "pressure", "psi", "kpa",
    "mm", "inch", "in.",
    "voltage", "ampere", "ohm",
]
