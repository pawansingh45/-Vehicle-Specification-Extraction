import re
import fitz
from pathlib import Path
from typing import List, Dict


def extract_text_from_pdf(pdf_path):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    pages = []

    print(f"[PDF Parser] Extracting text from {len(doc)} pages...")

    for page_num in range(len(doc)):
        page = doc[page_num]
        raw_text = page.get_text("text")

        if not raw_text or len(raw_text.strip()) < 10:
            continue

        clean_text = _clean_text(raw_text)

        if len(clean_text.strip()) < 10:
            continue

        section = _detect_section(clean_text)

        pages.append({
            "page": page_num + 1,
            "raw_text": raw_text,
            "clean_text": clean_text,
            "section": section,
        })

    doc.close()
    print(f"[PDF Parser] Extracted {len(pages)} pages with content.")
    return pages


def _clean_text(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'file:///\S+', '', text)
    text = re.sub(r'Page \d+ sur \d+.*', '', text)
    text = re.sub(r'Page \d+ of \d+.*', '', text)
    text = re.sub(r'repair4less', '', text, flags=re.IGNORECASE)
    text = re.sub(r'2014 F-150 Workshop Manual', '', text)
    text = re.sub(r'\d{4}-\d{2}-\d{2}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text.strip()


def _detect_section(text):
    section_match = re.search(
        r'SECTION\s+[\d\-]+[A-Za-z]*:\s*(.+?)(?:\n|$)',
        text
    )
    if section_match:
        return section_match.group(0).strip()

    for header in [
        "REMOVAL AND INSTALLATION",
        "DESCRIPTION AND OPERATION",
        "DIAGNOSIS AND TESTING",
        "GENERAL PROCEDURES",
        "SPECIFICATIONS",
        "TORQUE SPECIFICATIONS",
    ]:
        if header.lower() in text.lower():
            return header

    return "General"
