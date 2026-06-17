import re
from typing import List, Dict


def chunk_pages(pages, chunk_size=500, chunk_overlap=100):
    all_chunks = []
    chunk_id = 0

    print(f"[Chunker] Chunking {len(pages)} pages (size={chunk_size}, overlap={chunk_overlap})...")

    for page_data in pages:
        page_num = page_data["page"]
        section = page_data["section"]
        text = page_data["clean_text"]

        sub_sections = _split_into_subsections(text)

        for sub_section in sub_sections:
            text_chunks = _recursive_split(sub_section, chunk_size, chunk_overlap)

            for chunk_text in text_chunks:
                chunk_text = chunk_text.strip()
                if len(chunk_text) < 30:
                    continue

                has_specs = _contains_spec_keywords(chunk_text)

                all_chunks.append({
                    "chunk_id": f"chunk_{chunk_id:05d}",
                    "page": page_num,
                    "section": section,
                    "text": chunk_text,
                    "has_specs": has_specs,
                })
                chunk_id += 1

    print(f"[Chunker] Created {len(all_chunks)} chunks "
          f"({sum(1 for c in all_chunks if c['has_specs'])} contain specs).")
    return all_chunks


def _split_into_subsections(text):
    patterns = [
        r'\n(?=\d+\.\s*[A-Z])',
        r'\n(?=[A-Z]{3,}[\s:])',
        r'\n(?=Procedure revision date)',
        r'\n(?=NOTICE:)',
        r'\n(?=NOTE:)',
    ]

    combined_pattern = '|'.join(patterns)
    sections = re.split(combined_pattern, text)
    return [s.strip() for s in sections if s and len(s.strip()) > 20]


def _recursive_split(text, chunk_size, overlap):
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    separators = ['\n\n', '\n', '. ', ' ']

    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            current_chunk = ""
            for part in parts:
                test_chunk = current_chunk + sep + part if current_chunk else part
                if len(test_chunk) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + sep + part
                else:
                    current_chunk = test_chunk

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            if chunks:
                return chunks

    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


def _contains_spec_keywords(text):
    text_lower = text.lower()
    spec_patterns = [
        r'\d+\s*nm\b',
        r'\d+\s*lb[\-\s]?ft',
        r'\d+\s*lb[\-\s]?in',
        r'tighten\s+to\b',
        r'torque\b',
        r'capacity\b',
        r'specification',
        r'\d+\.?\d*\s*(?:quart|liter|gallon)',
        r'\d+\.?\d*\s*(?:psi|kpa)',
        r'part\s*(?:number|no|#)',
    ]

    for pattern in spec_patterns:
        if re.search(pattern, text_lower):
            return True
    return False
