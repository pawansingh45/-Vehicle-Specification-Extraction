import json
import re
from typing import List, Dict, Optional
from groq import Groq


SYSTEM_PROMPT = """You are an expert automotive technician and data extraction specialist.
Your job is to extract vehicle specifications from service manual text.

Given context from a vehicle service manual and a user query, extract ALL relevant
specifications as structured JSON.

RULES:
1. ONLY extract specifications that are explicitly stated in the provided context.
2. Do NOT infer, guess, or make up any values.
3. Each specification should include: component, spec_type, value, unit, and context.
4. For torque specs, always include both metric (Nm) and imperial (lb-ft or lb-in) if both are provided.
5. If multiple specs match the query, return ALL of them.
6. If no relevant specs are found in the context, return an empty array [].

OUTPUT FORMAT (JSON array only, no other text):
[
  {
    "component": "Name of the component (e.g., 'Brake Caliper Bolt')",
    "spec_type": "Type of specification (e.g., 'Torque', 'Capacity', 'Pressure')",
    "value": "Numeric value as a string (e.g., '35')",
    "unit": "Unit of measurement (e.g., 'Nm', 'lb-ft', 'quarts')",
    "context": "Brief context or note about where/when this spec applies"
  }
]

Return ONLY the JSON array. No markdown, no explanation, no code fences."""


QUERY_TEMPLATE = """CONTEXT FROM SERVICE MANUAL:
{context}

USER QUERY: {query}

Extract all relevant specifications from the context above that answer the user's query.
Return a JSON array of specifications. If no specs are found, return [].
"""


class LLMExtractor:

    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile"):
        if not api_key:
            raise ValueError(
                "Groq API key is required. Set GROQ_API_KEY in .env file."
            )

        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        print(f"[LLM Extractor] Initialized with model: {model_name}")

    def extract_specs(self, query: str, context_chunks: List[Dict]) -> List[Dict]:
        if not context_chunks:
            print("[LLM Extractor] No context chunks provided.")
            return []

        context_parts = []
        for i, chunk in enumerate(context_chunks):
            page = chunk.get("page", "?")
            section = chunk.get("section", "Unknown")
            text = chunk.get("text", "")
            context_parts.append(
                f"[Source: Page {page} | {section}]\n{text}"
            )

        context_str = "\n\n---\n\n".join(context_parts)

        user_prompt = QUERY_TEMPLATE.format(
            context=context_str,
            query=query,
        )

        print(f"[LLM Extractor] Querying {self.model_name}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )

            raw_output = response.choices[0].message.content.strip()
            specs = self._parse_json_output(raw_output)

            pages_used = sorted(set(
                str(c.get("page", "?")) for c in context_chunks
            ))
            for spec in specs:
                spec["source_pages"] = ", ".join(pages_used)

            print(f"[LLM Extractor] Extracted {len(specs)} specifications.")
            return specs

        except Exception as e:
            print(f"[LLM Extractor] Error: {e}")
            return self._fallback_extraction(query, context_chunks)

    def _parse_json_output(self, raw_output: str) -> List[Dict]:
        try:
            result = json.loads(raw_output)
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                for key in ["specifications", "specs", "results", "data"]:
                    if key in result and isinstance(result[key], list):
                        return result[key]
                if "component" in result:
                    return [result]
            return []
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        print("[LLM Extractor] Warning: Could not parse LLM output as JSON.")
        return []

    def _fallback_extraction(self, query: str, context_chunks: List[Dict]) -> List[Dict]:
        print("[LLM Extractor] Using fallback regex extraction...")
        specs = []

        for chunk in context_chunks:
            text = chunk.get("text", "")

            torque_matches = re.finditer(
                r'(?:tighten|torque)\s+to\s+(\d+)\s*(Nm|lb[\-\s]?ft|lb[\-\s]?in)'
                r'(?:\s*\((\d+)\s*(lb[\-\s]?ft|lb[\-\s]?in|Nm)\))?',
                text, re.IGNORECASE
            )
            for match in torque_matches:
                specs.append({
                    "component": "See context",
                    "spec_type": "Torque",
                    "value": match.group(1),
                    "unit": match.group(2),
                    "context": text[max(0, match.start()-50):match.end()+50].strip(),
                    "source_pages": str(chunk.get("page", "?")),
                })

        return specs


if __name__ == "__main__":
    from config import GROQ_API_KEY, MODEL_NAME

    extractor = LLMExtractor(GROQ_API_KEY, MODEL_NAME)

    test_chunks = [
        {
            "text": "Tighten to 90 Nm (66 lb-ft).\nUse the hex-holding feature.",
            "page": 41,
            "section": "Front Suspension",
        }
    ]
    result = extractor.extract_specs("suspension bolt torque", test_chunks)
    print(json.dumps(result, indent=2))
