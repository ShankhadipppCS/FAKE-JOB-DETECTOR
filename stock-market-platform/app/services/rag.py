from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DocumentChunk:
    source: str
    text: str


class FinancialRAG:
    def answer(self, question: str, chunks: list[DocumentChunk]) -> str:
        terms = {term.lower() for term in question.split() if len(term) > 3}
        scored = []
        for chunk in chunks:
            score = sum(1 for term in terms if term in chunk.text.lower())
            scored.append((score, chunk))
        context = [chunk for score, chunk in sorted(scored, key=lambda item: item[0], reverse=True)[:3] if score > 0]
        if not context:
            return "No relevant financial context was found in the local knowledge base."
        evidence = " ".join(chunk.text for chunk in context)
        return f"Based on the retrieved financial context: {evidence[:900]}"

