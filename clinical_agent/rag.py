from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from uuid import uuid4

from .schemas import DocumentRecord, EvidenceItem

TOKEN_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


@dataclass
class Chunk:
    document_id: str
    filename: str
    page: int
    chunk_id: str
    text: str
    vector: Counter[str]


class RagStore:
    def __init__(self, chunk_size: int = 120, overlap: int = 24) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._chunks: list[Chunk] = []
        self._documents: dict[str, DocumentRecord] = {}
        self.query_log: list[dict[str, object]] = []

    def upsert_document(self, filename: str, text: str) -> DocumentRecord:
        document_id = str(uuid4())
        self.delete_document_by_filename(filename)
        chunks = self._chunk_document(document_id, filename, text)
        self._chunks.extend(chunks)
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            chunk_count=len(chunks),
            status="indexed",
        )
        self._documents[document_id] = record
        return record

    def delete_document(self, document_id: str) -> bool:
        existed = document_id in self._documents
        self._documents.pop(document_id, None)
        self._chunks = [chunk for chunk in self._chunks if chunk.document_id != document_id]
        return existed

    def delete_document_by_filename(self, filename: str) -> None:
        for document_id, record in list(self._documents.items()):
            if record.filename == filename:
                self.delete_document(document_id)

    def list_documents(self) -> list[DocumentRecord]:
        return list(self._documents.values())

    def query(self, query_text: str, top_k: int = 4) -> list[EvidenceItem]:
        query_vector = Counter(tokenize(query_text))
        scored = []
        for chunk in self._chunks:
            score = self._cosine(query_vector, chunk.vector)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        evidence = [
            EvidenceItem(
                document_id=chunk.document_id,
                document=chunk.filename,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                quote_or_excerpt=chunk.text[:420],
                retrieval_score=round(score, 4),
                relevance=min(1.0, round(score * 2, 4)),
            )
            for score, chunk in scored[:top_k]
        ]
        self.query_log.append(
            {
                "query": query_text,
                "top_k": top_k,
                "retrieved": [item.model_dump() for item in evidence],
            }
        )
        return evidence

    def _chunk_document(self, document_id: str, filename: str, text: str) -> list[Chunk]:
        pages = text.split("\f")
        chunks: list[Chunk] = []
        for page_index, page_text in enumerate(pages, start=1):
            tokens = page_text.split()
            step = max(1, self.chunk_size - self.overlap)
            for start in range(0, len(tokens), step):
                window = tokens[start : start + self.chunk_size]
                if not window:
                    continue
                chunk_text = " ".join(window)
                chunk_id = f"{document_id}:p{page_index}:c{len(chunks) + 1}"
                chunks.append(
                    Chunk(
                        document_id=document_id,
                        filename=filename,
                        page=page_index,
                        chunk_id=chunk_id,
                        text=chunk_text,
                        vector=Counter(tokenize(chunk_text)),
                    )
                )
                if start + self.chunk_size >= len(tokens):
                    break
        return chunks

    @staticmethod
    def _cosine(left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0
        dot = sum(left[token] * right.get(token, 0) for token in left)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return dot / (left_norm * right_norm)
