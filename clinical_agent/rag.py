from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .schemas import DocumentRecord, EvidenceItem

try:  # pragma: no cover - optional dependency behind local runtime
    import chromadb
except Exception:  # pragma: no cover
    chromadb = None

try:  # pragma: no cover - Ollama embedding API is the local semantic backend here
    from ollama import embeddings as ollama_embeddings
except Exception:  # pragma: no cover
    ollama_embeddings = None

TOKEN_RE = re.compile(r"[a-záéíóúñü0-9]+", re.IGNORECASE)


@dataclass
class Chunk:
    document_id: str
    filename: str
    page: int
    chunk_id: str
    text: str


class RagStore:
    def __init__(self, chunk_size: int = 120, overlap: int = 24) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
        self.chroma_persist_dir = os.getenv("CHROMA_PERSIST_DIR", str(Path(__file__).resolve().parent.parent / "data" / "chroma"))
        self.collection_name = os.getenv("CHROMA_COLLECTION", "clinical_knowledge")
        self._documents: dict[str, DocumentRecord] = {}
        self._chunks: list[Chunk] = []
        self.query_log: list[dict[str, object]] = []

        if chromadb is None:
            raise RuntimeError("chromadb is required for vector retrieval")
        self._chroma_client = chromadb.PersistentClient(path=self.chroma_persist_dir)
        self._collection = self._chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._hydrate_from_collection()

    def upsert_document(self, filename: str, text: str) -> DocumentRecord:
        document_id = str(uuid4())
        self.delete_document_by_filename(filename)
        chunks = self._chunk_document(document_id, filename, text)

        if not chunks:
            return DocumentRecord(
                document_id=document_id,
                filename=filename,
                chunk_count=0,
                status="AVAILABLE",
            )

        ids = []
        embeddings = []
        documents = []
        metadatas = []
        for chunk in chunks:
            embedding = self._embed(chunk.text)
            ids.append(chunk.chunk_id)
            embeddings.append(embedding)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "document_id": chunk.document_id,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "source": "clinical_knowledge",
                }
            )

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            chunk_count=len(chunks),
            status="AVAILABLE",
        )
        self._documents[document_id] = record
        self._chunks.extend(chunks)
        return record

    def delete_document(self, document_id: str) -> bool:
        existed = document_id in self._documents
        if not existed:
            return False

        ids_to_remove = []
        collected = self._collection.get(where={"document_id": document_id}, include=["metadatas"])
        ids_to_remove = list(collected.get("ids", []))
        if ids_to_remove:
            self._collection.delete(ids=ids_to_remove)

        record = self._documents.pop(document_id)
        self._chunks = [chunk for chunk in self._chunks if chunk.document_id != document_id]
        self._documents = {key: value for key, value in self._documents.items() if value.filename != record.filename}
        return True

    def delete_document_by_filename(self, filename: str) -> None:
        for document_id, record in list(self._documents.items()):
            if record.filename == filename:
                self.delete_document(document_id)

    def list_documents(self) -> list[DocumentRecord]:
        return list(self._documents.values())

    def query(self, query_text: str, top_k: int = 4) -> list[EvidenceItem]:
        started = time.perf_counter()
        query_embedding = self._embed(query_text)
        if not query_text.strip():
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        distances = results.get("distances", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        evidence = []
        for distance, document, metadata in zip(distances, documents, metadatas):
            if distance is None:
                continue
            retrieval_score = max(0.0, 1.0 - float(distance))
            if retrieval_score <= 0.0:
                continue
            metadata = metadata or {}
            document_id = str(metadata.get("document_id") or "")
            filename = str(metadata.get("filename") or "")
            chunk_id = str(metadata.get("chunk_id") or "")
            page = int(metadata.get("page") or 1)
            evidence.append(
                EvidenceItem(
                    document_id=document_id,
                    document=filename,
                    page=page,
                    chunk_id=chunk_id,
                    quote_or_excerpt=document[:420] if document else "",
                    retrieval_score=round(retrieval_score, 4),
                    relevance=round(retrieval_score, 4),
                )
            )

        query_latency_ms = int((time.perf_counter() - started) * 1000)
        self.query_log.append(
            {
                "query": query_text,
                "top_k": top_k,
                "retrieved": [item.model_dump() for item in evidence],
                "retrieval_method": "vector",
                "embedding_model": self.embedding_model,
                "retrieval_latency_ms": query_latency_ms,
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
                    )
                )
                if start + self.chunk_size >= len(tokens):
                    break
        return chunks

    def _embed(self, text: str) -> list[float]:
        if not text.strip():
            return []
        if ollama_embeddings is None:
            raise RuntimeError("Ollama embedding API is not available")
        payload = ollama_embeddings(model=self.embedding_model, prompt=text)
        if isinstance(payload, dict):
            embedding = payload.get("embedding") or payload.get("data")
            if isinstance(embedding, dict):
                embedding = embedding.get("embedding")
            if isinstance(embedding, list):
                return [float(value) for value in embedding]
        if hasattr(payload, "embedding"):
            embedding = payload.embedding
            return [float(value) for value in embedding]
        if isinstance(payload, list):
            return [float(value) for value in payload]
        raise RuntimeError("Unsupported embedding payload shape from Ollama")

    def _hydrate_from_collection(self) -> None:
        try:
            records = self._collection.get(include=["metadatas", "documents"])
        except Exception:
            records = {"ids": [], "metadatas": [], "documents": []}

        ids = records.get("ids", [])
        metadatas = records.get("metadatas", [])
        documents = records.get("documents", [])
        if not ids:
            return

        for index, metadata in enumerate(metadatas):
            if not metadata:
                continue
            document_id = str(metadata.get("document_id") or "")
            filename = str(metadata.get("filename") or "")
            if not document_id or not filename:
                continue
            record = self._documents.get(document_id)
            if record is None:
                self._documents[document_id] = DocumentRecord(
                    document_id=document_id,
                    filename=filename,
                    chunk_count=0,
                    status="AVAILABLE",
                )
            self._documents[document_id].chunk_count = max(
                self._documents[document_id].chunk_count,
                len([item for item in metadatas if (item or {}).get("document_id") == document_id]),
            )
            self._chunks.append(
                Chunk(
                    document_id=document_id,
                    filename=filename,
                    page=int(metadata.get("page") or 1),
                    chunk_id=str(metadata.get("chunk_id") or f"{document_id}:chunk:{index}"),
                    text=str(documents[index] or ""),
                )
            )
