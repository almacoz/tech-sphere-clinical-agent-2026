from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile

from .agent import ClinicalAgent
from .rag import RagStore
from .schemas import AgentRequest, AgentResponse, DocumentRecord

app = FastAPI(title="Tech Sphere Clinical Agent", version="0.1.0")
rag_store = RagStore()
agent = ClinicalAgent(rag_store)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentRecord)
async def upload_document(file: UploadFile = File(...)) -> DocumentRecord:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded document is empty")
    text = content.decode("utf-8", errors="ignore")
    return rag_store.upsert_document(file.filename or "document.txt", text)


@app.get("/documents", response_model=list[DocumentRecord])
def list_documents() -> list[DocumentRecord]:
    return rag_store.list_documents()


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, bool]:
    deleted = rag_store.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True}


@app.post("/agent/respond", response_model=AgentResponse)
def respond(request: AgentRequest) -> AgentResponse:
    return agent.answer(session_id=request.session_id, message=request.message)
