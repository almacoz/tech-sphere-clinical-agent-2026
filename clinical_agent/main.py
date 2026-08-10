from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agent import ClinicalAgent
from .rag import RagStore
from .schemas import AgentRequest, AgentResponse, DocumentRecord, SessionResetRequest

app = FastAPI(title="Tech Sphere Clinical Agent", version="0.1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
rag_store = RagStore()
agent = ClinicalAgent(
    rag_store,
    use_llm_extraction=os.getenv("CLINICAL_AGENT_USE_LLM", "1") != "0",
)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


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


@app.post("/knowledge/upload", response_model=DocumentRecord)
async def upload_knowledge(file: UploadFile = File(...)) -> DocumentRecord:
    return await upload_document(file)


@app.get("/documents", response_model=list[DocumentRecord])
def list_documents() -> list[DocumentRecord]:
    return rag_store.list_documents()


@app.get("/knowledge", response_model=list[DocumentRecord])
def list_knowledge() -> list[DocumentRecord]:
    return list_documents()


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, bool]:
    deleted = rag_store.delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True}


@app.delete("/knowledge/{document_id}")
def delete_knowledge(document_id: str) -> dict[str, bool]:
    return delete_document(document_id)


@app.post("/agent/respond", response_model=AgentResponse)
def respond(request: AgentRequest) -> AgentResponse:
    return agent.answer(session_id=request.session_id, message=request.message)


@app.post("/session/reset")
def reset_session(request: SessionResetRequest) -> dict[str, bool]:
    return {"deleted": agent.session_store.delete(request.session_id)}
