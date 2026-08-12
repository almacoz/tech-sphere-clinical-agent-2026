from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pypdf import PdfReader

from . import tts as tts_module
from .agent import ClinicalAgent
from .rag import RagStore
from .runtime_manager import RuntimeManager
from .schemas import (
    AgentRequest,
    AgentResponse,
    DocumentRecord,
    SessionResetRequest,
    TtsRequest,
)

SUPPORTED_REQUEST_SUFFIXES = {".txt", ".pdf"}

app = FastAPI(title="Agente Clínico Postoperatorio", version="0.1.0")
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
rag_store = RagStore()
agent = ClinicalAgent(
    rag_store,
    use_llm_extraction=os.getenv("CLINICAL_AGENT_USE_LLM", "1") != "0",
)
runtime_manager = RuntimeManager()


def extract_document_text(filename: str, content: bytes) -> str:
    suffix = Path(filename or "document.txt").suffix.lower()
    if suffix not in SUPPORTED_REQUEST_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported document format. Supported formats: .txt, .pdf",
        )

    if suffix == ".txt":
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail="Unable to decode TXT document as UTF-8.",
            ) from exc

    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
            pages = []
            for page in reader.pages:
                extracted = page.extract_text() or ""
                pages.append(extracted)
            text = "\f".join(pages)
        except Exception as exc:
            print(f"RuntimeManager/PDF ingestion error: {type(exc).__name__}: {exc}", flush=True)
            raise HTTPException(
                status_code=400,
                detail="The PDF could not be read.",
            ) from exc

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="The PDF does not contain extractable text.",
            )
        return text

    raise HTTPException(
        status_code=400,
        detail="Unsupported document format. Supported formats: .txt, .pdf",
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    ico = STATIC_DIR / "favicon.ico"
    if ico.exists():
        return FileResponse(ico)
    # Return 204 No Content instead of 404 so browsers don't log a missing asset
    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=DocumentRecord)
async def upload_document(file: UploadFile = File(...)) -> DocumentRecord:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded document is empty")

    filename = file.filename or "document.txt"
    text = extract_document_text(filename, content)
    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded document is empty")

    return rag_store.upsert_document(filename, text)


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


@app.get("/runtime/status")
def runtime_status() -> dict[str, object]:
    return runtime_manager.get_status()


@app.post("/runtime/pull")
async def runtime_pull(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    status_code, result = runtime_manager.request_pull(payload)
    return JSONResponse(status_code=status_code, content=result)


@app.get("/runtime/pull/status")
def runtime_pull_status() -> dict[str, Any]:
    return runtime_manager.get_pull_status()


@app.post("/agent/respond", response_model=AgentResponse)
def respond(request: AgentRequest, voice: bool = False) -> AgentResponse:
    session_id = request.session_id.strip() if request.session_id else str(uuid4())
    result = agent.answer(session_id=session_id, message=request.message)
    if voice:
        # Sintetiza la respuesta con Kokoro-82M y la suma a la latencia
        # total del turno, para que P50/P95 reflejen de verdad "desde que el
        # paciente termina de hablar hasta que empieza a sonar el audio del
        # agente" (definición exacta de la métrica de la rúbrica §5). Si
        # Kokoro no está instalado, el turno sigue respondiendo en texto:
        # la voz es una capa opcional sobre el mismo pipeline clínico, no
        # un requisito para que /agent/respond funcione.
        try:
            audio_bytes, tts_latency_ms = tts_module.synthesize(result.response)
            result.audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
            result.metrics["tts_latency_ms"] = tts_latency_ms
            result.metrics["total_latency_ms"] = (
                result.metrics.get("total_latency_ms", 0) + tts_latency_ms
            )
        except (RuntimeError, ValueError) as error:
            result.metrics["tts_latency_ms"] = 0
            result.metrics.setdefault("tts_error", str(error))
    return result


@app.get("/tts/status")
def tts_status() -> dict[str, Any]:
    return tts_module.tts_status()


@app.post("/tts")
def synthesize_speech(request: TtsRequest) -> Response:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text vacío")
    try:
        audio_bytes, latency_ms = tts_module.synthesize(
            request.text,
            voice=request.voice or tts_module.DEFAULT_VOICE,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"X-TTS-Latency-Ms": str(latency_ms)},
    )


@app.post("/session/reset")
def reset_session(request: SessionResetRequest) -> dict[str, bool]:
    return {"deleted": agent.session_store.delete(request.session_id)}


@app.get("/audit/{session_id}")
def get_audit_trail(session_id: str) -> dict[str, Any]:
    """Ventana de auditoria: devuelve, para una sesion, la traza completa de
    cada turno (guardrails evaluados, handrail del LLM, rama de riesgo que
    disparo y verificaciones de seguridad), para poder ver en que momento y
    por que se tomo cada decision sin depender de leer logs de consola."""
    session = agent.session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "turns": session.audit_log}
