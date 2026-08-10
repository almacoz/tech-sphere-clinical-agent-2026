from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    UNKNOWN = "UNKNOWN"


class SupportLevel(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    NO_EVIDENCE = "NO_EVIDENCE"


class EvidenceItem(BaseModel):
    document_id: str
    document: str
    page: int
    chunk_id: str
    quote_or_excerpt: str
    retrieval_score: float = Field(ge=0.0)
    relevance: float = Field(ge=0.0, le=1.0)


class ClinicalExtraction(BaseModel):
    symptoms: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    severity: str | None = None
    duration: str | None = None
    trajectory: str | None = None
    associated_symptoms: list[str] = Field(default_factory=list)
    postoperative_context: dict[str, Any] = Field(default_factory=dict)
    alarm_signals: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    prompt_injection_detected: bool = False


class EvidenceEvaluation(BaseModel):
    patient_state: dict[str, Any] = Field(default_factory=dict)
    extraction: ClinicalExtraction | None = None
    symptoms: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    risk_level: RiskLevel
    needs_human: bool
    evidence: list[EvidenceItem] = Field(default_factory=list)
    clinical_claims: list[str] = Field(default_factory=list)
    support_level: SupportLevel = SupportLevel.NO_EVIDENCE
    evidence_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    reasoning_summary: str


class Decision(BaseModel):
    risk_level: RiskLevel
    needs_human: bool
    reason_codes: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    decision_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SafetyValidation(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    safe_response: str | None = None
    escalated: bool = False


class AgentRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    message: str


class AgentResponse(BaseModel):
    session_id: str
    response: str
    decision: Decision
    evidence: list[EvidenceItem]
    safety_validation: SafetyValidation
    summary: dict[str, Any]
    metrics: dict[str, Any]


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    status: str = "AVAILABLE"


class TurnMetrics(BaseModel):
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    session_id: str
    latency_ms: int = 0
    stt_latency_ms: int = 0
    rag_latency_ms: int = 0
    llm_latency_ms: int = 0
    tts_latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    token_accounting: str = "estimated"
    rag_queries: int = 0
    retrieved_documents: list[str] = Field(default_factory=list)
    decision: str = ""
    safety_validation: str = ""
    escalated: bool = False
