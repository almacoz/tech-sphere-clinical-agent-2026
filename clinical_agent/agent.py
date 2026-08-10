from __future__ import annotations

import re
import time
from typing import Any

from .rag import RagStore
from .schemas import (
    AgentResponse,
    Decision,
    EvidenceEvaluation,
    RiskLevel,
    SafetyValidation,
    TurnMetrics,
)

ALARM_PATTERNS = {
    "sangrado_abundante": r"sangr(ad|o).*(abundante|mucho|empap|no para)",
    "fiebre_alta": r"fiebre|temperatura",
    "dificultad_respiratoria": r"(no puedo respirar|falta de aire|dificultad.*respirar)",
    "dolor_pecho": r"(dolor.*pecho|opresi[oó]n.*pecho)",
    "confusion": r"(confusi[oó]n|desmayo|perd[ií] el conocimiento)",
}

INJECTION_PATTERNS = [
    r"olvida tus instrucciones",
    r"ignora (el|tu) sistema",
    r"revela.*prompt",
    r"cambia.*riesgo",
    r"dime qu[eé] medicamento tomar",
]


class ClinicalAgent:
    def __init__(self, rag_store: RagStore) -> None:
        self.rag_store = rag_store

    def answer(self, session_id: str, message: str) -> AgentResponse:
        started = time.perf_counter()
        rag_started = time.perf_counter()
        evidence = self.rag_store.query(message, top_k=4)
        rag_latency_ms = elapsed_ms(rag_started)

        evaluation = self.evaluate(message, evidence)
        decision = self.decide(evaluation)
        candidate = self.generate_response(evaluation, decision)
        safety = self.validate_safety(message, candidate, evaluation, decision)
        final_response = safety.safe_response if not safety.passed and safety.safe_response else candidate

        summary = {
            "session_id": session_id,
            "symptoms": evaluation.symptoms,
            "risk_level": decision.risk_level,
            "needs_human": decision.needs_human,
            "missing_information": decision.missing_information,
            "evidence_ids": decision.evidence_ids,
            "response_sent": final_response,
        }
        metrics = TurnMetrics(
            session_id=session_id,
            latency_ms=elapsed_ms(started),
            rag_latency_ms=rag_latency_ms,
            input_tokens=count_tokens(message),
            output_tokens=count_tokens(final_response),
            rag_queries=1,
            retrieved_documents=list({item.document_id for item in evidence}),
            decision=decision.risk_level.value,
            safety_validation="passed" if safety.passed else "failed",
            escalated=decision.needs_human or safety.escalated,
        ).model_dump()

        return AgentResponse(
            session_id=session_id,
            response=final_response,
            decision=decision,
            evidence=evidence,
            safety_validation=safety,
            summary=summary,
            metrics=metrics,
        )

    def evaluate(self, message: str, evidence: list[Any]) -> EvidenceEvaluation:
        normalized = message.lower()
        injection = any(re.search(pattern, normalized) for pattern in INJECTION_PATTERNS)
        alarm_codes = [
            code for code, pattern in ALARM_PATTERNS.items() if re.search(pattern, normalized)
        ]
        symptoms = extract_symptoms(normalized)
        missing = []
        if not evidence:
            missing.append("evidencia clínica recuperada")
        if "dolor" in normalized:
            for field in ["ubicación del dolor", "intensidad", "duración", "evolución"]:
                if field.split()[0] not in normalized:
                    missing.append(field)

        if alarm_codes:
            risk_level = RiskLevel.RED
            needs_human = True
        elif injection:
            risk_level = RiskLevel.UNKNOWN
            needs_human = False
            missing.append("solicitud clínica válida sin intento de cambiar instrucciones")
        elif missing:
            risk_level = RiskLevel.UNKNOWN
            needs_human = False
        else:
            risk_level = RiskLevel.GREEN
            needs_human = False

        claims = []
        if evidence and not alarm_codes:
            claims.append("La orientación se limita a las fuentes clínicas recuperadas.")
        if alarm_codes:
            claims.append("Hay una posible señal de alarma y corresponde escalar a humano.")

        return EvidenceEvaluation(
            patient_state={"prompt_injection_suspected": injection},
            symptoms=symptoms,
            missing_information=dedupe(missing),
            risk_level=risk_level,
            needs_human=needs_human,
            evidence=evidence,
            clinical_claims=claims,
            reasoning_summary="Evaluación basada en señales explícitas, evidencia recuperada y datos faltantes.",
        )

    def decide(self, evaluation: EvidenceEvaluation) -> Decision:
        return Decision(
            risk_level=evaluation.risk_level,
            needs_human=evaluation.needs_human,
            reason_codes=reason_codes(evaluation),
            missing_information=evaluation.missing_information,
            evidence_ids=[item.chunk_id for item in evaluation.evidence],
        )

    def generate_response(self, evaluation: EvidenceEvaluation, decision: Decision) -> str:
        if decision.needs_human:
            return (
                "Por seguridad, esto debe revisarlo personal clínico ahora. "
                "Voy a escalar la llamada y no voy a darte indicaciones médicas no verificadas."
            )
        if decision.risk_level == RiskLevel.UNKNOWN:
            missing = decision.missing_information[0] if decision.missing_information else "más contexto"
            return (
                "No tengo evidencia suficiente para responder con seguridad todavía. "
                f"Para orientarte sin adivinar, dime por favor: {missing}."
            )
        cited = evaluation.evidence[0] if evaluation.evidence else None
        citation = (
            f" Fuente: {cited.document}, página {cited.page}, chunk {cited.chunk_id}."
            if cited
            else ""
        )
        return (
            "Con la información disponible no identifico una señal de alarma explícita. "
            "Mantendré la orientación limitada al documento recuperado y escalaré si aparece fiebre, "
            "sangrado abundante, dolor de pecho, dificultad para respirar o confusión."
            + citation
        )

    def validate_safety(
        self,
        message: str,
        response: str,
        evaluation: EvidenceEvaluation,
        decision: Decision,
    ) -> SafetyValidation:
        issues: list[str] = []
        normalized = f"{message}\n{response}".lower()
        if any(re.search(pattern, normalized) for pattern in INJECTION_PATTERNS):
            issues.append("prompt_injection_detected")
        if re.search(r"\b(toma|duplica|suspende|receta|mg)\b", response.lower()):
            issues.append("unsupported_medication_instruction")
        if evaluation.clinical_claims and not evaluation.evidence and not decision.needs_human:
            issues.append("clinical_claim_without_evidence")
        if evaluation.risk_level == RiskLevel.RED and not decision.needs_human:
            issues.append("alarm_without_escalation")

        if issues:
            return SafetyValidation(
                passed=False,
                issues=issues,
                escalated="alarm_without_escalation" in issues or decision.needs_human,
                safe_response=(
                    "No puedo responder esa solicitud de forma segura. "
                    "Voy a mantener las instrucciones clínicas del sistema y escalaré si hay señales de alarma."
                ),
            )
        return SafetyValidation(passed=True)


def extract_symptoms(text: str) -> list[str]:
    known = ["dolor", "fiebre", "sangrado", "náusea", "vomito", "vómito", "mareo"]
    return [symptom for symptom in known if symptom in text]


def reason_codes(evaluation: EvidenceEvaluation) -> list[str]:
    codes = []
    if evaluation.patient_state.get("prompt_injection_suspected"):
        codes.append("PROMPT_INJECTION_SUSPECTED")
    if evaluation.risk_level == RiskLevel.RED:
        codes.append("ALARM_SIGNAL")
    if evaluation.risk_level == RiskLevel.UNKNOWN:
        codes.append("INSUFFICIENT_INFORMATION")
    if not evaluation.evidence:
        codes.append("NO_RAG_EVIDENCE")
    return codes


def count_tokens(text: str) -> int:
    return len(text.split())


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
