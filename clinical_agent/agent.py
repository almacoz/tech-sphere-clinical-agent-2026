import os
import json
import re
import time
from typing import Any

from .llm import generate_json_raw
from .prompts import clinical_extraction_prompt
from .rag import RagStore
from .schemas import (
    AgentResponse,
    ClinicalExtraction,
    ClinicalState,
    ConversationTurn,
    Decision,
    EvidenceStatus,
    EvidenceEvaluation,
    MissingInformationItem,
    RiskLevel,
    SafetyValidation,
    SessionState,
    SupportLevel,
    TurnMetrics,
)
from .session_store import SessionStore

ALARM_PATTERNS = {
    "sangrado_abundante": r"sangr(ad|o|a).*(abundante|mucho|empap|no para)",
    "fiebre_alta": r"fiebre|temperatura",
    "dificultad_respiratoria": r"(no puedo respirar|falta de aire|dificultad.*respirar)",
    "dolor_pecho": r"(dolor.*pecho|opresi[oó]n.*pecho)",
    "confusion": r"(confusi[oó]n|desmayo|perd[ií] el conocimiento)",
}

INJECTION_PATTERNS = [
    r"olvida tus instrucciones",
    r"ignora (el|tu) sistema",
    r"ignora.*señal",
    r"revela.*prompt",
    r"(cambia|marca).*riesgo",
    r"marca mi caso como verde",
    r"dime qu[eé] medicamento tomar",
    r"(ajustar|cambiar|subir|bajar).*(dosis|medicamento|antibi[oó]tico)",
    r"no escales",
]


class ClinicalAgent:
    def __init__(
        self,
        rag_store: RagStore,
        use_llm_extraction: bool = True,
        session_store: SessionStore | None = None,
    ) -> None:
        self.rag_store = rag_store
        self.use_llm_extraction = use_llm_extraction
        self.last_llm_latency_ms = 0
        self.last_llm_provider = "ollama"
        self.last_llm_model = "llama3.2"
        self.last_llm_status = "unavailable"
        self.last_llm_fallback_used = True
        self.session_store = session_store or SessionStore()

    def answer(self, session_id: str, message: str) -> AgentResponse:
        started = time.perf_counter()
        session = self.session_store.get_or_create(session_id)
        extraction = self.extract_clinical(message, session)
        previous_state = session.clinical_state
        clinical_state = merge_clinical_state(previous_state, extraction)
        rag_started = time.perf_counter()
        evidence = self.rag_store.query(message, top_k=4)
        rag_latency_ms = elapsed_ms(rag_started)
        if debug_llm_enabled():
            print("\n========== RAG RETRIEVED ==========")
            print([item.model_dump() for item in evidence])
            print("===================================\n")

        evaluation = self.evaluate(message, extraction, clinical_state, evidence)
        decision_started = time.perf_counter()
        decision = self.decide(evaluation)
        decision_latency_ms = elapsed_ms(decision_started)
        if debug_llm_enabled():
            print("\n========== DECISION ==========")
            print(decision.model_dump_json(indent=2))
            print("==============================\n")
        candidate = self.generate_response(evaluation, decision)
        safety_started = time.perf_counter()
        safety = self.validate_safety(message, extraction, evidence, candidate, evaluation, decision)
        safety_latency_ms = elapsed_ms(safety_started)
        final_response = safety.safe_response if not safety.passed and safety.safe_response else candidate
        session = update_session_after_turn(
            session=session,
            patient_message=message,
            assistant_response=final_response,
            clinical_state=clinical_state,
            evidence=evidence,
            decision=decision,
        )
        self.session_store.update(session_id, session)

        summary = {
            "session_id": session_id,
            "turn_count": session.turn_count,
            "clinical_extraction": extraction.model_dump(),
            "previous_clinical_state": previous_state.model_dump(),
            "clinical_state": clinical_state.model_dump(),
            "contradictions": clinical_state.contradictions,
            "known_information": decision.known_information,
            "symptoms": extraction.symptoms,
            "risk_level": decision.risk_level,
            "needs_human": decision.needs_human,
            "missing_information": decision.missing_information,
            "asked_information": session.asked_information,
            "missing_information_priority": [
                item.model_dump() for item in decision.missing_information_priority
            ],
            "evidence_status": decision.evidence_status,
            "evidence_ids": decision.evidence_ids,
            "evidence_coverage": evaluation.evidence_coverage,
            "unsupported_claims": evaluation.unsupported_claims,
            "response_sent": final_response,
        }
        metrics = TurnMetrics(
            session_id=session_id,
            latency_ms=elapsed_ms(started),
            total_latency_ms=elapsed_ms(started),
            rag_latency_ms=rag_latency_ms,
            llm_latency_ms=self.last_llm_latency_ms,
            decision_latency_ms=decision_latency_ms,
            safety_latency_ms=safety_latency_ms,
            input_tokens=None,
            output_tokens=None,
            token_accounting="unavailable",
            llm_provider=self.last_llm_provider,
            llm_model=self.last_llm_model,
            llm_status=self.last_llm_status,
            fallback_used=self.last_llm_fallback_used,
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

    def extract_clinical(
        self,
        message: str,
        session: SessionState | None = None,
    ) -> ClinicalExtraction:
        self.last_llm_latency_ms = 0
        self.last_llm_provider = "fallback"
        self.last_llm_model = "unknown"
        self.last_llm_status = "unavailable"
        self.last_llm_fallback_used = True
        if self.use_llm_extraction:
            try:
                llm_started = time.perf_counter()
                raw_response = generate_json_raw(
                    clinical_extraction_prompt,
                    contextual_extraction_message(message, session),
                )
                self.last_llm_latency_ms = elapsed_ms(llm_started)
                self.last_llm_provider = "ollama"
                self.last_llm_model = "llama3.2"
                self.last_llm_status = "success"
                self.last_llm_fallback_used = False
                if debug_llm_enabled():
                    print("\n========== LLM RAW RESPONSE ==========")
                    print(raw_response)
                    print("======================================\n")
                payload = json.loads(raw_response)
                extraction = ClinicalExtraction.model_validate(payload)
                extraction = apply_deterministic_safety_overrides(message, extraction)
                if debug_llm_enabled():
                    print("\n========== CLINICAL EXTRACTION ==========")
                    print(extraction.model_dump_json(indent=2))
                    print("=========================================\n")
                return extraction
            except RuntimeError as error:
                self.last_llm_latency_ms = elapsed_ms(llm_started) if "llm_started" in locals() else 0
                self.last_llm_provider = "ollama"
                self.last_llm_model = "llama3.2"
                self.last_llm_status = "unavailable"
                self.last_llm_fallback_used = True
                if debug_llm_enabled():
                    print("\n========== LLM EXTRACTION UNAVAILABLE ==========")
                    print(type(error).__name__, repr(error))
                    print("==============================================\n")
                extraction = contextual_deterministic_extract_clinical(message, session)
                if debug_llm_enabled():
                    print("\n========== CLINICAL EXTRACTION FALLBACK ==========")
                    print(extraction.model_dump_json(indent=2))
                    print("==================================================\n")
                return extraction
            except json.JSONDecodeError as error:
                self.last_llm_latency_ms = elapsed_ms(llm_started) if "llm_started" in locals() else 0
                self.last_llm_provider = "ollama"
                self.last_llm_model = "llama3.2"
                self.last_llm_status = "invalid_json"
                self.last_llm_fallback_used = True
                if debug_llm_enabled():
                    print("\n========== LLM EXTRACTION INVALID JSON ==========")
                    print(repr(error))
                    print("===============================================\n")
                extraction = contextual_deterministic_extract_clinical(message, session)
                if debug_llm_enabled():
                    print("\n========== CLINICAL EXTRACTION FALLBACK ==========")
                    print(extraction.model_dump_json(indent=2))
                    print("==================================================\n")
                return extraction
            except Exception as error:
                self.last_llm_latency_ms = elapsed_ms(llm_started) if "llm_started" in locals() else 0
                self.last_llm_provider = "ollama"
                self.last_llm_model = "llama3.2"
                self.last_llm_status = "schema_error"
                self.last_llm_fallback_used = True
                if debug_llm_enabled():
                    print("\n========== LLM EXTRACTION ERROR ==========")
                    print(type(error).__name__, repr(error))
                    print("==========================================\n")
                extraction = contextual_deterministic_extract_clinical(message, session)
                if debug_llm_enabled():
                    print("\n========== CLINICAL EXTRACTION FALLBACK ==========")
                    print(extraction.model_dump_json(indent=2))
                    print("==================================================\n")
                return extraction
        extraction = contextual_deterministic_extract_clinical(message, session)
        if debug_llm_enabled():
            print("\n========== CLINICAL EXTRACTION ==========")
            print(extraction.model_dump_json(indent=2))
            print("=========================================\n")
        return extraction
        extraction = contextual_deterministic_extract_clinical(message, session)
        if debug_llm_enabled():
            print("\n========== CLINICAL EXTRACTION ==========")
            print(extraction.model_dump_json(indent=2))
            print("=========================================\n")
        return extraction

    def evaluate(
        self,
        message: str,
        extraction: ClinicalExtraction,
        clinical_state: ClinicalState,
        evidence: list[Any],
    ) -> EvidenceEvaluation:
        patient_missing = list(clinical_state.missing_information)
        known = known_information_from_state(clinical_state)
        missing_priority = prioritize_missing_information(patient_missing)
        evidence_status = (
            EvidenceStatus.HAS_RELEVANT_EVIDENCE
            if evidence
            else EvidenceStatus.NO_RELEVANT_EVIDENCE
        )

        if clinical_state.contradictions:
            risk_level = RiskLevel.UNKNOWN
            needs_human = False
            patient_missing.append("aclaración")
        elif clinical_state.alarm_signals:
            risk_level = RiskLevel.RED
            needs_human = True
        elif extraction.prompt_injection_detected:
            risk_level = RiskLevel.UNKNOWN
            needs_human = False
            patient_missing.append("solicitud clínica válida")
        elif clinical_state.trajectory in {"empeorando", "empeora", "peor"}:
            risk_level = RiskLevel.YELLOW
            needs_human = False
        elif patient_missing:
            risk_level = RiskLevel.UNKNOWN
            needs_human = False
        else:
            risk_level = RiskLevel.GREEN
            needs_human = False

        claims = []
        if evidence and not clinical_state.alarm_signals:
            claims.append("La orientación se limita a las fuentes clínicas recuperadas.")
        if clinical_state.alarm_signals:
            claims.append("Hay una posible señal de alarma y corresponde escalar a humano.")
        support_level = SupportLevel.NO_EVIDENCE if not evidence else SupportLevel.SUPPORTED
        coverage = 0.0 if not evidence else min(1.0, sum(item.relevance for item in evidence) / len(evidence))

        return EvidenceEvaluation(
            patient_state={
                "prompt_injection_suspected": extraction.prompt_injection_detected,
                "contradictions": clinical_state.contradictions,
            },
            extraction=extraction,
            symptoms=clinical_state.symptoms,
            missing_information=dedupe(patient_missing),
            patient_information_missing=dedupe(patient_missing),
            missing_information_priority=missing_priority,
            known_information=known,
            evidence_status=evidence_status,
            risk_level=risk_level,
            needs_human=needs_human,
            evidence=evidence,
            clinical_claims=claims,
            support_level=support_level,
            evidence_coverage=round(coverage, 4),
            unsupported_claims=[],
            reasoning_summary="Evaluación basada en señales explícitas, evidencia recuperada y datos faltantes.",
        )

    def decide(self, evaluation: EvidenceEvaluation) -> Decision:
        return Decision(
            risk_level=evaluation.risk_level,
            needs_human=evaluation.needs_human,
            reason_codes=reason_codes(evaluation),
            missing_information=evaluation.patient_information_missing,
            missing_information_priority=evaluation.missing_information_priority,
            known_information=evaluation.known_information,
            evidence_status=evaluation.evidence_status,
            evidence_ids=[item.chunk_id for item in evaluation.evidence],
            decision_confidence=decision_confidence(evaluation),
        )

    def generate_response(self, evaluation: EvidenceEvaluation, decision: Decision) -> str:
        if decision.needs_human:
            return (
                "Por seguridad, esto debe revisarlo personal clínico ahora. "
                "Voy a escalar la llamada y no voy a darte indicaciones médicas no verificadas."
            )
        if decision.risk_level == RiskLevel.UNKNOWN:
            if evaluation.extraction and evaluation.extraction.prompt_injection_detected:
                pass
            if evaluation.patient_state.get("contradictions"):
                contradiction = evaluation.patient_state["contradictions"][-1]
                return (
                    "Para evitar asumir mal: antes entendí "
                    f"{contradiction['field']} como {contradiction['previous']}, "
                    f"pero ahora mencionas {contradiction['new']}. ¿Cuál dato debo usar?"
                )
            prioritized_fields = [item.field for item in decision.missing_information_priority]
            if prioritized_fields[:2] == ["ubicación", "evolución"]:
                return (
                    "Entiendo. ¿En qué parte te duele y notas que el dolor está empeorando, "
                    "mejorando o se mantiene igual?"
                )
            missing = next_question_for(decision.missing_information)
            return (
                "Entiendo. Para orientarte sin adivinar, dime por favor: "
                f"{missing}."
            )
        cited = evaluation.evidence[0] if evaluation.evidence else None
        citation = (
            f" Fuente: {cited.document}, página {cited.page}, chunk {cited.chunk_id}."
            if cited
            else ""
        )
        return (
            "Con la información y la fuente recuperada, no identifico una señal de alarma explícita. "
            "Mantendré la orientación limitada a esa evidencia y escalaré si aparece una señal de alarma."
            + citation
        )

    def validate_safety(
        self,
        message: str,
        extraction: ClinicalExtraction,
        evidence: list[Any],
        response: str,
        evaluation: EvidenceEvaluation,
        decision: Decision,
    ) -> SafetyValidation:
        issues: list[str] = []
        normalized_message = message.lower()
        normalized_response = response.lower()
        if extraction.prompt_injection_detected or any(
            re.search(pattern, normalized_message) for pattern in INJECTION_PATTERNS
        ):
            issues.append("prompt_injection_detected")
        if re.search(r"\b(toma|duplica|suspende|receta|mg)\b", normalized_response):
            issues.append("unsupported_medication_instruction")
        if re.search(r"\b(es|tienes|diagn[oó]stico)\b.*\b(infecci[oó]n|trombosis|infarto)\b", normalized_response):
            issues.append("diagnosis_without_human")
        if evaluation.clinical_claims and not evaluation.evidence and not decision.needs_human:
            issues.append("clinical_claim_without_evidence")
        if evaluation.risk_level == RiskLevel.RED and not decision.needs_human:
            issues.append("alarm_without_escalation")
        if (
            decision.risk_level == RiskLevel.UNKNOWN
            and "dime por favor" not in normalized_response
            and "?" not in response
        ):
            issues.append("unknown_without_question")
        if not evidence and decision.risk_level == RiskLevel.GREEN:
            issues.append("green_without_evidence")

        if issues:
            return SafetyValidation(
                passed=False,
                issues=issues,
                unsupported_claims=evaluation.unsupported_claims,
                escalated="alarm_without_escalation" in issues or decision.needs_human,
                safe_response=(
                    "No puedo responder esa solicitud de forma segura. "
                    "Voy a mantener las instrucciones clínicas del sistema y escalaré si hay señales de alarma."
                ),
            )
        return SafetyValidation(passed=True)


def deterministic_extract_clinical(message: str) -> ClinicalExtraction:
        normalized = message.lower()
        injection = any(re.search(pattern, normalized) for pattern in INJECTION_PATTERNS)
        alarm_codes = [
            code for code, pattern in ALARM_PATTERNS.items() if re.search(pattern, normalized)
        ]
        symptoms = extract_symptoms(normalized)
        missing = missing_information_for(normalized)
        locations = [value for value in ["pecho", "herida", "abdomen", "pierna"] if value in normalized]
        severity = extract_severity(normalized)
        duration = extract_duration(normalized)
        trajectory = extract_trajectory(normalized)
        return ClinicalExtraction(
            symptoms=symptoms,
            locations=locations,
            severity=severity,
            duration=duration,
            trajectory=trajectory,
            associated_symptoms=[symptom for symptom in symptoms if symptom not in {"dolor"}],
            postoperative_context={"mentioned": bool(re.search(r"\b(cirug[ií]a|postoperatorio|operaci[oó]n)\b", normalized))},
            alarm_signals=alarm_codes,
            missing_information=missing,
            prompt_injection_detected=injection,
        )


def contextual_deterministic_extract_clinical(
    message: str,
    session: SessionState | None,
) -> ClinicalExtraction:
    extraction = deterministic_extract_clinical(message)
    if session is None or not session.last_question:
        return extraction
    question = session.last_question.lower()
    normalized = message.lower()
    updates: dict[str, Any] = {}
    if "en qué parte" in question or "dónde" in question or "donde" in question:
        locations = extraction.locations or [
            value for value in ["pecho", "herida", "abdomen", "pierna"] if value in normalized
        ]
        if locations:
            updates["locations"] = locations
    if "desde cuándo" in question or "cuándo" in question or "cuando" in question:
        duration = extraction.duration or extract_duration(normalized)
        if duration:
            updates["duration"] = duration
    if "empeorando" in question or "mejorando" in question or "mantiene igual" in question:
        trajectory = extraction.trajectory or extract_trajectory(normalized)
        if trajectory:
            updates["trajectory"] = trajectory
    return extraction.model_copy(update=updates)


def contextual_extraction_message(message: str, session: SessionState | None) -> str:
    context = {
        "known_information": session.known_information if session else {},
        "previous_question": session.last_question if session else None,
        "recent_conversation": [
            turn.model_dump() for turn in (session.history[-6:] if session else [])
        ],
        "current_patient_message": message,
    }
    return json.dumps(context, ensure_ascii=False, indent=2)


def update_session_after_turn(
    session: SessionState,
    patient_message: str,
    assistant_response: str,
    clinical_state: ClinicalState,
    evidence: list[Any],
    decision: Decision,
) -> SessionState:
    session.turn_count += 1
    session.history.extend(
        [
            ConversationTurn(role="patient", content=patient_message),
            ConversationTurn(role="assistant", content=assistant_response),
        ]
    )
    session.history = session.history[-12:]
    session.clinical_state = clinical_state
    session.known_information = decision.known_information
    session.missing_information = decision.missing_information
    session.asked_information = dedupe(
        session.asked_information + [item.field for item in decision.missing_information_priority]
    )
    session.evidence = evidence
    session.current_decision = decision.model_dump()
    session.last_question = assistant_response if "?" in assistant_response else None
    return session


def apply_deterministic_safety_overrides(
    message: str,
    extraction: ClinicalExtraction,
) -> ClinicalExtraction:
    deterministic = deterministic_extract_clinical(message)
    return extraction.model_copy(
        update={
            "alarm_signals": dedupe(extraction.alarm_signals + deterministic.alarm_signals),
            "prompt_injection_detected": (
                extraction.prompt_injection_detected
                or deterministic.prompt_injection_detected
            ),
        }
    )


def merge_clinical_state(
    previous: ClinicalState,
    extraction: ClinicalExtraction,
) -> ClinicalState:
    already_known: list[str] = []
    symptoms = merge_list(previous.symptoms, extraction.symptoms, "symptoms", already_known)
    locations = merge_list(previous.locations, extraction.locations, "locations", already_known)
    associated = merge_list(
        previous.associated_symptoms,
        extraction.associated_symptoms,
        "associated_symptoms",
        already_known,
    )
    alarms = merge_list(
        previous.alarm_signals,
        extraction.alarm_signals,
        "alarm_signals",
        already_known,
    )
    contradictions: list[dict[str, Any]] = []
    severity = merge_scalar(previous.severity, extraction.severity, "severity", already_known, contradictions)
    duration = merge_scalar(previous.duration, extraction.duration, "duration", already_known, contradictions)
    trajectory = merge_scalar(
        previous.trajectory,
        normalize_trajectory(extraction.trajectory),
        "trajectory",
        already_known,
        contradictions,
    )
    state = ClinicalState(
        symptoms=symptoms,
        locations=locations,
        severity=severity,
        duration=duration,
        trajectory=trajectory,
        associated_symptoms=associated,
        alarm_signals=alarms,
        asked=dedupe(previous.asked + normalize_patient_missing(extraction.missing_information)),
        already_known=dedupe(previous.already_known + already_known),
        contradictions=previous.contradictions + contradictions,
    )
    state.missing_information = missing_information_for_state(state)
    return state


def merge_list(
    previous: list[str],
    incoming: list[str],
    field: str,
    already_known: list[str],
) -> list[str]:
    for value in incoming:
        if value in previous:
            already_known.append(field)
    return dedupe(previous + incoming)


def merge_scalar(
    previous: str | None,
    incoming: str | None,
    field: str,
    already_known: list[str],
    contradictions: list[dict[str, Any]],
) -> str | None:
    if incoming is None:
        return previous
    if previous == incoming:
        already_known.append(field)
        return previous
    if previous and incoming and previous != incoming:
        contradictions.append({"field": field, "previous": previous, "new": incoming})
        return previous
    return incoming or previous


def missing_information_for_state(state: ClinicalState) -> list[str]:
    missing = []
    if "dolor" in state.symptoms:
        if not state.locations:
            missing.append("ubicación")
        if not state.trajectory:
            missing.append("evolución")
        if not state.severity:
            missing.append("intensidad")
        if not state.duration:
            missing.append("duración")
    return dedupe([item for item in missing if item not in state.asked or item in {"ubicación", "evolución"}])


def normalize_patient_missing(values: list[str]) -> list[str]:
    normalized = []
    for value in values:
        text = value.lower().strip(" ¿?")
        if "evidencia" in text or "rag" in text or "document" in text:
            continue
        if "ubic" in text:
            normalized.append("ubicación")
        elif "evol" in text or "mejor" in text or "peor" in text:
            normalized.append("evolución")
        elif "intens" in text:
            normalized.append("intensidad")
        elif "duraci" in text or "cuándo" in text or "cuando" in text:
            normalized.append("duración")
        else:
            normalized.append(value.strip())
    return dedupe(normalized)


def known_information_from(extraction: ClinicalExtraction) -> dict[str, Any]:
    known: dict[str, Any] = {}
    if extraction.symptoms:
        known["symptoms"] = extraction.symptoms
    if extraction.locations:
        known["locations"] = extraction.locations
    if extraction.severity:
        known["severity"] = normalize_severity(extraction.severity)
    if extraction.duration:
        known["duration"] = extraction.duration
    if extraction.trajectory:
        known["trajectory"] = extraction.trajectory
    if extraction.associated_symptoms:
        known["associated_symptoms"] = extraction.associated_symptoms
    if extraction.alarm_signals:
        known["alarm_signals"] = extraction.alarm_signals
    return known


def known_information_from_state(state: ClinicalState) -> dict[str, Any]:
    known: dict[str, Any] = {}
    if state.symptoms:
        known["symptoms"] = state.symptoms
    if state.locations:
        known["locations"] = state.locations
    if state.severity:
        known["severity"] = normalize_severity(state.severity)
    if state.duration:
        known["duration"] = state.duration
    if state.trajectory:
        known["trajectory"] = state.trajectory
    if state.associated_symptoms:
        known["associated_symptoms"] = state.associated_symptoms
    if state.alarm_signals:
        known["alarm_signals"] = state.alarm_signals
    return known


def prioritize_missing_information(values: list[str]) -> list[MissingInformationItem]:
    priority = {
        "evolución": ("HIGH", "puede cambiar la clasificación de riesgo"),
        "ubicación": ("HIGH", "puede revelar localizaciones de alarma"),
        "intensidad": ("MEDIUM", "ayuda a caracterizar el síntoma"),
        "duración": ("LOW", "ya puede estar parcialmente mencionada o ser menos decisiva"),
    }
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    items = []
    for field in values:
        level, reason = priority.get(field, ("MEDIUM", None))
        items.append(MissingInformationItem(field=field, priority=level, reason=reason))
    return sorted(items, key=lambda item: (order.get(item.priority, 1), values.index(item.field)))


def normalize_severity(value: str) -> str:
    return "alta" if value.lower() in {"bastante", "muchísimo", "muchisimo", "intenso", "severo", "insoportable", "alta"} else value


def next_question_for(missing_information: list[str]) -> str:
    missing = set(missing_information)
    if {"ubicación", "evolución"}.issubset(missing):
        return "¿en qué parte te duele y el dolor está empeorando, mejorando o igual?"
    if "evolución" in missing:
        return "¿el dolor está empeorando, mejorando o se mantiene igual?"
    if "ubicación" in missing:
        return "¿en qué parte te duele?"
    if "intensidad" in missing:
        return "¿qué tan intenso es el dolor?"
    if "duración" in missing:
        return "¿desde cuándo empezó?"
    return "más contexto"


def debug_llm_enabled() -> bool:
    return os.getenv("CLINICAL_AGENT_DEBUG_LLM", "0") == "1"


def extract_symptoms(text: str) -> list[str]:
    known = ["dolor", "fiebre", "sangrado", "náusea", "vomito", "vómito", "mareo"]
    symptoms = [symptom for symptom in known if symptom in text]
    if re.search(r"\b(duele|doloroso|adolorido)\b", text):
        symptoms.append("dolor")
    return dedupe(symptoms)


def extract_severity(text: str) -> str | None:
    match = re.search(r"\b(bastante|much[ií]simo|intenso|severo|insoportable|alta)\b", text)
    return match.group(1) if match else None


def extract_duration(text: str) -> str | None:
    match = re.search(r"\bdesde que [^.?!,]+", text)
    if match:
        return match.group(0)
    if re.search(r"\bayer\b", text):
        return "ayer"
    if re.search(r"\bhoy\b", text):
        return "hoy"
    return "mencionada" if re.search(r"\b(hace|desde|horas?|d[ií]as?|ayer)\b", text) else None


def extract_trajectory(text: str) -> str | None:
    if re.search(r"\b(empeora|empeorando|peor|aumenta)\b", text):
        return "empeorando"
    if re.search(r"\b(mejora|mejorando|mejor|disminuye)\b", text):
        return "mejorando"
    if re.search(r"\b(igual|se mantiene|mantiene igual)\b", text):
        return "igual"
    return None


def normalize_trajectory(value: str | None) -> str | None:
    if value is None:
        return None
    return extract_trajectory(value.lower()) or value


def missing_information_for(text: str) -> list[str]:
    missing = []
    if "dolor" in extract_symptoms(text):
        checks = [
            ("ubicación del dolor", r"\b(pecho|herida|abdomen|pierna|cabeza|espalda)\b"),
            ("intensidad", r"\b(leve|moderado|bastante|intenso|severo|much[ií]simo|[0-9]\s*/\s*10)\b"),
            ("duración", r"\b(hace|desde|horas?|d[ií]as?|ayer)\b"),
            ("evolución del dolor", r"\b(mejor|peor|empeora|igual|aumenta|disminuye)\b"),
        ]
        for label, pattern in checks:
            if not re.search(pattern, text):
                missing.append(label)
    return missing


def reason_codes(evaluation: EvidenceEvaluation) -> list[str]:
    codes = []
    if evaluation.patient_state.get("prompt_injection_suspected"):
        codes.append("PROMPT_INJECTION_SUSPECTED")
    if evaluation.risk_level == RiskLevel.RED:
        codes.append("ALARM_SIGNAL")
    if evaluation.risk_level == RiskLevel.UNKNOWN:
        if evaluation.patient_information_missing:
            codes.append("PATIENT_INFORMATION_MISSING")
        else:
            codes.append("INSUFFICIENT_INFORMATION")
    if not evaluation.evidence and not evaluation.patient_information_missing:
        codes.append("NO_RAG_EVIDENCE")
    if (
        evaluation.support_level in {SupportLevel.UNSUPPORTED, SupportLevel.NO_EVIDENCE}
        and not evaluation.patient_information_missing
    ):
        codes.append(evaluation.support_level.value)
    return codes


def decision_confidence(evaluation: EvidenceEvaluation) -> float:
    if evaluation.risk_level == RiskLevel.RED:
        return 0.95
    if evaluation.risk_level == RiskLevel.UNKNOWN:
        return 0.35 if not evaluation.evidence else 0.55
    return min(0.85, 0.5 + evaluation.evidence_coverage / 2)


def count_tokens(text: str) -> int:
    return len(text.split())


def elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
