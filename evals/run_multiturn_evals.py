from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from clinical_agent.agent import ClinicalAgent
from clinical_agent.rag import RagStore


ROOT = Path(__file__).resolve().parent / "multiturn"
RESULTS = Path(__file__).resolve().parent / "results"


def main() -> None:
    rows = [
        run_session_persistence(),
        run_redundant_question(),
        run_contextual_answer(),
        run_session_isolation(),
        run_contradiction(),
    ]
    metrics = {
        "total_cases": len(rows),
        "state_persistence": score(rows, "state_persistence"),
        "correct_information_merge": score(rows, "correct_information_merge"),
        "redundant_question_rate": 1 - score(rows, "no_redundant_question"),
        "context_interpretation": score(rows, "context_interpretation"),
        "session_isolation": score(rows, "session_isolation"),
        "contradiction_detection": score(rows, "contradiction_detection"),
        "rows": rows,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "multiturn_latest.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def run_session_persistence() -> dict[str, object]:
    case = load("session_persistence")
    agent = ClinicalAgent(RagStore(), use_llm_extraction=False)
    result = run_turns(agent, "persist", case["turns"])
    state = result.summary["clinical_state"]
    passed = state_matches(state, case["expected_state"])
    return row(case["id"], state_persistence=passed, correct_information_merge=passed)


def run_redundant_question() -> dict[str, object]:
    case = load("redundant_question")
    agent = ClinicalAgent(RagStore(), use_llm_extraction=False)
    result = run_turns(agent, "redundant", case["turns"])
    passed = all(question not in result.response for question in case["forbidden_questions"])
    return row(case["id"], no_redundant_question=passed)


def run_contextual_answer() -> dict[str, object]:
    case = load("contextual_answer")
    agent = ClinicalAgent(RagStore(), use_llm_extraction=False)
    session = agent.session_store.get_or_create("contextual")
    session.last_question = case["last_question"]
    agent.session_store.update("contextual", session)
    result = run_turns(agent, "contextual", case["turns"])
    passed = state_matches(result.summary["clinical_state"], case["expected_state"])
    return row(case["id"], context_interpretation=passed)


def run_session_isolation() -> dict[str, object]:
    case = load("session_isolation")
    agent = ClinicalAgent(RagStore(), use_llm_extraction=False)
    run_turns(agent, "A", case["session_a_turns"][:1])
    b = run_turns(agent, "B", case["session_b_turns"])
    a = run_turns(agent, "A", case["session_a_turns"][1:])
    b_state = b.summary["clinical_state"]
    forbidden = case["session_b_forbidden_state"]
    passed = not state_contains(b_state, forbidden) and "dolor" in a.summary["clinical_state"]["symptoms"]
    return row(case["id"], session_isolation=passed)


def run_contradiction() -> dict[str, object]:
    case = load("contradiction")
    agent = ClinicalAgent(RagStore(), use_llm_extraction=False)
    result = run_turns(agent, "contradiction", case["turns"])
    passed = case["expected_contradiction"] in result.summary["clinical_state"]["contradictions"]
    return row(case["id"], contradiction_detection=passed)


def load(name: str) -> dict[str, object]:
    return json.loads((ROOT / f"{name}.json").read_text(encoding="utf-8"))


def run_turns(agent: ClinicalAgent, session_id: str, turns: list[str]):
    result = None
    for turn in turns:
        result = agent.answer(session_id, turn)
    return result


def state_matches(state: dict[str, object], expected: dict[str, object]) -> bool:
    return all(state.get(key) == value for key, value in expected.items())


def state_contains(state: dict[str, object], forbidden: dict[str, list[str]]) -> bool:
    return any(value in state.get(key, []) for key, values in forbidden.items() for value in values)


def row(case_id: str, **checks: bool) -> dict[str, object]:
    return {"id": case_id, **checks}


def score(rows: list[dict[str, object]], key: str) -> float:
    relevant = [row[key] for row in rows if key in row]
    return sum(bool(value) for value in relevant) / len(relevant) if relevant else 1.0


if __name__ == "__main__":
    main()
