"""Evalúa la lógica de decisión y escalamiento del agente contra el corpus
sintético con ground truth de `dataset/dataset_final.xlsx` (160 casos × 2
capas = 320 combinaciones, ver `dataset/README.md`).

Reproduce cada caso turno a turno (solo los mensajes de `paciente`/`tercero`;
los turnos de `agente` en el Excel son del generador sintético y no se
reinyectan — es tu agente real el que conduce la conversación) y compara el
`risk_level` final contra `label_ground_truth`.

Uso:
    python scripts/eval_ground_truth.py                     # los 320 casos, fallback determinista (rápido, sin Ollama)
    python scripts/eval_ground_truth.py --capa capa1_limpia # solo la capa sin ruido
    python scripts/eval_ground_truth.py --sample 40         # muestra aleatoria (para iterar rápido)
    python scripts/eval_ground_truth.py --use-llm           # usa el LLM real (Ollama/Llama 3.2) en vez del fallback determinista

No requiere el servidor FastAPI corriendo: usa `ClinicalAgent` directamente,
igual que `evals/run_evals.py` y `evals/run_multiturn_evals.py`.

Dependencias extra (no están en requirements.txt porque no las necesita la
app, solo este análisis): `uv pip install pandas openpyxl`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from clinical_agent.agent import ClinicalAgent
from clinical_agent.rag import RagStore

DATASET_PATH = PROJECT_ROOT / "dataset" / "dataset_final.xlsx"
RESULTS_PATH = PROJECT_ROOT / "evals" / "results" / "ground_truth_latest.json"

LABEL_MAP = {"verde": "GREEN", "amarillo": "YELLOW", "rojo": "RED"}
# Orden de severidad para distinguir sub-triage (el agente clasifica MENOS
# grave que el ground truth -- el falso negativo peligroso, penalizado más
# duro por la rúbrica, §1 "asimetría clínica") de sobre-triage (MÁS grave).
SEVERITY = {"GREEN": 0, "YELLOW": 1, "RED": 2, "UNKNOWN": 1}


def seed_rag(store: RagStore) -> None:
    # Mismo conocimiento base que evals/run_evals.py: este script mide
    # decisión de riesgo, no precisión de RAG, así que basta con las señales
    # de alarma estándar para que evaluate()/validate_safety() tengan contexto.
    store.upsert_document(
        "postoperatorio.txt",
        "Después de cirugía, fiebre alta, sangrado abundante, dolor de pecho, "
        "dificultad para respirar y confusión son señales de alarma.",
    )


def load_cases(capa: str | None, sample: int | None, seed: int) -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise SystemExit(
            f"No encuentro {DATASET_PATH}.\n"
            "Copia dataset_final.xlsx a dataset/ (ver dataset/README.md)."
        )
    df = pd.read_excel(DATASET_PATH)
    df = df[df["hablante"].isin(["paciente", "tercero"])].copy()
    if capa:
        df = df[df["capa"] == capa]
    case_keys = df[["caso_id", "capa"]].drop_duplicates()
    if sample and sample < len(case_keys):
        case_keys = case_keys.sample(n=sample, random_state=seed)
    return df.merge(case_keys, on=["caso_id", "capa"])


def run_case(agent: ClinicalAgent, caso_id: str, capa: str, turns: pd.DataFrame) -> dict:
    session_id = f"{caso_id}::{capa}"
    result = None
    for _, row in turns.sort_values("turno_idx").iterrows():
        result = agent.answer(session_id, str(row["texto"]))
    if result is None:
        raise ValueError(f"Caso sin turnos de paciente/tercero: {caso_id}::{capa}")
    expected = LABEL_MAP[str(turns["label_ground_truth"].iloc[0])]
    predicted = result.decision.risk_level.value
    return {
        "caso_id": caso_id,
        "capa": capa,
        "estilo_paciente": turns["estilo_paciente"].iloc[0],
        "expected": expected,
        "predicted": predicted,
        "match": predicted == expected,
        "sub_triage": SEVERITY[predicted] < SEVERITY[expected],
        "over_triage": SEVERITY[predicted] > SEVERITY[expected],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--capa", choices=["capa1_limpia", "capa2_ruidosa"], default=None)
    parser.add_argument("--sample", type=int, default=None, help="Muestra N casos al azar en vez de los 320")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Usa el LLM real (Ollama) en vez del fallback determinista. Requiere Ollama corriendo.",
    )
    args = parser.parse_args()

    df = load_cases(args.capa, args.sample, args.seed)
    case_keys = df[["caso_id", "capa"]].drop_duplicates().values.tolist()
    if not case_keys:
        raise SystemExit("No hay casos que coincidan con los filtros dados.")

    rows = []
    for i, (caso_id, capa) in enumerate(case_keys, start=1):
        store = RagStore()
        seed_rag(store)
        agent = ClinicalAgent(store, use_llm_extraction=args.use_llm)
        turns = df[(df["caso_id"] == caso_id) & (df["capa"] == capa)]
        rows.append(run_case(agent, caso_id, capa, turns))
        if i % 20 == 0 or i == len(case_keys):
            print(f"  {i}/{len(case_keys)} casos procesados...", file=sys.stderr)

    total = len(rows)
    matches = sum(r["match"] for r in rows)
    red_cases = [r for r in rows if r["expected"] == "RED"]
    false_negatives = [r for r in red_cases if not r["match"]]

    by_capa: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "matches": 0})
    by_estilo: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "matches": 0})
    for r in rows:
        by_capa[r["capa"]]["total"] += 1
        by_capa[r["capa"]]["matches"] += int(r["match"])
        by_estilo[r["estilo_paciente"]]["total"] += 1
        by_estilo[r["estilo_paciente"]]["matches"] += int(r["match"])

    metrics = {
        "total_cases": total,
        "accuracy": matches / total if total else None,
        "red_recall": (sum(r["match"] for r in red_cases) / len(red_cases)) if red_cases else None,
        "false_negative_count": len(false_negatives),
        "false_negative_rate": (len(false_negatives) / len(red_cases)) if red_cases else None,
        "sub_triage_count": sum(r["sub_triage"] for r in rows),
        "over_triage_count": sum(r["over_triage"] for r in rows),
        "accuracy_by_capa": {k: v["matches"] / v["total"] for k, v in by_capa.items()},
        "accuracy_by_estilo_paciente": {k: v["matches"] / v["total"] for k, v in by_estilo.items()},
        "false_negatives": [
            {
                "caso_id": r["caso_id"],
                "capa": r["capa"],
                "estilo_paciente": r["estilo_paciente"],
                "predicted": r["predicted"],
            }
            for r in false_negatives
        ],
        "rows": rows,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    summary = {k: v for k, v in metrics.items() if k not in ("rows", "false_negatives")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n{len(false_negatives)} falsos negativos (ROJO real, agente no dijo RED) -- ver {RESULTS_PATH}", file=sys.stderr)
    print(f"Reporte completo (incluye detalle por caso): {RESULTS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
