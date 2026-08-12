#!/usr/bin/env bash
# Paso 3/3 del arranque (ver README > "Ejecutar el Agente Clínico").
# Usa el intérprete del venv directamente: no hace falta "source
# .venv/bin/activate" como paso aparte.
set -euo pipefail

if [ ! -x ".venv/bin/uvicorn" ]; then
  echo "No encuentro .venv/bin/uvicorn. ¿Corriste ./setup.sh primero?"
  exit 1
fi

exec .venv/bin/uvicorn clinical_agent.main:app --reload --port 8000
