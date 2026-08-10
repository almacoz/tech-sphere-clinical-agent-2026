# Tech Sphere Clinical Agent 2026

Agente postoperatorio educativo en español para el Tech Sphere Challenge 2026.
Incluye un servicio FastAPI mínimo con RAG, generación condicionada por
evidencia, decisión clínica separada, validador de seguridad, trazabilidad de
fuentes y métricas verificables por turno.

> Proyecto simulado y educativo. No debe utilizarse para atención médica real.

## Ejecutar el Agente Clínico

```sh
cd tech-sphere-clinical-agent-2026
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn clinical_agent.main:app --reload --port 8000
```

## LLM

Modelo: Llama 3.2 3B
Runtime: Ollama
Inferencia: local
Credenciales API: ninguna

La extracción clínica usa Ollama por defecto y valida la salida con Pydantic.
Si Ollama no responde o devuelve JSON inválido, el agente vuelve a la extracción
determinista. Para desactivar Ollama durante pruebas locales:

```sh
CLINICAL_AGENT_USE_LLM=0 uvicorn clinical_agent.main:app --reload --port 8000
```

## Estado conversacional

La demo mantiene `ClinicalState` por `session_id` en memoria del proceso. Cada
turno fusiona la extracción nueva con el estado clínico previo para no repetir
preguntas sobre datos ya conocidos. Para producción, este estado debería moverse
a almacenamiento persistente.

Pruebas:

```sh
cd tech-sphere-clinical-agent-2026
pytest
```

Licencia: MIT. Ver [LICENSE](./LICENSE).
