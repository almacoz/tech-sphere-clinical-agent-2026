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

Pruebas:

```sh
cd tech-sphere-clinical-agent-2026
pytest
```

Licencia: MIT. Ver [LICENSE](./LICENSE).
