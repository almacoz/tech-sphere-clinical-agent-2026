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

## RAG

El RAG actual usa un vector store local con embeddings locales y ChromaDB persistente.

- LLM = Ollama / Llama 3.2
- Embeddings = modelo local de embeddings (por defecto `nomic-embed-text` a través de Ollama)
- Vector DB = ChromaDB persistente
- Chunking = `chunk_size = 120`, `overlap = 24`
- Retrieval = top-k semantic similarity con trazabilidad de `document_id`, `filename`, `page` y `chunk_id`
- Evidence = `EvidenceItem` con `retrieval_score` y `relevance`

La colección vectorial se persiste bajo `CHROMA_PERSIST_DIR` (por defecto `./data/chroma`), con un modelo de embedding configurado por `EMBEDDING_MODEL`.

```sh
cp .env.example .env
# ajustar CHROMA_PERSIST_DIR y EMBEDDING_MODEL si hace falta
```

Las rutas físicas se guardan en el vector store como metadata y el agente solo consume la forma pública `RagStore`.

## Modelos locales

LLM:
- `llama3.2`

Embeddings:
- `nomic-embed-text`

Vector DB:
- ChromaDB persistente

La aplicación detecta automáticamente si `Ollama` está disponible y si los modelos
locales exigidos para el agente están instalados. Si falta alguno, la interfaz puede
solicitar su preparación con el Runtime Manager sin tocar el pipeline clínico.

Ollama debe estar instalado previamente en el sistema. La instalación de Ollama es
responsabilidad del setup del proyecto.

## Knowledge ingestion

Supported knowledge formats:
- PDF
- TXT

El endpoint `POST /knowledge/upload` acepta archivos con la extensión `.pdf` o `.txt`.
PDF se lee localmente con `pypdf` y se extrae texto por página usando el separador `\f`
para conservar la trazabilidad de `page` y `chunk_id` en el RAG actual. TXT se decodifica
como UTF-8 y se pasa al mismo contrato `RagStore.upsert_document(filename, text)`.

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
