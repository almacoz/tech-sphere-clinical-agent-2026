# Tech Sphere Clinical Agent 2026

Agente postoperatorio educativo en español para el Tech Sphere Challenge 2026.
Incluye un servicio FastAPI con RAG sobre ChromaDB, extracción clínica asistida
por LLM con overrides deterministas, decisión de riesgo separada de la
generación de lenguaje, validador de seguridad, trazabilidad de fuentes y
métricas por turno.

> Proyecto simulado y educativo. No debe utilizarse para atención médica real.

## ⚠️ Estado actual frente a la rúbrica (léelo antes de evaluar)

| Compuerta | Estado | Detalle |
|---|---|---|
| G2 — Levantable ≤15 min | ✅ **3 pasos, `./setup.sh` + `./start.sh`** | `setup.sh` usa `uv` (venv + instalación de dependencias mucho más rápida que `pip` en frío) y es idempotente: pulls de Ollama y precalentado de Kokoro se saltan si ya están hechos. La primera vez en una máquina limpia el tiempo real depende de tu conexión (descarga de pesos de Ollama + Kokoro/torch, ver sección "Ejecutar el Agente Clínico"); correr `./setup.sh` una vez antes de la demo deja todo cacheado para una segunda ejecución de segundos. No cronometrado formalmente todavía — pendiente antes de la entrega. |
| G3 — Modelo permitido | ✅ | Llama 3.2 (Meta) vía Ollama, local. Kokoro (TTS) no es el LLM que razona y no está sujeto a G3 — `stack-tecnico.md` deja voz como libre elección. |
| G4 — Voz en tiempo real | ⚠️ **Parcial** | Ver "Voz" abajo: la respuesta hablada (TTS, Kokoro-82M) es real y local. La captura de la voz del paciente (STT) usa la Web Speech API del navegador — funciona hoy en Chrome, pero **no es local** (depende de internet y de un servicio de Google fuera de tu control). Antes de la entrega, decide si eso es aceptable para tu demo o si prefieres reemplazarlo por Whisper local/Groq. |
| G5 — Conocimiento vivo | ✅ | Subir/eliminar documento vía consola cambia lo que el agente recupera (`RagStore.upsert_document` / `delete_document`) |

**Este README documenta el sistema tal como está implementado, no como se
aspira a que quede.** Antes de dar por cerrado G4, prueba el loop completo
en Chrome: clic en "🎤 Hablar", di algo, confirma que el texto se transcribe,
marca "Responder con voz" y confirma que el agente contesta en audio.

## Arquitectura

```
Paciente (texto, por ahora)
   │  POST /agent/respond {session_id, message}
   ▼
ClinicalAgent.answer()
   │
   ├─ 1. extract_clinical()          → LLM (Llama 3.2 / Ollama) + overrides deterministas de seguridad
   ├─ 2. merge_clinical_state()      → fusiona con el estado previo de la sesión (en memoria)
   ├─ 3. rag_store.query()           → ChromaDB + embeddings Ollama (nomic-embed-text)
   ├─ 4. evaluate()                  → EvidenceEvaluation (riesgo, evidencia, missing_information)
   ├─ 5. decide()                    → Decision (risk_level, needs_human, reason_codes) — determinista, no LLM
   ├─ 6. generate_response()         → texto para el paciente, basado en plantillas + evidencia citada
   └─ 7. validate_safety()           → validador de seguridad basado en reglas (no LLM), puede bloquear la respuesta
   ▼
AgentResponse {response, decision, evidence, safety_validation, summary, metrics}
```

Puntos de diseño relevantes:

- **Solo una llamada al LLM por turno** (la extracción clínica). La
  clasificación de riesgo, la generación de la respuesta y la validación de
  seguridad son deterministas — no dependen de que el LLM "decida" nada
  clínicamente. Esto reduce latencia, costo y superficie de alucinación.
- **Fallback determinista.** Si Ollama no responde o devuelve JSON inválido,
  `extract_clinical` cae a `contextual_deterministic_extract_clinical` (regex
  sobre el mensaje) en vez de fallar la conversación.
- **Overrides de seguridad deterministas.** Independiente de lo que devuelva
  el LLM, `apply_deterministic_safety_overrides` vuelve a correr los patrones
  de `ALARM_PATTERNS` sobre el mensaje crudo y los fusiona con `alarm_signals`.
  Esto es intencional (cinturón de seguridad contra que el LLM omita una
  señal de alarma), pero también fue la causa de un bug corregido el
  2026-08-12: el patrón de `fiebre_alta` disparaba con solo mencionar la
  palabra "fiebre", incluso en preguntas informativas ("¿qué información
  tienes sobre la fiebre?"). Se corrigió exigiendo un calificador
  (`fiebre alta/elevada/...`) y detectando preguntas puras sin afirmación de
  síntoma antes de extraer. Ver `clinical_agent/agent.py::_is_pure_information_query`.

## Ejecutar el Agente Clínico

**Requisito previo (no cuenta como paso, es infraestructura — igual que
tener Python instalado):** [Ollama](https://ollama.com) instalado y
corriendo. Si falta, `./setup.sh` se detiene con un mensaje claro en vez de
fallar a medias.

**3 pasos:**

```sh
# 1) Entra al proyecto
cd tech-sphere-clinical-agent-2026

# 2) Prepara todo: crea el venv con uv, instala dependencias, descarga los
#    modelos de Ollama solo si faltan, instala espeak-ng si falta, y
#    precalienta Kokoro-82M (así la voz queda lista desde el primer request,
#    no falla en silencio la primera vez que alguien la pide)
./setup.sh

# 3) Levanta el servidor
./start.sh
```

Abre `http://localhost:8000` para la consola (chat + administración de
conocimiento).

**Sobre el tiempo real (honestidad ante todo, sin inflar números):**
`./setup.sh` es idempotente — los pulls de Ollama y el precalentado de
Kokoro se saltan si ya están hechos. Eso significa que:

- **Primera vez en una máquina limpia:** el tiempo depende de tu conexión a
  internet. Se descargan una sola vez los pesos de Ollama (`llama3.2` ≈2 GB,
  `nomic-embed-text` ≈270 MB) y las dependencias de voz (Kokoro ≈80 MB +
  `torch`, unos cientos de MB). Con banda ancha normal esto entra cómodo
  dentro de los 15 minutos de G2.
- **Ejecuciones posteriores** (por ejemplo, correr `./setup.sh` una vez
  antes de la demo para dejar todo cacheado, y volver a correrlo justo antes
  de la evaluación): segundos, porque `uv` reutiliza su cache de paquetes y
  ni Ollama ni Kokoro vuelven a descargar nada.

Para acercarte de verdad a los 5 minutos frente al jurado, la recomendación
es correr `./setup.sh` con anticipación (deja todo cacheado) y que la
ejecución cronometrada sea la segunda. No hemos fabricado un número de
"instalación en frío" porque ese tiempo está fuera de nuestro control (ancho
de banda del evaluador) — ver "Métricas obligatorias" más abajo para la
misma política aplicada a latencia/costo.

## LLM

**Modelo: Llama 3.2 3B, vía Ollama, inferencia 100% local.**

Por qué se eligió esta familia (documentar también en el informe final con el
detalle específico de tu caso):
- Cumple la compuerta G3 (familia permitida: Meta Llama, serie 3.x, local).
- Costo $0 — no depende de cuota ni de conectividad durante la evaluación en
  vivo, lo que reduce el riesgo de que la sesión evaluada falle por límites
  de un nivel gratuito de nube.
- Corre en el rango de hardware descrito en `stack-tecnico.md` (8–16 GB RAM,
  CPU), suficiente porque el agente le pide al LLM una única tarea acotada
  (extracción de JSON estructurado), no razonamiento clínico abierto.

La extracción clínica usa Ollama por defecto y valida la salida con Pydantic
(`ClinicalExtraction.model_validate`). Si Ollama no responde o devuelve JSON
inválido, el agente vuelve a la extracción determinista. Para desactivar
Ollama durante pruebas locales:

```sh
CLINICAL_AGENT_USE_LLM=0 uvicorn clinical_agent.main:app --reload --port 8000
```

Para ver en consola el prompt, la respuesta cruda del LLM y cada paso de la
extracción:

```sh
CLINICAL_AGENT_DEBUG_LLM=1 uvicorn clinical_agent.main:app --reload --port 8000
```

## RAG

Vector store local persistente con embeddings locales.

- LLM = Ollama / Llama 3.2 (solo para extracción, no para retrieval)
- Embeddings = `nomic-embed-text` vía Ollama
- Vector DB = ChromaDB persistente (`chromadb.PersistentClient`, similitud coseno)
- Chunking = `chunk_size = 120` tokens, `overlap = 24`, respeta separación de
  página (`\f`) para trazabilidad
- Retrieval = top-k (`top_k=4` por defecto) con `document_id`, `filename`,
  `page`, `chunk_id`, `quote_or_excerpt`, `retrieval_score` y `relevance` en
  cada `EvidenceItem`
- Cada consulta se registra en `RagStore.query_log` con latencia de
  recuperación, útil para las métricas de §5 de la rúbrica

La colección vectorial se persiste bajo `CHROMA_PERSIST_DIR` (por defecto
`./data/chroma`), con el modelo de embedding configurado por
`EMBEDDING_MODEL`.

```sh
cp .env.example .env
# ajustar CHROMA_PERSIST_DIR y EMBEDDING_MODEL si hace falta
```

### Conocimiento vivo (G5)

`POST /knowledge/upload` (alias de `POST /documents`) indexa un PDF o TXT;
`DELETE /knowledge/{document_id}` lo elimina del índice vectorial y de las
respuestas futuras. `tests/test_agent.py::test_document_lifecycle_removes_deleted_document_from_retrieval`
cubre exactamente este ciclo (subir → responder citando el documento →
eliminar → responder sin evidencia).

## Knowledge ingestion

Formatos soportados: PDF y TXT. PDF se lee localmente con `pypdf`, extrayendo
texto por página (separador `\f`) para conservar `page`/`chunk_id`. TXT se
decodifica como UTF-8. Ambos pasan por `RagStore.upsert_document(filename, text)`.

## Voz

### TTS — Kokoro-82M (respuesta hablada)

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0, 82M
parámetros, corre en CPU, sin token de HuggingFace) sintetiza la respuesta
del agente a WAV 24kHz, 100% local.

- Instalación: automática al correr `./setup.sh` (paso 2 de "Ejecutar el
  Agente Clínico") — instala `kokoro`/`soundfile` vía `uv`, instala
  `espeak-ng` (fallback de fonemización) vía Homebrew/apt, y precalienta el
  pipeline para descargar/cachear los pesos del modelo antes de que arranque
  el servidor. Ver también `stack-tecnico.md`.
- Voz por defecto: `ef_dora` (español, femenina). También disponibles
  `em_alex` y `em_santa` (español, masculinas) — ver
  [VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)
  del modelo para la lista completa de idiomas/voces.
- `GET /tts/status` — reporta si Kokoro/soundfile están instalados y listos.
- `POST /tts {"text": "...", "voice": "ef_dora"}` — sintetiza cualquier texto
  suelto, devuelve `audio/wav` crudo.
- `POST /agent/respond?voice=true` — el mismo endpoint clínico de siempre,
  pero además sintetiza `response` con Kokoro, la agrega en
  `audio_base64` (WAV en base64) y **suma la latencia de síntesis a
  `metrics.total_latency_ms` / `metrics.tts_latency_ms`** — así el P50/P95
  reportado en §5 refleja la definición real de la rúbrica ("desde que el
  paciente termina de hablar hasta que empieza a sonar el audio del
  agente"), no solo el razonamiento de texto.
- Si Kokoro no está instalado, `voice=true` **no rompe la respuesta**: el
  turno sigue devolviendo texto normal, con `metrics.tts_error` explicando
  por qué no hay audio. La voz es una capa opcional sobre el mismo pipeline
  clínico, nunca un requisito para que `/agent/respond` funcione.

### STT — captura de voz del paciente (Web Speech API del navegador)

El botón "🎤 Hablar" de la consola usa `webkitSpeechRecognition` /
`SpeechRecognition` del navegador (`lang="es-ES"`) para transcribir a texto
y rellenar el mensaje del paciente. **Esto es un atajo pragmático, no una
pieza del pipeline local**: funciona en Chrome, requiere conexión a
internet, y el audio se procesa fuera de este repositorio (servicio de
reconocimiento del navegador). Es la forma más simple de tener hoy un loop
de voz de punta a punta para pasar la verificación en vivo de G4 ("el
jurado habla y el agente responde con voz"), pero si tu demo debe correr
sin depender de un servicio externo de Google, o si prefieres declarar un
stack de STT 100% consistente con el resto (todo local u todo declarado),
reemplázalo por Whisper local (`faster-whisper`) o Whisper vía Groq — ver
`stack-tecnico.md`. Decláralo explícitamente en el informe final junto con
el LLM, tal como pide G3 para el modelo de razonamiento.

### Loop completo para probar G4

1. Abre `http://localhost:8000`.
2. Clic en "🎤 Hablar", di algo en español, confirma que el texto aparece en
   "Mensaje del paciente".
3. Marca "Responder con voz (Kokoro TTS, español)".
4. Clic en "Enviar" — el agente debe responder en texto **y** reproducir
   audio automáticamente (o con el botón de play si el navegador bloquea
   autoplay).

## Lógica de decisión y escalamiento

`risk_level` se calcula de forma determinista en `ClinicalAgent.evaluate()`,
en este orden de prioridad:

1. Contradicción detectada entre turnos → `UNKNOWN`, pide aclaración.
2. `alarm_signals` no vacío (sangrado abundante, fiebre alta, dificultad
   respiratoria, dolor de pecho, confusión) → `RED`, `needs_human=True`.
3. Inyección de prompt detectada → `UNKNOWN`, no ejecuta la instrucción del
   paciente.
4. Trayectoria "empeorando" → `YELLOW`.
5. Información faltante para caracterizar el síntoma → `UNKNOWN`, pregunta lo
   que falta (prioriza `evolución` y `ubicación` sobre `intensidad`/`duración`).
6. Si no hay nada de lo anterior → `GREEN`.

El resumen por llamada (`AgentResponse.summary`) incluye síntomas, riesgo,
escalamiento, `evidence_ids`, `missing_information` y la respuesta enviada —
es la base del "resumen al terminar la llamada" que pide el criterio de 20
pts de Lógica de decisión y escalamiento.

## Guardrails, handrails y ventana de auditoría

El pipeline distingue explícitamente dos tipos de control, y deja trazabilidad
de cada uno para poder auditar **en qué momento y por qué** se tomó una
decisión — incluida la del LLM:

- **Guardrails (duros, no bypasseables).** `ALARM_PATTERNS` e
  `INJECTION_PATTERNS` corren como regex sobre el mensaje crudo del paciente
  en **todos** los turnos, sin importar qué haya devuelto el LLM
  (`_guardrail_pattern_steps`). El resultado se fusiona con la extracción del
  LLM en `apply_deterministic_safety_overrides`, que nunca *quita* una señal
  que el LLM haya perdido — solo puede *agregar* lo que el guardrail
  detectó. La rama de riesgo en `evaluate()` también es un guardrail: es
  código determinista (if/elif), no una decisión del LLM.
- **Handrails (blandos, orientan sin bloquear).** La extracción vía LLM
  (Llama 3.2) es la única guía "suave": interpreta lenguaje libre para
  poblar `ClinicalExtraction`, pero nunca decide `risk_level` por sí sola —
  eso queda siempre en manos de la lógica determinista de `evaluate()`.

### Ventana de auditoría

Cada turno genera una traza (`DecisionAudit`) con un paso (`AuditStep`) por
cada guardrail evaluado (se haya activado o no), el paso de extracción del
LLM (con el `raw_response` recortado incluido), el override determinista
(qué agregó el guardrail sobre lo que dijo el LLM, si algo), la rama exacta
de `evaluate()` que fijó el `risk_level` y cada verificación de
`validate_safety`. Esto permite responder, para cualquier respuesta del
agente: *¿qué regla se activó, sobre qué texto, y qué rama de código decidió
el resultado final?* — sin depender de leer logs de consola.

- La API la devuelve en `AgentResponse.audit` en cada respuesta de
  `/agent/respond`.
- Se acumula por sesión (últimos 20 turnos) y se puede consultar completa en
  `GET /audit/{session_id}`.
- La consola (`/`) tiene una sección "Ventana de auditoría" que pinta la
  traza del turno actual (guardrails activados en rojo, el paso del LLM en
  azul) y puede cargar el histórico completo de la sesión.

### Caso real encontrado con la ventana de auditoría

Usando `GET /audit/{session_id}` en una sesión de prueba se detectó que, en
una conversación de varios turnos, el agente preguntaba **la misma** falta
de información (`ubicación`, `evolución`) turno tras turno sin avanzar,
aunque el paciente sí iba respondiendo. La traza mostró la causa exacta: el
LLM (Llama 3.2 3B) devolvía JSON válido pero dejaba `locations`/`trajectory`
vacíos en vez de poblarlos a partir de la pregunta anterior (por ejemplo,
respondía "cabeza" creando el síntoma `"dolor de cabeza"` en vez de fijar
`locations: ["cabeza"]`). El guardrail que sí rellena esos huecos con
contexto (`_fill_contextual_gaps`) existía, pero **solo se ejecutaba en el
camino de fallback** (cuando el LLM fallaba por completo) — nunca cuando el
LLM "tenía éxito" pero dejaba campos de contexto vacíos, que es justo lo
que hacía. Fix: `apply_contextual_backstop` corre ahora también en el
camino de éxito del LLM, y `KNOWN_BODY_LOCATIONS` se amplió (le faltaban
`cabeza`, `espalda`, `cuello`, `brazo`). Sin este guardrail, una trayectoria
de empeoramiento reportada por el paciente podía perderse silenciosamente
y nunca escalar a `YELLOW` — el tipo de falso negativo que la rúbrica
penaliza con más peso que un falso positivo.

## Estado conversacional

`SessionState` vive en memoria del proceso (`SessionStore`, diccionario
`session_id -> SessionState`), no hay singleton mutable compartido entre
sesiones distintas: cada `session_id` nuevo arranca con estado vacío. Para
producción este estado debería moverse a almacenamiento persistente
(Redis/Postgres), ya que se pierde al reiniciar el proceso.

## Métricas obligatorias (rúbrica §5)

Estas métricas son requisito, no opcionales. Se miden corriendo el servidor
real contra `scripts/measure_metrics.py` — no están fabricadas a mano:

```sh
# con el servidor corriendo en otra terminal
pip install httpx  # ya está en requirements.txt
python scripts/measure_metrics.py --base-url http://localhost:8000 --runs 20

# Con --voice, la latencia incluye la síntesis de audio con Kokoro (la
# definición exacta que pide la rúbrica: "hasta que empieza a sonar el
# audio del agente"). Requiere Kokoro instalado (ver sección "Voz").
python scripts/measure_metrics.py --base-url http://localhost:8000 --runs 20 --voice
```

El script pega contra `/agent/respond` con mensajes variados (síntoma simple,
pregunta informativa, señal de alarma, follow-up, intento de inyección),
lee `metrics.total_latency_ms` y `metrics.rag_queries` que ya devuelve la
API en cada respuesta, y guarda el reporte en
`evals/results/metrics_report.json`.

| Métrica | Valor | Cómo se midió |
|---|---|---|
| Latencia P50 | `<pendiente: correr script>` | extremo a extremo, fin del mensaje del paciente → respuesta del agente (`total_latency_ms`) |
| Latencia P95 | `<pendiente: correr script>` | idem |
| Tokens de entrada / turno | `<pendiente: correr script>` | aproximación palabras×1.3 (Ollama no expone tokenizer real vía API) |
| Tokens de salida / turno | `<pendiente: correr script>` | idem |
| Invocaciones al modelo / turno | `<pendiente: correr script>` | 1 si la extracción usó el LLM, 0 si cayó al fallback determinista |
| Consultas al RAG / llamada | `<pendiente: correr script>` | 1 consulta por turno × turnos promedio por llamada |
| Costo estimado por llamada | `<pendiente: correr script>` | corre local ($0 real); extrapolado a precio de referencia de un proveedor cloud equivalente — ver `--price-per-1k-input/output` en el script |

**Antes de entregar, corre el script y reemplaza estos placeholders con los
valores reales que te imprima.** La rúbrica es explícita: "reportar números
que no se sostienen es peor que no reportarlos", y se contrastan contra los
logs en la sesión de evaluación.

## Pruebas

```sh
pytest
```

`tests/test_agent.py` corre con `CLINICAL_AGENT_USE_LLM=0` (ruta determinista,
no requiere Ollama activo) usando `TestClient` de FastAPI in-process.

`evals/run_evals.py` corre un set de casos etiquetados (normales, ambiguos,
alarma, fuera de alcance, inyección de prompt, actualización de conocimiento)
y calcula `risk_accuracy`, `red_recall`, `false_negative_rate`,
`unsupported_claim_rate`, `injection_resistance`; el último resultado queda
en `evals/results/latest.json`. `evals/run_multiturn_evals.py` cubre
persistencia/aislamiento de sesión, contradicciones y preguntas redundantes.

```sh
python evals/run_evals.py
python evals/run_multiturn_evals.py
```

### Resultados de evaluación interna

Además de `evals/run_evals.py`, se corrió una evaluación más grande contra un
lote interno de conversaciones clínicas etiquetadas (riesgo real conocido de
antemano). **Por motivos de privacidad y legales, ese lote no se incluye en
este repositorio ni se documenta su origen, esquema o contenido aquí** — solo
se reportan los resultados agregados, igual que con cualquier otro conjunto
de prueba de terceros.

**Corrida más reciente — pipeline 100% determinista (sin LLM ni RAG activos;
es el camino de respaldo cuando Ollama no está disponible, no el modo de
producción):**

| Métrica | Valor |
|---|---:|
| Casos evaluados | 320 |
| Accuracy global | 26% |
| `red_recall` (casos de alarma real correctamente escalados) | 0% |
| De los casos de alarma real: clasificados `UNKNOWN` (pide más info, no tranquiliza) | 18 de 24 |
| De los casos de alarma real: clasificados `GREEN` (falso negativo peligroso) | 5 de 24 |
| De los casos de alarma real: clasificados `YELLOW` (parcial) | 1 de 24 |
| Sobre-triage (agente más cauteloso que el riesgo real) | 167 casos |
| Sub-triage (agente menos cauteloso que el riesgo real) | 40 casos |

**Lectura honesta de este resultado:** el `red_recall` de 0% en modo
determinista **no** significa que el sistema tranquiliza a un paciente en
riesgo real 24 de 24 veces — en 18 de esos 24 casos el pipeline responde
`UNKNOWN` y sigue preguntando (comportamiento seguro, aunque no escale),
y solo en 5 de 24 llega a `GREEN` (el fallo grave: falsa tranquilidad).
Aun así, el número confirma algo importante para la arquitectura: los
guardrails deterministas por sí solos (`ALARM_PATTERNS` sobre texto crudo)
no bastan para interpretar lenguaje clínico natural — dependen de que el LLM
(Llama 3.2) haga la extracción semántica primero. Si Ollama cae durante la
sesión evaluada, el sistema degrada a este modo, y ese es exactamente su
peor caso medido, no una hipótesis.

**Pendiente antes de la entrega:** correr la misma evaluación con el LLM y
el RAG activos (`scripts/eval_ground_truth.py --use-llm`, ver
`evals/results/ground_truth_latest.json`) y reemplazar la tabla de arriba
con esos números — son los que de verdad importan para el criterio de 20 pts
de *Lógica de decisión y escalamiento*, y los que se contrastan contra los
logs en la sesión de evaluación. No se fabrica un número optimista sin
correr el script real.

## Limitaciones conocidas / roadmap

- **STT depende del navegador (Chrome + internet), no es local.** La
  captura de voz del paciente usa la Web Speech API en vez de un modelo
  propio (Whisper local/Groq). Es suficiente para pasar la verificación en
  vivo de G4 hoy, pero es una dependencia externa no declarada en
  `stack-tecnico.md` como parte del pipeline — decláralo así en el informe
  final, o reemplázala antes de la entrega si prefieres un stack 100%
  consistente.
- **Instalación en frío no cronometrada formalmente todavía.** `./setup.sh`
  usa `uv` y es idempotente (ver "Ejecutar el Agente Clínico"), pero nadie
  ha medido con cronómetro una corrida en máquina 100% limpia (sin cache de
  `uv`, sin modelos de Ollama, sin pesos de Kokoro) contra la ventana de 15
  minutos de G2. Hazlo al menos una vez antes de la entrega.
- Estado de sesión en memoria del proceso, no persistente entre reinicios.
- El conteo de tokens reportado es una aproximación por palabras, no el
  tokenizer real de Llama 3.2 (Ollama no lo expone por esta vía).
- `evals/results/latest.json` muestra `safety_passed: false` en algunos casos
  de inyección/fuera-de-alcance — revisar antes de la entrega si eso refleja
  el comportamiento esperado (bloqueo correcto) o un falso positivo del
  validador de seguridad.
- **El camino 100% determinista (sin LLM) tiene `red_recall` bajo en
  lenguaje clínico natural** — ver "Resultados de evaluación interna". Es el
  camino de respaldo cuando Ollama falla, así que en la práctica el sistema
  depende fuertemente de que el LLM esté disponible durante la sesión
  evaluada. Vale la pena decidir antes de la entrega si esto es aceptable o
  si los guardrails deterministas deben ampliarse (más patrones, o
  extracción de valores numéricos como temperatura) para no depender tanto
  del LLM en el peor caso.

Licencia: MIT. Ver [LICENSE](./LICENSE).
