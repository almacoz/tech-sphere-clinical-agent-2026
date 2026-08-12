# Informe Final — Agente Clínico Postoperatorio 2026

## 1. Resumen ejecutivo

Este proyecto es un agente conversacional de seguimiento postoperatorio en español, con
interacción por voz, que acompaña al paciente después de su procedimiento para detectar
señales de alarma y decidir cuándo escalar a un humano.

El agente combina tres piezas separadas por diseño: un modelo de lenguaje (Llama 3.2 3B,
local vía Ollama) que interpreta el lenguaje natural del paciente y lo convierte en
estado clínico estructurado; un motor de recuperación aumentada por conocimiento (RAG,
con ChromaDB) que sustenta cada respuesta clínica en un documento rastreable; y una capa
de reglas deterministas —fuera del LLM— que clasifica el riesgo, decide el escalamiento y
valida la seguridad de cada respuesta antes de enviarla.

La decisión de diseño central es que el LLM interpreta pero no decide: la clasificación
de riesgo y el escalamiento son deterministas, auditables y no dependen del
comportamiento probabilístico del modelo. Esto es documento a fondo en la sección 5.

## 2. Arquitectura

Ver diagrama completo en [`docs/architecture.png`](architecture.png), y su versión en
texto (para lectura sin imágenes) en el README del repositorio, sección "Arquitectura".

En resumen, el flujo por turno es: el mensaje del paciente pasa en paralelo por
extracción clínica (LLM) y por guardrails deterministas sobre el texto original; ambas
salidas se combinan en el estado de sesión; ese estado se cruza contra el conocimiento
recuperado por RAG; con eso se evalúa el riesgo y la información faltante; se toma una
decisión determinista de escalamiento; se genera la respuesta con la evidencia citada; y
esa respuesta pasa por un validador de seguridad determinista antes de convertirse a voz
y devolverse al paciente. Cada turno queda registrado en una traza de auditoría completa.

## 3. Modelo utilizado

- Familia: Meta Llama
- Modelo exacto: Llama 3.2 3B
- Proveedor: Ollama (inferencia local)
- Por qué lo elegí: familia permitida, costo $0, ejecución local sin depender de APIs externas durante evaluación

## 4. Stack técnico completo

- LLM: Llama 3.2 3B (Ollama)
- Embeddings: nomic-embed-text (Ollama)
- Vector DB: ChromaDB
- TTS: Kokoro-82M
- STT: Web Speech API (navegador)
- Backend: FastAPI
- Frontend: HTML/JS

## 5. Decisiones de diseño

**El LLM interpreta; el código decide.** La decisión central del sistema es separar la
interpretación probabilística del lenguaje de las decisiones de seguridad: lenguaje
natural → LLM → estado clínico estructurado → reglas deterministas → riesgo +
escalamiento. Esto permite auditar la decisión sin depender exclusivamente del
razonamiento interno del modelo, y evita que una alucinación del LLM pueda, por sí sola,
bajar el nivel de riesgo o cancelar un escalamiento.

**Fallback determinista.** Si el LLM falla o no está disponible, el sistema no queda
inutilizable: la extracción degrada a una ruta determinista basada en reglas y
expresiones conocidas. Es un mecanismo de continuidad, no un sustituto equivalente de la
comprensión semántica del LLM — el propio README documenta esta diferencia de capacidad
como limitación conocida.

**Conocimiento separado del modelo.** El conocimiento médico vive fuera de los pesos del
modelo, en ChromaDB. Esto permite actualizar documentos, agregar protocolos y eliminar
información sin reentrenar nada, y que cada respuesta clínica sea rastreable hasta el
documento y la página que la sustenta.

## 6. Métricas

*Pendiente: correr `scripts/measure_metrics.py --base-url http://localhost:8000 --runs 20 --voice`
contra el servidor real (con Ollama y Kokoro activos) y pegar aquí la tabla resultante de
`evals/results/metrics_report.json` — latencia P50/P95, tokens de entrada/salida por
turno, invocaciones al modelo por turno, consultas RAG por llamada y costo estimado por
llamada. No se reportan números fabricados: estos deben salir de una corrida real antes
de la entrega.*

## 7. Limitaciones

**STT externo.** La captura de voz del paciente depende actualmente de la Web Speech API
del navegador, por lo que no es una solución STT completamente local.

**Estado de sesión.** El estado conversacional se mantiene en memoria y se pierde al
reiniciar el proceso.

**Conteo de tokens.** El cálculo usado en las métricas de tokens es una aproximación
basada en palabras, no una medición directa del tokenizer de Llama 3.2.

**Dependencia del LLM para interpretación semántica.** El fallback determinista ofrece
continuidad, pero interpreta peor el lenguaje clínico natural que la ruta con LLM,
especialmente cuando el paciente usa lenguaje indirecto o variaciones lingüísticas no
contempladas por las reglas.

**Uso educativo.** El sistema no ha sido diseñado, validado ni certificado para uso
clínico real.

## 8. Pregunta de cierre 1

**Problema que resuelvo:**
Los pacientes postoperatorios reciben altas sin acompañamiento continuo. Las llamadas de
seguimiento humano no escalan y los sistemas automáticos tradicionales carecen de
trazabilidad clínica. El problema real es el falso negativo: un paciente con fiebre que
no es alertado a tiempo puede tener complicaciones graves.

**Por qué mi solución:**
Un agente de voz disponible 24/7 que nunca se cansa, nunca olvida preguntar por red
flags, y escala con criterio clínico basado en evidencia recuperada en tiempo real.

**Valor diferencial:**
1. Conocimiento vivo — el hospital actualiza protocolos y el agente los adopta al
   instante sin reentrenamiento
2. Trazabilidad total — cada respuesta clínica cita el documento y página exactos
3. Decisiones auditables — un jurado clínico puede revisar por qué se escaló o no,
   independiente del LLM
4. Separación de responsabilidades — el LLM interpreta lenguaje, las reglas deciden
   seguridad

## 9. Pregunta de cierre 2

**Decisión técnica más relevante:**
Separar la interpretación probabilística del lenguaje (LLM) de las decisiones de
seguridad y escalamiento (reglas deterministas). El modelo extrae síntomas
estructurados, pero NO decide si escalar.

**Alternativas evaluadas:**
1. **Dejar que el LLM decidiera todo** — descartada porque los falsos negativos son
   catastróficos y los LLMs pueden alucinar o ser inconsistentes
2. **Pipeline 100% reglas sin LLM** — descartada porque no entendería lenguaje natural
   del paciente ni variaciones lingüísticas
3. **Fine-tuning médico del LLM** — descartada porque las familias de modelo permitidas
   están restringidas y el reto premia ingeniería, no presupuesto
4. **Usar modelos más grandes vía API** — descartada para mantener costo $0 y ejecución
   local reproducible

**Riesgos identificados:**
1. El fallback determinista es menos capaz que el LLM → mitigado con logs de cuándo se
   activa
2. Web Speech API depende del navegador → mitigado haciendo la capa TTS/STT opcional
3. nomic-embed-text puede ser menos preciso que BGE-M3 para español médico → mitigado
   con chunking conservador y top-k=4

**Con 2 semanas más:**
1. Evaluaría el agente con el dataset oficial de trayectorias usando un LLM-judge,
   midiendo precisión clínica vs. baseline de solo reglas
2. Implementaría STT local con Whisper para eliminar dependencia del navegador
3. Añadiría persistencia de estado conversacional con Redis
4. Implementaría streaming bidireccional real para permitir interrupciones naturales
5. Migraría de nomic-embed-text a BGE-M3 para mejorar precisión en español médico
