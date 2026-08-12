# dataset/

Corpus sintético del reto (Tech Sphere Challenge 2026) para construir y evaluar
escenarios de seguimiento postoperatorio. **No contiene datos reales de
pacientes** — nombres, direcciones, cédulas y EPS son sintéticos
(`perfiles_pacientes_co.xlsx` declara explícitamente `source_country: US` →
`adapted_country: CO`, es decir, identidades generadas y adaptadas a formato
colombiano, no personas reales).

## Los 4 archivos y cómo se relacionan

Los tres primeros describen 40 pacientes sintéticos (`paciente_id` es la clave
que los une); el cuarto es el corpus de diálogos generado a partir de esos 40
pacientes.

| Archivo | Filas | Qué contiene | Clave |
|---|---:|---|---|
| `perfiles_pacientes_co.xlsx` | 40 | Identidad sintética del paciente: nombre, dirección, ciudad, EPS, documento. | `paciente_id` |
| `perfiles_clinicos_pacientes_silver_contest.xlsx` | 40 | Ficha clínica: procedimiento quirúrgico, fecha de cirugía, edad, género, comorbilidades, si hubo complicación registrada. 5 procedimientos × 8 pacientes (apendicectomía, colecistectomía, colectomía, reemplazo de cadera/rodilla, mastectomía). | `paciente_id` |
| `trayectorias_postop_silver.xlsx` | 160 | Estado clínico por paciente en 4 días de seguimiento (1, 3, 7, 14): dolor (NRS 0-10), fiebre (°C), movilidad, estado de la herida, apetito, sueño. Clasificado por `arquetipo_trayectoria`: `recuperacion_normal` (76), `complicacion_leve_vigilancia` (60), `complicacion_real` (24). | `paciente_id` + `dia_postop` → `trayectoria_id` |
| `dataset_final.xlsx` | 3991 | **El corpus de evaluación.** Diálogos agente↔paciente turno a turno, generados a partir de cada trayectoria, con **`label_ground_truth`** (verde/amarillo/rojo) por caso. | `caso_id` (deriva de `trayectoria_id`) |

## `dataset_final.xlsx` en detalle

Es el archivo que importa para medir el criterio *Lógica de decisión y
escalamiento* (20 pts) contra un ground truth real, no solo contra los 5-6
casos hechos a mano que ya había en `evals/*.json`.

- **160 casos** (uno por cada fila de `trayectorias_postop_silver.xlsx`), cada
  uno en **2 capas**: `capa1_limpia` (diálogo sin ruido) y `capa2_ruidosa`
  (con interrupciones, muletillas, contradicciones, `[inaudible]` — simula
  transcripción STT imperfecta). 320 combinaciones caso×capa en total.
- **5 estilos de paciente**: `minimizador_sintomas`, `confundido`,
  `colaborativo`, `evasivo`, `ansioso` — el mismo cuadro clínico contado de
  formas muy distintas. El caso más instructivo del set es un
  `complicacion_real` (ground truth `rojo`) narrado por un paciente
  minimizador que insiste en que "no es nada grave" turno tras turno: es
  exactamente el escenario que la rúbrica penaliza más si tu agente le cree
  al paciente en vez de a los datos clínicos subyacentes (asimetría clínica,
  §1 de la rúbrica).
- Turnos con `hablante: "tercero"` (151 de 3991): un cuidador o familiar
  interviene en la conversación — vale la pena confirmar que tu agente no se
  confunde sobre quién está reportando síntomas de quién.
- **`label_ground_truth` es constante dentro de cada `caso_id`+`capa`** (no
  cambia turno a turno): es la clasificación de riesgo esperada para el caso
  completo, no por mensaje individual. Distribución: 3067 `verde`, 623
  `amarillo`, 301 `rojo` (a nivel de turno; a nivel de caso×capa son 320
  casos con esa misma proporción aproximada).
- `modelo_paciente`/`modelo_agente` documentan qué modelo generó cada lado
  del diálogo sintético (mayormente `claude-sonnet-5`, un resto con un
  modelo Nemotron) — es metadata de generación, no algo que tu agente deba
  imitar ni reproducir en su propia arquitectura.

## Cómo se usa

`scripts/eval_ground_truth.py` (en la raíz del repo) carga
`dataset/dataset_final.xlsx`, reproduce cada caso×capa turno a turno contra
`ClinicalAgent` (solo los turnos de `paciente`/`tercero`; los turnos de
`agente` en el Excel son del generador sintético, no se reinyectan — es tu
agente real el que debe conducir la conversación), y compara el
`risk_level` final contra `label_ground_truth`. Ver la sección "Métricas
obligatorias" del README principal para cómo se reporta esto.

## Lo que este dataset NO es

No es material para el RAG (`data/sample_docs/`). RAG debe alimentarse de
protocolos/guías clínicas, no de datos de pacientes — mezclar ambos sería
tanto un problema de diseño (el agente citando "casos anteriores" como si
fueran evidencia clínica) como de privacidad (aunque sea sintético, el
patrón no debe imitarse con datos reales).
