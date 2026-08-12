#!/usr/bin/env bash
# Paso 2/3 del arranque (ver README > "Ejecutar el Agente Clínico").
# Idempotente: se puede volver a correr sin costo (no reinstala ni
# redescarga lo que ya está listo). Esa idempotencia es la que hace que la
# SEGUNDA ejecución (p. ej. justo antes de la demo) sea de segundos en vez
# de minutos.
set -euo pipefail
SECONDS=0

echo "== [1/4] Entorno Python (uv) =="
if ! command -v uv >/dev/null 2>&1; then
  echo "uv no encontrado. Instalando (https://astral.sh/uv, sin compilación, ~5s)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d ".venv" ]; then
  uv venv
fi
# uv pip usa un cache global de wheels: la primera instalación baja los
# paquetes, las siguientes (incluso en otro proyecto) reutilizan el cache.
# Esto es varias veces más rápido que `pip install` en frío.
uv pip install -r requirements.txt --python .venv/bin/python

echo "== [2/4] Modelos de Ollama (LLM + embeddings, solo si faltan) =="
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama no está instalado. Instálalo desde https://ollama.com y vuelve a correr ./setup.sh"
  exit 1
fi

pull_if_missing() {
  local model="$1"
  if ollama list 2>/dev/null | awk '{print $1}' | grep -q "^${model}\(:\|$\)"; then
    echo "  '$model' ya está descargado, omito pull."
  else
    echo "  Descargando '$model' (primera vez; depende de tu conexión)..."
    ollama pull "$model"
  fi
}
pull_if_missing "llama3.2"
pull_if_missing "nomic-embed-text"

echo "== [3/4] espeak-ng (fonemización de español para Kokoro TTS) =="
# No es estrictamente necesario para que arranque el servidor (la voz se
# degrada a "no disponible" sin él, ver GET /tts/status), pero sin esto
# POST /tts devuelve 503.
if command -v espeak-ng >/dev/null 2>&1; then
  echo "  ya está instalado."
elif command -v brew >/dev/null 2>&1; then
  echo "  instalando con Homebrew..."
  brew install espeak-ng
elif command -v apt-get >/dev/null 2>&1; then
  echo "  instalando con apt-get..."
  sudo apt-get install -y espeak-ng
else
  echo "  no encontrado y no hay brew/apt-get disponibles. Instálalo manualmente:"
  echo "  https://github.com/espeak-ng/espeak-ng/releases"
fi

echo "== [4/4] Precalentando Kokoro-82M (descarga y cachea los pesos del modelo) =="
# Esto es lo que evita el problema de "Kokoro no instalado": si el import o
# la carga del modelo fallan, lo vemos AHORA, en setup, con un mensaje claro
# — no en silencio la primera vez que alguien pide voz. Si falla, el
# servidor arranca igual: la respuesta hablada se degrada automáticamente
# (GET /tts/status, ver sección "Voz" del README).
.venv/bin/python - <<'PYEOF'
import sys
try:
    from kokoro import KPipeline
    KPipeline(lang_code="e")
    print("  Kokoro: listo (pesos descargados/cacheados).")
except Exception as exc:  # noqa: BLE001 - queremos capturar cualquier fallo de import/carga
    print(f"  Kokoro: NO disponible ({exc}).")
    print("  El servidor arrancará igual; la voz de salida se degrada automáticamente.")
    print("  Revisa espeak-ng arriba y/o reintenta './setup.sh' con mejor conexión.")
PYEOF

echo
echo "Setup completo en ${SECONDS}s. Siguiente paso: ./start.sh"
