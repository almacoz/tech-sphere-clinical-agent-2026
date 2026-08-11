#!/usr/bin/env bash
set -euo pipefail

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama no está instalado. Instala Ollama y vuelve a abrir la aplicación."
  exit 1
fi

ollama pull llama3.2
ollama pull nomic-embed-text
