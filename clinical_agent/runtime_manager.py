from __future__ import annotations

import json
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

ALLOWED_MODELS = {
    "llama3.2",
    "nomic-embed-text",
}

DEFAULT_MODEL_NAMES = (
    "llama3.2",
    "nomic-embed-text",
)


class RuntimeManager:
    def __init__(self) -> None:
        self._pull_status = "idle"
        self._pull_model: str | None = None
        self._pull_message = "Sin descarga activa"
        self._pull_error: str | None = None
        self._pull_thread: threading.Thread | None = None
        self._pull_lock = threading.Lock()

    def get_status(self) -> dict[str, Any]:
        ollama_available = self._ollama_available()
        installed = self._list_installed_models() if ollama_available else set()
        models_report = {}
        for model_name in DEFAULT_MODEL_NAMES:
            models_report[model_name] = {
                "installed": self._model_is_installed(model_name, installed),
            }
        ready = bool(ollama_available and all(models_report[name]["installed"] for name in DEFAULT_MODEL_NAMES))
        return {
            "ollama": {
                "available": ollama_available,
            },
            "models": models_report,
            "ready": ready,
        }

    def get_pull_status(self) -> dict[str, Any]:
        if self._pull_status == "idle" and self._pull_thread is None:
            return {
                "status": "idle",
                "model": None,
                "message": "Sin descarga activa",
            }
        if self._pull_status == "running" and self._pull_model:
            return {
                "status": "running",
                "model": self._pull_model,
                "message": "Descargando modelo...",
            }
        if self._pull_status == "completed":
            payload = {
                "status": "completed",
                "model": self._pull_model,
                "message": "Entorno IA preparado",
            }
            return payload
        if self._pull_status == "error":
            return {
                "status": "error",
                "model": self._pull_model,
                "message": self._pull_message or "No fue posible descargar el modelo.",
            }
        return {
            "status": self._pull_status,
            "model": self._pull_model,
            "message": self._pull_message,
        }

    def request_pull(self, payload: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        status = self.get_status()
        if not status["ollama"]["available"]:
            return 400, {
                "status": "error",
                "model": None,
                "message": "Ollama no está instalado. Instala Ollama y vuelve a abrir la aplicación.",
            }

        requested = payload.get("models") if payload else None
        selected = self._normalize_requested_models(requested)
        if not selected:
            selected = [name for name in DEFAULT_MODEL_NAMES if not status["models"].get(name, {}).get("installed")]

        if not selected:
            return 202, {
                "status": "completed",
                "model": None,
                "message": "Todos los modelos requeridos ya están instalados",
            }

        with self._pull_lock:
            if self._pull_status == "running" and self._pull_thread and self._pull_thread.is_alive():
                return 202, {
                    "status": "running",
                    "model": self._pull_model,
                    "message": "Descarga ya en curso",
                }
            if self._pull_status == "running":
                self._pull_status = "idle"

            models_to_pull = []
            installed = self._list_installed_models()
            for model_name in selected:
                if self._model_is_installed(model_name, installed):
                    continue
                models_to_pull.append(model_name)

            if not models_to_pull:
                self._pull_status = "completed"
                self._pull_model = None
                self._pull_message = "Todos los modelos requeridos ya están instalados"
                return 202, self.get_pull_status()

            self._pull_status = "running"
            self._pull_model = models_to_pull[0]
            self._pull_error = None
            self._pull_message = "Descargando modelo..."
            self._pull_thread = threading.Thread(
                target=self._run_pull,
                args=(models_to_pull,),
                daemon=True,
            )
            self._pull_thread.start()
            return 202, {
                "status": "running",
                "model": self._pull_model,
                "message": "Descargando modelo...",
            }

    def _run_pull(self, model_names: list[str]) -> None:
        try:
            for model_name in model_names:
                self._pull_status = "running"
                self._pull_model = model_name
                self._pull_message = "Descargando modelo..."
                process = subprocess.Popen(
                    ["ollama", "pull", model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    self._pull_status = "error"
                    self._pull_message = "No fue posible descargar el modelo."
                    self._pull_error = stderr or stdout or str(process.returncode)
                    print(f"RuntimeManager: failed to pull {model_name}: {self._pull_error}", flush=True)
                    return
            self._pull_status = "completed"
            self._pull_message = "Entorno IA preparado"
            self._pull_model = None
        except Exception as exc:
            self._pull_status = "error"
            self._pull_message = "No fue posible descargar el modelo."
            self._pull_error = str(exc)
            print(f"RuntimeManager: unexpected pull failure: {exc}", flush=True)

    def _normalize_requested_models(self, requested: Any) -> list[str]:
        if requested is None:
            return []
        if isinstance(requested, str):
            candidates = [requested]
        elif isinstance(requested, list):
            candidates = requested
        else:
            candidates = []

        normalized = []
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            name = candidate.strip().lower()
            if name in ALLOWED_MODELS:
                normalized.append(name)
        return normalized

    def _ollama_available(self) -> bool:
        return shutil.which("ollama") is not None

    def _model_is_installed(self, model_name: str, installed: set[str]) -> bool:
        if not model_name:
            return False
        canonical = model_name.strip().lower()
        for candidate in installed:
            normalized = candidate.strip().lower()
            if normalized == canonical:
                return True
            if normalized.startswith(f"{canonical}:"):
                return True
            if canonical.startswith(f"{normalized}:"):
                return True
        return False

    def _list_installed_models(self) -> set[str]:
        if not self._ollama_available():
            return set()
        try:
            result = subprocess.run(
                ["ollama", "list"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=5,
            )
            if result.returncode != 0:
                return set()
            model_names: set[str] = set()
            lines = result.stdout.splitlines()
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("NAME"):
                    continue
                parts = stripped.split()
                if not parts:
                    continue
                model_name = parts[0]
                if model_name:
                    model_names.add(model_name)
            return model_names
        except Exception:
            return set()
