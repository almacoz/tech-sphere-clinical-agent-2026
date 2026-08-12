from __future__ import annotations

import io
import os
import threading
import time
from typing import Any

try:  # pragma: no cover - optional dependency, mismo patron que llm.py/rag.py
    from kokoro import KPipeline
except Exception:  # pragma: no cover
    KPipeline = None

try:  # pragma: no cover - optional dependency
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None

try:  # pragma: no cover - optional dependency (viene con kokoro, pero por si acaso)
    import numpy as np
except Exception:  # pragma: no cover
    np = None

# Kokoro-82M (https://huggingface.co/hexgrad/Kokoro-82M) — TTS local, Apache-2.0,
# CPU-friendly, sin token de HuggingFace. lang_code='e' = español; voces en
# español disponibles: ef_dora (femenina), em_alex y em_santa (masculinas).
# Ver VOICES.md del modelo para la lista completa.
DEFAULT_LANG_CODE = os.getenv("KOKORO_LANG_CODE", "e")
DEFAULT_VOICE = os.getenv("KOKORO_VOICE", "ef_dora")
SAMPLE_RATE = 24000
SPANISH_VOICES = ("ef_dora", "em_alex", "em_santa")

_pipeline_cache: dict[str, Any] = {}
_pipeline_lock = threading.Lock()


def tts_available() -> bool:
    """True si las dependencias de Kokoro están instaladas. No implica que
    espeak-ng (usado como fallback de fonemización) esté disponible — eso
    solo se sabe al intentar sintetizar."""
    return KPipeline is not None and sf is not None and np is not None


def tts_status() -> dict[str, Any]:
    return {
        "kokoro_installed": KPipeline is not None,
        "soundfile_installed": sf is not None,
        "numpy_installed": np is not None,
        "ready": tts_available(),
        "default_voice": DEFAULT_VOICE,
        "default_lang_code": DEFAULT_LANG_CODE,
        "spanish_voices": list(SPANISH_VOICES),
        "sample_rate": SAMPLE_RATE,
    }


def _get_pipeline(lang_code: str):
    if KPipeline is None:
        raise RuntimeError(
            "El paquete 'kokoro' no está instalado. Instala con: "
            "pip install kokoro soundfile — y en macOS además "
            "`brew install espeak-ng` (en Linux: `apt-get install espeak-ng`)."
        )
    with _pipeline_lock:
        if lang_code not in _pipeline_cache:
            # KPipeline carga pesos la primera vez que se usa un lang_code;
            # se cachea el pipeline para no recargar el modelo en cada turno.
            _pipeline_cache[lang_code] = KPipeline(lang_code=lang_code)
        return _pipeline_cache[lang_code]


def synthesize(
    text: str,
    voice: str = DEFAULT_VOICE,
    lang_code: str = DEFAULT_LANG_CODE,
) -> tuple[bytes, int]:
    """Sintetiza `text` a WAV (24kHz, mono) con Kokoro-82M.

    Devuelve (wav_bytes, latency_ms). Lanza RuntimeError con un mensaje
    accionable si kokoro/soundfile/numpy no están instalados, en vez de
    tumbar el proceso — el llamador (main.py) lo traduce a un 503, así el
    resto del agente (texto, RAG, decisión) sigue funcionando sin voz si el
    entorno de TTS no está listo."""
    if not text or not text.strip():
        raise ValueError("texto vacío: nada que sintetizar")
    if not tts_available():
        raise RuntimeError(
            "Kokoro TTS no está disponible en este entorno. Instala con: "
            "pip install kokoro soundfile — y en macOS `brew install espeak-ng`."
        )

    started = time.perf_counter()
    pipeline = _get_pipeline(lang_code)
    chunks = []
    for _graphemes, _phonemes, audio in pipeline(text, voice=voice, split_pattern=r"\n+"):
        chunks.append(audio)
    if not chunks:
        raise RuntimeError("Kokoro no generó audio para el texto dado.")
    full_audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    buffer = io.BytesIO()
    sf.write(buffer, full_audio, SAMPLE_RATE, format="WAV")
    latency_ms = int((time.perf_counter() - started) * 1000)
    return buffer.getvalue(), latency_ms
