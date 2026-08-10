from __future__ import annotations

import json

try:
    from ollama import chat
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    chat = None


MODEL = "llama3.2"


def generate(system_prompt: str, user_message: str) -> str:
    if chat is None:
        raise RuntimeError("ollama package is not installed")
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        options={
            "temperature": 0,
        },
    )

    return response.message.content


def generate_json(system_prompt: str, user_message: str) -> dict:
    raw_response = generate_json_raw(system_prompt, user_message)
    return json.loads(raw_response)


def generate_json_raw(system_prompt: str, user_message: str) -> str:
    if chat is None:
        raise RuntimeError("ollama package is not installed")
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        format="json",
        options={
            "temperature": 0,
        },
    )

    return response.message.content
