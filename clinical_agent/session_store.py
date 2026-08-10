from __future__ import annotations

from .schemas import SessionState


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def create(self, session_id: str) -> SessionState:
        state = SessionState(session_id=session_id)
        self._sessions[session_id] = state
        return state

    def get_or_create(self, session_id: str) -> SessionState:
        return self.get(session_id) or self.create(session_id)

    def update(self, session_id: str, state: SessionState) -> SessionState:
        self._sessions[session_id] = state
        return state

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def clear(self) -> None:
        self._sessions.clear()
