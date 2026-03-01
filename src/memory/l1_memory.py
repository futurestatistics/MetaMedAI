from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Dict, List


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: str = field(default_factory=_now)


class L1MemoryStore:
    def __init__(self, max_messages: int = 40):
        self.max_messages = max_messages
        self._lock = Lock()
        self._session_messages: Dict[str, List[ChatMessage]] = {}

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            messages = self._session_messages.setdefault(session_id, [])
            messages.append(ChatMessage(role=role, content=content))
            if len(messages) > self.max_messages:
                self._session_messages[session_id] = messages[-self.max_messages :]

    def get_messages(self, session_id: str, limit: int = 20) -> List[Dict[str, str]]:
        with self._lock:
            messages = self._session_messages.get(session_id, [])[-max(1, limit) :]
        return [
            {"role": message.role, "content": message.content, "timestamp": message.timestamp}
            for message in messages
        ]

    def get_dialogue_text(self, session_id: str, limit: int = 10) -> str:
        rows = self.get_messages(session_id=session_id, limit=limit)
        return "\n".join([f"{item['role']}: {item['content']}" for item in rows])
