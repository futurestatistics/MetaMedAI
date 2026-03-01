from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4

from langchain_core.chat_history import InMemoryChatMessageHistory


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class ConversationSession:
    session_id: str
    memory: InMemoryChatMessageHistory = field(default_factory=InMemoryChatMessageHistory)
    messages: List[Message] = field(default_factory=list)
    paper_index: Dict[str, dict] = field(default_factory=dict)
    query_counters: Dict[str, int] = field(default_factory=dict)
    evidence_notes: List[str] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        message = Message(role=role, content=content)
        self.messages.append(message)
        if role == "user":
            self.memory.add_user_message(content)
        elif role == "assistant":
            self.memory.add_ai_message(content)

    def get_recent_messages(self, limit: int = 8) -> List[Message]:
        if limit <= 0:
            return []
        return self.messages[-limit:]

    def upsert_paper(self, key: str, paper: dict) -> bool:
        if key in self.paper_index:
            return False
        self.paper_index[key] = paper
        return True

    def all_papers(self) -> List[dict]:
        return list(self.paper_index.values())

    def retrieve_related_papers(self, query: str, top_k: int = 6) -> List[dict]:
        if not query.strip() or not self.paper_index:
            return []

        query_tokens = set(query.lower().split())
        scored = []
        for paper in self.paper_index.values():
            combined = " ".join(
                [
                    str(paper.get("title", "")),
                    str(paper.get("summary", "")),
                    str(paper.get("conclusion", "")),
                    str(paper.get("implication", "")),
                    str(paper.get("critique", "")),
                ]
            ).lower()

            if not combined:
                continue

            text_tokens = set(combined.split())
            overlap = len(query_tokens.intersection(text_tokens))
            contains_bonus = 2 if query.lower() in combined else 0
            score = overlap + contains_bonus
            if score > 0:
                scored.append((score, paper))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [paper for _, paper in scored[: max(1, top_k)]]

    def build_memory_context(self, query: str, top_k_messages: int = 10, top_k_papers: int = 6) -> Dict[str, object]:
        recent_messages = self.get_recent_messages(limit=top_k_messages)
        recent_dialogue = "\n".join([f"{m.role}: {m.content}" for m in recent_messages])
        related_papers = self.retrieve_related_papers(query=query, top_k=top_k_papers)
        return {
            "recent_dialogue": recent_dialogue,
            "related_papers": related_papers,
            "recent_evidence_notes": self.get_recent_evidence_notes(limit=top_k_papers),
        }

    def next_query_offset(self, query: str, batch_size: int) -> int:
        key = query.strip().lower()
        current_count = self.query_counters.get(key, 0)
        self.query_counters[key] = current_count + 1
        return current_count * max(1, batch_size)

    def add_evidence_note(self, note: str) -> None:
        if not note:
            return
        self.evidence_notes.append(note)
        if len(self.evidence_notes) > 30:
            self.evidence_notes = self.evidence_notes[-30:]

    def get_recent_evidence_notes(self, limit: int = 8) -> List[str]:
        if limit <= 0:
            return []
        return self.evidence_notes[-limit:]


class SessionMemoryStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, ConversationSession] = {}
        self._lock = Lock()

    def create_session(self) -> str:
        session_id = uuid4().hex
        with self._lock:
            self._sessions[session_id] = ConversationSession(session_id=session_id)
        return session_id

    def get_or_create(self, session_id: Optional[str]) -> ConversationSession:
        with self._lock:
            if not session_id or session_id not in self._sessions:
                new_id = session_id or uuid4().hex
                self._sessions[new_id] = ConversationSession(session_id=new_id)
                return self._sessions[new_id]
            return self._sessions[session_id]


session_memory_store = SessionMemoryStore()
