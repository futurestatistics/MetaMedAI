from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Dict, List
from uuid import uuid4

from src.memory.l1_memory import L1MemoryStore
from src.memory.l2_memory import L2SearchMemoryStore
from src.memory.l3_rag import L3RAGStore


class SessionState:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.latest_citations: Dict[str, Dict[str, Any]] = {}


class LayeredMemoryStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: Dict[str, SessionState] = {}
        self.l1 = L1MemoryStore(max_messages=40)
        self.l2 = L2SearchMemoryStore(max_records=30)
        self.l3 = L3RAGStore(persist_directory="./data/vectorstore")
        self._citation_registry: Dict[str, Dict[str, Any]] = {}
        self._report_registry: Dict[str, str] = {}

    def create_session(self) -> str:
        session_id = uuid4().hex
        with self._lock:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return session_id

    def get_or_create_session(self, session_id: str | None) -> SessionState:
        with self._lock:
            if not session_id:
                new_id = uuid4().hex
                self._sessions[new_id] = SessionState(session_id=new_id)
                return self._sessions[new_id]
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(session_id=session_id)
            return self._sessions[session_id]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self.get_or_create_session(session_id)
        self.l1.add_message(session_id=session_id, role=role, content=content)

    def get_session_messages(self, session_id: str, limit: int = 20) -> List[Dict[str, str]]:
        self.get_or_create_session(session_id)
        return self.l1.get_messages(session_id=session_id, limit=limit)

    def get_recent_dialogue_text(self, session_id: str, limit: int = 10) -> str:
        self.get_or_create_session(session_id)
        return self.l1.get_dialogue_text(session_id=session_id, limit=limit)

    def add_l2_search_record(
        self,
        session_id: str,
        topic: str,
        query: str,
        papers: List[Dict[str, Any]],
        summary: str,
        markdown_path: str,
    ):
        self.get_or_create_session(session_id)
        return self.l2.add_record(
            session_id=session_id,
            topic=topic,
            query=query,
            papers=papers,
            summary=summary,
            markdown_path=markdown_path,
        )

    def search_l2_records(self, session_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        self.get_or_create_session(session_id)
        return self.l2.search(session_id=session_id, query=query, top_k=top_k)

    def add_report_to_l3(
        self,
        report_content: str,
        report_path: str,
        topic: str,
        search_id: str,
        session_id: str,
        papers: List[Dict[str, Any]],
    ) -> int:
        report_id = Path(report_path).stem if report_path else f"report_{uuid4().hex[:12]}"
        self._report_registry[report_id] = report_path

        paper_meta = papers[0] if papers else {}
        metadata = {
            "topic": topic,
            "report_file": report_path,
            "search_id": search_id,
            "session_id": session_id,
            "title": paper_meta.get("title", ""),
            "journal": paper_meta.get("journal_name", ""),
            "year": str(paper_meta.get("publish_date", ""))[:4],
        }
        return self.l3.add_report(
            report_id=report_id,
            report_content=report_content,
            metadata=metadata,
            papers=papers,
        )

    def search_l3_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.l3.search(query=query, top_k=top_k)

    def get_l3_backend(self) -> str:
        return self.l3.backend

    def register_citations(self, session_id: str, citations: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        session = self.get_or_create_session(session_id)
        registered: Dict[str, Dict[str, Any]] = {}
        for display_id, item in citations.items():
            citation_id = item.get("citation_id") or f"cit_{uuid4().hex}"
            enriched = {**item, "citation_id": citation_id, "display_id": display_id, "session_id": session_id}
            self._citation_registry[citation_id] = enriched
            registered[display_id] = enriched
        session.latest_citations = registered
        return registered

    def get_citation(self, citation_id: str) -> Dict[str, Any] | None:
        return self._citation_registry.get(citation_id)

    def get_session_citation(self, session_id: str, display_id: str) -> Dict[str, Any] | None:
        session = self.get_or_create_session(session_id)
        return session.latest_citations.get(display_id)

    def register_report(self, report_path: str) -> str:
        report_id = Path(report_path).stem
        self._report_registry[report_id] = report_path
        return report_id

    def get_report_path(self, report_id: str) -> str | None:
        return self._report_registry.get(report_id)


layered_memory_store = LayeredMemoryStore()
