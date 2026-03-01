from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List
from uuid import uuid4

from src.memory.l3_rag import HashEmbeddings


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower()))


def _similarity(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


@dataclass
class SearchRecord:
    search_id: str
    session_id: str
    topic: str
    query: str
    timestamp: str
    papers_count: int
    papers: List[Dict[str, Any]]
    summary: str
    markdown_path: str


class L2SearchMemoryStore:
    def __init__(self, max_records: int = 30, persist_directory: str = "./data/l2_vectorstore"):
        self.max_records = max_records
        self.persist_directory = persist_directory
        self._lock = Lock()
        self._records: Dict[str, List[SearchRecord]] = {}
        self._record_index: Dict[str, SearchRecord] = {}
        self.backend = "in_memory"
        self._store = None
        self._init_vector_store()

    def _init_vector_store(self) -> None:
        try:
            from langchain_chroma import Chroma

            self._store = Chroma(
                collection_name="l2_search_records",
                embedding_function=HashEmbeddings(),
                persist_directory=self.persist_directory,
            )
            self.backend = "chroma"
        except Exception:
            self._store = None
            self.backend = "in_memory"

    def _record_to_doc(self, record: SearchRecord) -> str:
        return f"{record.topic}\n{record.query}\n{record.summary}"

    def _add_record_to_vector_store(self, record: SearchRecord) -> None:
        if self.backend != "chroma" or self._store is None:
            return
        metadata = {
            "search_id": record.search_id,
            "session_id": record.session_id,
            "topic": record.topic,
            "query": record.query,
            "timestamp": record.timestamp,
            "papers_count": record.papers_count,
            "markdown_path": record.markdown_path,
        }
        self._store.add_texts(
            texts=[self._record_to_doc(record)],
            metadatas=[metadata],
            ids=[record.search_id],
        )

    def _delete_record_from_vector_store(self, search_id: str) -> None:
        if self.backend != "chroma" or self._store is None:
            return
        try:
            self._store.delete(ids=[search_id])
        except Exception:
            pass

    def _prune_session_records(self, session_id: str) -> None:
        rows = self._records.get(session_id, [])
        while len(rows) > self.max_records:
            dropped = rows.pop(0)
            self._record_index.pop(dropped.search_id, None)
            self._delete_record_from_vector_store(dropped.search_id)
        self._records[session_id] = rows

    def add_record(
        self,
        session_id: str,
        topic: str,
        query: str,
        papers: List[Dict[str, Any]],
        summary: str,
        markdown_path: str,
    ) -> SearchRecord:
        record = SearchRecord(
            search_id=f"search_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}",
            session_id=session_id,
            topic=topic,
            query=query,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            papers_count=len(papers),
            papers=papers,
            summary=summary,
            markdown_path=markdown_path,
        )
        with self._lock:
            rows = self._records.setdefault(session_id, [])
            rows.append(record)
            self._record_index[record.search_id] = record
            self._add_record_to_vector_store(record)
            self._prune_session_records(session_id)
        return record

    def _search_in_memory(self, session_id: str, query: str, top_k: int) -> List[Dict[str, Any]]:
        rows = list(self._records.get(session_id, []))
        scored = []
        for record in rows:
            score = _similarity(query, self._record_to_doc(record))
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)

        output = []
        for score, record in scored[: max(1, top_k)]:
            output.append(
                {
                    "similarity": round(score, 4),
                    "search_id": record.search_id,
                    "topic": record.topic,
                    "query": record.query,
                    "summary": record.summary,
                    "papers_count": record.papers_count,
                    "papers": record.papers,
                    "markdown_path": record.markdown_path,
                    "timestamp": record.timestamp,
                }
            )
        return output

    def search(self, session_id: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        with self._lock:
            if self.backend == "chroma" and self._store is not None:
                try:
                    docs_scores = self._store.similarity_search_with_relevance_scores(
                        query=query,
                        k=max(1, top_k),
                        filter={"session_id": session_id},
                    )
                except TypeError:
                    docs = self._store.similarity_search(
                        query=query,
                        k=max(1, top_k * 2),
                        filter={"session_id": session_id},
                    )
                    docs_scores = [(doc, 0.5) for doc in docs]
                except Exception:
                    docs_scores = []

                output: List[Dict[str, Any]] = []
                for doc, score in docs_scores:
                    search_id = doc.metadata.get("search_id", "")
                    record = self._record_index.get(search_id)
                    if not record:
                        continue
                    output.append(
                        {
                            "similarity": round(float(score), 4),
                            "search_id": record.search_id,
                            "topic": record.topic,
                            "query": record.query,
                            "summary": record.summary,
                            "papers_count": record.papers_count,
                            "papers": record.papers,
                            "markdown_path": record.markdown_path,
                            "timestamp": record.timestamp,
                        }
                    )
                if output:
                    return output[: max(1, top_k)]

            return self._search_in_memory(session_id=session_id, query=query, top_k=top_k)
