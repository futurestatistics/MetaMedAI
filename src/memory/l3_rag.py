from __future__ import annotations

import math
import re
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

from langchain_core.embeddings import Embeddings


class HashEmbeddings(Embeddings):
    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower())
        if not tokens:
            return vec
        for token in tokens:
            idx = hash(token) % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


class L3RAGStore:
    def __init__(self, persist_directory: str = "./data/vectorstore") -> None:
        self._lock = Lock()
        self.persist_directory = persist_directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self._fallback_docs: Dict[str, Dict[str, Any]] = {}
        self.backend = "in_memory"
        self._store = None
        self._init_store()

    def _init_store(self) -> None:
        try:
            from langchain_chroma import Chroma

            self._store = Chroma(
                collection_name="l3_reports",
                embedding_function=HashEmbeddings(),
                persist_directory=self.persist_directory,
            )
            self.backend = "chroma"
        except Exception:
            self._store = None
            self.backend = "in_memory"

    def _chunk_report(self, report_content: str) -> List[str]:
        chunks = [part.strip() for part in re.split(r"\n\s*\n", report_content or "") if part.strip()]
        if not chunks and report_content:
            chunks = [report_content]
        return chunks

    def add_report(
        self,
        report_id: str,
        report_content: str,
        metadata: Dict[str, Any],
    ) -> int:
        chunks = self._chunk_report(report_content)
        if not chunks:
            return 0

        with self._lock:
            if self.backend == "chroma" and self._store is not None:
                ids = [f"{report_id}_chunk_{i}" for i in range(1, len(chunks) + 1)]
                metadatas = []
                for idx, _ in enumerate(chunks, start=1):
                    metadatas.append({**metadata, "chunk_id": ids[idx - 1]})
                self._store.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
                return len(chunks)

            for idx, content in enumerate(chunks, start=1):
                chunk_id = f"{report_id}_chunk_{idx}"
                self._fallback_docs[chunk_id] = {
                    "content": content,
                    "metadata": {**metadata, "chunk_id": chunk_id},
                }
            return len(chunks)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        with self._lock:
            if self.backend == "chroma" and self._store is not None:
                docs_scores = self._store.similarity_search_with_relevance_scores(query=query, k=max(1, top_k))
                output = []
                for doc, score in docs_scores:
                    output.append(
                        {
                            "chunk_id": doc.metadata.get("chunk_id", ""),
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "similarity": round(float(score), 4),
                        }
                    )
                return output

            q_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower()))
            scored = []
            for chunk_id, item in self._fallback_docs.items():
                c_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", item["content"].lower()))
                if not q_tokens or not c_tokens:
                    continue
                score = len(q_tokens & c_tokens) / max(len(q_tokens), len(c_tokens))
                if score > 0:
                    scored.append((score, chunk_id, item))
            scored.sort(key=lambda x: x[0], reverse=True)

            output = []
            for score, chunk_id, item in scored[: max(1, top_k)]:
                output.append(
                    {
                        "chunk_id": chunk_id,
                        "content": item["content"],
                        "metadata": item["metadata"],
                        "similarity": round(score, 4),
                    }
                )
            return output
