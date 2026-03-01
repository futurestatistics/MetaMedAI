from __future__ import annotations

from typing import Any, Dict, List

from src.memory.layered_memory import layered_memory_store


class MemoryRetrievalAgent:
    def __init__(self, rag_threshold: float = 0.7) -> None:
        self.rag_threshold = rag_threshold

    def retrieve(
        self,
        session_id: str,
        query: str,
        intent: Dict[str, Any],
        expanded_queries: List[str],
    ) -> Dict[str, Any]:
        l2_records = layered_memory_store.search_l2_records(session_id=session_id, query=query, top_k=3)

        rag_hits: List[Dict[str, Any]] = []
        for text in [query, *expanded_queries[:3]]:
            rag_hits.extend(layered_memory_store.search_l3_chunks(query=text, top_k=3))

        dedup = {}
        for item in rag_hits:
            dedup[item["chunk_id"]] = item
        from_rag = sorted(dedup.values(), key=lambda x: x.get("similarity", 0), reverse=True)[:5]

        rag_similarity = from_rag[0]["similarity"] if from_rag else 0.0
        has_good_l2 = bool(l2_records and l2_records[0].get("similarity", 0) >= 0.55)

        should_use_pubmed = bool(intent.get("need_retrieval", False)) and (not has_good_l2) and (rag_similarity < self.rag_threshold)

        return {
            "from_l2_memory": l2_records,
            "from_rag": from_rag,
            "rag_similarity": rag_similarity,
            "should_use_pubmed": should_use_pubmed,
        }
