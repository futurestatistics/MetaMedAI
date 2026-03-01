from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


class AnswerGenerationAgent:
    def __init__(self, llm) -> None:
        self.llm = llm

    def _paper_key(self, paper: Dict[str, Any]) -> str:
        return (paper.get("pmid") or paper.get("doi") or paper.get("title") or "").strip().lower()

    def _chunk_key(self, chunk: Dict[str, Any]) -> str:
        metadata = chunk.get("metadata") or {}
        return (
            metadata.get("chunk_id")
            or metadata.get("report_file")
            or metadata.get("search_id")
            or (chunk.get("content") or "")[:120]
        ).strip().lower()

    def _build_citations(
        self,
        papers: List[Dict[str, Any]],
        rag_chunks: List[Dict[str, Any]],
        limit: int = 8,
    ) -> Dict[str, Dict[str, Any]]:
        citations: Dict[str, Dict[str, Any]] = {}
        dedup = set()
        index = 1

        for paper in papers:
            key = self._paper_key(paper)
            if not key or key in dedup:
                continue
            dedup.add(key)
            citations[str(index)] = {
                "citation_id": "",
                "source_type": "paper",
                "pmid": paper.get("pmid", ""),
                "title": paper.get("title", "未知标题"),
                "journal": paper.get("journal_name", "未知期刊"),
                "year": str(paper.get("publish_date", ""))[:4],
                "first_author": (paper.get("authors") or ["未知作者"])[0],
                "authors": paper.get("authors", []),
                "abstract": paper.get("conclusion", "未明确提及"),
            }
            index += 1
            if index > limit:
                break

        if index <= limit:
            for chunk in rag_chunks:
                key = self._chunk_key(chunk)
                if not key or key in dedup:
                    continue

                metadata = chunk.get("metadata") or {}
                content = (chunk.get("content") or "").strip()
                if not content:
                    continue

                authors = metadata.get("authors") or []
                first_author = metadata.get("first_author") or (authors[0] if authors else "历史来源")
                journal = metadata.get("journal") or metadata.get("journal_name") or "历史报告"
                year = metadata.get("year") or str(metadata.get("publish_date", ""))[:4]

                dedup.add(key)
                citations[str(index)] = {
                    "citation_id": "",
                    "source_type": metadata.get("source_type", "l3_chunk"),
                    "pmid": metadata.get("pmid", ""),
                    "doi": metadata.get("doi", ""),
                    "title": metadata.get("title", "历史报告片段"),
                    "journal": journal,
                    "year": year,
                    "first_author": first_author,
                    "authors": authors,
                    "abstract": content[:800],
                    "chunk_id": metadata.get("chunk_id", ""),
                    "report_file": metadata.get("report_file", ""),
                    "search_id": metadata.get("search_id", ""),
                }
                index += 1
                if index > limit:
                    break

        return citations

    def _build_citation_context(self, citations: Dict[str, Dict[str, Any]]) -> str:
        lines = []
        for num, item in citations.items():
            id_text = f"PMID:{item.get('pmid', '')}" if item.get("pmid") else f"DOI:{item.get('doi', '')}"
            lines.append(
                f"[{num}] {item.get('first_author', '未知作者')}, {item.get('journal', '未知期刊')}, {item.get('year', '未知年份')} ({id_text})\n"
                f"内容: {item.get('abstract', '未明确提及')}"
            )
        return "\n\n".join(lines)

    def _validate_citation_marks(self, answer: str, citations: Dict[str, Dict[str, Any]]) -> str:
        valid_nums = set(citations.keys())

        def _replace(match: re.Match) -> str:
            value = match.group(1)
            nums = [part.strip() for part in value.split(",")]
            kept = [num for num in nums if num in valid_nums]
            if not kept:
                return ""
            return f"[{','.join(kept)}]"

        return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", _replace, answer)

    def generate(
        self,
        user_query: str,
        recent_dialogue: str,
        intent: Dict[str, Any],
        l2_records: List[Dict[str, Any]],
        rag_chunks: List[Dict[str, Any]],
        papers: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        citations = self._build_citations(papers=papers, rag_chunks=rag_chunks)
        citation_context = self._build_citation_context(citations)

        l2_text = json.dumps(l2_records, ensure_ascii=False)
        rag_text = json.dumps(rag_chunks, ensure_ascii=False)
        prompt = f"""
                你是一个专业的医学文献助手。请基于以下信息回答问题。

                ## 意图信息
                {json.dumps(intent, ensure_ascii=False)}

                ## 最近对话
                {recent_dialogue}

                ## L2检索会话记忆
                {l2_text}

                ## L3 RAG片段
                {rag_text}

                ## 可参考文献片段
                {citation_context}

                ## 用户问题
                {user_query}

                ## 要求
                1. 如果是会话回顾问题，优先使用L1/L2记忆直接回答。
                2. 若使用文献结论，必须在句末标注[编号]。
                3. 只能使用已有编号，不得虚构。
                4. 证据不足时明确说明。
                """
        try:
            answer = self.llm.invoke(prompt).content
        except Exception:
            answer = "当前根据已缓存记忆给出回答；如需更强证据，请触发新的文献检索。"

        answer = self._validate_citation_marks(answer, citations)
        return answer, citations
