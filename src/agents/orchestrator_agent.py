from __future__ import annotations

from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from src.agents.answer_generation_agent import AnswerGenerationAgent
from src.agents.data_agent import DataAgent
from src.agents.intent_agent import IntentAgent
from src.agents.literature_agent import LiteratureAgent
from src.agents.memory_retrieval_agent import MemoryRetrievalAgent
from src.agents.query_expansion_agent import QueryExpansionAgent
from src.agents.report_agent import ReportAgent
from src.memory.layered_memory import layered_memory_store


class OrchestratorAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = self._init_llm()
        self.intent_agent = IntentAgent(self.llm)
        self.query_expansion_agent = QueryExpansionAgent(self.llm)
        self.memory_agent = MemoryRetrievalAgent(rag_threshold=0.7)
        self.answer_agent = AnswerGenerationAgent(self.llm)
        self.literature_agent = LiteratureAgent(config)
        self.data_agent = DataAgent(config, llm=self.llm)
        self.report_agent = ReportAgent(config, self.llm)

    def _init_llm(self):
        llm_config = self.config["llm"]
        if llm_config["type"] != "tongyi":
            raise ValueError("当前仅支持tongyi配置")
        return ChatOpenAI(
            model=llm_config["tongyi"]["model_name"],
            api_key=llm_config["tongyi"]["api_key"],
            base_url=llm_config["tongyi"]["base_url"],
            temperature=0.2,
        )

    def create_session(self) -> str:
        return layered_memory_store.create_session()

    def get_session_history(self, session_id: str) -> Dict[str, Any]:
        return {
            "session_id": session_id,
            "messages": layered_memory_store.get_session_messages(session_id, limit=100),
        }

    def get_citation(self, citation_id: str, session_id: str | None = None) -> Dict[str, Any] | None:
        citation = layered_memory_store.get_citation(citation_id)
        if citation:
            return citation
        if session_id:
            return layered_memory_store.get_session_citation(session_id, citation_id)
        return None

    def _collect_papers_from_memory(
        self,
        memory_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        papers: List[Dict[str, Any]] = []
        for record in memory_result.get("from_l2_memory", []):
            papers.extend(record.get("papers", []))
        return papers

    def _search_pubmed_with_expanded_queries(
        self,
        expanded_queries: List[str],
    ) -> Dict[str, Any]:
        max_papers = self.config["agent"]["literature"].get("max_papers", 10)
        start_date = self.config["agent"]["literature"].get("start_date")
        end_date = self.config["agent"]["literature"].get("end_date")
        per_query_limit = max(1, max_papers // max(1, len(expanded_queries)))

        all_papers: List[Dict[str, Any]] = []
        query_status = []
        for query in expanded_queries:
            result = self.literature_agent.run(
                keywords=query,
                max_papers_override=per_query_limit,
                start_date=start_date,
                end_date=end_date,
                sort="relevance",
            )
            query_status.append(
                {
                    "query": query,
                    "status": result.get("status", "error"),
                    "message": result.get("message", ""),
                }
            )
            if result.get("status") in {"success", "warning"}:
                all_papers.extend(result.get("data", []))

        dedup: Dict[str, Dict[str, Any]] = {}
        for paper in all_papers:
            key = (paper.get("pmid") or paper.get("doi") or paper.get("title") or "").strip().lower()
            if key and key not in dedup:
                dedup[key] = paper

        papers = list(dedup.values())[:max_papers]

        status = "success" if papers else "warning"
        return {
            "status": status,
            "message": "PubMed检索完成" if papers else "未检索到有效文献",
            "data": papers,
            "expanded_queries": expanded_queries,
            "query_status": query_status,
        }

    def _build_search_summary(self, papers: List[Dict[str, Any]], user_message: str) -> str:
        if not papers:
            return f"主题：{user_message}；本轮未检索到有效文献。"
        years = [str(item.get("publish_date", ""))[:4] for item in papers if item.get("publish_date")]
        year_text = f"，时间覆盖约{min(years)}-{max(years)}" if years else ""
        return f"主题：{user_message}；检索到{len(papers)}篇文献{year_text}。"

    def process_message(self, session_id: str | None, user_message: str) -> Dict[str, Any]:
        session = layered_memory_store.get_or_create_session(session_id)
        session_id = session.session_id

        layered_memory_store.add_message(session_id=session_id, role="human", content=user_message)
        recent_dialogue = layered_memory_store.get_recent_dialogue_text(session_id=session_id, limit=10)

        has_cached = bool(layered_memory_store.search_l2_records(session_id=session_id, query=user_message, top_k=1))
        intent = self.intent_agent.classify(
            user_message=user_message,
            recent_context=recent_dialogue,
            has_cached_papers=has_cached,
        )

        expanded_queries = []
        if intent.get("need_retrieval", False):
            expanded_queries = self.query_expansion_agent.expand(
                user_message=user_message,
                recent_context=recent_dialogue,
                max_queries=5,
            )
        intent["expanded_queries"] = expanded_queries

        memory_result = self.memory_agent.retrieve(
            session_id=session_id,
            query=user_message,
            intent=intent,
            expanded_queries=expanded_queries,
        )

        search_performed = False
        report_result: Dict[str, Any] = {
            "status": "skipped",
            "report_content": "",
            "report_path": "",
            "metadata": {},
        }

        papers = self._collect_papers_from_memory(memory_result)
        if memory_result.get("should_use_pubmed", False):
            search_performed = True
            lit_result = self._search_pubmed_with_expanded_queries(expanded_queries=expanded_queries)
            papers = lit_result.get("data", [])

            data_result = self.data_agent.run({"data": papers})
            report_result = self.report_agent.run(
                keywords=user_message,
                literature_data={"status": lit_result.get("status", "success"), "data": papers},
                data_process_data=data_result,
            )

            report_path = report_result.get("report_path", "")
            if report_path:
                search_record = layered_memory_store.add_l2_search_record(
                    session_id=session_id,
                    topic=" ".join(intent.get("keywords", [])) or user_message,
                    query=user_message,
                    papers=papers,
                    summary=self._build_search_summary(papers, user_message),
                    markdown_path=report_path,
                )
                layered_memory_store.register_report(report_path)
                layered_memory_store.add_report_to_l3(
                    report_content=report_result.get("report_content", ""),
                    report_path=report_path,
                    topic=search_record.topic,
                    search_id=search_record.search_id,
                    session_id=session_id,
                    papers=papers,
                )

        answer, citations = self.answer_agent.generate(
            user_query=user_message,
            recent_dialogue=recent_dialogue,
            intent=intent,
            l2_records=memory_result.get("from_l2_memory", []),
            rag_chunks=memory_result.get("from_rag", []),
            papers=papers,
        )

        citations = layered_memory_store.register_citations(session_id=session_id, citations=citations)

        layered_memory_store.add_message(session_id=session_id, role="ai", content=answer)

        return {
            "session_id": session_id,
            "answer": answer,
            "citations": citations,
            "search_performed": search_performed,
            "markdown_path": report_result.get("report_path", ""),
            "intent": intent,
            "memory": {
                "rag_similarity": memory_result.get("rag_similarity", 0.0),
                "should_use_pubmed": memory_result.get("should_use_pubmed", False),
                "l3_vector_backend": layered_memory_store.get_l3_backend(),
            },
        }
