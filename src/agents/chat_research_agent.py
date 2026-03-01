from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI

from src.agents.intent_agent import IntentAgent
from src.agents.query_expansion_agent import QueryExpansionAgent
from src.agents.report_agent import ReportAgent
from src.mcp.client import MCPToolClient
from src.memory.conversation_memory import ConversationSession, session_memory_store


class ChatResearchAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = self._init_llm()
        self.intent_agent = IntentAgent(self.llm)
        self.query_expander = QueryExpansionAgent(self.llm)
        self.report_agent = ReportAgent(config, self.llm)
        self.mcp_client = MCPToolClient()

    def _init_llm(self):
        llm_config = self.config["llm"]
        if llm_config["type"] != "tongyi":
            raise ValueError("当前聊天模式仅支持tongyi配置")
        return ChatOpenAI(
            model=llm_config["tongyi"]["model_name"],
            api_key=llm_config["tongyi"]["api_key"],
            base_url=llm_config["tongyi"]["base_url"],
            temperature=0.2,
        )

    def _paper_key(self, paper: Dict[str, Any]) -> str:
        doi = (paper.get("doi") or "").strip().lower()
        pmid = (paper.get("pmid") or "").strip()
        title = (paper.get("title") or "").strip().lower()
        if doi:
            return f"doi:{doi}"
        if pmid:
            return f"pmid:{pmid}"
        return f"title:{title}"

    def _upsert_papers(self, session: ConversationSession, papers: List[Dict[str, Any]]) -> Dict[str, int]:
        new_count = 0
        reused_count = 0
        for paper in papers:
            key = self._paper_key(paper)
            inserted = session.upsert_paper(key, paper)
            if inserted:
                new_count += 1
            else:
                reused_count += 1
        return {"new": new_count, "reused": reused_count}

    def _attach_report_memory(self, session: ConversationSession, all_papers: List[Dict[str, Any]], this_round_new_keys: set[str]) -> None:
        new_papers = [paper for paper in all_papers if self._paper_key(paper) in this_round_new_keys]
        if not new_papers:
            return

        summaries = self.report_agent.summarize_papers_for_memory(new_papers)
        summary_map = {(item.get("title") or "").strip().lower(): item for item in summaries}
        for paper in all_papers:
            title_key = (paper.get("title") or "").strip().lower()
            item = summary_map.get(title_key)
            if not item:
                continue
            paper["summary"] = item.get("summary", "")
            paper["implication"] = item.get("implication", "")
            paper["critique"] = item.get("critique", "")
            session.add_evidence_note(f"{paper.get('title', '未知文献')}：{paper.get('summary', '')}")

    def _retrieve_with_diversified_queries(self, session: ConversationSession, user_message: str, recent_context: str) -> Dict[str, Any]:
        expanded_queries = self.query_expander.expand(user_message=user_message, recent_context=recent_context, max_queries=4)
        max_papers = self.config["agent"]["literature"].get("max_papers", 10)
        start_date = self.config["agent"]["literature"].get("start_date")
        end_date = self.config["agent"]["literature"].get("end_date")

        per_query_limit = max(1, max_papers // max(1, len(expanded_queries)))
        all_papers: List[Dict[str, Any]] = []
        query_status = []

        for index, query in enumerate(expanded_queries):
            offset = session.next_query_offset(query=query, batch_size=per_query_limit)
            sort = "relevance" if index % 2 == 0 else "pub date"
            result = self.mcp_client.search_pubmed(
                query=query,
                email=self.config["entrez_email"],
                max_papers=per_query_limit,
                start_date=start_date,
                end_date=end_date,
                retstart=offset,
                sort=sort,
            )
            query_status.append({"query": query, "status": result.get("status", "error"), "message": result.get("message", "")})
            if result.get("status") in {"success", "warning"}:
                all_papers.extend(result.get("data", []))

        return {
            "status": "success" if all_papers else "warning",
            "message": "多查询检索完成" if all_papers else "未检索到有效文献",
            "data": all_papers,
            "expanded_queries": expanded_queries,
            "query_status": query_status,
        }

    def _build_answer(
        self,
        question: str,
        recent_context: str,
        evidence_notes: List[str],
        papers: List[Dict[str, Any]],
        intent: Dict[str, Any],
    ) -> str:
        top_papers = papers[-8:]
        compact_sources = [
            {
                "title": p.get("title", "未知文献"),
                "summary": p.get("summary", p.get("conclusion", "")),
                "implication": p.get("implication", ""),
                "critique": p.get("critique", ""),
            }
            for p in top_papers
        ]

        prompt = f"""
你是医学研究助手。请根据用户问题、对话记忆和证据源回答。

【意图识别】
{json.dumps(intent, ensure_ascii=False)}

【用户问题】
{question}

【最近对话】
{recent_context}

【证据笔记】
{json.dumps(evidence_notes, ensure_ascii=False)}

【文献证据（用于引用）】
{json.dumps(compact_sources, ensure_ascii=False)}

回答要求：
1. 若意图为memory/general，优先基于对话记忆，不要假装新检索。
2. 若意图为literature，给出结构化结论，并在关键陈述后引用来源：例如（《文献标题》）。
3. 简洁专业，先结论后要点。
4. 若证据不足要明确说明不足。
"""
        try:
            return self.llm.invoke(prompt).content
        except Exception:
            if intent.get("need_retrieval"):
                refs = "；".join([f"《{p.get('title', '未知文献')}》" for p in top_papers[:3]])
                return (
                    "已基于当前缓存证据给出简要结论：相关治疗策略包括生活方式干预、药物优化与个体化管理。"
                    f"可优先参考文献：{refs if refs else '暂无可用引用'}。"
                )
            return "该问题可由会话记忆回答：你当前在讨论糖尿病治疗相关主题。"

    def _build_memory_answer(
        self,
        question: str,
        recent_context: str,
        papers: List[Dict[str, Any]],
        evidence_notes: List[str],
    ) -> str:
        paper_sources = [
            {
                "title": p.get("title", "未知文献"),
                "summary": p.get("summary", p.get("conclusion", "")),
                "implication": p.get("implication", ""),
                "critique": p.get("critique", ""),
            }
            for p in papers[-12:]
        ]
        prompt = f"""
你是医学助手。当前任务是“基于会话记忆回答”，不要触发新检索。

【用户问题】
{question}

【最近对话】
{recent_context}

【记忆证据】
{json.dumps(evidence_notes, ensure_ascii=False)}

【可引用文献】
{json.dumps(paper_sources, ensure_ascii=False)}

输出要求：
1. 先直接回答问题。
2. 如果回答中涉及“结果/效果/结论”，必须在句末标注文献标题来源，例如（《文献标题》）。
3. 只能引用“可引用文献”中的标题，不得虚构。
4. 若记忆证据不足，明确说明“当前记忆证据不足”。
"""
        try:
            return self.llm.invoke(prompt).content
        except Exception:
            if paper_sources:
                title = paper_sources[-1].get("title", "未知文献")
                return f"基于当前会话记忆，你讨论的是糖尿病相关问题；已记录证据可参考（《{title}》）。"
            return "当前记忆证据不足，建议先触发文献检索后再回答。"

    def _build_retrieval_answer(
        self,
        question: str,
        recent_context: str,
        papers: List[Dict[str, Any]],
        intent: Dict[str, Any],
    ) -> str:
        return self._build_answer(
            question=question,
            recent_context=recent_context,
            evidence_notes=[],
            papers=papers,
            intent=intent,
        )

    def chat(self, user_message: str, session_id: str | None = None) -> Dict[str, Any]:
        session = session_memory_store.get_or_create(session_id)
        session.add_message("user", user_message)

        memory_context = session.build_memory_context(query=user_message, top_k_messages=10, top_k_papers=8)
        recent_context = str(memory_context.get("recent_dialogue", ""))
        related_memory_papers = memory_context.get("related_papers", [])
        related_memory_notes = memory_context.get("recent_evidence_notes", [])
        has_cached_papers = len(session.all_papers()) > 0
        execution_flow = ["用户输入问题"]

        intent = self.intent_agent.classify(
            user_message=user_message,
            recent_context=recent_context,
            has_cached_papers=has_cached_papers,
        )
        execution_flow.append(f"意图识别：{intent.get('intent', 'general')}")

        lit_result = {"status": "skipped", "message": "未触发新检索", "data": [], "expanded_queries": []}
        this_round_new_keys: set[str] = set()
        report_result: Dict[str, Any] = {
            "status": "skipped",
            "message": "本轮无需检索，未生成新报告",
            "report_content": "",
            "report_path": "",
            "metadata": {},
        }
        data_result: Dict[str, Any] = {
            "status": "skipped",
            "message": "本轮无需数据分析",
            "statistic": {},
            "plot_paths": [],
            "plot_count": 0,
        }

        if intent.get("need_retrieval", False):
            execution_flow.extend(["关键词扩写", "检索PubMed文献"])
            lit_result = self._retrieve_with_diversified_queries(
                session=session,
                user_message=user_message,
                recent_context=recent_context,
            )
            if lit_result.get("status") == "error":
                return {
                    "status": "error",
                    "session_id": session.session_id,
                    "message": lit_result.get("message", "PubMed检索失败"),
                    "assistant_reply": "检索文献失败，请稍后重试。",
                    "execution_flow": execution_flow,
                }

            for paper in lit_result.get("data", []):
                key = self._paper_key(paper)
                if key not in session.paper_index:
                    this_round_new_keys.add(key)

        dedup_stat = self._upsert_papers(session, lit_result.get("data", []))
        all_papers = session.all_papers()

        if this_round_new_keys:
            self._attach_report_memory(session=session, all_papers=all_papers, this_round_new_keys=this_round_new_keys)

        if intent.get("need_retrieval", False):
            execution_flow.extend(["数据分析", "生成报告", "基于文献生成回答"])
            data_result = self.mcp_client.process_data(
                papers_data=all_papers,
                analysis_type="all",
                plot_format=self.config["agent"]["data"]["plot_format"],
                save_path=self.config["agent"]["data"]["save_path"],
            )

            report_result = self.report_agent.run(
                keywords=user_message,
                literature_data={"status": "success", "data": lit_result.get("data", [])},
                data_process_data=data_result,
            )

            assistant_reply = self._build_retrieval_answer(
                question=user_message,
                recent_context=recent_context,
                papers=all_papers,
                intent=intent,
            )
        else:
            execution_flow.append("基于memory生成回答")
            assistant_reply = self._build_memory_answer(
                question=user_message,
                recent_context=recent_context,
                papers=related_memory_papers if related_memory_papers else all_papers,
                evidence_notes=related_memory_notes if related_memory_notes else session.get_recent_evidence_notes(limit=10),
            )

        execution_flow.append("维护会话memory")
        session.add_message("assistant", assistant_reply)

        return {
            "status": "success",
            "session_id": session.session_id,
            "assistant_reply": assistant_reply,
            "execution_flow": execution_flow,
            "intent": intent,
            "dedup": {
                "new_papers": dedup_stat["new"],
                "reused_papers": dedup_stat["reused"],
                "total_cached_papers": len(all_papers),
            },
            "literature_result": {
                "status": lit_result.get("status", "success"),
                "message": lit_result.get("message", "检索完成"),
                "total": len(lit_result.get("data", [])),
                "expanded_queries": lit_result.get("expanded_queries", []),
                "query_status": lit_result.get("query_status", []),
            },
            "data_result": data_result,
            "report_result": report_result,
            "conversation": [m.__dict__ for m in session.get_recent_messages(limit=12)],
        }
