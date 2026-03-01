from __future__ import annotations

from typing import Any, Dict

from src.agents.chat_research_agent import ChatResearchAgent


class ResearchRouterChain:
    """兼容入口：保留原有类名，内部切换为会话式聊天编排。"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.chat_agent = ChatResearchAgent(config)

    def run(self, keywords: str) -> Dict[str, Any]:
        result = self.chat_agent.chat(user_message=keywords, session_id=None)
        if result.get("status") != "success":
            return {
                "chain_status": "failed",
                "stage": "chat_research_agent",
                "message": result.get("message", "执行失败"),
                "results": {},
            }

        return {
            "chain_status": "success",
            "stage": "completed",
            "message": "会话式Agent链执行完成",
            "results": {
                "literature_result": result.get("literature_result", {}),
                "data_result": result.get("data_result", {}),
                "report_result": result.get("report_result", {}),
                "assistant_reply": result.get("assistant_reply", ""),
                "conversation": result.get("conversation", []),
            },
            "summary": {
                "keywords": keywords,
                "total_papers": result.get("data_result", {}).get("statistic", {}).get("total_papers", 0),
                "report_path": result.get("report_result", {}).get("report_path", ""),
                "plot_paths": result.get("data_result", {}).get("plot_paths", []),
                "plot_count": result.get("data_result", {}).get("plot_count", 0),
                "session_id": result.get("session_id", ""),
            },
        }
