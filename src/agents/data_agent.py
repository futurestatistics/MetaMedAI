from __future__ import annotations

import json
from typing import Any, Dict

from src.mcp.client import MCPToolClient


class DataAgent:
    """LangChain 1.0 兼容实现：通过MCP client调用数据处理工具。"""

    def __init__(self, config: Dict[str, Any], llm=None):
        self.config = config
        self.llm = llm
        self.mcp_client = MCPToolClient()

    def run(self, papers_data: str | Dict[str, Any]) -> Dict[str, Any]:
        try:
            if isinstance(papers_data, dict):
                payload = papers_data
            else:
                cleaned = papers_data.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1].replace("json", "").strip()
                payload = json.loads(cleaned)

            data = payload.get("data", payload if isinstance(payload, list) else [])
            if not isinstance(data, list):
                data = []

            data_cfg = self.config.get("agent", {}).get("data", {})
            return self.mcp_client.process_data(
                papers_data=data,
                analysis_type="stat",
                plot_format=data_cfg.get("plot_format", "png"),
                save_path=data_cfg.get("save_path", "./plots"),
            )
        except Exception as exc:
            return {
                "status": "error",
                "message": f"数据Agent执行失败：{str(exc)}",
                "statistic": {},
                "plot_paths": [],
            }
