from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from fastmcp import Client
from src.mcp.server import mcp as local_mcp_server


class MCPToolClient:
    def __init__(self, server_source: Any | None = None):
        self.server_source = server_source or local_mcp_server

    async def _call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        async with Client(self.server_source) as client:
            result = await client.call_tool(name=name, arguments=arguments)
        return self._normalize_result(result)

    def _normalize_result(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            return result

        if hasattr(result, "data"):
            data = result.data
            if isinstance(data, dict):
                return data
            if isinstance(data, list):
                if len(data) == 1 and isinstance(data[0], dict):
                    return data[0]
                return {"status": "success", "data": data}
            if isinstance(data, str):
                try:
                    parsed = json.loads(data)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return {"status": "success", "message": data}

        if hasattr(result, "content") and isinstance(result.content, list):
            texts = []
            for item in result.content:
                text = getattr(item, "text", None)
                if text:
                    texts.append(text)
            if texts:
                joined = "\n".join(texts)
                try:
                    parsed = json.loads(joined)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return {"status": "success", "message": joined}

        return {"status": "error", "message": f"无法解析MCP返回结果: {result}"}

    def search_pubmed(
        self,
        query: str,
        email: str,
        max_papers: int,
        start_date: str | None = None,
        end_date: str | None = None,
        retstart: int = 0,
        sort: str = "relevance",
    ) -> Dict[str, Any]:
        try:
            return asyncio.run(
                self._call_tool(
                    "pubmed_search",
                    {
                        "query": query,
                        "email": email,
                        "max_papers": max_papers,
                        "start_date": start_date,
                        "end_date": end_date,
                        "retstart": retstart,
                        "sort": sort,
                    },
                )
            )
        except Exception as exc:
            message = str(exc)
            if "Connection closed" in message:
                raise RuntimeError(
                    "MCP工具连接中断（Connection closed）。常见原因：1）依赖缺失（如 biopython）；"
                    "2）MCP server 初始化异常。请先执行 `python -m pip install -r requirements.txt`。"
                ) from exc
            raise

    def process_data(self, papers_data: List[Dict[str, Any]], analysis_type: str = "all", plot_format: str = "png", save_path: str = "./plots") -> Dict[str, Any]:
        try:
            return asyncio.run(
                self._call_tool(
                    "data_process",
                    {
                        "papers_data": papers_data,
                        "analysis_type": analysis_type,
                        "plot_format": plot_format,
                        "save_path": save_path,
                    },
                )
            )
        except Exception as exc:
            message = str(exc)
            if "Connection closed" in message:
                raise RuntimeError(
                    "MCP工具连接中断（Connection closed）。请检查依赖安装和MCP server初始化日志。"
                ) from exc
            raise
