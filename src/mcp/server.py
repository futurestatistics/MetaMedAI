from __future__ import annotations

from typing import Any, Dict, List

from fastmcp import FastMCP

from src.tools.data_process_tool import DataProcessTool
from src.tools.pubmed_tool import PubMedSearchTool

mcp = FastMCP("medical_research_tools")


def _build_config(email: str, max_papers: int = 10, plot_format: str = "png", save_path: str = "./plots") -> Dict[str, Any]:
    return {
        "entrez_email": email,
        "agent": {
            "literature": {"max_papers": max_papers},
            "data": {"plot_format": plot_format, "save_path": save_path},
        },
    }


@mcp.tool
def pubmed_search(
    query: str,
    email: str,
    max_papers: int = 10,
    start_date: str | None = None,
    end_date: str | None = None,
    retstart: int = 0,
    sort: str = "relevance",
) -> Dict[str, Any]:
    """检索PubMed文献并返回结构化结果。"""
    tool = PubMedSearchTool(_build_config(email=email, max_papers=max_papers))
    return tool._run(
        keywords=query,
        start_date=start_date,
        end_date=end_date,
        retstart=retstart,
        sort=sort,
    )


@mcp.tool
def data_process(papers_data: List[Dict[str, Any]], analysis_type: str = "all", plot_format: str = "png", save_path: str = "./plots") -> Dict[str, Any]:
    """对论文列表进行统计分析（仅返回分布统计，不生成图表）。"""
    config = _build_config(email="local@localhost", plot_format=plot_format, save_path=save_path)
    tool = DataProcessTool(config)
    return tool._run(papers_data=papers_data, analysis_type=analysis_type)


if __name__ == "__main__":
    mcp.run()
