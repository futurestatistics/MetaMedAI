from __future__ import annotations

import json
import math
import re
from typing import Any, Dict

from langchain_openai import ChatOpenAI

from src.mcp.client import MCPToolClient


class LiteratureAgent:
    """LangChain 1.0 兼容实现：通过MCP client调用工具并做轻量结构化处理。"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = self._init_llm()
        self.mcp_client = MCPToolClient()

    def _init_llm(self):
        llm_config = self.config["llm"]
        if llm_config["type"] != "tongyi":
            raise ValueError("当前LiteratureAgent仅支持tongyi配置")
        return ChatOpenAI(
            model=llm_config["tongyi"]["model_name"],
            api_key=llm_config["tongyi"]["api_key"],
            base_url=llm_config["tongyi"]["base_url"],
            temperature=0.1,
        )

    def _classify_research_method(self, method_text: str) -> str:
        text = (method_text or "").lower()
        if any(keyword in text for keyword in ["rct", "randomized controlled trial", "随机对照试验"]):
            return "RCT研究"
        if any(keyword in text for keyword in ["cohort", "队列", "前瞻性", "回顾性队列"]):
            return "队列研究"
        if any(keyword in text for keyword in ["case-control", "病例对照"]):
            return "病例对照研究"
        if any(keyword in text for keyword in ["cross-sectional", "横断面", "现况调查"]):
            return "横断面研究"
        if any(keyword in text for keyword in ["case report", "病例报告", "个案报告"]):
            return "病例报告"
        return "其他研究"

    def _strip_json_fence(self, text: str) -> str:
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            if len(parts) >= 2:
                cleaned = parts[1].replace("json", "").strip()
        return cleaned

    def _format_bullets(self, items: list[str]) -> str:
        valid = [item.strip() for item in items if isinstance(item, str) and item.strip()]
        if not valid:
            return "- 未明确提及"
        return "\n".join([f"- {item}" for item in valid])

    def _llm_standardize_papers(self, papers: list[dict]) -> list[dict]:
        if not papers:
            return papers

        payload = []
        for idx, paper in enumerate(papers):
            payload.append(
                {
                    "index": idx,
                    "title": paper.get("title", ""),
                    "methods_original": paper.get("methods_original", ""),
                    "conclusion": paper.get("conclusion", ""),
                }
            )

        prompt = f"""
                你是医学文献结构化助手。请对输入文献做标准化，输出中文分条总结并完成方法学分类。

                【输入文献】
                {json.dumps(payload, ensure_ascii=False)}

                【输出要求】
                1. 严格输出JSON数组，每个元素包含字段：
                - index: int（与输入一致）
                - background_points_zh: string[]（研究背景中文分条，2-3条）
                - methods_points_zh: string[]（研究方法中文分条，2-3条）
                - conclusion_points_zh: string[]（研究结论中文分条，2-3条）
                - limitation_points_zh: string[]（局限性中文分条，1-2条）
                - methods_classified: string（仅可选：RCT研究/队列研究/病例对照研究/横断面研究/病例报告/系统综述与Meta分析/其他研究）
                2. 输入中若是英文，必须翻译为中文后再概括。
                3. 不要输出任何解释性文本，不要使用Markdown代码块。
                """
        try:
            raw = self.llm.invoke(prompt).content
            parsed = json.loads(self._strip_json_fence(raw))
            if not isinstance(parsed, list):
                return papers

            mapped: dict[int, dict] = {
                item.get("index"): item
                for item in parsed
                if isinstance(item, dict) and isinstance(item.get("index"), int)
            }

            for idx, paper in enumerate(papers):
                item = mapped.get(idx, {})
                background_points = item.get("background_points_zh", [])
                methods_points = item.get("methods_points_zh", [])
                conclusion_points = item.get("conclusion_points_zh", [])
                limitation_points = item.get("limitation_points_zh", [])

                paper["background"] = self._format_bullets(background_points)
                paper["limitations"] = self._format_bullets(limitation_points)

                paper["methods_original_raw"] = paper.get("methods_original", "")
                paper["conclusion_raw"] = paper.get("conclusion", "")

                paper["methods_original"] = self._format_bullets(methods_points)
                paper["conclusion"] = self._format_bullets(conclusion_points)

                classified = item.get("methods_classified", "").strip()
                if classified:
                    paper["methods_classified"] = classified
                else:
                    paper["methods_classified"] = self._classify_research_method(paper.get("methods_original_raw", ""))
            return papers
        except Exception:
            return papers

    def _is_review_paper(self, paper: Dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(paper.get("title", "")),
                str(paper.get("methods_original_raw", paper.get("methods_original", ""))),
                str(paper.get("conclusion_raw", paper.get("conclusion", ""))),
            ]
        ).lower()
        markers = [
            r"\breview\b",
            r"systematic review",
            r"meta[-\s]?analysis",
            r"umbrella review",
            r"scoping review",
            r"综述",
            r"系统评价",
            r"荟萃分析",
            r"meta分析",
        ]
        return any(re.search(pattern, text) for pattern in markers)

    def _apply_review_ratio_cap(self, papers: list[dict], max_papers: int, review_ratio_cap: float) -> list[dict]:
        if not papers:
            return papers
        review_ratio_cap = min(max(float(review_ratio_cap), 0.0), 1.0)
        reviews = [paper for paper in papers if self._is_review_paper(paper)]
        non_reviews = [paper for paper in papers if not self._is_review_paper(paper)]

        allowed_review_count = int(math.floor(max_papers * review_ratio_cap))
        selected = non_reviews[:max_papers]
        remaining_slots = max(0, max_papers - len(selected))
        selected.extend(reviews[: min(allowed_review_count, remaining_slots)])

        if not selected:
            return papers[:max_papers]
        return selected[:max_papers]

    def run(
        self,
        keywords: str,
        max_papers_override: int | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        sort: str = "relevance",
    ) -> Dict[str, Any]:
        literature_cfg = self.config.get("agent", {}).get("literature", {})
        max_papers = int(max_papers_override or literature_cfg.get("max_papers", 10))
        result = self.mcp_client.search_pubmed(
            query=keywords,
            email=self.config.get("entrez_email", ""),
            max_papers=max_papers,
            start_date=start_date if start_date is not None else literature_cfg.get("start_date"),
            end_date=end_date if end_date is not None else literature_cfg.get("end_date"),
            retstart=0,
            sort=sort,
        )
        if result.get("status") not in {"success", "warning"}:
            return result

        papers = result.get("data", [])
        papers = self._llm_standardize_papers(papers)

        for paper in papers:
            if not paper.get("methods_classified"):
                paper["methods_classified"] = self._classify_research_method(paper.get("methods_original_raw", paper.get("methods_original", "")))
            if not paper.get("background"):
                paper["background"] = "- 未明确提及"
            if not paper.get("limitations"):
                paper["limitations"] = "- 未明确提及"

        review_ratio_cap = float(literature_cfg.get("review_ratio_cap", 0.3))
        papers = self._apply_review_ratio_cap(papers=papers, max_papers=max_papers, review_ratio_cap=review_ratio_cap)

        return {
            "status": result.get("status", "success"),
            "message": result.get("message", "检索完成"),
            "data": papers,
        }
