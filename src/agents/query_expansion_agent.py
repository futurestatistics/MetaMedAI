from __future__ import annotations

import json
from typing import List


class QueryExpansionAgent:
    def __init__(self, llm):
        self.llm = llm

    def expand(self, user_message: str, recent_context: str, max_queries: int = 4) -> List[str]:
        prompt = f"""
                将用户医学问题扩展为适合PubMed的多样英文检索短语，避免同质化。

                【最近对话】
                {recent_context}

                【用户问题】
                {user_message}

                要求：
                1. 输出2-{max_queries}个短语，每个3-8个词。
                2. 需覆盖：疾病词、干预词、结局词、研究类型词（如 RCT / cohort）。
                3. 避免完全同义重复。
                4. 严格输出JSON：{{"queries": ["...", "..."]}}
                """
        try:
            raw = self.llm.invoke(prompt).content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].replace("json", "").strip()
            parsed = json.loads(raw)
            queries = [q.strip() for q in parsed.get("queries", []) if isinstance(q, str) and q.strip()]
            deduped = []
            seen = set()
            for query in queries:
                key = query.lower()
                if key not in seen:
                    seen.add(key)
                    deduped.append(query)
            return deduped[:max_queries] if deduped else [user_message]
        except Exception:
            return [user_message]
