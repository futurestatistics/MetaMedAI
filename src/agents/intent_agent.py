from __future__ import annotations

import json
import re
from typing import Any, Dict


class IntentAgent:
    def __init__(self, llm):
        self.llm = llm

    def _extract_keywords(self, text: str) -> list[str]:
        words = re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower())
        stopwords = {"什么", "如何", "怎么", "请问", "研究", "趋势", "目前", "关于", "the", "and", "for", "with"}
        uniq = []
        seen = set()
        for word in words:
            if len(word) <= 1 or word in stopwords:
                continue
            if word not in seen:
                seen.add(word)
                uniq.append(word)
            if len(uniq) >= 6:
                break
        return uniq

    def _rule_based(self, user_message: str, has_cached_papers: bool) -> Dict[str, Any]:
        lower = user_message.lower()
        memory_markers = ["刚才", "上一轮", "你前面", "之前问", "what did i ask", "previous question", "回忆"]
        retrieval_markers = ["治疗", "趋势", "证据", "文献", "研究", "trial", "pubmed", "meta", "疗效", "副作用"]
        keywords = self._extract_keywords(user_message)

        if any(mark in lower for mark in memory_markers):
            return {
                "intent": "memory",
                "intent_type": "memory_query",
                "need_retrieval": False,
                "can_use_memory": True,
                "can_use_rag": False,
                "keywords": keywords,
                "expanded_queries": [],
                "reason": "问题是回顾会话内容",
            }

        if any(mark in lower for mark in retrieval_markers):
            return {
                "intent": "literature",
                "intent_type": "research_query",
                "need_retrieval": True,
                "can_use_memory": True,
                "can_use_rag": True,
                "keywords": keywords,
                "expanded_queries": [],
                "reason": "问题需要外部证据",
            }

        if has_cached_papers:
            return {
                "intent": "memory",
                "intent_type": "factual_query",
                "need_retrieval": False,
                "can_use_memory": True,
                "can_use_rag": True,
                "keywords": keywords,
                "expanded_queries": [],
                "reason": "优先使用已缓存证据与记忆",
            }

        return {
            "intent": "general",
            "intent_type": "factual_query",
            "need_retrieval": False,
            "can_use_memory": True,
            "can_use_rag": True,
            "keywords": keywords,
            "expanded_queries": [],
            "reason": "一般问答",
        }

    def _fallback(self, user_message: str, has_cached_papers: bool) -> Dict[str, Any]:
        return self._rule_based(user_message=user_message, has_cached_papers=has_cached_papers)

    def classify(self, user_message: str, recent_context: str, has_cached_papers: bool) -> Dict[str, Any]:
        rule_result = self._rule_based(user_message=user_message, has_cached_papers=has_cached_papers)
        if rule_result["intent"] in {"memory", "literature"}:
            return rule_result

        prompt = f"""
                    你是医学对话系统的意图识别器。请判断该问题是否必须触发新的PubMed检索。

                    【最近对话】
                    {recent_context}

                    【当前问题】
                    {user_message}

                    输出严格JSON：
                    {{
                    "intent": "literature|memory|general",
                    "need_retrieval": true/false,
                    "reason": "不超过25字"
                    }}

                    规则：
                    1. 若问题是“回顾之前问了什么、复述上文、解释刚刚回答”等，need_retrieval=false。
                    2. 若问题询问治疗趋势、疗效证据、最新研究、对比方案等，need_retrieval=true。
                    3. 若已有缓存文献且问题可由缓存回答，need_retrieval=false。
                """
        try:
            raw = self.llm.invoke(prompt).content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].replace("json", "").strip()
            parsed = json.loads(raw)
            intent = parsed.get("intent", "general")
            need = bool(parsed.get("need_retrieval", False))
            reason = parsed.get("reason", "")
            if intent not in {"literature", "memory", "general"}:
                intent = "general"
            intent_type_map = {
                "literature": "research_query",
                "memory": "memory_query",
                "general": "factual_query",
            }
            return {
                "intent": intent,
                "intent_type": intent_type_map.get(intent, "factual_query"),
                "need_retrieval": need,
                "can_use_memory": True,
                "can_use_rag": intent != "memory",
                "keywords": self._extract_keywords(user_message),
                "expanded_queries": [],
                "reason": reason,
            }
        except Exception:
            return rule_result
