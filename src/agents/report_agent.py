from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
from typing import Dict, Any
import os
from src.callbacks.log_handler import AgentLogHandler
from datetime import datetime

class ReportAgent:
    def __init__(self, config: Dict[str, Any], llm):
        self.config = config
        self.llm = llm
        self.log_handler = AgentLogHandler()
        self.chain = self._init_chain()
        self.save_path = config["agent"]["report"]["save_path"]
        os.makedirs(self.save_path, exist_ok=True)

    def _init_chain(self):
        """初始化报告生成链"""
        # 系统提示词：严格规定报告结构和内容要求
        system_prompt = """你是专业的医学科研报告生成专家，需整合文献分析和数据处理结果，生成结构化、专业的Markdown报告。
        
          【字段优先级规则（必须遵守）】
          对每篇论文，优先使用以下新字段：
          1) 研究背景：优先使用 `background`；若缺失，再根据标题+方法+结论补写；仍不足则“未明确提及”。
          2) 研究方法（中文概括）：优先使用 `methods_original`（该字段已是中文分条概括）。
          3) 研究方法原文：优先使用 `methods_original_raw`；若缺失则显示“未明确提及”。
          4) 研究结论（中文概括）：优先使用 `conclusion`（该字段已是中文分条概括）。
          5) 研究结论原文：优先使用 `conclusion_raw`；若缺失则显示“未明确提及”。
          6) 方法学分类：优先使用 `methods_classified`。
          7) 局限性：优先使用 `limitations`；若缺失则“未明确提及”。

          【核心要求】
          1. 报告必须包含以下模块（按顺序）：
              - 🔍 检索概述：包含检索关键词、文献总数、数据来源（PubMed）
              - 📑 论文详情：逐篇列出每篇论文的「题目、发表时间、期刊名称、研究背景、研究方法（中文概括+原文+分类）、研究结论（中文概括+原文）、局限性、借鉴意义、锐评」
              - 📊 统计分析：展示「发表时间分布、研究方法分类分布、期刊分布、作者数量统计」
              - 🎯 核心结论：总结研究趋势（如哪种研究方法占比最高、发表时间趋势等）
          2. 论文详情模块要求：
              - 每篇论文单独分段，标注序号（如 1. 论文标题：XXX）
              - 研究背景、研究方法、研究结论、局限性均优先直接复用输入字段，不要与原字段语义冲突
              - “研究方法”必须拆为三行：
                 - 中文概括：...
                 - 原文：...
                 - 分类：...
              - “研究结论”必须拆为两行：
                 - 中文概括：...
                 - 原文：...
              - 借鉴意义：总结该文献可落地借鉴点（1-2句）
              - 锐评：指出局限、偏倚风险或外推边界（1-2句），优先结合`limitations`
          3. 统计分析模块要求：
              - 用表格/列表形式展示分布数据，清晰易读
              - 数值保留2位小数（如作者平均数量）
          4. 格式要求：
              - 全程使用Markdown格式，标题层级清晰（一级标题#，二级##，三级###）
              - 语言专业、简洁，无冗余内容
              - 避免使用口语化表达，符合科研报告规范
          5. 数据缺失处理：
              - 若某字段为空/未知，标注「未明确提及」
              - 统计数据为空时标注「无有效数据」"""

        # 用户提示词：明确传入参数格式
        user_prompt = """### 输入数据
        【文献分析结果（含新字段）】：{literature_data}
        【数据处理结果】：{data_process_data}
        【检索关键词】：{keywords}

        ### 新字段说明
        - background：研究背景（中文分条）
        - methods_original：研究方法中文分条概括
        - methods_original_raw：研究方法原始文本（可能英文）
        - conclusion：研究结论中文分条概括
        - conclusion_raw：研究结论原始文本（可能英文）
        - methods_classified：方法学分类
        - limitations：局限性中文分条

        ### 输出要求
        严格按照系统规则生成Markdown报告；必须优先使用新字段，不要忽略。无需额外解释，直接输出报告内容。"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt),
        ])

        return prompt | self.llm | StrOutputParser()

    def _parse_input_data(self, data: str | Dict[str, Any]) -> Dict[str, Any]:
        """统一解析输入数据（兼容字符串/字典，处理JSON格式）"""
        if isinstance(data, dict):
            return data
        try:
            # 清洗可能的markdown包裹（```json ... ```）
            clean_data = data.strip()
            if clean_data.startswith("```"):
                clean_data = clean_data.split("```")[1].replace("json", "").strip()
            return json.loads(clean_data)
        except (json.JSONDecodeError, TypeError):
            # 解析失败返回空字典，让LLM处理缺失
            return {}

    def summarize_papers_for_memory(self, papers: list[dict]) -> list[dict]:
        """由report_agent统一生成文献记忆条目，供chat阶段检索使用。"""
        if not papers:
            return []

        payload = [
            {
                "title": p.get("title", ""),
                "pmid": p.get("pmid", ""),
                "doi": p.get("doi", ""),
                "publish_date": p.get("publish_date", ""),
                "journal_name": p.get("journal_name", ""),
                "methods_original": p.get("methods_original", ""),
                "conclusion": p.get("conclusion", ""),
            }
            for p in papers
        ]

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "你是医学文献总结助手。请将输入文献转为结构化摘要，便于后续memory检索。"
                "输出必须是JSON数组，每个元素包含title, summary, implication, critique。",
            ),
            ("user", "输入文献：{papers_json}"),
        ])

        chain = prompt | self.llm | StrOutputParser()
        try:
            raw = chain.invoke({"papers_json": json.dumps(payload, ensure_ascii=False)})
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("```")[1].replace("json", "").strip()
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except Exception:
            pass

        return [
            {
                "title": p.get("title", ""),
                "summary": (p.get("conclusion") or "未明确提及")[:180],
                "implication": "可作为后续研究设计与临床决策的参考证据。",
                "critique": "摘要信息有限，仍需结合全文与样本质量评估。",
            }
            for p in papers
        ]

    def run(self, keywords: str, literature_data: str | Dict[str, Any], data_process_data: str | Dict[str, Any]) -> Dict[str, Any]:
        """生成报告（适配Agent链输入，优化数据解析和保存）"""
        try:
            # 1. 解析输入数据（统一转为字典）
            lit_data = self._parse_input_data(literature_data)
            data_data = self._parse_input_data(data_process_data)

            # 2. 生成报告内容（传入结构化参数）
            report_content = self.chain.invoke({
                "keywords": keywords,
                "literature_data": json.dumps(lit_data, ensure_ascii=False, indent=2),
                "data_process_data": json.dumps(data_data, ensure_ascii=False, indent=2)
            })

            # 3. 生成唯一文件名（时间戳+关键词，避免重复）
            safe_keywords = keywords.replace(" ", "_").replace("/", "_").replace("\\", "_")[:20]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"research_report_{safe_keywords}_{timestamp}.md"
            report_path = os.path.join(self.save_path, report_filename)

            # 4. 保存报告（UTF-8编码避免中文乱码）
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            return {
                "status": "success",
                "message": f"报告生成成功，已保存至：{report_path}",
                "report_content": report_content,
                "report_path": report_path,
                "metadata": {
                    "keywords": keywords,
                    "generate_time": timestamp,
                    "total_papers": data_data.get("statistic", {}).get("total_papers", 0),
                    "plot_count": len(data_data.get("plot_paths", []))
                }
            }

        except Exception as e:
            error_msg = f"报告生成失败：{str(e)}"
            return {
                "status": "error",
                "message": error_msg,
                "report_content": "",
                "report_path": "",
                "metadata": {}
            }