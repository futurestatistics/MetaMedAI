from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage
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

    def _init_chain(self) -> LLMChain:
        """初始化报告生成链（精准定义报告结构）"""
        # 系统提示词：严格规定报告结构和内容要求
        system_prompt = """你是专业的医学科研报告生成专家，需整合文献分析和数据处理结果，生成结构化、专业的Markdown报告。
        【核心要求】
        1. 报告必须包含以下模块（按顺序）：
           - 🔍 检索概述：包含检索关键词、文献总数、数据来源（PubMed）
           - 📑 论文详情：逐篇列出每篇论文的「题目、发表时间、期刊名称、研究背景、研究方法（原文+分类）、研究结论」
           - 📊 统计分析：展示「发表时间分布、研究方法分类分布、期刊分布、作者数量统计」
           - 📈 可视化说明：列出生成的图表路径及对应的分析维度
           - 🎯 核心结论：总结研究趋势（如哪种研究方法占比最高、发表时间趋势等）
        2. 论文详情模块要求：
           - 每篇论文单独分段，标注序号（如 1. 论文标题：XXX）
           - 研究背景：基于论文标题+研究方法+结论，提炼1-2句话的背景（无则填「未明确提及」）
           - 研究方法：同时展示原文和分类结果（如「原文：XXX | 分类：RCT研究」）
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
        【文献分析结果】：{literature_data}
        【数据处理结果】：{data_process_data}
        【检索关键词】：{keywords}

        ### 输出要求
        严格按照上述规则生成Markdown格式的科研报告，无需额外解释，直接输出报告内容。"""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            ("user", user_prompt),
        ])

        return LLMChain(
            llm=self.llm,
            prompt=prompt,
            callbacks=[self.log_handler],
            verbose=True  # 开启verbose便于调试
        )

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

    def run(self, keywords: str, literature_data: str | Dict[str, Any], data_process_data: str | Dict[str, Any]) -> Dict[str, Any]:
        """生成报告（适配Agent链输入，优化数据解析和保存）"""
        try:
            # 1. 解析输入数据（统一转为字典）
            lit_data = self._parse_input_data(literature_data)
            data_data = self._parse_input_data(data_process_data)

            # 2. 生成报告内容（传入结构化参数）
            report_content = self.chain.run({
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