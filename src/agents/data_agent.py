from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from typing import Dict, Any
import json
from src.tools.data_process_tool import DataProcessTool
from src.callbacks.log_handler import AgentLogHandler

class DataAgent:
    def __init__(self, config: Dict[str, Any], llm):
        self.config = config
        self.llm = llm
        self.tool = DataProcessTool(config)
        self.log_handler = AgentLogHandler()
        self.agent_executor = self._init_agent()
        

    def _init_agent(self) -> AgentExecutor:
        """初始化Agent（强化Prompt+适配literature_agent输出）"""
        system_prompt = """你是专业的医学文献数据分析师，负责处理literature_agent返回的结构化论文数据。
        核心规则：
        1. 输入数据是JSON格式，包含status、message、data字段，其中data是论文列表
        2. 必须提取data字段中的论文列表，传入data_process工具进行处理
        3. 分析类型默认选择"all"（同时生成统计结果+可视化图表）
        4. 输出要求：
           - 严格返回JSON格式，包含status、message、statistic、plot_paths字段
           - statistic字段包含：总文献数、研究方法分类分布、发表年份分布、期刊分布、作者数量统计
           - plot_paths字段是可视化图表的本地路径列表
           - 禁止输出JSON以外的任何内容（如解释、备注、markdown格式）
        5. 若输入数据格式错误/为空，返回status=error并说明原因"""

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=[self.tool],
            prompt=prompt
        )

        return AgentExecutor(
            agent=agent,
            tools=[self.tool],
            callbacks=[self.log_handler],
            verbose=True,
            handle_parsing_errors=True,  # 增加解析错误处理
            return_intermediate_steps=False  # 简化输出
        )

    def run(self, papers_data: str | Dict[str, Any]) -> Dict[str, Any]:
        """执行数据处理（兼容字符串/字典输入，适配Agent链传递）"""
        try:
            # 统一将输入转为字符串（适配Agent调用格式）
            if isinstance(papers_data, dict):
                input_str = json.dumps(papers_data, ensure_ascii=False)
            else:
                # 清洗输入（去除可能的markdown包裹）
                input_str = papers_data.strip()
                if input_str.startswith("```"):
                    input_str = input_str.split("```")[1].replace("json", "").strip()
                # 验证JSON格式
                json.loads(input_str)

            # 调用Agent
            result = self.agent_executor.invoke({
                "input": f"处理以下论文数据：{input_str}，分析类型为all，生成完整的统计结果和可视化图表"
            })

            # 解析Agent输出为字典
            output_text = result["output"].strip()
            if output_text.startswith("```"):
                output_text = output_text.split("```")[1].replace("json", "").strip()
            structured_result = json.loads(output_text)

            return structured_result

        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "message": f"输入数据格式错误（非有效JSON）：{str(e)}",
                "statistic": {},
                "plot_paths": []
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"数据Agent执行失败：{str(e)}",
                "statistic": {},
                "plot_paths": []
            }