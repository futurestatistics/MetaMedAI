from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
import json
from typing import Dict, Any
from src.tools.pubmed_tool import PubMedSearchTool, RESEARCH_METHOD_TYPES
from src.callbacks.log_handler import AgentLogHandler

class LiteratureAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = self._init_llm()
        self.tool = PubMedSearchTool(config)
        self.log_handler = AgentLogHandler()
        self.agent_executor = self._init_agent()
        

    def _init_llm(self) -> OpenAI:
        """初始化LLM"""
        llm_config = self.config["llm"]
        if llm_config["type"] == "tongyi":
            return ChatOpenAI(
                model = llm_config["tongyi"]["model_name"],
                api_key=llm_config["tongyi"]["api_key"],
                base_url=llm_config["tongyi"]["base_url"],
                temperature=0.1
            )
        elif llm_config["type"] == "zhipu":
            from langchain_community.chat_models import ChatZhipuAI
            return ChatZhipuAI(
                api_key=llm_config["zhipu"]["api_key"],
                model=llm_config["zhipu"]["model"],
                temperature=0
            )
        elif llm_config["type"] == "chatglm":
            return OpenAI(
                openai_api_base=llm_config["chatglm"]["api_base"],
                openai_api_key=llm_config["chatglm"]["api_key"],
                model_name=llm_config["chatglm"]["model"],
                temperature=0
            )
        raise ValueError(f"不支持的LLM类型：{llm_config['type']}")

    def _classify_research_method(self, method_text: str) -> str:
            """辅助函数：研究方法分类（可单独调用或嵌入prompt）"""
            method_text_lower = method_text.lower()
            if any(keyword in method_text_lower for keyword in ["rct", "randomized controlled trial", "随机对照试验"]):
                return "RCT研究"
            elif any(keyword in method_text_lower for keyword in ["cohort", "队列", "前瞻性", "回顾性队列"]):
                return "队列研究"
            elif any(keyword in method_text_lower for keyword in ["case-control", "病例对照"]):
                return "病例对照研究"
            elif any(keyword in method_text_lower for keyword in ["cross-sectional", "横断面", "现况调查"]):
                return "横断面研究"
            elif any(keyword in method_text_lower for keyword in ["case report", "病例报告", "个案报告"]):
                return "病例报告"
            else:
                return "其他研究"

    def _init_agent(self) -> AgentExecutor:
        """初始化Agent（强化prompt+强制结构化输出）"""
        # 系统提示词：明确要求+研究方法分类+JSON格式
        system_prompt = f"""你是专业的循证医学文献分析师，严格按照以下要求处理PubMed论文数据：
        核心要求：
        1. 调用pubmed_search工具获取论文原始数据，确保数量不多于{self.tool.max_papers}篇
        2. 对每篇论文提取以下字段并结构化：
           - title: 论文题目
           - publish_date: 发表时间
           - journal_name: 期刊名称
           - methods_original: 研究方法原文
           - methods_classified: 研究方法分类（必须从以下列表选择：{', '.join(RESEARCH_METHOD_TYPES)}）
           - conclusion: 研究结论
           - authors: 作者列表
        3. 研究方法分类规则：
           - RCT研究：包含随机对照试验/RCT/randomized controlled trial
           - 队列研究：包含队列/cohort/前瞻性/回顾性队列
           - 病例对照研究：包含病例对照/case-control
           - 横断面研究：包含横断面/cross-sectional/现况调查
           - 病例报告：包含病例报告/case report/个案报告
           - 其他研究：不符合以上类型的均归为此类
        4. 输出格式必须是严格的JSON，包含3个顶级字段：
           - status: success/warning/error
           - message: 结果说明
           - data: 论文列表（每篇包含上述所有字段）
        5. 禁止输出JSON以外的任何内容（如解释、备注等）
        """

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
            handle_parsing_errors=True  # 增加解析错误处理
        )

    def run(self, keywords: str) -> Dict[str, Any]:
        """执行文献分析（修复返回值+JSON解析）"""
        try:
            result = self.agent_executor.invoke({
                "input": f"检索关键词为：{keywords}，严格按照要求返回结构化JSON结果"
            })
            
            # 解析LLM输出的JSON字符串为字典
            output_text = result["output"].strip()
            # 去除可能的markdown包裹（如```json ... ```）
            if output_text.startswith("```"):
                output_text = output_text.split("```")[1].replace("json", "").strip()
            
            structured_result = json.loads(output_text)
            return structured_result

        except json.JSONDecodeError as e:
            return {
                "status": "error",
                "message": f"结果解析失败：{str(e)}，原始输出：{result['output']}",
                "data": []
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Agent执行失败：{str(e)}",
                "data": []
            }