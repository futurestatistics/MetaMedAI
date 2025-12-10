from langchain.chains import SequentialChain
from langchain.chains.base import Chain
from typing import Dict, Any, List
import json
import yaml
from src.agents.literature_agent import LiteratureAgent
from src.agents.data_agent import DataAgent
from src.agents.report_agent import ReportAgent

class ResearchRouterChain:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.literature_agent = LiteratureAgent(config)
        self.data_agent = DataAgent(config, self.literature_agent.llm)
        self.report_agent = ReportAgent(config, self.literature_agent.llm)

    def run(self, keywords: str) -> Dict[str, Any]:
        """执行完整Agent链：文献检索/数据处理/报告生成"""
        try:
            # Step 1: 文献检索与结构化
            print("===== 【阶段1/3】检索并结构化PubMed文献 =====")
            lit_result = self.literature_agent.run(keywords)
            if lit_result["status"] not in ["success", "warning"]:
                return {
                    "chain_status": "failed",
                    "stage": "literature_agent",
                    "message": lit_result["message"],
                    "results": {}
                }

            # Step 2: 数据处理与可视化
            print("===== 【阶段2/3】处理文献数据并生成可视化 =====")
            data_result = self.data_agent.run(lit_result)  # 直接传入literature_agent的完整结果
            if data_result["status"] != "success":
                return {
                    "chain_status": "failed",
                    "stage": "data_agent",
                    "message": data_result["message"],
                    "results": {
                        "literature_result": lit_result,
                        "data_result": data_result
                    }
                }

            # Step 3: 生成最终报告
            print("===== 【阶段3/3】整合结果生成科研报告 =====")
            report_result = self.report_agent.run(
                keywords=keywords,
                literature_data=lit_result,
                data_process_data=data_result
            )

            # 整合所有结果
            final_result = {
                "chain_status": "success" if report_result["status"] == "success" else "failed",
                "stage": "report_agent" if report_result["status"] != "success" else "completed",
                "message": "完整Agent链执行完成，报告已生成",
                "results": {
                    "literature_result": lit_result,
                    "data_result": data_result,
                    "report_result": report_result
                },
                "summary": {
                    "keywords": keywords,
                    "total_papers": data_result["statistic"].get("total_papers", 0),
                    "main_research_method": max(
                        data_result["statistic"].get("methods_classified_distribution", {}),
                        key=lambda k: data_result["statistic"]["methods_classified_distribution"][k],
                        default="未知"
                    ),
                    "report_path": report_result["report_path"],
                    "plot_paths": data_result["plot_paths"]
                }
            }

            print(f"\n===== 执行完成 =====\n📄 报告路径：{report_result['report_path']}\n📊 生成图表数：{len(data_result['plot_paths'])}\n📚 分析论文数：{data_result['statistic'].get('total_papers', 0)}")
            return final_result

        except Exception as e:
            return {
                "chain_status": "failed",
                "stage": "full_chain",
                "message": f"Agent链执行失败：{str(e)}",
                "results": {}
            }

# 测试示例
if __name__ == "__main__":
    with open("./config/model_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 执行完整链
    chain = ResearchRouterChain(config)
    result = chain.run_full_chain("diabetes mellitus RCT treatment")
    
    # 打印最终结果摘要
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))