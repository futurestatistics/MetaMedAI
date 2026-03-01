import pandas as pd
import numpy as np
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import ClassVar, List, Dict, Any
import os

class DataProcessInput(BaseModel):
    papers_data: List[Dict[str, Any]] = Field(description="文献分析师返回的论文数据")
    analysis_type: str = Field(default="stat", description="分析类型：stat(统计)，plot参数将被忽略")

class DataProcessTool(BaseTool):
    name: ClassVar[str] = "data_process"
    description: ClassVar[str] = """处理文献分析师返回的论文数据，支持：
    1. 统计分析：文献数量、研究方法分类分布、发表时间分布、期刊分布、作者数量统计
    2. 不生成图表，仅返回分布统计结果"""
    args_schema: ClassVar[type[BaseModel]] = DataProcessInput

    config: Dict[str, Any] = Field(default_factory=dict)  # 配置字典，默认空字典
    plot_format: str = Field(default="png")  # 图表格式
    save_path: str = Field(default="./plots")

    def __init__(self, config: Dict[str, Any]):
        # 提取配置参数
        plot_format = config["agent"]["data"]["plot_format"]
        save_path = config["agent"]["data"]["save_path"]
        
        # 调用父类初始化，传入声明的字段
        super().__init__(
            config=config,
            plot_format=plot_format,
            save_path=save_path
        )
        
        # 保持目录初始化兼容，但当前版本不写入图表
        os.makedirs(self.save_path, exist_ok=True)

    def _parse_publish_year(self, date_str: str) -> str:
        """解析发表时间为年份（用于时间分布统计）"""
        if not date_str or date_str == "未知":
            return "未知"
        try:
            # 匹配 YYYY-MM-DD / YYYY-MM / YYYY 格式
            if "-" in date_str:
                return date_str.split("-")[0]
            elif date_str.isdigit() and len(date_str) == 4:
                return date_str
            else:
                return "未知"
        except:
            return "未知"

    def _run(self, papers_data: List[Dict[str, Any]], analysis_type: str = "all") -> Dict[str, Any]:
        """执行数据处理（适配literature_agent的输出字段，仅做统计不绘图）。"""
        try:
            if not papers_data:
                return {
                    "status": "warning",
                    "message": "没有可分析的论文数据",
                    "statistic": {
                        "total_papers": 0,
                        "methods_classified_distribution": {},
                        "publish_year_distribution": {},
                        "journal_distribution": {},
                        "author_count_stat": {"avg": 0.0, "max": 0, "min": 0},
                    },
                    "plot_paths": [],
                    "plot_count": 0,
                }

            # 数据清洗（核心：匹配literature_agent的字段）
            df = pd.DataFrame(papers_data)
            # 填充缺失值
            df = df.fillna({
                "title": "未知",
                "publish_date": "未知",
                "journal_name": "未知",
                "methods_original": "未知",
                "methods_classified": "其他研究",
                "conclusion": "未知",
                "authors": []
            })
            
            # 衍生字段：发表年份、作者数量
            df["publish_year"] = df["publish_date"].apply(self._parse_publish_year)
            df["author_count"] = df["authors"].apply(lambda x: len(x) if isinstance(x, list) else 0)

            # 统计分析
            stat_result = {
                "total_papers": len(df),
                # 研究方法分类分布（匹配literature_agent的分类）
                "methods_classified_distribution": df["methods_classified"].value_counts().to_dict(),
                # 发表年份分布
                "publish_year_distribution": df["publish_year"].value_counts().to_dict(),
                # 期刊分布（取Top10）
                "journal_distribution": df["journal_name"].value_counts().head(10).to_dict(),
                # 作者数量统计
                "author_count_stat": {
                    "avg": round(np.mean(df["author_count"]), 2),
                    "max": df["author_count"].max(),
                    "min": df["author_count"].min()
                }
            }

            plot_paths = []

            return {
                "status": "success",
                "message": f"数据处理完成：共分析{len(df)}篇论文，已返回分布统计（未生成图表）",
                "statistic": stat_result,
                "plot_paths": plot_paths,
                "plot_count": len(plot_paths),
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"数据处理失败：{str(e)}",
                "statistic": {},
                "plot_paths": []
            }