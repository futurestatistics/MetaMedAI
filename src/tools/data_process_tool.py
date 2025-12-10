import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

class DataProcessInput(BaseModel):
    papers_data: List[Dict[str, Any]] = Field(description="文献分析师返回的论文数据")
    analysis_type: str = Field(default="stat", description="分析类型：stat(统计)/plot(可视化)")

class DataProcessTool(BaseTool):
    name = "data_process"
    description = """处理文献分析师返回的论文数据，支持：
    1. 统计分析：文献数量、研究方法分类分布、发表时间分布、期刊分布
    2. 可视化：研究方法分类饼图、发表时间趋势图、期刊分布柱状图"""
    args_schema = DataProcessInput

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
        
        # 创建保存目录（确保路径存在）
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
        """执行数据处理（适配literature_agent的输出字段）"""
        try:
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

            # 可视化
            plot_paths = []
            if analysis_type in ["plot", "all"]:
                # 研究方法分类分布饼图
                plt.figure(figsize=(10, 6))
                method_counts = df["methods_classified"].value_counts()
                # 解决空值/无数据问题
                if not method_counts.empty:
                    method_counts.plot(kind="pie", autopct="%1.1f%%", startangle=90)
                    plt.title("研究方法分类分布", fontsize=12)
                    plt.ylabel("")
                    plt.tight_layout()
                    pie_path = os.path.join(self.save_path, "methods_classified_distribution." + self.plot_format)
                    plt.savefig(pie_path, dpi=300, bbox_inches="tight")
                    plot_paths.append(pie_path)
                plt.close()

                # 发表年份分布柱状图
                plt.figure(figsize=(12, 6))
                year_counts = df["publish_year"].value_counts().sort_index()
                if not year_counts.empty:
                    year_counts.plot(kind="bar", color="#1f77b4")
                    plt.title("论文发表年份分布", fontsize=12)
                    plt.xlabel("年份")
                    plt.ylabel("论文数量")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    year_path = os.path.join(self.save_path, "publish_year_distribution." + self.plot_format)
                    plt.savefig(year_path, dpi=300, bbox_inches="tight")
                    plot_paths.append(year_path)
                plt.close()

                # 作者数量分布直方图
                plt.figure(figsize=(10, 6))
                author_counts = df["author_count"].values
                if len(author_counts) > 0:
                    plt.hist(author_counts, bins=min(10, len(author_counts)), edgecolor="black", alpha=0.7)
                    plt.title("论文作者数量分布", fontsize=12)
                    plt.xlabel("作者数量")
                    plt.ylabel("论文数量")
                    plt.tight_layout()
                    author_path = os.path.join(self.save_path, "author_count_distribution." + self.plot_format)
                    plt.savefig(author_path, dpi=300, bbox_inches="tight")
                    plot_paths.append(author_path)
                plt.close()

            return {
                "status": "success",
                "message": f"数据处理完成：共分析{len(df)}篇论文，生成{len(plot_paths)}张可视化图表",
                "statistic": stat_result,
                "plot_paths": plot_paths
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"数据处理失败：{str(e)}",
                "statistic": {},
                "plot_paths": []
            }