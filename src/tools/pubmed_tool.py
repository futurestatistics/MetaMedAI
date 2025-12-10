import requests
import xml.etree.ElementTree as ET
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os
from Bio import Entrez  # Biopython简化PubMed检索

RESEARCH_METHOD_TYPES = [
    "横断面研究", "队列研究", "病例报告", "RCT研究", "病例对照研究", "综述", "其他研究"
]

class PubMedSearchInput(BaseModel):
    keywords: str = Field(description="检索关键词")
    # max_results: int = Field(default=10, description="最大检索论文数量")

class PubMedSearchTool(BaseTool):
    name = "pubmed_search"
    description = "检索PubMed数据库获取相关论文，返回题目、研究方法、结论等信息"
    args_schema = PubMedSearchInput
    max_papers: int = Field(default=10, description="最大检索论文数量")
    
    

    def __init__(self, config: Dict):
        
        Entrez.email = config["entrez_email"]  # 必须填写邮箱
        max_papers = config.get("agent", {}).get("literature", {}).get("max_papers", 10)
        # 父类初始化
        super().__init__(max_papers=max_papers)

    def _extract_publish_date(self, article: ET.Element) -> str:
        """提取发表时间（优先取电子出版日期，无则取印刷出版日期）"""
        # 解析PubDate节点
        pub_date = article.find(".//Journal/JournalIssue/PubDate")
        if not pub_date:
            pub_date = article.find(".//ArticleDate")
        
        if pub_date:
            year = pub_date.find("Year")
            month = pub_date.find("Month")
            day = pub_date.find("Day")
            parts = []
            if year is not None:
                parts.append(year.text)
            if month is not None:
                parts.append(month.text)
            if day is not None:
                parts.append(day.text)
            return "-".join(parts) if parts else "未知"
        return "未知"

    def _extract_journal_name(self, article: ET.Element) -> str:
        """提取期刊名称"""
        journal = article.find(".//Journal/Title")
        return journal.text if journal is not None else "未知"

    def _extract_abstract_sections(self, article: ET.Element) -> Dict[str, str]:
        """提取摘要各部分（方法、结论等）"""
        abstract_sections = {}
        abstract_texts = article.findall(".//Abstract/AbstractText")
        
        for section in abstract_texts:
            label = section.get("Label", "").upper()
            text = section.text.strip() if section.text else ""
            if label == "METHODS":
                abstract_sections["methods"] = text
            elif label == "CONCLUSION":
                abstract_sections["conclusion"] = text
        
        # 若无标签的摘要
        if not abstract_sections.get("methods"):
            full_abstract = " ".join([t.text.strip() for t in abstract_texts if t.text])
            abstract_sections["methods"] = full_abstract if full_abstract else "未提及"
        if not abstract_sections.get("conclusion"):
            abstract_sections["conclusion"] = "未提及"
        
        return abstract_sections

    def _run(self, keywords: str) -> Dict[str, Any]:
        """执行PubMed检索（完善字段提取）"""
        try:
            # 检索论文ID
            handle = Entrez.esearch(
                db="pubmed",
                term=keywords,
                retmax=self.max_papers,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            id_list = record["IdList"]
            
            # 数量校验
            # if len(id_list) < self.min_papers:
            #     return {
            #         "status": "warning",
            #         "message": f"仅检索到{len(id_list)}篇论文（最少需要{self.min_papers}篇），请补充关键词或扩大检索范围",
            #         "data": []
            #     }

            # 获取论文详情
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(id_list),
                rettype="xml",
                retmode="text"
            )
            xml_data = handle.read()
            handle.close()

            # 解析论文信息（完善字段）
            root = ET.fromstring(xml_data)
            papers = []
            for article in root.findall(".//PubmedArticle/MedlineCitation/Article"):
                abstract_info = self._extract_abstract_sections(article)
                
                paper = {
                    "title": article.find("ArticleTitle").text.strip() if article.find("ArticleTitle") is not None else "未知标题",
                    "publish_date": self._extract_publish_date(article),  # 发表时间
                    "journal_name": self._extract_journal_name(article),  # 期刊名
                    "methods_original": abstract_info["methods"],  # 研究方法原文
                    "conclusion": abstract_info["conclusion"],
                    "authors": [
                        f"{author.find('LastName').text} {author.find('Initials').text}" 
                        for author in article.findall("AuthorList/Author") 
                        if author.find("LastName") is not None
                    ] or ["未知作者"]
                }
                papers.append(paper)

            return {
                "status": "success",
                "message": f"成功检索到{len(papers)}篇论文",
                "data": papers
            }

        except Exception as e:
            # 异常处理
            return {
                "status": "error",
                "message": f"PubMed检索失败：{str(e)}，使用模拟数据",
                "data": [
                    {
                        "title": f"模拟论文{i}",
                        "publish_date": "2024-01-01",
                        "journal_name": "模拟期刊",
                        "methods_original": "随机对照试验（RCT）研究，选取100例患者分为实验组和对照组",
                        "conclusion": "模拟结论：该治疗方案有效",
                        "authors": ["Author A"]
                    }
                    for i in range(10)
                ]
            }