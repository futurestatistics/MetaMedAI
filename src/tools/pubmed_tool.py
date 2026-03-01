import xml.etree.ElementTree as ET
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import ClassVar, Dict, Any
from Bio import Entrez  # Biopython简化PubMed检索

RESEARCH_METHOD_TYPES = [
    "横断面研究", "队列研究", "病例报告", "RCT研究", "病例对照研究", "综述", "其他研究"
]

class PubMedSearchInput(BaseModel):
    keywords: str = Field(description="检索关键词")
    start_date: str | None = Field(default=None, description="发表起始日期，格式YYYY/MM/DD")
    end_date: str | None = Field(default=None, description="发表结束日期，格式YYYY/MM/DD")
    retstart: int = Field(default=0, description="结果偏移量")
    sort: str = Field(default="relevance", description="排序方式：relevance/pub date")

class PubMedSearchTool(BaseTool):
    name: ClassVar[str] = "pubmed_search"
    description: ClassVar[str] = "检索PubMed数据库获取相关论文，返回题目、研究方法、结论等信息"
    args_schema: ClassVar[type[BaseModel]] = PubMedSearchInput
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

    def _normalize_pubmed_date(self, date_text: str | None) -> str | None:
        if not date_text:
            return None
        return date_text.replace("-", "/")

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

    def _extract_article_id(self, medline_citation: ET.Element, pubmed_article: ET.Element, id_type: str) -> str:
        if id_type == "pmid":
            pmid = medline_citation.find("PMID")
            return pmid.text.strip() if pmid is not None and pmid.text else ""

        for node in pubmed_article.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if node.get("IdType", "").lower() == id_type and node.text:
                return node.text.strip()
        return ""

    def _run(
        self,
        keywords: str,
        start_date: str | None = None,
        end_date: str | None = None,
        retstart: int = 0,
        sort: str = "relevance",
    ) -> Dict[str, Any]:
        """执行PubMed检索（完善字段提取）"""
        try:
            search_kwargs = {
                "db": "pubmed",
                "term": keywords,
                "retmax": self.max_papers,
                "retstart": max(0, retstart),
                "sort": sort,
            }
            norm_start_date = self._normalize_pubmed_date(start_date)
            norm_end_date = self._normalize_pubmed_date(end_date)
            if norm_start_date and norm_end_date:
                search_kwargs["datetype"] = "pdat"
                search_kwargs["mindate"] = norm_start_date
                search_kwargs["maxdate"] = norm_end_date

            # 检索论文ID
            handle = Entrez.esearch(**search_kwargs)
            record = Entrez.read(handle)
            handle.close()
            id_list = record["IdList"]

            if not id_list:
                return {
                    "status": "warning",
                    "message": "未检索到文献，请调整关键词或时间范围",
                    "data": [],
                }
            
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
            for pubmed_article in root.findall(".//PubmedArticle"):
                medline_citation = pubmed_article.find("MedlineCitation")
                article = medline_citation.find("Article") if medline_citation is not None else None
                if article is None:
                    continue

                abstract_info = self._extract_abstract_sections(article)
                title_node = article.find("ArticleTitle")
                title_text = "未知标题"
                if title_node is not None:
                    title_text = "".join(title_node.itertext()).strip() or "未知标题"

                doi = self._extract_article_id(medline_citation, pubmed_article, "doi")
                pmid = self._extract_article_id(medline_citation, pubmed_article, "pmid")
                
                paper = {
                    "title": title_text,
                    "publish_date": self._extract_publish_date(article),  # 发表时间
                    "journal_name": self._extract_journal_name(article),  # 期刊名
                    "methods_original": abstract_info["methods"],  # 研究方法原文
                    "conclusion": abstract_info["conclusion"],
                    "doi": doi,
                    "pmid": pmid,
                    "source": "PubMed",
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
                "query": keywords,
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