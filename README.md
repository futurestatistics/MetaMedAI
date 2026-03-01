## MetaMedAI — 医学文献对话助手

项目提供一个多 agent 架构的医学文献问答与检索系统，整合会话记忆、长期 RAG 知识库与 PubMed 检索能力，面向科研/临床证据快速查询与结构化报告生成。

---

**主要特性**

- 分层记忆（L1/L2/L3）：短期对话、会话级检索记录与跨会话长期 RAG（向量化持久化）。
- 多 Agent 协作：Intent 判别、Memory 检索、Query 扩展、文献检索（PubMed via MCP）、数据处理、报告与回答生成。
- 向量后端优先使用 Chroma，失败回退为内存检索，保证可用性。
- 前端提供轻量对话界面（templates/index.html），带头像、加载提示与报告预览。

---

## 目录结构（简要）

- `app.py` — Flask 服务入口，路由与运行时配置。
- `requirements.txt` — 依赖清单（建议创建虚拟环境安装）。
- `templates/index.html` — 前端聊天界面。
- `src/agents/` — 各 Agent 实现：`orchestrator_agent.py`, `intent_agent.py`, `memory_retrieval_agent.py`, `literature_agent.py`, `data_agent.py`, `report_agent.py`, `answer_generation_agent.py` 等。
- `src/memory/` — 分层记忆实现：`l1_memory.py`, `l2_memory.py`, `l3_rag.py`, `layered_memory.py`, `conversation_memory.py`。
- `src/tools/` — 工具（如 `data_process_tool.py`, `pubmed_tool.py`），由 MCP server 暴露。
- `src/mcp/` — 本地 MCP server/client 实现，用以统一调用外部工具。
- `reports/` — 生成的 Markdown 报告保存路径。
- `data/` — 向量存储目录（运行时生成）：`vectorstore/`（L3），`l2_vectorstore/`（L2）。

---

## 设计理念

- 分层与最小惊讶：优先使用快速、会话内的 L1/L2；L3 持久化知识库用于跨会话复用；只有在必要时才发动耗时的 PubMed 检索。
- 容错优先：对外部向量库（Chroma）做优先接入，任何初始化失败都会回退到内存实现，保持服务可用。
- 可解释与可追溯：检索与报告带 metadata（session_id、search_id、report_path），并在前端能查看引用详情。
- 可扩展的 Agent 架构：各功能模块（意图识别 / 检索 / 生成 / 报告）解耦，便于替换模型或扩展工具。

---

## 快速开始（开发环境）

1. 克隆仓库并进入目录：

```bash
cd path/to/project
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

2. 配置（示例）

- 在 `app.py` 的运行配置中填入 LLM（Tongyi）的 base_url、api_key 等。
- 确保 `fastmcp`/MCP server 可用，或在本地启动 `src/mcp/server.py`（见项目 README 扩展）。

3. 运行服务：

```bash
python app.py
```

4. 在浏览器打开 `http://localhost:5000/` 使用前端界面。

---

## 主要用法与 API

- `/` — 前端页面。
- `POST /api/chat` — 发送用户消息，后端返回回答、session_id、引用与报告路径。Payload 包含 `message`, `session_id` (可空), `llm_config`, `agent_config`, `pubmed_config`。
- `GET /api/report/<id>` — 获取生成的 Markdown 报告内容（通过 `layered_memory_store.get_report_path` 注册并读取）。
- `GET /api/citation/<citation_id>` — 获取单条引用的详情（由 `LayeredMemoryStore` 管理）。
- `POST /research` 等内部接口用于批量/离线研究任务（详见 `app.py`）。

---

## 开发与调试要点

- 当出现 LangChain 与 Chroma 的 deprecation 警告，建议安装 `langchain-chroma`（已在 `requirements.txt` 中添加）。
- 启动顺序建议：1) 启动 MCP server（若使用） 2) 启动应用 3) 在前端发送请求并观察 `reports/` 与 `data/` 中的持久化文件。
- 若 Chroma 无法初始化，系统会回退到内存检索（L2/L3 均有回退策略），可在 `src/memory/l3_rag.py` 和 `src/memory/l2_memory.py` 查看实现。

---

## 注意事项

- 需要有效的 LLM 凭证与可用的 MCP 服务（用于 PubMed、数据处理等工具）。
- 在生产环境请对 `reports/` 与 `data/` 路径设置持久化存储与权限管理。
- 若需要高质量向量表示，请替换内置 `HashEmbeddings` 为更合适的 embeddings（例如 OpenAI/其他服务）。

# MetaMedAI（会话式多Agent版）

## 功能

- 聊天式医学问答：用户可直接提问（如“目前糖尿病治疗研究趋势是什么”）
- 文献检索：通过 FastMCP 本地 MCP Server 暴露 `pubmed_search` 工具
- 数据分析：通过 FastMCP MCP 工具 `data_process` 做统计与图表生成
- 记忆与上下文：按 `session_id` 维护多轮对话历史
- 去重复用：基于 DOI（缺失时回退 PMID/Title）缓存文献，避免重复抓取
- 报告生成：自动输出 Markdown 报告并在前端展示
- L3 RAG向量库：报告分块后写入 Chroma 持久化目录 `data/vectorstore/`

## 接口

- `POST /api/chat`：多轮会话主接口（新）
- `POST /api/session`：创建新会话
- `GET /api/session/<session_id>`：获取会话历史
- `GET /api/citation/<citation_id>`：获取引用详情
- `GET /api/report/<report_id>`：获取报告Markdown内容
- `POST /chat`：兼容旧接口
- `POST /research`：兼容旧的一次性关键词入口

## 启动

1. 安装依赖
   - `pip install -r requirements.txt`
2. 运行服务
   - `python app.py`
3. 打开网页
   - `http://127.0.0.1:5000`
