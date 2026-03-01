from pathlib import Path
import importlib

from flask import Flask, jsonify, render_template, request
_cors_spec = importlib.util.find_spec("flask_cors")
if _cors_spec:
    CORS = importlib.import_module("flask_cors").CORS
else:
    def CORS(_app):
        return _app

from src.agents.orchestrator_agent import OrchestratorAgent
from src.chains.router_chain import ResearchRouterChain
from src.memory.layered_memory import layered_memory_store

# 初始化Flask应用
app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route("/")
def index():
    """渲染前端页面"""
    return render_template("index.html")  # 前端文件放在templates目录下

def _build_runtime_config(request_data: dict) -> dict:
    llm_config = request_data.get("llm_config", {})
    agent_config = request_data.get("agent_config", {})
    pubmed_config = request_data.get("pubmed_config", {})

    return {
        "llm": {
            "type": "tongyi",
            "tongyi": {
                "model_name": llm_config.get("model_name", "").strip(),
                "api_key": llm_config.get("api_key", "").strip(),
                "base_url": llm_config.get("base_url", "").strip(),
            },
        },
        "agent": {
            "literature": {
                "max_papers": int(agent_config.get("max_papers", 10)),
                "review_ratio_cap": float(agent_config.get("review_ratio_cap", 0.3)),
                "start_date": (agent_config.get("start_date") or "").strip() or None,
                "end_date": (agent_config.get("end_date") or "").strip() or None,
            },
            "data": {
                "plot_format": "png",
                "save_path": "./plots",
            },
            "report": {
                "save_path": "./reports",
            },
        },
        "entrez_email": pubmed_config.get("email", "").strip(),
    }


def _get_orchestrator(config: dict) -> OrchestratorAgent:
    return OrchestratorAgent(config)


def _validate_runtime_config(config: dict):
    if not config["llm"]["tongyi"]["api_key"]:
        return "LLM API Key不能为空"
    if not config["llm"]["tongyi"]["base_url"]:
        return "LLM Base URL不能为空"
    if not config["llm"]["tongyi"]["model_name"]:
        return "模型名称不能为空"
    if config["agent"]["literature"]["max_papers"] < 1:
        return "检索数量必须是大于0的整数"
    ratio_cap = config["agent"]["literature"].get("review_ratio_cap", 0.3)
    if ratio_cap < 0 or ratio_cap > 1:
        return "综述比例上限需在0到1之间"
    start_date = config["agent"]["literature"].get("start_date")
    end_date = config["agent"]["literature"].get("end_date")
    if bool(start_date) ^ bool(end_date):
        return "起始日期和结束日期需同时填写或同时留空"
    email = config["entrez_email"]
    if not email or "@" not in email:
        return "请输入有效的PubMed邮箱"
    return None


@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        request_data = request.get_json() or {}
        message = request_data.get("message", "").strip()
        session_id = request_data.get("session_id")

        if not message:
            return jsonify({"status": "error", "message": "消息不能为空"}), 400

        config = _build_runtime_config(request_data)
        config_error = _validate_runtime_config(config)
        if config_error:
            return jsonify({"status": "error", "message": config_error}), 400

        agent = _get_orchestrator(config)
        result = agent.process_message(session_id=session_id, user_message=message)
        return jsonify({
            "status": "success",
            "data": result,
            "session_id": result.get("session_id", session_id or ""),
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": f"服务器处理失败：{str(exc)}"}), 500


@app.route("/chat", methods=["POST"])
def chat_compat():
    response = api_chat()
    if isinstance(response, tuple):
        return response
    payload = response.get_json() if hasattr(response, "get_json") else None
    if not payload:
        return response
    if payload.get("status") != "success":
        return response
    data = payload.get("data", {})
    return jsonify(
        {
            "status": "success",
            "session_id": payload.get("session_id", ""),
            "assistant_reply": data.get("answer", ""),
            "citations": data.get("citations", {}),
            "search_performed": data.get("search_performed", False),
            "markdown_path": data.get("markdown_path", ""),
            "intent": data.get("intent", {}),
            "memory": data.get("memory", {}),
        }
    )


@app.route("/api/session", methods=["POST"])
def create_session():
    try:
        session_id = layered_memory_store.create_session()
        return jsonify({"status": "success", "data": {"session_id": session_id}})
    except Exception as exc:
        return jsonify({"status": "error", "message": f"创建会话失败：{str(exc)}"}), 500


@app.route("/api/session/<session_id>", methods=["GET"])
def get_session(session_id: str):
    try:
        history = {
            "session_id": session_id,
            "messages": layered_memory_store.get_session_messages(session_id, limit=100),
        }
        return jsonify({"status": "success", "data": history})
    except Exception as exc:
        return jsonify({"status": "error", "message": f"获取会话失败：{str(exc)}"}), 500


@app.route("/api/citation/<citation_id>", methods=["GET"])
def get_citation(citation_id: str):
    try:
        session_id = request.args.get("session_id")
        citation = layered_memory_store.get_citation(citation_id)
        if not citation and session_id:
            citation = layered_memory_store.get_session_citation(session_id, citation_id)
        if not citation:
            return jsonify({"status": "error", "message": "未找到引用文献"}), 404
        return jsonify({"status": "success", "data": citation})
    except Exception as exc:
        return jsonify({"status": "error", "message": f"获取引用失败：{str(exc)}"}), 500


@app.route("/api/report/<report_id>", methods=["GET"])
def get_report(report_id: str):
    try:
        report_path = layered_memory_store.get_report_path(report_id)
        if not report_path:
            candidate = Path("reports") / f"{report_id}.md"
            if candidate.exists():
                report_path = str(candidate)
            else:
                return jsonify({"status": "error", "message": "报告不存在"}), 404

        content = Path(report_path).read_text(encoding="utf-8")
        return jsonify(
            {
                "status": "success",
                "data": {
                    "report_id": report_id,
                    "markdown_content": content,
                    "papers_count": content.count("###") or 0,
                    "report_path": report_path,
                },
            }
        )
    except Exception as exc:
        return jsonify({"status": "error", "message": f"获取报告失败：{str(exc)}"}), 500


@app.route("/research", methods=["POST"])
def research():
    """兼容旧接口：将keywords作为首条消息执行一次完整链。"""
    try:
        request_data = request.get_json() or {}
        keywords = request_data.get("keywords", "").strip()
        if not keywords:
            return jsonify({
                "chain_status": "failed",
                "stage": "params",
                "message": "检索关键词不能为空",
            }), 400

        config = _build_runtime_config(request_data)
        config_error = _validate_runtime_config(config)
        if config_error:
            return jsonify({
                "chain_status": "failed",
                "stage": "params",
                "message": config_error,
            }), 400

        chain = ResearchRouterChain(config)
        return jsonify(chain.run(keywords))
    except Exception as exc:
        return jsonify({
            "chain_status": "failed",
            "stage": "server",
            "message": f"服务器处理失败：{str(exc)}",
        }), 500

if __name__ == "__main__":
    # 启动Flask服务（调试模式，生产环境需关闭）
    app.run(host="0.0.0.0", port=5000, debug=True)