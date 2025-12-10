from flask import Flask, render_template, request, jsonify
import yaml
import os
from src.chains.router_chain import ResearchRouterChain

# 初始化Flask应用
app = Flask(__name__, template_folder="templates")

@app.route("/")
def index():
    """渲染前端页面"""
    return render_template("index.html")  # 前端文件放在templates目录下

@app.route("/research", methods=["POST"])
def research():
    """处理科研检索请求（核心接口）"""
    try:
        # 1. 接收前端传递的JSON参数
        request_data = request.get_json()
        if not request_data:
            return jsonify({
                "chain_status": "failed",
                "stage": "request",
                "message": "请求参数为空，请检查输入"
            })

        # 2. 提取参数并校验
        keywords = request_data.get("keywords", "").strip()
        llm_config = request_data.get("llm_config", {})
        agent_config = request_data.get("agent_config", {})
        pubmed_config = request_data.get("pubmed_config", {})

        # 基础校验
        if not keywords:
            return jsonify({
                "chain_status": "failed",
                "stage": "params",
                "message": "检索关键词不能为空"
            })
        if not llm_config.get("api_key"):
            return jsonify({
                "chain_status": "failed",
                "stage": "params",
                "message": "LLM API Key不能为空"
            })
        if not llm_config.get("base_url"):
            return jsonify({
                "chain_status": "failed",
                "stage": "params",
                "message": "LLM Base URL不能为空"
            })
        
        if not llm_config.get("model_name"):
            return jsonify({
                "chain_status": "failed",
                "stage": "params",
                "message": "模型名称不能为空"
            })
        
        max_papers = agent_config.get("max_papers", 10)
        if not isinstance(max_papers, int) or max_papers < 1:
            return jsonify({
                "chain_status": "failed",
                "stage": "params",
                "message": "检索数量必须是大于0的整数"
            })
    
        if not pubmed_config.get("email") or "@" not in pubmed_config["email"]:
            return jsonify({
                "chain_status": "failed",
                "stage": "params",
                "message": "请输入有效的PubMed邮箱"
            })

        # 构建Agent配置
        config = {
            "llm": {
                "type": "tongyi",  # 固定为openai兼容模式
                "tongyi": {
                    "model_name": llm_config["model_name"],
                    "api_key": llm_config["api_key"],
                    "base_url": llm_config["base_url"]
                }
            },
            "agent": {
                "literature": {
                    "max_papers": agent_config["max_papers"]  
                },
                "data": {
                    "plot_format": "png",
                    "save_path": "./plots"
                },
                "report": {
                    "save_path": "./reports"
                }
            },
            "entrez_email": pubmed_config["email"]  # 传递给PubMedSearchTool
        }
        # print(config)

        # 4. 初始化并执行Agent链
        chain = ResearchRouterChain(config)
        chain_result = chain.run(keywords)

        # 5. 返回结果给前端
        return jsonify(chain_result)

    except Exception as e:
        # 全局异常捕获
        return jsonify({
            "chain_status": "failed",
            "stage": "server",
            "message": f"服务器处理失败：{str(e)}"
        })

if __name__ == "__main__":
    # 启动Flask服务（调试模式，生产环境需关闭）
    app.run(host="0.0.0.0", port=5000, debug=True)