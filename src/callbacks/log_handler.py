import time
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict, List

class AgentLogHandler(BaseCallbackHandler):
    def __init__(self):
        self.logs = []
        self.start_time = None

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """链启动时记录"""
        self.start_time = time.time()
        self.logs.append({
            "type": "chain_start",
            "chain_name": serialized.get("name", "unknown"),
            "inputs": inputs,
            "timestamp": time.time()
        })

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """链结束时记录"""
        elapsed = time.time() - self.start_time
        self.logs.append({
            "type": "chain_end",
            "outputs": outputs,
            "elapsed_time": elapsed,
            "timestamp": time.time()
        })

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        """工具调用启动时记录"""
        self.logs.append({
            "type": "tool_start",
            "tool_name": serialized.get("name", "unknown"),
            "input": input_str,
            "timestamp": time.time()
        })

    def on_tool_end(self, output_str: str, **kwargs: Any) -> None:
        """工具调用结束时记录"""
        self.logs.append({
            "type": "tool_end",
            "output": output_str,
            "timestamp": time.time()
        })

    def on_error(self, error: Exception, **kwargs: Any) -> None:
        """错误时记录"""
        self.logs.append({
            "type": "error",
            "error": str(error),
            "timestamp": time.time()
        })

    def get_logs(self) -> List[Dict[str, Any]]:
        """获取所有日志"""
        return self.logs