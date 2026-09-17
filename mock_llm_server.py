"""本地 Mock LLM 服务（OpenAI 兼容），仅用于离线验证 AlgoSolver 全流程。

【作用】
它复用真实的 LLMProvider 代码路径（POST /v1/chat/completions），因此端到端
跑通即证明：Analyzer->Coder->Tester->(Diagnose->Coder) 编排、沙箱执行、测试
解析、报告生成全部正确。等 10.0.131.251:9005 真实服务恢复后，一键脚本无需改动
即可切回真实模型。

【用法】（在 WSL 项目目录内）
  python mock_llm_server.py --port 8911
  # 对 two_sum 首次注入错误以验证 Reflexion 闭环：
  MSE_MOCK_FAULT_FN=two_sum python mock_llm_server.py --port 8911

然后运行：
  python src/main.py --problems problems --base-url http://127.0.0.1:8911/v1 --output mock_report.json

【实现要点】
  - 使用 Python 标准库 http.server 的 ThreadingHTTPServer 支持并发请求；
  - 根据系统提示词识别当前请求属于哪个智能体（Analyzer/Coder/Diagnose/Tester），
    再分别返回模拟回复；
  - 对 Coder 请求返回"参考解"（正确代码），实现"全部通过"链路验证；
  - 通过环境变量 MSE_MOCK_FAULT_FN 指定某个函数首次故意返回错误代码，
    从而验证"测试未通过 -> 诊断 -> 重写 -> 通过"的 Reflexion 闭环。
"""
from __future__ import annotations  # 延迟注解解析

import argparse  # 解析 --host / --port 参数
import json      # 解析请求体、构造 JSON 响应
import re        # 从消息内容中提取函数名
import threading  # （ThreadingHTTPServer 内部使用，此处为依赖之一）
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # 基础 HTTP 服务

# ---------------- 参考解（已知正确的实现，供返回给"编码器"）----------------
# 每道题给出一份最小可用的正确答案，让 Mock 服务在离线环境下也能让测试通过。
REFERENCE = {
    "two_sum": '''def two_sum(nums, target):
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
''',
    "is_valid": '''def is_valid(s):
    stack = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack
''',
    "climb_stairs": '''def climb_stairs(n):
    if n <= 1:
        return 1
    a, b = 1, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
''',
    "max_sub_array": '''def max_sub_array(nums):
    best = cur = nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best
''',
    "reverse": '''def reverse(x):
    sign = -1 if x < 0 else 1
    s = str(abs(x))[::-1].lstrip("0")
    if not s:
        return 0
    val = sign * int(s)
    if val < -2 ** 31 or val > 2 ** 31 - 1:
        return 0
    return val
''',
    "search": '''def search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
}

# 故障注入：对指定函数首次 Coder 调用返回错误实现（用于验证 Reflexion 闭环）
FAULT_FN = ""       # 全局：由环境变量 MSE_MOCK_FAULT_FN 指定要注入错误的函数名
_ATTEMPTS = {}      # 记录每个函数被 Coder 请求的次数（用于"仅首次注入错误"）


def _fn_from_messages(messages):
    """从聊天消息里尝试提取"def 函数名("，返回函数名（找不到返回 None）。"""
    for m in reversed(messages):  # 从最新一条消息开始（用户消息里一般包含函数名）
        c = m.get("content", "")
        m2 = re.search(r"def\s+(\w+)\s*\(", c)  # 匹配 "def demo("
        if m2:
            return m2.group(1)
    return None


def _role_from_messages(messages):
    """根据系统提示词内容判断当前请求属于哪个智能体角色。

    规则：系统消息里若包含 "Analyzer Agent" 则识别为 Analyzer，依此类推；
    都没有则返回 "Other"。
    """
    if messages and messages[0].get("role") == "system":  # 第一条应为系统消息
        sys = messages[0].get("content", "")
        for name, key in (("Analyzer", "Analyzer Agent"), ("Coder", "Coder Agent"),
                          ("Diagnose", "Diagnose Agent"), ("Tester", "Tester Agent")):
            if key in sys:  # 命中则返回角色名
                return name
    return "Other"  # 兜底

def _coder_reply(messages):
    """给"编码器"智能体的模拟回复：返回参考答案（可注入一次错误）。"""
    fn = _fn_from_messages(messages)   # 提取被测函数名
    if fn and fn in REFERENCE:          # 若函数在参考答案库中
        cnt = _ATTEMPTS.get(fn, 0) + 1  # 该函数被请求的次数 +1
        _ATTEMPTS[fn] = cnt             # 记录
        code = REFERENCE[fn]            # 取参考答案
        if FAULT_FN and fn == FAULT_FN and cnt == 1:
            # 注入错误（首次只对指定函数）：把下标对调，制造 wrong_answer
            code = code.replace("return [i, j]", "return [j, i]")
            code = code.replace("return [0, 1]", "return [1, 0]")
        return "```python\n" + code + "```"   # 包成 Python 代码块返回
    # 未知函数：返回一个占位实现（会触发 import_error -> 诊断）
    return "```python\ndef solution_placeholder():\n    raise NotImplementedError()\n```"


def _analyzer_reply(messages):
    """给"分析器"的模拟回复：结构化规格文本。"""
    return ("[Mock Analyzer]\nSummary: 解析题目并给出结构化规格。\n"
            "Functional Requirements: 实现目标函数，满足所有示例与边界用例。\n"
            "Edge Cases: 空输入、单元素、无解、极端值。\n"
            "Acceptance Criteria: 所有测试用例通过。")


def _diagnose_reply(messages):
    """给"诊断器"的模拟回复：根因分析与建议。"""
    return ("[Mock Diagnose]\nIssue Summary: 测试未通过。\nRoot Cause: 上一版代码逻辑有误。\n"
            "Affected Component: Coder output.\nRecommended Action: 修正下标/边界处理，重新生成正确实现。")


class Handler(BaseHTTPRequestHandler):
    """HTTP 请求处理器：模拟 OpenAI 兼容的 /v1/models 与 /v1/chat/completions。"""

    def log_message(self, *a):
        pass  # 静默，避免控制台刷屏

    def _send(self, status, obj):
        """统一的响应发送：把字典序列化为 JSON，设置 Content-Type/Length 后写出。"""
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)             # 状态码
        self.send_header("Content-Type", "application/json")  # 内容类型
        self.send_header("Content-Length", str(len(body)))   # 长度（必须有，否则连接不结束）
        self.end_headers()
        self.wfile.write(body)                 # 写入响应体

    def do_GET(self):
        """GET 处理：/v1/models（模型列表）与 /v1/health、/health、/（健康检查）。"""
        if self.path.rstrip("/") in ("/v1/models",):   # 模型列表接口
            self._send(200, {"object": "list", "data": [{"id": "mock-model"}]})
        elif self.path.rstrip("/") in ("/v1/health", "/health", ""):  # 健康检查
            self._send(200, {"status": "ok"})
        else:                                    # 未知 GET 一律返回 ok（兼容探测）
            self._send(200, {"status": "ok"})

    def do_POST(self):
        """POST 处理：解析 body -> 根据角色返回模拟 chat completion 响应。"""
        length = int(self.headers.get("Content-Length", 0))  # 读取请求体长度
        raw = self.rfile.read(length) if length else b"{}"   # 读取请求体
        try:
            payload = json.loads(raw.decode("utf-8", "replace"))  # 解析请求体 JSON
        except Exception:
            payload = {}  # 解析失败按空对象处理
        if self.path.rstrip("/") in ("/v1/chat/completions", "/chat/completions"):
            messages = payload.get("messages", [])   # 聊天消息列表
            role = _role_from_messages(messages)     # 识别角色
            if role == "Coder":                      # 编码器 -> 参考答案（可注入错误）
                content = _coder_reply(messages)
            elif role == "Analyzer":                 # 分析器 -> 规格文本
                content = _analyzer_reply(messages)
            elif role == "Diagnose":                 # 诊断器 -> 诊断文本
                content = _diagnose_reply(messages)
            else:                                    # 其它角色 -> 普通占位
                content = "[mock] ok"
            # 按 OpenAI chat.completion 格式构造响应
            self._send(200, {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })
        else:  # 未知 POST 路径 -> 404
            self._send(404, {"error": "not found"})


def main():
    """启动 Mock LLM 服务。"""
    ap = argparse.ArgumentParser()                 # 命令行参数解析
    ap.add_argument("--host", default="127.0.0.1")  # 监听地址
    ap.add_argument("--port", type=int, default=8911)  # 监听端口
    args = ap.parse_args()
    global FAULT_FN                               # 修改模块级故障注入变量
    FAULT_FN = os_environ("MSE_MOCK_FAULT_FN", "")  # 从环境变量读取要注入错误的函数名
    # 创建多线程 HTTP 服务器（能处理并发请求，避免串行阻塞）
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[mock-llm] listening on http://{args.host}:{args.port}/v1  "
          f"fault_fn={FAULT_FN or '(none)'}", flush=True)
    httpd.serve_forever()                            # 常驻监听

def os_environ(k, d=""):
    """安全的环境变量读取（兼容老 Python 的写法，避免多次 import os）。"""
    import os
    return os.environ.get(k, d)


if __name__ == "__main__":  # 该脚本作为主程序运行时才进入
    main()