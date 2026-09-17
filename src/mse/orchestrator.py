"""轻量 StateGraph 编排引擎 + AlgoSolver 解题流水线。

设计说明
--------
原计划（Plan.md）使用 LangGraph 的 StateGraph 管理 Analyzer -> Coder -> Tester -> Diagnose
以及 Reflexion 闭环。本环境为 Python 3.14、无 Docker、且要求依赖走国内源，
为最大化可靠性，这里用**零三方依赖**实现了一套语义等价的轻量状态图：
节点(node) + 普通边(edge) + 条件边(conditional_edges)，与 LangGraph StateGraph 一致。

流水线：
  analyzer --> coder --> tester --(通过)--> END(成功)
                          |
                       (未通过 & 有迭代次数)
                          |
                       diagnose --> coder   （Reflexion 闭环：诊断 -> 重写 -> 再测）

【多智能体闭环说明】
  1. Analyzer（分析器）：把题目描述分析成结构化规格（摘要/功能需求/边界情况/验收标准）；
  2. Coder（编码器）：根据规格写出候选 Python 代码；
  3. Tester（测试器）：把代码放进沙箱跑测试用例；
  4. 若失败且有剩余迭代次数 -> Diagnose（诊断器）进行根因分析，再回到 Coder 重写
     （这就是 Reflexion 反思闭环；iteration 计数 + 最大次数限制保证终止）。
"""
from __future__ import annotations  # 延迟注解解析，兼容向后类型注解

import re  # 正则表达式，用于从模型输出中提取代码块
from dataclasses import dataclass, field  # dataclass 简化状态类的定义
from typing import Callable, Optional  # 类型标注工具

from mse.agents.agent_manager import (  # 引入每个智能体角色的系统提示词
    ANALYZER_SYSTEM_PROMPT,
    CODER_SYSTEM_PROMPT,
    DIAGNOSE_SYSTEM_PROMPT,
)
from mse.ledger import record as _ledger  # 台账：记录每个 Agent 的输入/输出与节点结果


# ---------------- 共享状态 ----------------
@dataclass
class AlgoState:
    """解题流水线的共享状态对象。

    该对象会在状态图的每个节点之间传递，各节点通过读写这些字段进行协作。
    LangGraph 的做法与此类似：所有节点共享一个可变的状态.
    """
    problem: str = ""          # 题目原始描述（来自 problem.md）
    signature: str = ""        # 目标函数签名（来自 tests.json，例如 def two_sum(nums, target):）
    spec: str = ""             # Analyzer 产出的结构化规格说明
    code: str = ""             # Coder 产出的当前解法代码
    test_report: str = ""      # Tester（沙箱执行）产出的测试报告摘要
    diagnosis: str = ""        # Diagnose 产出的根因诊断与修改建议
    iteration: int = 0         # 当前编码迭代次数（Coder 第几次写代码）
    max_iterations: int = 4    # 允许的最大迭代次数，防止无限循环
    status: str = "pending"    # 运行状态：pending | passed | failed | error
    history: list = field(default_factory=list)  # 记录每轮测试的 {iter,status,report}，便于审计


# ---------------- 轻量状态图引擎 ----------------
class StateGraph:
    """零三方依赖的微型状态图引擎。

    概念与 LangGraph 一致：
      - 节点（node）：一个处理函数，接收 state 返回（或就地修改）state；
      - 普通边（edge）：从节点 A 无条件跳转到节点 B；
      - 条件边（conditional edge）：从节点 A 依据路由函数结果，动态跳转到某个目标节点；
      - 入口（entry）：图从哪个节点开始执行；
      - "__end__"：特殊标记，遇到它即停止执行。
    """

    def __init__(self):
        # 存储所有节点：{"analyzer": fn, "coder": fn, ...}
        self.nodes = {}
        # 普通边：节点名 -> 下一个节点名
        self.edges = {}
        # 条件边：节点名 -> (路由函数, {返回值 -> 下一个节点名})；支持 __default__ 兜底
        self.cond = {}
        # 入口节点名
        self.entry = None

    def add_node(self, name, fn):
        """注册一个节点。*name* 为节点名，*fn* 为 `fn(state, **ctx) -> state` 的处理函数。"""
        self.nodes[name] = fn
        return self  # 返回 self 以支持链式调用

    def add_edge(self, a, b):
        """添加普通边：从节点 *a* 无条件跳到节点 *b*（*b* 可为 "__end__" 结束）。"""
        self.edges[a] = b
        return self

    def add_conditional_edges(self, a, router, then_map):
        """添加条件边：从节点 *a*，调用 *router(state)* 得 key，再按 then_map 决定下一节点。"""
        self.cond[a] = (router, then_map)
        return self

    def set_entry(self, name):
        """设置图的入口节点。"""
        self.entry = name
        return self

    def run(self, state, **ctx):
        """从入口节点开始循环执行，直到遇到 "__end__" 或没有后续节点。

        *ctx 中可传共享上下文（如 provider / sandbox / problem / timeout），
        每个节点可以通过 `state=fn(state, **ctx)` 访问。
        """
        name = self.entry  # 从入口节点开始
        while name not in (None, "__end__"):  # 到达终点前持续迭代
            fn = self.nodes[name]             # 取出当前节点处理函数
            state = fn(state, **ctx) or state  # 执行节点；若节点返回空则保留原 state
            if name in self.cond:             # 若是条件边，则根据路由函数决定下一步
                router, then_map = self.cond[name]
                key = router(state)           # 路由函数读取状态并返回 key
                # 查跳转表，若 key 不存在则使用 __default__，都没有则结束
                name = then_map.get(key, then_map.get("__default__"))
            elif name in self.edges:          # 若是普通边，直接跳到下一个节点
                name = self.edges[name]
            else:                             # 没有出边 -> 结束
                name = None
        return state  # 返回最终状态


# ---------------- 节点实现 ----------------
def _now() -> float:
    """返回当前时间戳（秒），用于节点耗时统计。"""
    import time

    return time.time()


def _extract_code(text: str) -> str:
    """从模型回包文本中提取被 ```python``` 代码块包裹的 Python 源码。

    若没有代码块，则直接返回去除首尾空白的整个文本（允许模型直接输出裸代码）。"""
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)  # 非贪婪匹配第一个代码块
    if m:
        return m.group(1).strip()
    return text.strip()  # 兜底：整段当代码


def analyzer_node(state: AlgoState, **ctx) -> AlgoState:
    """分析器节点：调用 LLM 把题目描述解析为结构化规格说明。

    系统提示词固定为 ANALYZER_SYSTEM_PROMPT；用户消息中包含题目原文与函数签名。
    产出保存在 state.spec 中，供后续 Coder/Tester 使用。"""
    provider = ctx["provider"]  # 从共享上下文中取出 LLM 客户端
    problem = ctx.get("problem", {})  # 当前题目信息（用于台账关联题目名）
    msgs = [  # 构造一条 system + user 的对话消息列表
        {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "分析以下算法题，输出结构化规格（摘要 / 功能需求 / 边界情况 / 验收标准）：\n\n"
                f"{state.problem}\n\n目标函数签名：{state.signature}"
            ),
        },
    ]
    # 【台账】记录分析器的输入：发往 LLM 的完整消息列表
    _ledger("analyzer.input", problem=problem.get("name"), iteration=state.iteration, messages=msgs)
    t0 = _now()  # 计时开始
    # 调用 LLM，temperature 偏低（0.3）保证分析稳定性，最多输出 1500 tokens
    state.spec = provider.chat(msgs, temperature=0.3, max_tokens=1500)
    # 【台账】记录分析器的输出：模型返回的原始文本 + 解析后的规格
    _ledger("analyzer.output", problem=problem.get("name"), iteration=state.iteration,
            duration_s=round(_now() - t0, 3), raw=state.spec, spec=state.spec)
    return state


def coder_node(state: AlgoState, **ctx) -> AlgoState:
    """编码器节点：根据规格/诊断生成候选 Python 代码。

    分支逻辑：
      - 第一次（iteration == 1）：仅提供题目与签名，要求实现目标函数；
      - 后续迭代（iteration > 1）：附带上一版代码、诊断与测试报告，要求"修正"缺陷。
    生成结果提取代码块后存入 state.code。
    """
    provider = ctx["provider"]      # LLM 客户端
    problem = ctx.get("problem", {})  # 当前题目信息（用于台账关联题目名）
    state.iteration += 1           # 每调用一次迭代 +1
    if state.iteration == 1:       # 首次编码：只需要题目与签名
        msgs = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"题目：\n{state.problem}\n\n请实现以下函数"
                    f"（仅输出可运行的 Python 代码，用 ```python 代码块包裹）：\n{state.signature}"
                ),
            },
        ]
    else:  # 后续迭代（Reflexion）：提供历史信息以便针对性修复
        msgs = [
            {"role": "system", "content": CODER_SYSTEM_PROMPT},
            {"role": "user", "content": f"题目：\n{state.problem}\n\n目标函数：{state.signature}"},
            {"role": "assistant", "content": f"上一版代码：\n```python\n{state.code}\n```"},
            {
                "role": "user",
                "content": (
                    f"测试未通过，诊断信息如下：\n{state.diagnosis}\n\n"
                    f"测试报告：\n{state.test_report}\n\n"
                    "请修正代码，仅输出修正后的完整 Python 代码（```python 代码块）。"
                ),
            },
        ]
    # 【台账】记录编码器的输入：发往 LLM 的完整消息列表 + 当前规格/诊断上下文
    _ledger("coder.input", problem=problem.get("name"), iteration=state.iteration,
            messages=msgs, spec=state.spec, diagnosis=state.diagnosis, test_report=state.test_report)
    t0 = _now()  # 计时开始
    text = provider.chat(msgs, temperature=0.1, max_tokens=4096)  # 低温更偏确定性，限制输出长度
    state.code = _extract_code(text)  # 只保留代码块内的源码
    # 【台账】记录编码器的输出：模型原始返回 + 提取后的可执行代码
    _ledger("coder.output", problem=problem.get("name"), iteration=state.iteration,
            duration_s=round(_now() - t0, 3), raw=text, code=state.code)
    return state


def tester_node(state: AlgoState, **ctx) -> AlgoState:
    """测试器节点：在沙箱中执行候选代码跑以全部测试用例，收集结构化结果。"""
    provider = ctx["provider"]  # 本文节点不直接用 `provider`，但也从 ctx 中拿 sandbox 等
    sandbox = ctx["sandbox"]    # 沙箱执行器（LocalSandbox）
    problem = ctx["problem"]    # 当前题目信息（含 function 与 cases）
    from mse.sandbox import generate_test_runner  # 延迟导入生成测试运行器，避免循环引用

    # 生成一个"测试运行器"脚本：它 import solution，逐条运行 cases，并以 RESULT_JSON 结构输出
    runner = generate_test_runner(problem["function"], problem["cases"])
    files = {"solution.py": state.code, "test_runner.py": runner}  # 把候选代码和测试脚本写入沙箱
    timeout = ctx.get("timeout", 60)
    # 【台账】记录测试器的输入：待测代码、目标函数、用例列表、测试命令
    _ledger("tester.input", problem=problem.get("name"), iteration=state.iteration,
            function=problem["function"], cases=problem["cases"], code=state.code,
            test_command="python test_runner.py", timeout=timeout)
    t0 = _now()  # 计时开始
    result = sandbox.execute(files=files, test_command="python test_runner.py", timeout=timeout)
    state.test_report = result.summary()  # 生成给 Agent 看的简洁摘要
    if result.passed:  # 全部用例通过 -> passed
        state.status = "passed"
    else:  # 否则取沙箱分类出的失败类型（wrong_answer / timeout / runtime_error ...）
        state.status = result.status or "failed"
    # 记录本轮测试结果到历史，便于追溯
    state.history.append(
        {"iter": state.iteration, "status": state.status, "report": state.test_report}
    )
    # 【台账】记录测试器的输出：状态、退出码、逐用例结果、完整测试报告
    _ledger("tester.output", problem=problem.get("name"), iteration=state.iteration,
            duration_s=round(_now() - t0, 3), status=result.status, exit_code=result.exit_code,
            passed=result.passed, error_message=result.error_message, tests=result.tests,
            summary=state.test_report)
    return state


def diagnose_node(state: AlgoState, **ctx) -> AlgoState:
    """诊断器节点：结合题目、规格、当前代码和测试报告，让 LLM 做根因分析并给出修改建议。"""
    provider = ctx["provider"]  # 从共享上下文中取出 LLM 客户端
    problem = ctx.get("problem", {})  # 当前题目信息（用于台账关联题目名）
    # 组装完整的对话上下文给 LLM：系统提示 + 题目/规格/代码/测试报告
    msgs = [
        {"role": "system", "content": DIAGNOSE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"题目：{state.problem}\n函数签名：{state.signature}\n\n"
                f"规格：{state.spec}\n\n当前代码：\n```python\n{state.code}\n```\n\n"
                f"测试报告：\n{state.test_report}\n\n请给出根因诊断与修改建议。"
            ),
        },
    ]
    # 【台账】记录诊断器的输入：发往 LLM 的完整消息列表
    _ledger("diagnose.input", problem=problem.get("name"), iteration=state.iteration, messages=msgs)
    t0 = _now()  # 计时开始
    state.diagnosis = provider.chat(msgs, temperature=0.3, max_tokens=1500)
    # 【台账】记录诊断器的输出：模型返回的诊断文本
    _ledger("diagnose.output", problem=problem.get("name"), iteration=state.iteration,
            duration_s=round(_now() - t0, 3), raw=state.diagnosis, diagnosis=state.diagnosis)
    return state


def tester_router(state: AlgoState) -> str:
    """测试器之后的路由决策。

    返回 "__end__"（结束）或 "diagnose"（再走一轮诊断->重写）。
    判断依据：全部通过 -> 结束成功；迭代耗尽 -> 结束（状态为 failed）；否则继续诊断。
    """
    decision = "__end__"
    if state.status == "passed":
        decision = "__end__"  # 测试已通过，流水线成功结束
    elif state.iteration >= state.max_iterations:
        decision = "__end__"  # 迭代耗尽 -> 以失败状态结束（避免死循环）
    else:
        decision = "diagnose"  # 还有次数，继续 -> diagnose
    return decision


def build_graph() -> StateGraph:
    """构建解题状态图（等价于 LangGraph 的图）：

         analyzer -> coder -> tester --> "__end__"  (通过)
                                    |--> diagnose -> coder（递归，受迭代上限约束）
    """
    g = StateGraph()
    g.add_node("analyzer", analyzer_node)   # 分析器
    g.add_node("coder", coder_node)         # 编码器
    g.add_node("tester", tester_node)       # 测试器
    g.add_node("diagnose", diagnose_node)   # 诊断器
    g.set_entry("analyzer")                 # 从分析器开始
    g.add_edge("analyzer", "coder")         # 分析 -> 编码
    g.add_edge("coder", "tester")           # 编码 -> 测试
    # 测试后的分支：passed -> 结束；否则 -> 诊断（由 tester_router 决定）
    g.add_conditional_edges(
        "tester", tester_router, {"__end__": "__end__", "diagnose": "diagnose"}
    )
    g.add_edge("diagnose", "coder")         # 诊断 -> 回到编码器重写（Reflexion 闭环入口）
    return g


def solve_problem(
    problem: dict,
    provider,
    sandbox,
    max_iterations: int = 4,
    timeout: int = 60,
) -> AlgoState:
    """对单道题目运行整条流水线，返回最终状态。

    流程：构造 AlgoState -> 构建状态图 -> 运行。
    参数：
      problem:        题目字典，必须包含 statement / signature / function / cases 等字段
      provider:       LLM 客户端（任何具有 chat() 方法的对象）
      sandbox:        沙箱执行器（任何具有 execute() 方法的对象）
      max_iterations: 最大编码迭代次数（默认 4）
      timeout:        单次沙箱执行超时秒数（默认 60）
    返回值：
      AlgoState：最终状态，含 status / code / diagnosis / iteration / test_report 等。
    """
    # 初始化题目相关的状态字段
    state = AlgoState(
        problem=problem["statement"],   # 题目原文
        signature=problem["signature"], # 目标函数签名
        max_iterations=max_iterations,   # 迭代上限
    )
    # 【台账】记录单题流水线开始（含题目元信息与完整题目原文）
    _ledger("problem.pipeline_start", name=problem.get("name"),
            function=problem.get("function"), signature=state.signature,
            max_iterations=max_iterations, statement=state.problem)
    t0 = _now()  # 计时开始
    graph = build_graph()  # 构建一次流水线
    # 运行整条图，传入共享上下文：provider、sandbox、problem、timeout
    graph.run(state, provider=provider, sandbox=sandbox, problem=problem, timeout=timeout)
    # 【台账】记录单题流水线结束：最终状态 / 迭代次数 / 测试历史
    _ledger("problem.pipeline_end", name=problem.get("name"), status=state.status,
          iterations=state.iteration, duration_s=round(_now() - t0, 3),
          history=state.history, code=state.code, diagnosis=state.diagnosis)
    return state