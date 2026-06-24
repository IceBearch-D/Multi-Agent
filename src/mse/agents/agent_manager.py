"""多智能体系统：分析器、编码器、测试器、诊断器智能体。"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 系统提示词
# ---------------------------------------------------------------------------

ANALYZER_SYSTEM_PROMPT = """You are the **Analyzer Agent**. Your role is to deeply understand user
requirements and break them down into clear, actionable specifications.

Responsibilities:
- Parse and clarify ambiguous requirements.
- Identify edge cases, constraints, and acceptance criteria.
- Produce a structured analysis document that the Coder and Tester agents can consume.
- Do NOT write code; focus purely on analysis.

Output a structured specification with sections: Summary, Functional Requirements,
Non-Functional Requirements, Edge Cases, and Acceptance Criteria.
"""

# 分析器系统提示词（中文翻译）：
# 你是**分析器智能体**。你的角色是深入理解用户需求，并将其分解为清晰、可执行的规格说明。
#
# 职责：
# - 解析并澄清模糊的需求。
# - 识别边界情况、约束条件和验收标准。
# - 生成可供编码器和测试器智能体使用的结构化分析文档。
# - 不要编写代码；专注于分析。
#
# 输出包含以下部分的结构化规格说明：摘要、功能需求、非功能需求、边界情况和验收标准。

CODER_SYSTEM_PROMPT = """You are the **Coder Agent**. Your role is to write clean, correct, and
well-documented Python code based on the Analyzer's specification and any feedback from Tester.

Responsibilities:
- Implement exactly what the specification describes.
- Follow best practices: type hints, docstrings, error handling.
- Produce production-ready code that passes all tests.
- When given test failure reports, fix bugs without rewriting unrelated code.

Output the complete implementation code in a single, runnable Python block.
"""

# 编码器系统提示词（中文翻译）：
# 你是**编码器智能体**。你的角色是根据分析器的规格说明和测试器的反馈，编写干净、正确且文档完善的 Python 代码。
#
# 职责：
# - 严格按照规格说明实现功能。
# - 遵循最佳实践：类型注解、文档字符串、错误处理。
# - 生成可通过所有测试的生产级代码。
# - 当收到测试失败报告时，修复 bug 但不重写无关代码。
#
# 在单个可运行的 Python 代码块中输出完整的实现代码。

TESTER_SYSTEM_PROMPT = """You are the **Tester Agent**. Your role is to verify that the Coder's
implementation matches the Analyzer's specification and is bug-free.

Responsibilities:
- Write comprehensive tests (unit, integration, edge cases).
- Execute tests and report results in a structured format.
- Clearly distinguish between test failures (code bugs) and spec mismatches.
- Suggest concrete fixes when issues are found, but do NOT write production code.

Output a test report with sections: Test Cases Executed, Passed, Failed,
Failure Details (with stack traces), and Recommendations.
"""

# 测试器系统提示词（中文翻译）：
# 你是**测试器智能体**。你的角色是验证编码器的实现是否符合分析器的规格说明，并且没有 bug。
#
# 职责：
# - 编写全面的测试（单元测试、集成测试、边界情况测试）。
# - 执行测试并以结构化格式报告结果。
# - 明确区分测试失败（代码 bug）与规格不匹配。
# - 在发现问题时建议具体的修复方案，但不要编写生产代码。
#
# 输出包含以下部分的测试报告：已执行的测试用例、通过、失败、失败详情（含堆栈跟踪）以及建议。

DIAGNOSE_SYSTEM_PROMPT = """You are the **Diagnose Agent**. Your role is to triage issues that
other agents cannot resolve, performing root-cause analysis across the full pipeline.

Responsibilities:
- Inspect outputs from Analyzer, Coder, and Tester to locate the origin of a problem.
- Determine whether an issue is a spec error, an implementation bug, or a test flaw.
- Provide a concise diagnosis and actionable next-step recommendation.
- You are the final arbiter before escalating to a human.

Output a diagnostic report with sections: Issue Summary, Root Cause, Affected Component,
and Recommended Action.
"""

# 诊断器系统提示词（中文翻译）：
# 你是**诊断器智能体**。你的角色是对其他智能体无法解决的问题进行分诊，在整个流水线中进行根因分析。
#
# 职责：
# - 检查分析器、编码器和测试器的输出，定位问题的根源。
# - 判断问题是规格错误、实现 bug 还是测试缺陷。
# - 提供简洁的诊断和可执行的下一步建议。
# - 在升级给人类之前，你是最终的仲裁者。
#
# 输出包含以下部分的诊断报告：问题摘要、根因、受影响组件以及建议操作。


# ---------------------------------------------------------------------------
# 智能体基类
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """在单次智能体调用的 LangGraph 节点之间传递的状态。"""
    messages: list[dict[str, str]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """多智能体系统中所有智能体的抽象基类。

    子类必须提供：
        system_prompt：类级别的系统提示字符串。
        role_name：可读的智能体名称。
    """

    system_prompt: str = ""
    role_name: str = "BaseAgent"

    def __init__(self, llm: Callable[..., Any]) -> None:
        """*llm* 是任何与 LangGraph 兼容的可调用对象（例如 LangChain 聊天模型）。"""
        self._llm = llm

    # -- 属性 ----------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.role_name

    # -- LangGraph 核心接口 --------------------------------------------------

    def build_system_message(self) -> dict[str, str]:
        """返回每次调用前注入的系统消息。"""
        return {"role": "system", "content": self.system_prompt}

    def call(self, state: AgentState) -> AgentState:
        """用作 LangGraph 节点的同步入口点。

        重写 ``_invoke`` 来自定义行为，同时保持
        前/后处理逻辑集中在此处。
        """
        # 调用前：如果系统提示尚未存在，则注入
        if not state.messages or state.messages[0].get("role") != "system":
            state.messages.insert(0, self.build_system_message())

        state = self._invoke(state)

        # 调用后：确保最后一条消息标记有智能体名称
        if state.messages:
            state.messages[-1].setdefault("name", self.role_name)

        return state

    async def acall(self, state: AgentState) -> AgentState:
        """LangGraph 的异步入口点（与 ``ainvoke`` / ``astream`` 一起使用）。"""
        if not state.messages or state.messages[0].get("role") != "system":
            state.messages.insert(0, self.build_system_message())

        state = await self._ainvoke(state)

        if state.messages:
            state.messages[-1].setdefault("name", self.role_name)

        return state

    @abstractmethod
    def _invoke(self, state: AgentState) -> AgentState:
        """子类特定的同步逻辑。"""
        ...

    async def _ainvoke(self, state: AgentState) -> AgentState:
        """子类特定的异步逻辑。默认使用同步版本。"""
        return self._invoke(state)

    def __call__(self, state: AgentState) -> AgentState:
        """允许在 LangGraph 图中使用 ``agent(state)`` 简写。"""
        return self.call(state)


# ---------------------------------------------------------------------------
# Concrete Agents
# ---------------------------------------------------------------------------

class AnalyzerAgent(BaseAgent):
    system_prompt = ANALYZER_SYSTEM_PROMPT
    role_name = "Analyzer"

    def _invoke(self, state: AgentState) -> AgentState:
        response = self._llm.invoke(state.messages)
        state.messages.append({"role": "assistant", "content": response.content, "name": self.role_name})
        return state

    async def _ainvoke(self, state: AgentState) -> AgentState:
        response = await self._llm.ainvoke(state.messages)
        state.messages.append({"role": "assistant", "content": response.content, "name": self.role_name})
        return state


class CoderAgent(BaseAgent):
    system_prompt = CODER_SYSTEM_PROMPT
    role_name = "Coder"

    def _invoke(self, state: AgentState) -> AgentState:
        response = self._llm.invoke(state.messages)
        state.messages.append({"role": "assistant", "content": response.content, "name": self.role_name})
        return state

    async def _ainvoke(self, state: AgentState) -> AgentState:
        response = await self._llm.ainvoke(state.messages)
        state.messages.append({"role": "assistant", "content": response.content, "name": self.role_name})
        return state


class TesterAgent(BaseAgent):
    system_prompt = TESTER_SYSTEM_PROMPT
    role_name = "Tester"

    def _invoke(self, state: AgentState) -> AgentState:
        response = self._llm.invoke(state.messages)
        state.messages.append({"role": "assistant", "content": response.content, "name": self.role_name})
        return state

    async def _ainvoke(self, state: AgentState) -> AgentState:
        response = await self._llm.ainvoke(state.messages)
        state.messages.append({"role": "assistant", "content": response.content, "name": self.role_name})
        return state


class DiagnoseAgent(BaseAgent):
    system_prompt = DIAGNOSE_SYSTEM_PROMPT
    role_name = "Diagnose"

    def _invoke(self, state: AgentState) -> AgentState:
        response = self._llm.invoke(state.messages)
        state.messages.append({"role": "assistant", "content": response.content, "name": self.role_name})
        return state

    async def _ainvoke(self, state: AgentState) -> AgentState:
        response = await self._llm.ainvoke(state.messages)
        state.messages.append({"role": "assistant", "content": response.content, "name": self.role_name})
        return state


