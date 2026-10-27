"""多智能体系统：分析器、编码器、测试器、诊断器智能体。

本模块是项目"多智能体协作"的核心：定义了四种不同"角色"的 AI 智能体
（Analyzer / Coder / Tester / Diagnose），并通过"系统提示词 + 一个 LLM 回调"
的方式复用同一套基类逻辑。每个智能体都可以作为一个 LangGraph 节点被调用。

四种角色的分工：
  - Analyzer（分析器）：理解题目，产出结构化规格说明；
  - Coder（编码器）  ：依据规格/反馈编写 Python 代码；
  - Tester（测试器）  ：设计并执行测试，验证实现是否正确；
  - Diagnose（诊断器）：对失败做根因分析，输出修改建议。
"""

from abc import ABC, abstractmethod  # ABC：定义抽象基类；abstractmethod：声明抽象方法
from typing import Any, Callable, Optional  # 类型标注
from dataclasses import dataclass, field  # dataclass：简化数据类定义


# ---------------------------------------------------------------------------
# 系统提示词
# ---------------------------------------------------------------------------

# 分析器系统提示词（英文原版，直接发给 LLM）：
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

# 分析器系统提示词（中文翻译，仅用于注释说明）：
# 你是**分析器智能体**。你的角色是深入理解用户需求，并将其分解为清晰、可执行的规格说明。
#
# 职责：
# - 解析并澄清模糊的需求。
# - 识别边界情况、约束条件和验收标准。
# - 生成可供编码器和测试器智能体使用的结构化分析文档。
# - 不要编写代码；专注于纯粹分析。
#
# 输出包含以下部分的结构化规格说明：摘要、功能需求、非功能需求、边界情况和验收标准。

# 编码器系统提示词（英文原版）：
CODER_SYSTEM_PROMPT = """You are the **Coder Agent**. Your role is to write clean, correct, and
well-documented Python code based on the Analyzer's specification and any feedback from Tester.

Responsibilities:
- Implement exactly what the specification describes.
- Follow best practices: type hints, docstrings, error handling.
- Produce production-ready code that passes all tests.
- When given test failure reports, fix bugs without rewriting unrelated code.

Output the complete implementation code in a single, runnable Python block.
"""

# 编码器系统提示词（中文翻译，仅用于理解说明）：
# 你是**编码器智能体**。你的角色是根据分析器的规格说明和测试器的反馈，编写干净、正确且文档完善的 Python 代码。
#
# 职责：
# - 严格按照规格说明实现功能。
# - 遵循最佳实践：类型注解、文档字符串、错误处理。
# - 生成可通过所有测试的生产级代码。
# - 当收到测试失败报告时，只修复 bug，不重写无关代码。
#
# 在单个可运行的 Python 代码块中输出完整的实现代码。

# 测试器系统提示词（英文原版）：
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

# 测试器系统提示词（中文翻译，仅用于理解说明）：
# 你是**测试器智能体**。你的角色是验证编码器的实现是否符合分析器的规格说明，并且没有 bug。
#
# 职责：
# - 编写全面的测试（单元测试、集成测试、边界情况测试）。
# - 执行测试并以结构化格式报告结果。
# - 明确区分测试失败（代码 bug）与规格不匹配。
# - 在发现问题时给出具体修复建议，但不编写生产代码。
#
# 输出包含以下部分的测试报告：已执行的测试用例、通过、失败、失败详情（含堆栈跟踪）以及建议。

# 诊断器系统提示词（英文原版）：
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

# 诊断器系统提示词（中文翻译，仅用于理解说明）：
# 你是**诊断器智能体**。你的角色是对其他智能体无法解决的问题进行分诊处理，
# 在整个流水线上进行根因分析。
#
# 职责：
# - 检查分析器、编码器和测试器的输出，定位问题根源。
# - 判断问题是规格错误、实现 bug 还是测试缺陷。
# - 提供简洁的诊断和可执行的下一步建议。
# - 在升级给人类处理之前，你是最终的仲裁者。
#
# 输出包含以下部分的诊断报告：问题摘要、根因、受影响组件以及建议操作。


# ---------------------------------------------------------------------------
# 智能体基类
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """在单次智能体调用的 LangGraph 节点之间传递的状态。"""
    messages: list[dict[str, str]] = field(default_factory=list)  # 对话历史（系统/用户/助手消息）
    context: dict[str, Any] = field(default_factory=dict)          # 附加上下文（键值对）


class BaseAgent(ABC):
    """多智能体系统中所有智能体的抽象基类。

    子类必须提供：
        system_prompt：类级别的系统提示字符串。
        role_name：可读的智能体名称。

    基类已经实现了一套标准的"调用前注入系统消息 + 调用后标注助手角色名"流程，
    子类只需实现 `_invoke`（同步）或选择性重写 `_ainvoke`（异步）。
    """

    # 类级属性：每个子类覆写为自己的系统提示与角色名
    system_prompt: str = ""
    role_name: str = "BaseAgent"

    def __init__(self, llm: Callable[..., Any]) -> None:
        """保存 LLM 封装的引用。

        *llm* 是任何与 LangGraph 兼容的可调用对象（例如 LangChain 聊天模型、
        或本项目 LLMProvider 实例，它实现了 invoke()/ainvoke()）。
        """
        self._llm = llm  # 实际与大模型交互的客户端

    # -- 属性 ----------------------------------------------------------------

    @property
    def name(self) -> str:
        """返回智能体的可读角色名（只读属性）。"""
        return self.role_name

    # -- LangGraph 核心接口 --------------------------------------------------

    def build_system_message(self) -> dict[str, str]:
        """返回系统消息字典，该消息会在每次调用前注入对话头部。"""
        return {"role": "system", "content": self.system_prompt}

    def call(self, state: AgentState) -> AgentState:
        """作为 LangGraph 节点的同步入口点。

        统一处理两件事：
          1. 调用前：若无系统消息则在最前面注入系统提示；
          2. 调用后：给最后一条消息打上 `name=role_name` 标记，
             方便后续追踪每条消息出自哪个智能体。
        子类通过覆写 ``_invoke`` 自定义具体行为。
        """
        # 调用前：如果对话头部还不是 system 系统消息，则插入系统提示
        if not state.messages or state.messages[0].get("role") != "system":
            state.messages.insert(0, self.build_system_message())

        state = self._invoke(state)  # 执行子类专属的"一次 LLM 调用"

        # 调用后：给最后一条助手消息补上智能体名（用于归属标记）
        if state.messages:
            state.messages[-1].setdefault("name", self.role_name)

        return state

    async def acall(self, state: AgentState) -> AgentState:
        """LangGraph 的异步入口点（配合 LangGraph 的 ``ainvoke`` / ``astream`` 使用）。"""
        # 调用前：同样先确保系统消息就位
        if not state.messages or state.messages[0].get("role") != "system":
            state.messages.insert(0, self.build_system_message())

        state = await self._ainvoke(state)  # 执行子类专属的异步逻辑

        if state.messages:  # 调用后同样的角色名标记
            state.messages[-1].setdefault("name", self.role_name)

        return state

    @abstractmethod
    def _invoke(self, state: AgentState) -> AgentState:
        """子类必须实现的同步专属逻辑（真正与大模型交互的部分）。"""
        ...

    async def _ainvoke(self, state: AgentState) -> AgentState:
        """子类可选的异步逻辑。默认回退到同步实现。"""
        return self._invoke(state)

    def __call__(self, state: AgentState) -> AgentState:
        """使智能体对象可被直接调用：``agent(state)``，方便在图中使用。"""
        return self.call(state)


# ---------------------------------------------------------------------------
# 具体智能体
# ---------------------------------------------------------------------------

class AnalyzerAgent(BaseAgent):
    """分析器智能体：理解题目 -> 输出结构化规格说明。"""
    system_prompt = ANALYZER_SYSTEM_PROMPT  # 覆写系统提示
    role_name = "Analyzer"                  # 角色名（消息标记用）

    def _invoke(self, state: AgentState) -> AgentState:
        # 同步调用 LLM：把整个对话历史交给模型
        response = self._llm.invoke(state.messages)
        # 把模型返回内容追加到对话历史中，并标注来源智能体
        state.messages.append({"role": "assistant", "content": response.content, "name": self.role_name})
        return state

    async def _ainvoke(self, state: AgentState) -> AgentState:
        # 异步版本：配合 async 图执行
        response = await self._llm.ainvoke(state.messages)
        state.messages.append({"role": "assistant", "content": response.content, "name": self.role_name})
        return state


class CoderAgent(BaseAgent):
    """编码智能体：利用规格/反馈信息编写 Python 实现。"""
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
    """测试智能体：验证 Coder 的实现是否满足规格、无 bug。"""
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
    """诊断智能体：对其它智能体无法解决的问题做根因分析。"""
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