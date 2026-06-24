"""多智能体系统：分析器、编码器、测试器、诊断器智能体。

所有核心实现位于 ``agent_manager`` 模块中，本文件仅做重导出以保持
``from mse.agents import AnalyzerAgent`` 等用法不变。
"""

from mse.agents.agent_manager import (  # noqa: F401
    ANALYZER_SYSTEM_PROMPT,
    CODER_SYSTEM_PROMPT,
    TESTER_SYSTEM_PROMPT,
    DIAGNOSE_SYSTEM_PROMPT,
    AgentState,
    BaseAgent,
    AnalyzerAgent,
    CoderAgent,
    TesterAgent,
    DiagnoseAgent,
)

__all__ = [
    "BaseAgent",
    "AgentState",
    "AnalyzerAgent",
    "CoderAgent",
    "TesterAgent",
    "DiagnoseAgent",
    "ANALYZER_SYSTEM_PROMPT",
    "CODER_SYSTEM_PROMPT",
    "TESTER_SYSTEM_PROMPT",
    "DIAGNOSE_SYSTEM_PROMPT",
]

