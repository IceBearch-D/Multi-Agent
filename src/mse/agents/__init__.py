"""多智能体系统：分析器、编码器、测试器、诊断器智能体。

所有核心实现位于 ``agent_manager`` 模块中，本文件仅做重导出
（re-export）以保持 ``from mse.agents import AnalyzerAgent`` 等简短导入用法不变：
  即把 agent_manager 里定义的类和常量"搬"到 mse.agents 包的命名空间下，
  让调用方无需关心内部模块路径。

导出的主要内容：
  - BaseAgent / AgentState：智能体基类与状态数据类
  - AnalyzerAgent / CoderAgent / TesterAgent / DiagnoseAgent：四种角色的具体智能体
  - 各角色的系统提示词常量（供 orchestrator 等模块直接使用）
"""

# noqa: F401 —— 提醒 flake8 忽略"import 了但未使用"的告警，
# 因为这里 import 的目的就是"重导出"，它们会被 __all__ 显式列出
from mse.agents.agent_manager import (  # noqa: F401
    ANALYZER_SYSTEM_PROMPT,   # 分析器系统提示词
    CODER_SYSTEM_PROMPT,      # 编码器系统提示词
    TESTER_SYSTEM_PROMPT,     # 测试器系统提示词
    DIAGNOSE_SYSTEM_PROMPT,   # 诊断器系统提示词
    AgentState,               # 智能体状态（消息列表 + 上下文）
    BaseAgent,                # 抽象基类
    AnalyzerAgent,            # 分析器智能体
    CoderAgent,               # 编码器智能体
    TesterAgent,              # 测试器智能体
    DiagnoseAgent,            # 诊断器智能体
)

# 定义包的公开对外接口（配合 `from x import *` 使用，也声明模块的公共 API）
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