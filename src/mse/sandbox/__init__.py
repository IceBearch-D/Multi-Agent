"""
MSE 沙箱模块。

提供代码执行的隔离沙箱环境，支持安全的代码运行与结果捕获。

核心组件:
    - Sandbox: 沙箱执行器，负责在隔离环境中运行代码。
    - ExecutionResult: 执行结果封装，包含输出、错误及退出码等信息。
"""

from .sandbox import Sandbox, ExecutionResult

__all__ = ["Sandbox", "ExecutionResult"]



