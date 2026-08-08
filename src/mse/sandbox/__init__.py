"""沙箱包：默认导出本地子进程执行器（无 Docker 也可运行）。

沙箱的作用是"隔离执行 + 资源限制 + 结构化结果收集"，用来运行候选代码的测试。
本项目默认使用本地实现（local_sandbox），无需安装 Docker 即可运行。
"""
# 从 local_sandbox 模块导入本地执行器及其相关工具，作为本包的公开 API
from mse.sandbox.local_sandbox import LocalSandbox, ExecResult, create_sandbox, generate_test_runner

# __all__ 声明本包对外暴露的符号，供 `from mse.sandbox import *` 使用
__all__ = ["LocalSandbox", "ExecResult", "create_sandbox", "generate_test_runner"]