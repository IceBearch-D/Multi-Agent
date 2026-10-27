"""本地子进程沙箱执行器（替代 Docker，零三方依赖）。

【背景】
原计划使用 Docker 沙箱（见 sandbox.py），但当前 WSL 环境未安装 Docker，
这里提供一个等价实现：把候选代码与测试运行器写入临时目录，用 subprocess 执行，
带墙钟超时与 CPU/内存资源上限，解析运行结果并返回结构化 ExecutionResult。

【工作流程】
  1. 用 tempfile 创建临时工作目录；
  2. 把 files 字典中的每个文件（候选代码 solution.py、测试运行器 test_runner.py 等）写入该目录；
  3. 用 subprocess 执行 test_command 指定的命令（默认 python test_runner.py）；
  4. 在子进程启动前通过 preexec_fn 做资源限制（CPU 上限 + 内存上限）；
  5. 从标准输出里提取标记为 RESULT_JSON 的结构化 JSON，解析成 ExecResult；
  6. 无论成败，finally 里删除临时目录。

接口与 Docker Sandbox 保持一致：execute(files, test_command, ...) -> ExecResult
"""
from __future__ import annotations  # 延迟注解解析

import json  # JSON 解析：解析 RUNNER 输出的结构化结果
import os    # 路径拼接、环境相关操作
import re    # 正则：从 stdout 中匹配 RESULT_JSON
import subprocess  # 子进程：执行测试命令
import sys   # sys.executable：定位当前 Python 解释器
import tempfile  # 临时目录
from dataclasses import dataclass, field  # dataclass 简化数据类
from typing import Optional  # 可选类型注解

# 测试运行器模板：一个独立的 Python 脚本。
# 它会被替换占位符后写到临时目录，里面 import 候选代码 solution 模块，逐条执行测试用例，
# 并把结果以一行 "RESULT_JSON {json}" 的形式打印到标准输出，供沙箱解析。
# 注意：这个是字符串模板，不是用来执行的模块，其中 __CASES__ / __FN__ 会被替换。
RUNNER_TMPL = r'''
import json, math, sys, traceback
CASES = __CASES__                            # 测试用例列表（json repr 替换而来）
def equal(a, b):                             # 递归地比较实际值与期望值
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        # 数值用 math.isclose 近似比较：相对误差 1e-6，绝对误差 1e-9
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        # 列表/元组逐项递归比较
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, (dict)):
        # 字典比较键与对应值（递归）
        return set(a.keys()) == set(b.keys()) and all(equal(a[k], b[k]) for k in a)
    return a == b                             # 其它类型直接 ==
FN = "__FN__"                                # 被测函数名（替换而来）
results = []                                 # 各用例的结果
overall = True                               # 总通过标志
try:
    import solution                          # 导入候选代码模块
    fn = getattr(solution, FN, None)         # 取目标函数
    if fn is None:                           # 函数不存在
        raise AttributeError("solution 中未找到函数 " + repr(FN))
except Exception as e:
    err = "".join(traceback.format_exception_only(type(e), e)).strip()
    results.append({"name": "import", "passed": False, "expected": None, "actual": err, "error": err})
    print("RESULT_JSON " + json.dumps({"passed": False, "status": "import_error",
          "error_message": err, "cases": results}, ensure_ascii=False))
    sys.exit(0)
for i, c in enumerate(CASES):                # 遍历每个测试用例
    name = c.get("name", "case" + str(i))   # 用例名（缺省用 index）
    args = c["input"]                       # 函数入参（list 时拆包传入，否则整体传入）
    expected = c["expected"]                # 期望输出
    try:
        actual = fn(*args) if isinstance(args, list) else fn(args)  # 调用目标函数
        ok = equal(actual, expected)        # 与期望值比较
    except Exception as e:                 # 该用例抛异常
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        results.append({"name": name, "passed": False, "expected": expected, "actual": err, "error": err})
        overall = False
        continue
    results.append({"name": name, "passed": ok, "expected": expected, "actual": actual,
                    "error": "" if ok else "mismatch"})
    if not ok:
        overall = False
status = "passed" if overall else "wrong_answer"
print("RESULT_JSON " + json.dumps({"passed": overall, "status": status,
      "error_message": "" if overall else "部分用例未通过", "cases": results}, ensure_ascii=False))
'''


def generate_test_runner(function_name: str, cases: list) -> str:
    """把测试运行器模板中的占位符替换成具体函数名与用例后返回。

    这是给 orchestrator 的 tester_node 用的：它拿到返回的脚本字符串，
    随 solution.py 一起写入沙箱执行。
    """
    return (
        RUNNER_TMPL.replace("__CASES__", repr(cases))  # 把用例列表以 repr 文本嵌入
        .replace("__FN__", function_name)             # 替换被测函数名
    )


@dataclass
class ExecResult:
    """一次沙箱执行的结构化结果（给 Agent 诊断用）。"""
    status: str = "unknown"          # passed / wrong_answer / runtime_error / timeout / import_error / syntax_error
    passed: bool = False             # 是否全部通过
    error_message: str = ""          # 关键错误信息（供 Agent 阅读）
    tests: list = field(default_factory=list)  # 单用例粒度的结果列表
    exit_code: int = 0               # 子进程退出码（0=成功）

    def summary(self) -> str:
        """生成供 LLM/Agent 阅读的简洁文本摘要。"""
        lines = [f"Status: {self.status}"]        # 首行：总体状态
        if self.error_message:                    # 有错误信息则追加
            lines.append(f"Error: {self.error_message}")
        for t in self.tests:                      # 逐条列出各用例结果
            icon = "OK " if t.get("passed") else "FAIL"
            lines.append(f"  [{icon}] {t.get('name')}: expected={t.get('expected')!r} actual={t.get('actual')!r}")
        return "\n".join(lines)


class LocalSandbox:
    """本地子进程沙箱：在隔离临时目录中用子进程执行候选代码并限制资源。"""

    def __init__(self, timeout: int = 60, memory_limit_mb: int = 1024, cpu_seconds: int = 60):
        self.timeout = timeout            # 命令执行超时（秒）
        self.memory_limit_mb = memory_limit_mb  # 子进程内存上限（MB）
        self.cpu_seconds = cpu_seconds    # CPU 使用时间上限（秒）

    def _preexec(self):
        """在子进程启动前设置资源限制（仅类 Unix 平台）。

        通过 resource 模块设置 RLIMIT_CPU（CPU 秒数）与 RLIMIT_AS（虚拟内存字节数）。
        各设置单独 try，因为不同系统对某些限制可能不支持，失败不阻塞执行。
        """
        try:
            import resource  # Unix 下的资源限制接口

            try:  # CPU 时间上限
                resource.setrlimit(resource.RLIMIT_CPU, (self.cpu_seconds, self.cpu_seconds))
            except Exception:
                pass
            try:  # 内存上限（地址空间字节数）
                mem = self.memory_limit_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            except Exception:
                pass
        except Exception:
            pass  # Windows 等无 resource 的环境直接忽略

    def execute(
        self,
        files: dict,
        test_command: str = "python test_runner.py",
        requirements: Optional[str] = None,
        timeout: Optional[int] = None,
        setup_commands: Optional[list] = None,
    ) -> ExecResult:
        """执行一次沙箱：写文件 --> 跑命令 --> 解析结果。

        参数：
          files: 文件名 -> 内容 的字典，会写入临时工作目录
          test_command: 要执行的测试命令（默认运行 test_runner.py）
          requirements: pip requirements 文本（本地沙箱不安装，仅兼容接口）
          timeout: 覆盖默认执行超时秒数
          setup_commands: 额外 shell 命令（本地沙箱不执行，仅兼容接口）
        返回：ExecResult（含结构化状态 / 错误信息 / 各用例结果）
        """
        timeout = timeout or self.timeout           # 取有效超时
        work = tempfile.mkdtemp(prefix="mse_sbx_")  # 创建临时工作目录
        try:
            # ---- 1. 把文件写入临时目录 ----
            for name, content in files.items():
                # 写入完整路径，UTF-8 编码
                with open(os.path.join(work, name), "w", encoding="utf-8") as f:
                    f.write(content)

            # ---- 2. 解析 test_command 为可执行命令 ----
            cmd = test_command.strip().split()   # 按空格拆分成命令列表
            if cmd and cmd[0] == "python":       # 把 "python" 替换为当前解释器路径
                cmd[0] = sys.executable

            # ---- 3. 用子进程运行，带超时与资源限制 ----
            proc = subprocess.run(               # 同步子进程
                cmd,
                cwd=work,               # 工作目录 = 临时目录（使 solution 可导入）
                capture_output=True,    # 捕获 stdout/stderr
                text=True,              # 以文本方式解码
                timeout=timeout,        # 墙钟超时
                preexec_fn=self._preexec,  # 启动前设置资源限制（POSIX）
            )
            stdout = proc.stdout or ""  # 标准输出文本

            # ---- 4. 从输出中解析 RESULT_JSON 标记的结构化结果 ----
            m = re.search(r"RESULT_JSON\s*(\{.*\})", stdout, re.DOTALL)  # 匹配 JSON 片段
            if m:
                try:
                    data = json.loads(m.group(1))  # 反序列化 JSON
                    # 直接还原为 ExecResult（test 携带全部用例级结果）
                    return ExecResult(
                        status=data.get("status", "wrong_answer"),
                        passed=data.get("passed", False),
                        error_message=data.get("error_message", ""),
                        tests=data.get("cases", []),
                        exit_code=proc.returncode,
                    )
                except Exception:
                    pass  # JSON 解析失败则走下方兜底逻辑

            # ---- 5. 兜底：没有结构化输出时，返回原始错误尾部 ----
            err = (proc.stderr or stdout)[-800:]          # 截取最后 800 字符
            # 非零退出码视为 runtime_error，否则视为 wrong_answer（视为用例未通过）
            status = "runtime_error" if proc.returncode != 0 else "wrong_answer"
            return ExecResult(status=status, passed=False, error_message=err, tests=[], exit_code=proc.returncode)

        except subprocess.TimeoutExpired:
            # 超时：可能是死循环或复杂度太高，Agent 可据此提示"优化算法"
            return ExecResult(status="timeout", passed=False, error_message=f"执行超时（>{timeout}s）", tests=[])
        except Exception as e:
            # 其它异常（如子进程启动失败）统一归为 error
            return ExecResult(status="error", passed=False, error_message=f"{type(e).__name__}: {e}", tests=[])
        finally:
            # ---- 6. 清理临时目录（无论成功/失败/异常都执行） ----
            try:
                import shutil
                shutil.rmtree(work, ignore_errors=True)  # 递归删除临时目录
            except Exception:
                pass


def create_sandbox(kind: str = "local", **kw) -> LocalSandbox:
    """工厂函数：创建沙箱实例（统一入口）。

    目前仅支持本地实现（kind 忽略），预留 future 支持 Docker。
    *kw（如 timeout）会传给 LocalSandbox 构造器。
    """
    return LocalSandbox(**kw)