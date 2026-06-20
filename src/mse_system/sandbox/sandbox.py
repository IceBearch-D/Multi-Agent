from dataclasses import dataclass
import docker
import tarfile
import io
import time


@dataclass
class ExecutionResult:
    """一次执行的所有结果——只读，不修改"""

    # 基本结果
    exit_code: int          # 0=通过, 1=失败, -1=异常
    stdout: str             # 标准输出（测试报告在这里）
    stderr: str             # 错误输出（报错信息在这里）

    # 性能数据
    duration: float         # 执行耗时（秒）

    # 诊断分类（给 Agent 看的）
    status: str             # "passed" / "wrong_answer" / "timeout" / "runtime_error"
    error_message: str      # 提取的关键错误信息

    @property
    def passed(self) -> bool:
        return self.exit_code == 0

    def summary(self) -> str:
        """给 Agent 看的简洁摘要"""
        lines = [f"Status: {self.status}"]
        if self.error_message:
            lines.append(f"Error: {self.error_message}")
        lines.append(f"Duration: {self.duration:.2f}s")
        if self.stdout:
            # 截取关键部分
            output = self.stdout[-2000:] if len(self.stdout) > 2000 else self.stdout
            lines.append(f"Output:\n{output}")
        return "\n".join(lines)


class Sandbox:
    """
    Docker 沙盒——每个实例管理一次完整的执行生命周期

    使用模式:
        sandbox = Sandbox()
        result = sandbox.execute(
            files={"solution.py": "...", "test_solution.py": "..."},
            test_command="python -m pytest -v",
        )
        sandbox.close()
    """

    IMAGE = "python:3.11-slim"

    def __init__(self, memory_limit: str = "256m", timeout: int = 10):
        """
        不在这里创建容器。
        只是记录配置，真正创建容器在 execute() 里。
        """
        self.memory_limit = memory_limit
        self.timeout = timeout
        self._client = docker.from_env() # Docker 客户端连接
        self._container = None

    def execute(
        self,
        files: dict[str, str],
        test_command: str = "python -m pytest test_solution.py -v --tb=short",
        setup_commands: list[str] | None = None,
    ) -> ExecutionResult:
        """
        核心方法: 创建容器 → 传文件 → 跑测试 → 收集结果 → 销毁容器

        这一个方法完成了所有功能:
          - 加载镜像（创建新容器）
          - 清除环境（新容器天然干净）
          - 安装依赖（通过 setup_commands）
          - 运行代码（执行 test_command）
          - 存储结果（返回 ExecutionResult）
        """
        start = time.time()

        try:
            # ── 1. 创建容器（全新环境 = 清除之前的） ──
            self._container = self._client.containers.run(
                image=self.IMAGE,
                command="sleep 60",
                detach=True,
                network_disabled=True,
                mem_limit=self.memory_limit,
                cpu_period=100000,
                cpu_quota=100000,     # 1 核
                working_dir="/app",
            )

            # ── 2. 传入文件 ──
            tar_data = self._pack_files(files) # 把文件打包成 tar 格式
            self._container.put_archive("/app", tar_data) # 把文件放到容器的 /app 目录

            # ── 3. 安装依赖（如果有） ──
            if setup_commands:
                for cmd in setup_commands:
                    self._container.exec_run(cmd, timeout=30)

            # ── 4. 执行测试 ──
            exit_code, output = self._container.exec_run(
                cmd=f"sh -c '{test_command}'",
                timeout=self.timeout,
            )

            stdout = output.decode("utf-8", errors="replace")
            duration = time.time() - start

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr="",
                duration=round(duration, 2),
                status=self._classify_result(exit_code, stdout),
                error_message=self._extract_error(stdout),
            )

        except Exception as e:
            duration = time.time() - start
            error_str = str(e)
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=error_str,
                duration=round(duration, 2),
                status="timeout" if "timeout" in error_str.lower() else "error",
                error_message=error_str[:500],
            )

        finally:
            # ── 5. 无论成败，销毁容器 ──
            self._cleanup()

    def close(self):
        """关闭 Docker 连接"""
        self._cleanup()
        self._client.close()

    # ── 内部方法 ──

    def _pack_files(self, files: dict[str, str]) -> bytes:
        """把文件字典打包成 tar"""
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as tar:
            for name, content in files.items():
                data = content.encode("utf-8")
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        stream.seek(0)
        return stream.read()

    def _classify_result(self, exit_code: int, stdout: str) -> str:
        """
        把测试结果分类——这是给 Agent 诊断用的关键信息

        不只是"通过/失败"，还要告诉 Agent 失败的类型:
          passed:         全部通过
          wrong_answer:   测试断言失败（逻辑错误）
          timeout:        超时（可能需要优化算法复杂度）
          runtime_error:  运行时崩溃（语法错误、类型错误等）
          import_error:   导入错误（函数名写错了）
        """
        if exit_code == 0:
            return "passed"

        stdout_lower = stdout.lower()

        if "timeout" in stdout_lower or "timed out" in stdout_lower:
            return "timeout"
        if "importerror" in stdout_lower or "modulenotfounderror" in stdout_lower:
            return "import_error"
        if "syntaxerror" in stdout_lower:
            return "syntax_error"
        if "assertionerror" in stdout_lower or "assert" in stdout_lower:
            return "wrong_answer"
        if "typeerror" in stdout_lower or "nameerror" in stdout_lower:
            return "runtime_error"

        return "failed"

    def _extract_error(self, stdout: str) -> str:
        """
        从测试输出中提取关键错误信息

        pytest 的输出可能很长，Agent 不需要看全部，
        只需要看失败原因那一段。
        """
        lines = stdout.split("\n")
        error_lines = []
        capture = False

        for line in lines:
            if "FAILED" in line or "ERROR" in line:
                capture = True
            if capture:
                error_lines.append(line)
                # 遇到空行或分隔线就停
                if line.strip() == "" and len(error_lines) > 2:
                    break

        if error_lines:
            return "\n".join(error_lines[-10:])  # 最多保留 10 行

        # fallback: 返回 stderr 的最后几行
        return stdout[-500:] if stdout else "Unknown error"

    def _cleanup(self):
        """销毁容器"""
        if self._container:
            try:
                self._container.remove(force=True)
            except Exception:
                pass
            self._container = None
