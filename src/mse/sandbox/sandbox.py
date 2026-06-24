from dataclasses import dataclass, field
import docker
import tarfile
import io
import time
import subprocess
import xml.etree.ElementTree as ET


@dataclass
class TestCaseResult:
    """单个测试用例的运行数据"""

    name: str               # e.g. "test_solution.py::test_add"
    classname: str          # e.g. "test_solution"
    status: str             # "passed" / "failed" / "error" / "skipped"
    duration: float         # 该用例耗时（秒）
    message: str            # 失败/错误的具体信息（通过则为空）

    @property
    def passed(self) -> bool:
        return self.status == "passed"


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

    # 单用例粒度数据（仅 pytest 且启用 junitxml 时有值）
    tests: list[TestCaseResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.exit_code == 0

    @property
    def test_count(self) -> dict[str, int]:
        """统计各状态用例数: {"passed": 2, "failed": 1, ...}"""
        counts: dict[str, int] = {}
        for t in self.tests:
            counts[t.status] = counts.get(t.status, 0) + 1
        return counts

    def summary(self) -> str:
        """给 Agent 看的简洁摘要"""
        lines = [f"Status: {self.status}"]
        if self.error_message:
            lines.append(f"Error: {self.error_message}")
        lines.append(f"Duration: {self.duration:.2f}s")

        # 有单用例数据时，展示每个用例的详情
        if self.tests:
            lines.append(f"\nTest Cases ({len(self.tests)} total):")
            for t in self.tests:
                icon = "✅" if t.passed else "❌"
                lines.append(f"  {icon} {t.name}  ({t.duration:.3f}s)")
                if t.message:
                    lines.append(f"     {t.message[:200]}")
        elif self.stdout:
            # 没有结构化数据时，回退到原文截取
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
            requirements="pytest\nnumpy==1.26.0\n",
            test_command="python -m pytest test_solution.py -v --tb=short",
        )
        sandbox.close()

    每次 execute() 调用都会:
      1. 创建全新容器（干净环境）
      2. 传入代码文件和 requirements.txt
      3. 运行 pip install -r requirements.txt 安装依赖
      4. 执行 test_command
      5. 收集结果并销毁容器
    """

    IMAGE = "python-test:3.11"

    def __init__(self, memory_limit: str = "256m", timeout: int = 10, network_disabled: bool = True):
        """
        不在这里创建容器。
        只是记录配置，真正创建容器在 execute() 里。
        """
        self.memory_limit = memory_limit
        self.timeout = timeout
        self.network_disabled = network_disabled
        self._client = docker.from_env()  # Docker 客户端连接
        self._container = None

    def execute(
        self,
        files: dict[str, str],
        test_command: str = "python -m pytest test_solution.py -v --tb=short",
        requirements: str | None = None,
        setup_commands: list[str] | None = None,
    ) -> ExecutionResult:
        """
        核心方法: 创建容器 → 传文件 → 安装依赖 → 跑测试 → 收集结果 → 销毁容器

        参数:
          files:          文件名 → 内容 的映射，会写入容器 /app 目录
          test_command:   在容器内执行的测试命令
          requirements:   pip requirements.txt 格式的字符串，
                          会写入 requirements.txt 并自动 pip install
          setup_commands: 额外的 shell 命令（在 pip install 之后、test_command 之前执行）
        """
        start = time.time()

        try:
            # ── 1. 创建容器（全新环境 = 清除之前的） ──
            # 如果需要安装依赖，必须允许网络访问 pip install
            # 容器执行完立即销毁，不会留下安全隐患
            _network_disabled = self.network_disabled and not requirements
            self._container = self._client.containers.run(
                image=self.IMAGE,
                command="sleep 60",
                detach=True,
                network_disabled=_network_disabled,
                mem_limit=self.memory_limit,
                cpu_period=100000,
                cpu_quota=100000,     # 1 核
                working_dir="/app",
            )

            # ── 2. 传入文件（代码 + requirements.txt） ──
            all_files = dict(files)  # 不修改调用方传入的字典
            if requirements:
                all_files["requirements.txt"] = requirements

            tar_data = self._pack_files(all_files)
            self._container.put_archive("/app", tar_data)

            # ── 3. 安装 Python 依赖 ──
            if requirements:
                pip_result = subprocess.run(
                    ["docker", "exec", self._container.id, "sh", "-c", "pip install -r requirements.txt -q"],
                    capture_output=True,
                    timeout=60,
                )
                if pip_result.returncode != 0:
                    # pip install 失败直接返回，不继续执行测试
                    duration = time.time() - start
                    return ExecutionResult(
                        exit_code=-1,
                        stdout="",
                        stderr=pip_result.stdout.decode("utf-8", errors="replace") + pip_result.stderr.decode("utf-8", errors="replace"),
                        duration=round(duration, 2),
                        status="setup_error",
                        error_message=f"pip install failed:\n{(pip_result.stdout + pip_result.stderr).decode('utf-8', errors='replace')[:500]}",
                    )

            # ── 4. 额外的 setup 命令（如果有） ──
            if setup_commands:
                for cmd in setup_commands:
                    subprocess.run(
                        ["docker", "exec", self._container.id, "sh", "-c", cmd],
                        capture_output=True,
                        timeout=30,
                    )

            # ── 5. 执行测试 ──
            # 如果是 pytest，自动追加 --junitxml 以获取结构化单用例数据
            actual_command = test_command
            is_pytest = "pytest" in test_command
            if is_pytest:
                actual_command = f"{test_command} --junitxml=/app/report.xml"

            exec_result = subprocess.run(
                ["docker", "exec", self._container.id, "sh", "-c", actual_command],
                capture_output=True,
                timeout=self.timeout,
            )
            exit_code = exec_result.returncode
            combined = exec_result.stdout + exec_result.stderr
            stdout = combined.decode("utf-8", errors="replace")
            duration = time.time() - start

            # 解析 junitxml 报告，提取单用例粒度数据
            tests: list[TestCaseResult] = []
            if is_pytest:
                tests = self._read_junitxml(self._container.id)

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr="",
                duration=round(duration, 2),
                status=self._classify_result(exit_code, stdout),
                error_message=self._extract_error(stdout),
                tests=tests,
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
            # ── 6. 无论成败，销毁容器 ──
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
        if "zerodivisionerror" in stdout_lower or "valueerror" in stdout_lower:
            return "runtime_error"
        if "indexerror" in stdout_lower or "keyerror" in stdout_lower:
            return "runtime_error"
        if "attributeerror" in stdout_lower or "runtimeerror" in stdout_lower:
            return "runtime_error"
        if "filenotfounderror" in stdout_lower or "eoferror" in stdout_lower:
            return "runtime_error"
        # 任何未分类的异常都视为 runtime_error
        if "error" in stdout_lower and exit_code != 0:
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

    @staticmethod
    def _read_junitxml(container_id: str) -> list[TestCaseResult]:
        """
        从容器中读取 /app/report.xml（pytest --junitxml 生成），
        解析为 TestCaseResult 列表。
        如果文件不存在或解析失败，返回空列表。
        """
        try:
            cat_result = subprocess.run(
                ["docker", "exec", container_id, "cat", "/app/report.xml"],
                capture_output=True,
                timeout=5,
            )
            if cat_result.returncode != 0:
                return []

            xml_text = cat_result.stdout.decode("utf-8", errors="replace")
            root = ET.fromstring(xml_text)

            results: list[TestCaseResult] = []
            for testcase in root.iter("testcase"):
                name = testcase.attrib.get("name", "unknown")
                classname = testcase.attrib.get("classname", "unknown")
                duration = float(testcase.attrib.get("time", 0))

                # 检查子元素判断状态
                failure = testcase.find("failure")
                error = testcase.find("error")
                skipped = testcase.find("skipped")

                if failure is not None:
                    status = "failed"
                    message = (failure.attrib.get("message", "") + "\n" +
                               (failure.text or "").strip()).strip()
                elif error is not None:
                    status = "error"
                    message = (error.attrib.get("message", "") + "\n" +
                               (error.text or "").strip()).strip()
                elif skipped is not None:
                    status = "skipped"
                    message = skipped.attrib.get("message", "")
                else:
                    status = "passed"
                    message = ""

                results.append(TestCaseResult(
                    name=f"{classname}::{name}",
                    classname=classname,
                    status=status,
                    duration=round(duration, 4),
                    message=message,
                ))

            return results

        except (ET.ParseError, OSError, ValueError) as e:
            # XML 解析失败或文件不存在，返回空列表
            return []

    def _cleanup(self):
        """销毁容器"""
        if self._container:
            try:
                self._container.remove(force=True)
            except Exception:
                pass
            self._container = None
