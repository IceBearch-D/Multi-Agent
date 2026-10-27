"""Docker 沙箱实现（当前环境未安装 Docker，正式运行请使用 local_sandbox.LocalkSandbox）。

本文件实现"基于 Docker 容器的代码隔离执行"方案：通过把候选代码写入全新容器、
安装依赖、执行测试命令、收集结果并销毁容器的完整生命周期，提供严格隔离的环境。
尽管当前 WSL 未安装 Docker，本实现仍保留，方便在有 Docker 的环境中启用。

接口与 LocalSandbox 保持一致：execute() / close()。
"""
from dataclasses import dataclass, field  # dataclass 简化数据类定义
import docker      # Docker SDK：管理容器
import tarfile     # 把文件打包成 tar 传入容器
import io          # 内存字节流（tar 打包用）
import time        # 计时
import subprocess  # 调用 docker CLI（pip install、执行测试等）
import xml.etree.ElementTree as ET  # 解析 pytest junitxml 报告


@dataclass
class TestCaseResult:
    """单个测试用例的运行数据（由 pytest junitxml 解析而来）"""

    name: str               # 用例名，e.g. "test_solution.py::test_add"
    classname: str          # 所属类/模块，e.g. "test_solution"
    status: str             # "passed" / "failed" / "error" / "skipped"
    duration: float         # 该用例耗时（秒）
    message: str            # 失败/错误的具体信息（通过则为空）

    @property
    def passed(self) -> bool:
        """该用例是否通过（status == passed）。"""
        return self.status == "passed"


@dataclass
class ExecutionResult:
    """一次执行的所有结果——只读，不修改（作为数据载体使用）"""

    # 基本结果
    exit_code: int          # 0=通过, 1=测试失败, -1=异常（setup/其它错误）
    stdout: str             # 标准输出（测试报告通常在这里）
    stderr: str             # 错误输出（报错堆栈通常在这里）

    # 性能数据
    duration: float         # 执行耗时（秒）

    # 诊断分类（给 Agent 看的）：
    status: str             # "passed" / "wrong_answer" / "timeout" / "runtime_error" ...
    error_message: str      # 提取的关键错误信息（浓缩后给 Agent）

    # 单用例粒度数据（仅 pytest 且启用 junitxml 时有值）
    tests: list[TestCaseResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """整体是否通过：以退出码 0 为准。"""
        return self.exit_code == 0

    @property
    def test_count(self) -> dict[str, int]:
        """统计各状态用例数，例如 {"passed": 2, "failed": 1, ...}"""
        counts: dict[str, int] = {}
        for t in self.tests:
            counts[t.status] = counts.get(t.status, 0) + 1
        return counts

    def summary(self) -> str:
        """生成给 Agent/LLM 阅读的简洁摘要文本。"""
        lines = [f"Status: {self.status}"]
        # 有关键错误信息则追加
        if self.error_message:
            lines.append(f"Error: {self.error_message}")
        lines.append(f"Duration: {self.duration:.2f}s")

        # 有单用例粒度数据时，逐条展示每个用例的通过/失败状态与时间
        if self.tests:
            lines.append(f"\nTest Cases ({len(self.tests)} total):")
            for t in self.tests:
                icon = "✅" if t.passed else "❌"
                lines.append(f"  {icon} {t.name}  ({t.duration:.3f}s)")
                if t.message:
                    lines.append(f"     {t.message[:200]}")  # 错误信息截断到 200 字符
        elif self.stdout:
            # 没有结构化数据时，回退到原始输出截取（太长时只保留末尾 2000 字符）
            output = self.stdout[-2000:] if len(self.stdout) > 2000 else self.stdout
            lines.append(f"Output:\n{output}")

        return "\n".join(lines)


class Sandbox:
    """Docker 沙盒——每个实例管理一次完整的执行生命周期。

    使用模式:
        sandbox = Sandbox()
        result = sandbox.execute(
            files={"solution.py": "...", "test_solution.py": "..."},
            requirements="pytest\nnumpy==1.26.0\n",
            test_command="python -m pytest test_solution.py -v --tb=short",
        )
        sandbox.close()

    每次 execute() 调用都会:
      1. 创建全新容器（干净环境，杜绝脏数据影响）
      2. 传入代码文件和 requirements.txt
      3. 运行 pip install -r requirements.txt 安装依赖
      4. 执行 test_command 运行测试
      5. 收集结果并销毁容器
    """

    IMAGE = "python-test:3.11"  # 沙箱基础镜像（含 Python 3.11）

    def __init__(self, memory_limit: str = "256m", timeout: int = 10, network_disabled: bool = True):
        """构造函数：不在这里创建容器，只记录配置并建立 Docker 客户端连接。

        真正的容器创建推迟到 execute() 中，目的是统一管理一次执行的生命周期。
        - memory_limit:  容器内存上限（Docker 格式，如 256m / 512m）
        - timeout:       单次测试命令的执行超时秒数
        - network_disabled: 默认禁止容器联网，安装依赖时再临时放开
        """
        self.memory_limit = memory_limit
        self.timeout = timeout
        self.network_disabled = network_disabled
        self._client = docker.from_env()  # 从环境变量连接 Docker 守护进程
        self._container = None            # 当前活动容器（None 表示没有）

    def execute(
        self,
        files: dict[str, str],
        test_command: str = "python -m pytest test_solution.py -v --tb=short",
        requirements: str | None = None,
        setup_commands: list[str] | None = None,
    ) -> ExecutionResult:
        """核心方法: 创建容器 → 传文件 → 安装依赖 → 跑测试 → 收集结果 → 销毁容器。

        参数:
          files:          文件名 → 内容 的映射，会写入容器 /app 目录
          test_command:   在容器内执行的测试命令
          requirements:   pip requirements.txt 格式的字符串，会写入
                          requirements.txt 并自动 pip install
          setup_commands: 额外的 shell 命令（在 pip install 之后、test_command 之前执行）
        返回:
          ExecutionResult 结构化结果
        """
        start = time.time()  # 记录执行起点，用于计算耗时

        try:
            # ── 1. 创建容器（全新环境 = 清除之前的状态）──
            # 若需要安装依赖，则必须临时放开网络（_network_disabled=False）让 pip 可用
            # 容器生命周期只持续到本次 execute 结束，立即销毁，不遗留安全隐患
            _network_disabled = self.network_disabled and not requirements
            self._container = self._client.containers.run(
                image=self.IMAGE,
                command="sleep 60",     # 让容器保持存活，等待我们 exec 命令进去
                detach=True,            # 后台启动
                network_disabled=_network_disabled,
                mem_limit=self.memory_limit,
                cpu_period=100000,      # 限制 CPU：period 100ms
                cpu_quota=100000,       # —— quota 100ms => 1 核
                working_dir="/app",
            )

            # ── 2. 传入文件（代码 + requirements.txt）──
            # 深拷贝一份文件字典，避免修改调用方传入的对象
            all_files = dict(files)
            if requirements:  # 若需要安装依赖，把 requirements.txt 一起写入
                all_files["requirements.txt"] = requirements

            tar_data = self._pack_files(all_files)  # 打包成 tar 字节流
            self._container.put_archive("/app", tar_data)  # 解压到容器 /app 目录

            # ── 3. 安装 Python 依赖 ──
            if requirements:
                # 通过 docker CLI 在容器内执行 pip install（静默模式）
                pip_result = subprocess.run(
                    ["docker", "exec", self._container.id, "sh", "-c", "pip install -r requirements.txt -q"],
                    capture_output=True,
                    timeout=60,  # 安装依赖最多给 60 秒
                )
                if pip_result.returncode != 0:
                    # pip install 失败直接返回 setup_error，不再继续测试
                    duration = time.time() - start
                    return ExecutionResult(
                        exit_code=-1,
                        stdout="",
                        stderr=pip_result.stdout.decode("utf-8", errors="replace")
                             + pip_result.stderr.decode("utf-8", errors="replace"),
                        duration=round(duration, 2),
                        status="setup_error",
                        error_message=(
                            f"pip install failed:\n"
                            f"{(pip_result.stdout + pip_result.stderr).decode('utf-8', errors='replace')[:500]}"
                        ),
                    )

            # ── 4. 额外的 setup 命令（如果有）──
            if setup_commands:
                for cmd in setup_commands:
                    subprocess.run(
                        ["docker", "exec", self._container.id, "sh", "-c", cmd],
                        capture_output=True,
                        timeout=30,
                    )

            # ── 5. 执行测试 ──
            # 若测试命令是 pytest，自动追加 --junitxml 以拿到结构化单用例数据
            actual_command = test_command
            is_pytest = "pytest" in test_command
            if is_pytest:
                actual_command = f"{test_command} --junitxml=/app/report.xml"

            ev_result = subprocess.run(
                ["docker", "exec", self._container.id, "sh", "-c", actual_command],
                capture_output=True,
                timeout=self.timeout,  # 执行超时（秒）
            )
            exit_code = ev_result.returncode
            combined = ev_result.stdout + ev_result.stderr  # 合并 stdout 与 stderr
            stdout = combined.decode("utf-8", errors="replace")  # 解码（容错非法字符）
            duration = time.time() - start

            # 解析 junitxml 报告，提取单用例粒度测试结果
            tests: list[TestCaseResult] = []
            if is_pytest:
                tests = self._read_junitxml(self._container.id)

            # 组装最终结果
            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr="",
                duration=round(duration, 2),
                status=self._classify_result(exit_code, stdout),  # 分类失败类型
                error_message=self._extract_error(stdout),        # 提取关键错误
                tests=tests,
            )

        except Exception as e:
            # 任何异常（容器创建失败、子进程超时等）都转成结构化结果
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
            # ── 6. 无论成败，销毁容器，释放资源 ──
            self._cleanup()

    def close(self):
        """释放资源：销毁容器并关闭 Docker 客户端连接。"""
        self._cleanup()
        self._client.close()

    # ── 内部方法 ──

    def _pack_files(self, files: dict[str, str]) -> bytes:
        """把 {文件名: 内容} 字典打包成 tar 字节流，供 put_archive 使用。"""
        stream = io.BytesIO()          # 内存中的字节流
        with tarfile.open(fileobj=stream, mode="w") as tar:  # 以写模式创建缩包
            for name, content in files.items():
                data = content.encode("utf-8")  # 内容编码为字节
                info = tarfile.TarInfo(name=name)  # 创建 tar 条目元信息
                info.size = len(data)             # 设置文件大小
                tar.addfile(info, io.BytesIO(data))  # 写入条目及内容
        stream.seek(0)                 # 指针回到开头
        return stream.read()           # 返回整个 tar 字节

    def _classify_result(self, exit_code: int, stdout: str) -> str:
        """把测试结果分类——这是给 Agent 诊断用的关键信息。

        不只是"通过/失败"，还要告诉 Agent 失败的类型，例如：
          passed:         全部通过
          wrong_answer:   测试断言失败（逻辑错误）
          timeout:        超时（可能需要优化算法复杂度）
          runtime_error:  运行时崩溃（类型错误、空指针等）
          import_error:   导入错误（函数名写错等）
          syntax_error:   语法错误
        """
        if exit_code == 0:
            return "passed"  # 退出码 0 + 无异常即通过

        stdout_lower = stdout.lower()  # 统一转小写便于匹配

        # 按关键字依次判断失败大类（注意顺序：从最具体的开始）
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
        # 任何其它包含 error 关键字且非零退出的情况，兜底归为 runtime_error
        if "error" in stdout_lower and exit_code != 0:
            return "runtime_error"

        return "failed"  # 兜底：未知失败

    def _extract_error(self, stdout: str) -> str:
        """从测试输出中提取关键错误信息。

        pytest 的输出可能很长，Agent 不需要看全部；这里只截取
        "FAILED/ERROR 所在行开始的那一段"，并在遇到空行时停止，最多保留 10 行。
        """
        lines = stdout.split("\n")  # 按行切分
        error_lines = []   # 收集错误相关行
        capture = False    # 是否进入"捕获模式"

        for line in lines:
            if "FAILED" in line or "ERROR" in line:  # 遇到标记开始捕获
                capture = True
            if capture:
                error_lines.append(line)
                # 遇到空行且已捕获至少 2 行，认为错误段落结束
                if line.strip() == "" and len(error_lines) > 2:
                    break

        if error_lines:
            return "\n".join(error_lines[-10:])  # 只保留最后 10 行作为摘要

        # fallback: 若无标记，返回 stdout 的最后 500 字符
        return stdout[-500:] if stdout else "Unknown error"

    @staticmethod
    def _read_junitxml(container_id: str) -> list[TestCaseResult]:
        """从容器中读取 /app/report.xml（pytest --junitxml 生成），解析为单用例结果列表。

        如果文件不存在或解析失败（返回空列表），供调用方做兜底处理。
        """
        try:
            cat_result = subprocess.run(  # 用 docker exec cat 读取容器内文件
                ["docker", "exec", container_id, "cat", "/app/report.xml"],
                capture_output=True,
                timeout=5,
            )
            if cat_result.returncode != 0:  # 文件不存在或读取失败
                return []

            # 解码并解析 XML
            xml_text = cat_result.stdout.decode("utf-8", errors="replace")
            root = ET.fromstring(xml_text)  # 解析 XML 树

            results: list[TestCaseResult] = []
            for testcase in root.iter("testcase"):  # 遍历每个 testcase 节点
                name = testcase.attrib.get("name", "unknown")          # 用例名
                classname = testcase.attrib.get("classname", "unknown")  # 归属类名
                duration = float(testcase.attrib.get("time", 0))        # 耗时

                # 通过子元素判断该用例的状态
                failure = testcase.find("failure")  # 存在 => 断言失败
                error = testcase.find("error")      # 存在 => 执行出错
                skipped = testcase.find("skipped")  # 存在 => 被跳过

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
                else:  # 无失败/错误/跳过 = 通过
                    status = "passed"
                    message = ""

                # 组装 TestCaseResult 并追加
                results.append(TestCaseResult(
                    name=f"{classname}::{name}",   # 带归属前缀的唯一名称
                    classname=classname,
                    status=status,
                    duration=round(duration, 4),
                    message=message,
                ))

            return results  # 返回解析后的结果列表

        except (ET.ParseError, OSError, ValueError) as e:
            # XML 解析失败或文件不存在，返回空列表
            return []

    def _cleanup(self):
        """销毁当前容器（force=True 强制删除）。"""
        if self._container:
            try:
                self._container.remove(force=True)  # 强制移除容器
            except Exception:
                pass  # 删除失败也继续（容器可能已不存在）
            self._container = None  # 清空引用