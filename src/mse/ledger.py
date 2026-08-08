"""台账（Ledger）日志系统：完整记录解题流水线的每一步。

【设计目标】
  详细记录多智能体解题过程中的**所有信息**，便于复盘、审计与调试：
  - 每次运行（run）的开始/结束、使用的模型队列；
  - 每道题的开始/结束与结果；
  - 每个节点（analyzer / coder / tester / diagnose）的输入与输出：
      * 输入：发给 LLM 的消息（含 system/user 及历史上下文）或喂给沙箱的文件；
      * 输出：模型返回的文本、结构化规格、代码、测试报告、诊断结论等；
  - 每个 Agent 的每次调用（谁被调用、用的哪个模型、耗时、降级情况）。

【格式】
  使用 Python 标准库 logging，把每条记录序列化为**一行 JSON** 追加写入
  logs/ledger.jsonl（JSON Lines），方便用 jq / rg / grep 做针对性查询。
  每次运行另存一份带时间戳的归档文件 logs/ledger_YYYYmmdd_HHMMSS.jsonl，
  所有历史文件保留、不覆盖，保证"全程留痕"。

【用法】
  from mse.ledger import logger, record
  logger.info({...})        # 写入一条结构化 JSON 行
  record('problem_start', name='two_sum')  # 便捷封装（自动带事件名）

【控制台输出】
  默认不在控制台打印详细台本（避免刷屏）；设环境变量 ALGOSOLVER_CONSOLE=1
  可在控制台同步看到每一条结构化日志（每行 JSON）。
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# 日志根目录：项目根下的 logs/（当前文件位于 src/mse/ → parents[2] 即项目根）
_LOG_ROOT = Path(__file__).resolve().parents[2] / "logs"
_LOG_ROOT.mkdir(parents=True, exist_ok=True)

# 本次运行的时间戳，用作日志文件名的一部分
_RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
# 人类可读的最新副本（便于 tail 查看）
_LATEST = _LOG_ROOT / "ledger.jsonl"
# 本轮运行专属的归档文件
_ROUND = _LOG_ROOT / f"ledger_{_RUN_STAMP}.jsonl"


class _JsonLineFormatter(logging.Formatter):
    """把 log 记录格式化为 JSON Lines（每条一行紧凑 JSON）。"""

    def format(self, record: logging.LogRecord) -> str:
        # 若消息本身是 dict，则作为结构化字段展开；否则作为 msg 文本
        if isinstance(record.msg, dict):
            obj: dict[str, Any] = {"ts": _ts(record.created), "level": record.levelname}
            obj.update(record.msg)
        else:
            obj = {"ts": _ts(record.created), "level": record.levelname, "msg": record.getMessage()}
        return json.dumps(obj, ensure_ascii=False, default=_default)


def _ts(epoch: float) -> str:
    """把时间戳格式化为 ISO 8601 字符串。"""
    return datetime.fromtimestamp(epoch).isoformat(timespec="milliseconds")


def _default(o: Any) -> Any:
    """把不可序列化的对象兜底转为字符串（如第三方异常对象）。"""
    try:
        return str(o)
    except Exception:
        return repr(o)


def _console_enabled() -> bool:
    """是否在控制台同步输出台本（由环境变量 ALGOSOLVER_CONSOLE 控制）。"""
    import os

    return os.getenv("ALGOSOLVER_CONSOLE", "").strip().lower() in ("1", "true", "yes", "on")


def _setup_logger() -> logging.Logger:
    """初始化台账 logger：输出到本轮归档文件 + 最新别名文件（可选控制台）。"""
    logger = logging.getLogger("algosolver.ledger")  # 唯一命名空间
    if logger.handlers:  # 已被初始化过（例如多次 import）直接复用
        return logger
    logger.setLevel(logging.INFO)  # 记录 INFO 及以上（含每步详细）
    logger.propagate = False       # 避免冒泡到根 logger 产生重复输出

    fmt = _JsonLineFormatter()

    # 1) 本轮归档文件（保留明细）
    h1 = logging.FileHandler(_ROUND, encoding="utf-8")
    h1.setFormatter(fmt)
    logger.addHandler(h1)

    # 2) 最新别名文件（方便 tail -f / 工具读取）
    h2 = logging.FileHandler(_LATEST, encoding="utf-8")
    h2.setFormatter(fmt)
    logger.addHandler(h2)

    # 3) 控制台（可选，默认关闭避免刷屏）
    if _console_enabled():
        h3 = logging.StreamHandler(sys.stdout)
        h3.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h3)

    return logger


# 全局唯一台账 logger 实例
logger = _setup_logger()


def record(event: str, **fields) -> None:
    """记录一条结构化台账事件（JSON Lines）。

    参数:
      event:  事件名（如 "problem_start"、"analyzer.input"、"analyzer.output"）
      fields: 事件附加字段（键值，均可 JSON 序列化）
    示例:
      record("problem_start", name="two_sum", stamp=...) 写一行 JSON
    """
    data: dict[str, Any] = {"event": event}
    data.update(fields)
    logger.info(data)


def log_agent_input(agent: str, input: Any, **extra) -> None:
    """记录一个 Agent 的输入（事件名形如 analyzer_input / coder_input）。"""
    record(f"{agent.lower()}_input", agent=agent, input=input, **extra)


def log_agent_output(agent: str, output: Any, **extra) -> None:
    """记录一个 Agent 的输出（事件名形如 analyzer_output / coder_output）。"""
    record(f"{agent.lower()}_output", agent=agent, output=output, **extra)


def latest_log_path() -> Path:
    """返回最新台账文件路径（供使用者提示/读取）。"""
    return _LATEST