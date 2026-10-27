"""加载 problems/ 目录下的算法题。

【目录约定】
  每道题用一个子目录存放，子目录命名即为题目名，内含两个文件：
    - problem.md   题目陈述 + 目标函数签名 + 示例
    - tests.json   {"function": 函数名, "signature": 签名, "cases": [{name, input, expected}]}

【tests.json 字段说明】
  - function : 需要实现的函数名
  - signature: 目标函数签名，例如 "def two_sum(nums, target) -> list:"，
               缺省时会自动构造 "def {function}(...):"
  - cases    : 测试用例数组，每个用例是 {"name": 用例名, "input": 参数列表, "expected": 期望输出}
"""
from __future__ import annotations  # 延迟注解解析（Optional 等注解向后兼容）

import json  # 解析 tests.json
from pathlib import Path  # 跨平台路径操作
from typing import Optional  # 可选类型注解


def load_problems(problems_dir: str) -> list:
    """扫描题目根目录，加载所有结构合法的算法题。

    只加载满足以下条件的子目录：
      - 是目录；
      - 同时包含 problem.md 与 tests.json；
    否则跳过该子目录（不影响其它题目）。

    返回列表，元素为字典：
      {"name", "statement", "function", "signature", "cases"}
    """
    root = Path(problems_dir)  # 题目根目录
    out = []  # 收集结果列表
    if not root.exists():  # 目录不存在时直接返回空列表
        return out
    for d in sorted(root.iterdir()):  # 遍历子目录（按名字排序保证顺序稳定）
        if not d.is_dir():  # 跳过非目录项（如 .py、.md 文件）
            continue
        md = d / "problem.md"   # 题目描述文件路径
        tj = d / "tests.json"   # 测试用例文件路径
        if not (md.exists() and tj.exists()):  # 缺少任一文件则跳过该题
            continue
        meta = json.loads(tj.read_text(encoding="utf-8"))  # 读取并解析 tests.json
        out.append(  # 组装题目字典（精简为流水线所需的信息）：
            {
                "name": d.name,                        # 题目名 = 子目录名
                "statement": md.read_text(encoding="utf-8"),   # 完整题目描述文本
                "function": meta["function"],          # 目标函数名
                "signature": meta.get("signature",    # 目标函数签名（缺省自动构造）
                                    f"def {meta['function']}(...):"),
                "cases": meta["cases"],                # 测试用例列表
            }
        )
    return out


def load_problem(name: str, problems_dir: str = "problems") -> Optional[dict]:
    """按题目名查找单个题目，找到则返回其字典，找不到返回 None。"""
    for p in load_problems(problems_dir):  # 遍历所有已加载的题目
        if p["name"] == name:  # 匹配题目名
            return p
    return None  # 未找到