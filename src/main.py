"""AlgoSolver CLI：一键对若干算法题跑多智能体解题闭环。

【本项目简介】
  这是一个"多智能体（Multi-Agent）算法题自动求解"系统。对于每道算法题，系统会
  依次派出多种角色的 AI 智能体（分析器 Analyzer -> 编码器 Coder -> 测试器 Tester
  -> 诊断器 Diagnose）进行协同工作，形成一个"分析 -> 编码 -> 测试 ->（诊断 -> 重写
  -> 再测）"的闭环，直到代码通过全部测试用例，或达到最大迭代次数为止。

用法：
  python src/main.py --problems problems --max-iter 4 --output report.json      # 默认走真实 API
  python src/main.py --name two_sum --max-iter 4                               # 只跑一道题
  python src/main.py --mock                                                   # 使用本地 mock-LLM

行为：
  1. **默认使用真实 API**：从 .env 读取多模型配置——GLM 云端（GLM-4.7/GLM-4.6V/
     GLM-4.5-Air，OpenAI 兼容）+ Ollama 本地（deepseek-r1:8b / qwen3.5:4b），
     按能力从强到弱自动降级：某模型失败后本次运行不再使用，自动切换到更弱的模型。
  2. 加 `--mock` 参数则改用本地 mock LLM（需先启动 mock_llm_server.py），
     便于离线验证全流程；也可同时用 --base-url/--model 覆盖 mock 的地址/模型名。
  3. 健康检查：探测所有 LLM 端点，全部不可用（5xx / 无法连接）才直接退出。
  4. 对每道题执行 Analyzer -> Coder -> Tester -> (Diagnose -> Coder)* 闭环。
  5. 输出表格 + 写 report.json。

退出码：
  0 = 全部通过
  1 = 已运行但部分未通过
  2 = LLM 服务不可用
"""
from __future__ import annotations  # 启用延迟注解解析，让类型注解可以写字符串/可空类型，兼容旧版本 Python

import argparse  # 标准库：解析命令行参数
import json
import os  # 读取环境变量
import sys  # 系统相关的操作，例如 sys.path、sys.exit
import time  # 计时，用于统计每道题的求解耗时
from pathlib import Path  # 跨平台路径操作

# 让 `import mse` 可用（运行 `python src/main.py` 时 src 在 sys.path[0]）
# 此处把 src 目录插入到模块搜索路径的最前面，保证能从任何位置直接 import 本项目包
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 导入项目内部模块
from mse.llm.provider import LLMProvider, load_env        # LLM 统一接口 + .env 环境变量加载
from mse.sandbox import LocalSandbox, create_sandbox      # 代码沙箱（本地子进程执行器）
from mse.orchestrator import solve_problem                # 全局编排：状态图流水线入口
from mse import problems as problems_mod                  # 题目加载模块
from mse.ledger import record as _ledger, latest_log_path  # 台账：全流程留痕


def main(argv=None) -> int:
    # ---- 1. 解析命令行参数 ----
    # 命令行工具入口，返回退出码（供外层 sys.exit 使用）
    ap = argparse.ArgumentParser(description="AlgoSolver: Multi-Agent 算法题自动解答")
    ap.add_argument("--problems", default="problems", help="题目目录")  # 存放所有题目的根目录
    ap.add_argument("--max-iter", type=int, default=4, help="每道题最大编码迭代次数")  # 诊断->重写 的最大轮数
    ap.add_argument("--output", default="report.json", help="结果报告路径")            # 输出 JSON 报告的路径
    ap.add_argument("--name", default=None, help="只运行指定题目")                    # 按题目名过滤，只跑一道题
    ap.add_argument("--mock", action="store_true", default=False,
                    help="使用本地 mock LLM（需先启动 mock_llm_server.py），默认使用真实 API")  # 离线验证开关
    ap.add_argument("--base-url", default=None)          # LLM 服务的地址（可覆盖 .env / mock 端口）
    ap.add_argument("--api-key", default=None)           # LLM 的 API Key（可覆盖 .env）
    ap.add_argument("--model", default=None)             # 使用 mock 时默认 mock-model，否则取 .env
    ap.add_argument("--timeout", type=int, default=60, help="单步沙箱执行超时(秒)")
    args = ap.parse_args(argv)

    # ---- 2. 加载环境变量与构造 LLM 客户端 ----
    load_env()  # 读取 .env 文件中的配置（如果存在）
    if args.mock:
        # 使用本地 mock LLM：默认指向 mock_llm_server.py 的地址；--base-url 可覆盖端口
        # mock 服务固定返回 mock-model 或 Coder 的参考答案，用于离线验证整条流水线
        base = args.base_url or "http://127.0.0.1:8911/v1"
        provider = LLMProvider(base_url=base, api_key=args.api_key or "",
                               model=args.model or "mock-model", timeout=args.timeout)
    elif args.base_url or args.model or args.api_key:
        # 仅显式指定了 --base-url/--model/--api-key：构造单一模型（临时指向）
        base = args.base_url or os.getenv("GLM_BASE_URL", "http://10.0.131.251:9005/v1")
        key = args.api_key or os.getenv("GLM_API_KEY", "") or os.getenv("LLM_API_KEY", "")
        model = args.model or None
        # 创建单模型 Provider：负责对话、自动模型发现、重试（OpenAI 兼容）
        provider = LLMProvider(base_url=base, api_key=key, model=model, timeout=args.timeout)
    else:
        # 默认：从环境变量构建"多模型 + 能力降级"客户端（GLM 云端 + Ollama 本地）
        provider = LLMProvider(timeout=args.timeout)
    # 打印当前模型队列，方便观察降级策略
    model_list = ", ".join(c.name for c in provider.models)
    print(f"[模型队列] {model_list}", flush=True)

    # ---- 3. 健康检查：LLM 服务不可用直接退出 ----
    print(f"[健康检查] 探测所有 LLM 端点 ...", end=" ", flush=True)
    if not provider.health():  # 任一可用端点返回 True，全部不可达才退出
        print("失败（所有 LLM 端点不可用 / 5xx / 无法连接）。请检查 .env 配置后重试。")
        return 2
    print(f"OK (当前使用 model={provider.model})", flush=True)
    # 【台账】记录本次运行的开始：模式、模型队列、健康检查结果
    _ledger("run.start", mode="mock" if args.mock else "real",
            models=[m.name for m in provider.models], active_model=provider.model,
            input_dir=args.problems, max_iterations=args.max_iter, output=args.output,
            ledger_file=str(latest_log_path()))

    # ---- 4. 加载题目列表（可选按名称过滤） ----
    probs = problems_mod.load_problems(args.problems)  # 扫描题目目录，解析每道题的 description/tests.json
    if args.name:  # 指定了 --name 时，只保留同名题目
        probs = [p for p in probs if p["name"] == args.name]
    if not probs:  # 没有可运行的题目则直接退出
        print("没有可运行的题目。")
        return 1

    # ---- 5. 创建沙箱（代码隔离执行环境） ----
    # 默认"本地沙箱"：把候选代码和测试脚本写入临时目录，用子进程执行，带超时和资源限制
    sandbox = create_sandbox(timeout=args.timeout)

    # ---- 6. 逐题执行完整流水线 ----
    results = []  # 收集每道题的运行结果，用于最后的汇总报告
    for p in probs:  # 遍历每道题
        print(f"\n===== 题目 {p['name']} =====", flush=True)
        t0 = time.time()
        try:
            # 核心调用：对这道题运行 分析器->编码器->测试器->(诊断->编码)*闭环 的状态图
            st = solve_problem(p, provider, sandbox, max_iterations=args.max_iter, timeout=args.timeout)
        except Exception as e:  # 偶发异常（如网络抖动）时记录 error 状态，不中断整批运行
            print(f"运行异常：{type(e).__name__}: {e}")
            results.append({"name": p["name"], "status": "error", "iterations": 0,
                           "time_s": round(time.time() - t0, 1), "code": "", "diagnosis": str(e)})
            # 【台账】记录抛出异常被捕获的题目级事件
            _ledger("problem.error", name=p["name"], duration_s=round(time.time() - t0, 3),
                    error=f"{type(e).__name__}: {e}")
            continue
        dt = time.time() - t0  # 统计单题耗时
        results.append({   # 组装一条结构化结果记录
            "name": p["name"],
            "status": st.status,          # passed / failed / error ...
            "iterations": st.iteration,  # 实际迭代次数
            "time_s": round(dt, 1),
            "code": st.code,             # 最终代码
            "diagnosis": st.diagnosis,   # 最终诊断信息
        })
        print(f"结果: {st.status}  用时 {dt:.1f}s  迭代 {st.iteration}", flush=True)

    # ---- 7. 输出总结并写报告文件 ----
    passed = sum(1 for r in results if r["status"] == "passed")  # 统计通过的数量
    print(f"\n===== 总结：{passed}/{len(results)} 通过 =====", flush=True)
    # 把全部结果写为 JSON 文件（ensure_ascii=False 保留中文，indent=2 美化格式）
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入 {args.output}", flush=True)
    # 【台账】记录整个批次的运行结束：结果汇总、通过率与报告路径
    _ledger("run.end", passed=passed, total=len(results),
            by_name=[(r["name"], r["status"]) for r in results], report=str(args.output))
    # 退出码：全部通过返回 0（成功），否则返回 1（部分未通过）
    return 0 if passed == len(results) else 1


if __name__ == "__main__":  # 脚本作为主程序运行时才执行（被 import 时不执行）
    sys.exit(main())        # 以 mse 的方式返回退出码（0/1/2）