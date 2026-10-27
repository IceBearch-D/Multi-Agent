# AlgoSolver 构建说明（本次搭建）

基于 `Plan.md` 完成「多智能体算法题自动解答系统」的可运行骨架，LLM 指向 `10.0.131.251:9005`（OpenAI 兼容）。

## 架构（对齐 Plan.md）
- **4 个智能体**：Analyzer / Coder / Tester（沙箱执行）/ Diagnose，系统提示词沿用 `agent_manager.py`。
- **Reflexion 闭环**：Analyzer → Coder → Tester，未通过则 Diagnose 诊断 → Coder 重写 → 再测，最多 `max-iter` 轮。
- **轻量 StateGraph 引擎**（`orchestrator.py`）：节点 + 普通边 + 条件边，语义等价于 LangGraph 的 StateGraph。

## 与原 Plan 的偏差（及原因）
| Plan | 本次实现 | 原因 |
|------|----------|------|
| LangGraph | 自研零依赖 StateGraph | 环境为 Python 3.14，langgraph 及其依赖在 3.14 上安装风险高；自研引擎更稳 |
| Docker 沙箱 | 本地子进程沙箱（`LocalSandbox`） | WSL 未安装 Docker；本地执行器带超时 + CPU/内存上限，等价可用 |
| Pydantic / Instructor 结构化输出 | 提示词 + JSON 解析 | 避免额外依赖 |
| pytest 测试 | 自带测试运行器 | 零依赖、输出可控 |

> 全部代码**仅用 Python 标准库**，因此无需 `pip install` 即可运行，天然满足「依赖走国内源」要求。

## 依赖与镜像
- 当前实现零三方运行时依赖。
- 若切换到完整技术栈，请使用国内源：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple langgraph pydantic docker openai pytest`

## 运行
```bash
bash run.sh            # 或 python run.py
# 自定义：python src/main.py --name two_sum --max-iter 4
```
- 服务不可用时（如 502）脚本会直接退出（退出码 2），不空耗。
- 通过 `problems/` 下的题目运行，结果写入 `report.json`。

## 题目集
`problems/` 下 6 道：`two_sum`、`valid_parentheses`、`climb_stairs`、`max_sub_array`、`reverse`、`binary_search`，每题含 `problem.md` 与 `tests.json`（含输入/输出示例与隐藏用例）。
