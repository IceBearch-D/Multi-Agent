# MSE-System 项目计划

本计划面向“基于 LangGraph 的多智能体协同软件工程与自动修复系统”。项目总周期控制在 8 周内：前 4 周完成可运行的系统框架搭建，后 4 周完成功能细节丰富、工程化完善、基准测评与展示材料。

## 总体目标

- 第 1-4 周：完成项目骨架、LangGraph 多 Agent 主流程、代码库检索原型、沙箱执行原型和最小可用闭环。
- 第 5-8 周：完善 AST/RAG、Diff 修改、安全沙箱、可观测 Dashboard、Benchmark 数据集、对比实验和最终报告。
- 最终交付：可运行代码库、单元测试与覆盖率报告、Streamlit/Gradio 可视化界面、20-30 个 Bug 修复任务的评测报告、简历与面试展示材料。

## 技术路线总览

项目采用 Python 作为主语言，核心技术栈如下：

- Agent 编排：LangGraph，用显式 StateGraph 管理 Product Agent、Coder Agent、Tester Agent 和 Reflexion Loop。
- 结构化输出：Pydantic 或 Instructor，约束 Agent 输出 Issue 分析、修改计划、Diff、测试诊断等结构化对象。
- 代码理解：Tree-sitter 或 Python `ast` 模块优先完成 Python 项目的函数、类、导入关系解析，后续扩展 Tree-sitter。
- 代码检索：Chroma 作为本地向量库，Embedding 模型可先使用 OpenAI Embedding 或 BGE 系列，本地资源不足时先保留接口。
- 上下文优化：语义检索 + 调用依赖补充 + 文件摘要，避免把整个代码库直接塞入 LLM。
- 修改应用：统一使用 Unified Diff 或 Search/Replace Blocks，避免整文件重写。
- 执行沙箱：Docker SDK for Python，限制网络、CPU、内存、运行时间和挂载目录。
- 测试框架：pytest + coverage，用测试结果驱动 Agent 自修复。
- 可观测性：LangSmith 或 Arize Phoenix 追踪调用链路；Streamlit/Gradio 展示图状态、Agent 消息、日志和评测结果。

## 第 1 周：需求冻结与工程骨架

### 本周目标

完成项目定位、模块边界和代码仓库基础结构，确保项目从一开始就按“工业级工程”组织，而不是散乱脚本。

### 主要任务

- 明确系统输入输出：
  - 输入：用户 Issue、目标代码仓库路径、可选测试命令。
  - 输出：任务拆解、相关上下文、代码修改 Diff、测试日志、最终修复报告。
- 建立项目目录：
  - `src/mse_system/agents/`：Product、Coder、Tester Agent。
  - `src/mse_system/graph/`：LangGraph 图定义和 State。
  - `src/mse_system/retrieval/`：代码解析、索引、检索。
  - `src/mse_system/sandbox/`：Docker 沙箱执行。
  - `src/mse_system/patching/`：Diff 解析与应用。
  - `src/mse_system/evaluation/`：Benchmark 与指标统计。
  - `tests/`：pytest 测试。
  - `examples/`：演示代码仓库和示例 Issue。
- 初始化配置文件：
  - `pyproject.toml` 或 `requirements.txt`
  - `.env.example`
  - `README.md`
  - `.gitignore`
- 定义核心数据模型：
  - `IssueSpec`
  - `RepoContext`
  - `PatchPlan`
  - `TestResult`
  - `ThreadState`

### 技术指导

- `ThreadState` 是全系统关键，应包含：
  - `issue`
  - `repo_path`
  - `retrieved_context`
  - `plan`
  - `patches`
  - `test_result`
  - `retry_count`
  - `max_retries`
  - `agent_messages`
  - `status`
- 所有 Agent 的输入输出都优先使用 Pydantic 模型，避免靠自然语言字符串互相传递状态。
- 先支持 Python 项目修复，不要一开始追求多语言。Python 场景足够展示 AST、pytest、Docker、Diff 和 Agent 闭环。

### 验收标准

- 项目能通过 `pytest` 跑通基础测试。
- `ThreadState`、核心 Pydantic 模型和目录结构完成。
- README 中能说明系统架构、运行方式和第一个最小 Demo 目标。

## 第 2 周：LangGraph 多 Agent 主流程

### 本周目标

完成 Product Agent、Coder Agent、Tester Agent 的最小可运行图，让 Issue 能按固定路径流转。

### 主要任务

- 实现 LangGraph 节点：
  - `product_node`：解析 Issue，输出任务拆解和修改目标。
  - `retrieve_context_node`：先用简单全文检索或文件扫描返回相关文件片段。
  - `coder_node`：根据任务和上下文生成修改方案或 Diff。
  - `patch_node`：应用 Diff 或先模拟应用结果。
  - `tester_node`：运行测试命令或模拟测试结果。
  - `diagnose_node`：分析失败日志，给出下一轮修复建议。
- 实现条件边：
  - 测试通过：进入 `END`。
  - 测试失败且 `retry_count < max_retries`：回到 `coder_node`。
  - 测试失败且达到重试上限：进入人工介入或失败总结节点。
- 保留可替换的 LLM 调用接口，方便后续接 OpenAI、LangChain ChatModel 或本地模型。

### 技术指导

- 图结构建议：

```text
START
  -> product_node
  -> retrieve_context_node
  -> coder_node
  -> patch_node
  -> tester_node
  -> route_by_test_result
       -> END
       -> diagnose_node -> coder_node
       -> fail_node -> END
```

- 条件边不要只依赖字符串，建议用枚举状态：
  - `SUCCESS`
  - `NEED_RETRY`
  - `FAILED`
- 每个节点只做一类事情，避免把检索、生成、执行、诊断塞进一个函数。
- 在本周可以使用 Mock LLM 或规则逻辑，使图先稳定运行。

### 验收标准

- 能用一个示例 Issue 触发完整 LangGraph 流程。
- 能打印或保存每个 Agent 的消息、节点跳转路径和最终状态。
- 至少有 3 个图流程测试：成功、失败后重试成功、超过重试上限失败。

## 第 3 周：代码库解析与上下文检索原型

### 本周目标

完成 Codebase RAG 的第一版，让系统能根据 Issue 找到相关文件、函数、类和调用关系。

### 主要任务

- 实现代码索引：
  - 遍历目标仓库。
  - 过滤 `.git`、虚拟环境、缓存、构建产物。
  - 提取 Python 文件。
- 实现 AST 解析：
  - 使用 Python `ast` 提取函数、类、导入、函数签名、docstring。
  - 为每个代码块生成 `CodeChunk`。
- 实现检索：
  - 第一阶段：关键词 BM25 或简单文本相似度。
  - 第二阶段：接入 Chroma 向量库。
  - 第三阶段：预留 reranker 接口。
- 构建 `Project Outline`：
  - 文件树摘要。
  - 模块关系。
  - 函数/类索引。
  - 入口测试命令。

### 技术指导

- `CodeChunk` 建议字段：
  - `file_path`
  - `symbol_name`
  - `symbol_type`
  - `start_line`
  - `end_line`
  - `signature`
  - `content`
  - `imports`
  - `docstring`
- 检索返回不要只给代码文本，还要给文件路径和行号，方便 Coder Agent 生成局部 Diff。
- 对上下文做预算控制，例如：
  - Issue 摘要：500 tokens。
  - 项目大纲：1000 tokens。
  - Top-K 代码片段：3000-6000 tokens。
  - 依赖补充：1000 tokens。

### 验收标准

- 给定一个本地示例项目和 Issue，能召回 Top-K 相关代码块。
- 能输出项目大纲和符号索引。
- 检索模块有单元测试，覆盖文件过滤、AST 解析、Top-K 返回。

## 第 4 周：沙箱执行与最小闭环

### 本周目标

在 4 周内完成项目框架搭建要求：系统能完成“Issue -> 检索 -> 生成补丁 -> 应用补丁 -> 沙箱测试 -> 失败重试/成功结束”的最小闭环。

### 主要任务

- 实现 Docker 沙箱：
  - 使用 Docker SDK 启动容器。
  - 将目标仓库复制到临时目录或以只读/受控方式挂载。
  - 执行 `pytest`、指定测试命令或静态检查命令。
  - 捕获 stdout、stderr、exit code、耗时。
- 实现 Patch 应用：
  - 支持 Unified Diff。
  - Diff 应用失败时返回结构化错误。
  - 每轮修改前保存工作区快照。
- 打通 LangGraph 闭环：
  - Coder 生成 Diff。
  - Patch 节点应用 Diff。
  - Tester 调沙箱运行测试。
  - Diagnose 节点根据测试日志生成修复建议。
- 建立最小示例：
  - 1 个带 Bug 的 Python 小项目。
  - 1 个 Issue。
  - 1 个测试用例。

### 技术指导

- Docker 沙箱至少限制：
  - `network_disabled=True`
  - CPU 或运行时间限制
  - 内存限制
  - 超时 kill
  - 工作目录隔离
- 第 4 周不要追求复杂 Benchmark，重点是闭环稳定。
- 对文件修改要保守：
  - 默认只允许修改检索命中的文件。
  - 超出范围时需要 Product Agent 明确批准。
  - 每次补丁记录 patch id、来源 Agent、应用结果。

### 验收标准

- 第 4 周结束时必须可以运行一个端到端 Demo。
- Demo 能展示至少一次测试失败后的自动修复重试。
- 具备基础日志：节点轨迹、Agent 输出、测试日志、最终报告。

## 第 5 周：Diff 可靠性与 Reflexion 增强

### 本周目标

让自动修复从“能跑”变成“更稳”，重点解决 Diff 应用失败、错误诊断粗糙和无效重试问题。

### 主要任务

- 增强 Patch 系统：
  - 支持 Search/Replace Blocks 作为 Unified Diff 的备用格式。
  - 检查补丁是否命中唯一位置。
  - 应用后运行格式化或语法检查。
- 增强 Reflexion Loop：
  - Tester Agent 提取错误类型：语法错误、导入错误、断言失败、超时、依赖缺失。
  - Diagnose Agent 生成根因分析和下一轮约束。
  - Coder Agent 在下一轮必须引用上一轮失败原因。
- 增加状态控制：
  - `retry_count`
  - `seen_errors`
  - `patch_history`
  - `last_stable_snapshot`
  - `failure_reason`
- 实现回退机制：
  - 连续失败时回退到上一稳定版本。
  - 超过 3 次失败后结束并输出人工可读报告。

### 技术指导

- 不建议让 Coder 每次自由重写。应把诊断结果转成明确约束：
  - 只能修改哪些文件。
  - 必须保留哪些函数签名。
  - 失败测试名是什么。
  - 上一次补丁为什么失败。
- 错误分类可以先用规则实现，再交给 LLM 总结：
  - `SyntaxError`
  - `ModuleNotFoundError`
  - `AssertionError`
  - `TimeoutExpired`
  - `PatchApplyError`

### 验收标准

- 至少 5 个示例 Bug 能自动跑完流程。
- Patch 应用失败时不会破坏原项目。
- 失败报告能清晰说明尝试过的补丁、测试日志和最终失败原因。

## 第 6 周：可观测性与 Dashboard

### 本周目标

完成可视化展示，让系统具备“面试可讲、演示可看”的效果。

### 主要任务

- 集成 LangSmith 或 Phoenix：
  - 记录每次 LLM 调用。
  - 记录 Agent 节点耗时。
  - 记录 Token 消耗。
  - 记录工具调用成功率。
- 实现 Streamlit 或 Gradio Dashboard：
  - 输入 Issue 和仓库路径。
  - 展示 LangGraph 节点状态。
  - 展示 Product/Coder/Tester 消息。
  - 展示检索到的代码片段。
  - 展示 Diff。
  - 展示沙箱测试日志。
  - 展示最终修复报告。
- 增加运行记录持久化：
  - 每次任务保存为一个 run id。
  - 保存 `state.json`、`patch.diff`、`test.log`、`report.md`。

### 技术指导

- Dashboard 不需要做成复杂前端，重点是信息层次清晰：
  - 左侧：输入和运行配置。
  - 中间：当前图状态、Agent 消息、Diff。
  - 右侧或底部：测试日志、指标、最终结论。
- 节点状态可用表格或 Mermaid 图展示，不必一开始做动态动画。
- 日志要注意脱敏，不要把 `.env` 和 API Key 输出到界面。

### 验收标准

- 可以通过 Web 页面触发一次端到端修复。
- 页面能展示 Agent 决策轨迹、测试日志和 Diff。
- 每次运行结果能保存，方便后续 Benchmark 复盘。

## 第 7 周：Benchmark 数据集与量化评测

### 本周目标

构建本地 Bug 修复评测集，并用数据证明多 Agent 闭环相比单 Agent 的优势。

### 主要任务

- 构建 20-30 个本地 Bug 任务：
  - 语法错误。
  - 单函数逻辑错误。
  - 跨文件调用错误。
  - 测试缺失。
  - 依赖导入错误。
  - 简单重构任务。
- 为每个任务定义：
  - 初始代码版本。
  - Issue 描述。
  - 目标测试命令。
  - 预期通过测试。
  - 难度标签。
- 实现 Baseline：
  - 单 Agent Zero-shot 修复。
  - 本系统多 Agent 修复。
- 统计指标：
  - 修复成功率。
  - 平均重试次数。
  - 平均耗时。
  - Token 成本。
  - Patch 应用失败率。
  - 测试通过率。

### 技术指导

- Benchmark 优先本地可控，不必直接接 SWE-bench。
- 每个任务应尽量小而清晰，保证失败原因来自目标 Bug，而不是环境不稳定。
- 用 `evaluation/run_benchmark.py` 批量执行，输出 CSV 或 JSONL。
- 用 notebook、matplotlib 或 Streamlit 图表生成结果图。

### 验收标准

- 至少完成 20 个 Bug 任务。
- 能一键运行对比实验。
- 生成包含关键指标的 `benchmark_results.csv` 和图表。

## 第 8 周：工程打磨、报告与面试材料

### 本周目标

完成细节丰富与测评要求，将项目整理成可展示、可复现、可写进简历的最终版本。

### 主要任务

- 工程质量：
  - 补齐单元测试。
  - 覆盖率目标达到 80% 左右。
  - 补充异常处理和日志。
  - 整理配置和启动命令。
- 文档交付：
  - README 完整化。
  - 系统架构图。
  - 使用教程。
  - Benchmark Report。
  - 面试讲解稿。
- 展示材料：
  - 录制或准备一次完整 Demo 流程。
  - 准备 3 个代表性案例：简单 Bug、跨文件 Bug、失败后重试成功。
  - 准备简历项目描述。
- 最终复盘：
  - 总结系统优势。
  - 总结失败案例。
  - 总结未来优化方向。

### 技术指导

- README 建议包含：
  - 项目背景。
  - 系统架构。
  - 技术栈。
  - 快速开始。
  - Demo 截图。
  - Benchmark 结果。
  - 设计亮点。
  - 已知限制。
- Benchmark Report 建议包含：
  - 数据集构造方法。
  - Baseline 设计。
  - 指标定义。
  - 实验结果表格。
  - 成功/失败案例分析。
- 面试高频问题准备：
  - 为什么用 LangGraph 而不是普通 Chain？
  - 如何避免 Agent 死循环？
  - 如何控制上下文长度？
  - 如何保证沙箱安全？
  - 为什么用 Diff 而不是重写整个文件？
  - 如何评价系统是否真的有效？

### 验收标准

- 项目可以按 README 从零启动。
- 端到端 Demo 稳定可运行。
- Benchmark 报告包含真实数据和图表。
- 简历描述能量化体现修复成功率、成本、耗时或检索效果。

## 关键风险与应对

| 风险 | 表现 | 应对方案 |
| --- | --- | --- |
| LLM 输出不可解析 | Diff 格式错误、JSON 不合法 | 使用 Pydantic/Instructor 约束输出；失败后让模型只修复格式 |
| 自动修改破坏项目 | 大范围重写、误改无关文件 | 限制可修改文件；保存快照；使用 Diff；支持回退 |
| 沙箱不稳定 | Docker 环境差异导致测试失败 | 固定基础镜像；记录依赖安装日志；提供本地非 Docker fallback |
| 检索噪声高 | 召回无关代码，修复失败 | AST 符号索引 + Top-K + rerank + 依赖补充 |
| Agent 死循环 | 多轮修复无进展 | 设置最大重试次数；记录重复错误；触发 Product Agent 重新规划 |
| Benchmark 不公平 | Baseline 与本系统条件不一致 | 固定模型、温度、任务、测试命令和超时配置 |

## 推荐里程碑

- Week 2 Milestone：LangGraph 多 Agent 流程跑通。
- Week 4 Milestone：最小自动修复闭环完成，满足“4 周内框架搭建”要求。
- Week 6 Milestone：Dashboard 和可观测链路完成。
- Week 8 Milestone：Benchmark、报告、文档和演示完成，满足“8 周内细节丰富与测评”要求。

## 最小可行版本范围

如果时间紧张，优先保证以下能力：

1. Python 项目 Bug 修复。
2. LangGraph 三 Agent 闭环。
3. AST 基础检索。
4. Diff 应用。
5. pytest 沙箱执行。
6. 失败日志驱动重试。
7. 10 个以上 Benchmark 任务。
8. 一份可展示的 Dashboard 或运行报告。

这些能力足以支撑项目作为简历亮点，并能应对大部分技术追问。
