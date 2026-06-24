# AlgoSolver 项目计划

本计划面向"基于 LangGraph 的多智能体协同算法题目自动解答系统"。项目总周期控制在 8 周内：前 4 周完成可运行的解题闭环，后 4 周完成 Reflexion 增强、评测体系、工程打磨和展示材料。

## 总体目标

- 第 1-4 周：完成项目骨架、LangGraph 多 Agent 主流程（Analyzer → Coder → Tester → Diagnose）、Docker 沙箱集成和最小可用闭环。
- 第 5-8 周：增强 Reflexion 诊断能力、构建算法题目评测集、多基线对比实验、工程打磨和最终报告。
- 最终交付：可运行代码库、单元测试与覆盖率报告、20-30 道算法题目的评测报告、简历与面试展示材料。

## 技术路线总览

项目采用 Python 作为主语言，核心技术栈如下：

- Agent 编排：LangGraph，用显式 StateGraph 管理 Analyzer Agent、Coder Agent、Tester Agent 和 Reflexion Loop。
- 结构化输出：Pydantic 或 Instructor，约束 Agent 输出题目分析、解题方案、代码、测试诊断等结构化对象。
- 代码生成：从零生成完整 Python 解答代码（非 Diff），包含函数定义、类型注解、边界条件处理。
- 执行沙箱：Docker SDK for Python，限制网络、CPU、内存、运行时间和临时目录。
- 测试框架：pytest，用测试结果驱动 Agent 自修复。
- LLM 适配：统一的 LLM Provider 抽象层，支持 OpenAI / DeepSeek / 智谱等多 Provider。

---

## 补充设计：系统架构深度解析

### 整体分层架构

系统采用经典的三层架构，每层职责清晰、接口明确，便于独立测试和替换：

```
┌──────────────────────────────────────────────────────────────────┐
│                     Presentation Layer                           │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │   CLI (click/typer)                                        │  │
│  │   algosolver solve --problem "..." --test tests/            │  │
│  └──────────────────────────┬─────────────────────────────────┘  │
│                             │ Python API                          │
├─────────────────────────────┼────────────────────────────────────┤
│                     Application Layer                            │
│  ┌──────────────────────────┴──────────────────────────────────┐│
│  │              LangGraph StateGraph Engine                     ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐     ││
│  │  │Analyzer │  │ Coder   │  │ Tester  │  │ Diagnose  │     ││
│  │  │Agent    │─>│Agent    │─>│Agent    │─>│Agent      │     ││
│  │  └─────────┘  └─────────┘  └─────┬───┘  └─────┬─────┘     ││
│  │                      ┌───────────┘            │           ││
│  │                      │  Reflexion Loop         │           ││
│  │                      │  (Diagnose → Coder →    │           ││
│  │                      │   Tester)               │           ││
│  │                      └─────────────────────────┘           ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│                     Domain / Service Layer                        │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐  │
│  │ LLM Provider Layer   │  │ Docker Sandbox                    │  │
│  │ - OpenAI / DeepSeek  │  │ - Container Lifecycle             │  │
│  │ - Structured Output  │  │ - Test Executor                   │  │
│  │ - Prompt Templates   │  │ - Result Parser                   │  │
│  │ - Token Counter      │  │ - Resource Limits                 │  │
│  └──────────────────────┘  └──────────────────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Evaluation Engine                                           │  │
│  │ - Benchmark Runner / Metric Calculator / Report Generator   │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### LangGraph StateGraph 详细设计

#### 全局 State 定义 (AlgoState)

`AlgoState` 是全系统的核心数据结构，使用 Pydantic BaseModel 定义，LangGraph 在每个节点间传递并增量更新。State 按职责分为六个维度：

**1. 题目定义维度：**
- `problem_description`：用户输入的算法题目原文
- `test_file`：测试用例文件路径
- `constraints`：题目中的约束条件（如数组长度、数值范围）

**2. 分析结果维度：**
- `problem_analysis`：Analyzer Agent 的结构化输出（`ProblemAnalysis` 对象），包含：
  - `algorithm_type`：算法类型（DP / Greedy / Graph / Tree / BinarySearch / TwoPointers / Math / Sorting / 等）
  - `time_complexity_target`：目标时间复杂度
  - `space_complexity_target`：目标空间复杂度
  - `edge_cases`：边界条件列表
  - `approach_summary`：解题思路摘要
  - `example_input_output`：题目示例的输入输出

**3. 代码轨迹维度：**
- `solution_code`：当前轮的 Python 解答代码
- `code_history`：历轮代码快照列表
- `best_solution`：通过测试用例最多的代码版本
- `best_passed_count`：最佳代码通过的测试用例数

**4. 测试结果维度：**
- `test_result`：当前测试结果（`TestResult` 对象），包含：
  - `exit_code`、`stdout`、`stderr`、`duration_seconds`
  - `passed_count`、`failed_count`、`total_count`
  - `failed_test_names`：失败的测试函数名列表
  - `error_category`：错误分类（WrongAnswer / TimeLimitExceeded / RuntimeError / SyntaxError / ImportError）
- `test_history`：历次测试结果

**5. 流程控制维度：**
- `retry_count`：当前重试次数
- `max_retries`：最大重试次数（默认 3）
- `seen_error_signatures`：已见过的错误签名集合，用于死循环检测
- `status`：当前状态（init → analyzing → coding → testing → success / failed）
- `termination_reason`：终止原因

**6. 可观测维度：**
- `agent_messages`：所有 Agent 的对话记录
- `node_trajectory`：节点访问轨迹
- `total_tokens` / `total_cost`：累计 Token 消耗和成本
- `start_time` / `end_time`：任务起止时间

#### 节点职责与接口契约

每个节点遵循统一的接口契约：

- **Node 函数**：接收 `AlgoState`，返回 `dict`（需要增量更新的字段），不修改传入的 state 对象
- **条件边函数**：接收 `AlgoState`，返回字符串字面量（`"success"` / `"need_retry"` / `"max_retries_exceeded"`），决定下一步跳转

**节点详细设计：**

| 节点 | 输入依赖 | 输出字段 | 失败策略 |
|------|---------|---------|---------|
| `analyzer_node` | `problem_description` | `problem_analysis` | 让 LLM 重试 2 次，仍失败则终止 |
| `coder_node` | `problem_analysis`, `solution_code`(如有), `failure_feedback`(如有) | `solution_code` | 让 LLM 修复格式，最多 2 次 |
| `tester_node` | `solution_code`, `test_file` | `test_result` | 超时 kill；沙箱异常则标记 `error_category="SandboxError"` |
| `diagnose_node` | `test_result`, `solution_code`, `retry_count`, `problem_analysis` | `failure_feedback`(错误分类 + 修改建议) | 规则引擎 + LLM 兜底 |
| `fail_node` | 全量 state | `termination_reason`, 最终报告 | 输出最佳代码 + 失败分析 |

### 图结构定义（含条件路由）

使用 LangGraph 的 `StateGraph` 构建工作流图：

**固定边（顺序执行）：**
`analyzer` → `coder` → `tester`

**条件边（tester 之后的三分支路由）：**
- 测试全部通过（`exit_code == 0` 且 `failed_count == 0`）→ `END`（成功终止）
- 测试失败且未超重试上限 → `diagnose` → `coder`（Reflexion 重试回路）
- 测试失败且超过重试上限 → `fail` → `END`（失败终止）

**入口点：** `analyzer` 节点

### 条件路由逻辑

路由决策基于 `AlgoState` 中的测试结果和重试状态：

1. **成功判定**：若 `test_result.exit_code == 0` 且 `test_result.failed_count == 0`，返回 `"success"`，流程进入 END
2. **死循环检测**：对当前错误生成签名（error_category + failed_test_names），若该签名已在 `seen_error_signatures` 集合中，说明是重复错误，加速 `retry_count` 计数（额外 +1）
3. **上限判定**：若 `retry_count >= max_retries`，返回 `"max_retries_exceeded"`，进入失败总结节点
4. **最佳代码更新**：每轮测试后比较 `passed_count`，若当前代码通过更多用例，则更新 `best_solution`
5. **默认重试**：以上条件均不满足时返回 `"need_retry"`，触发 Diagnose → Coder 回路

---

## 补充设计：安全模型

### 威胁模型

系统面临的主要安全威胁：

| 威胁 | 攻击面 | 风险等级 |
|------|--------|---------|
| 恶意代码执行 | Docker 沙箱内的用户代码 | 高 |
| Prompt 注入 | 用户输入的题目描述 | 中 |
| 敏感信息泄露 | LLM 上下文、日志输出 | 中 |
| 资源耗尽 | 无限循环、内存泄漏 | 中 |
| 供应链攻击 | 依赖包、Docker 基础镜像 | 低 |

### Docker 沙箱安全策略

沙箱采用多层安全限制，通过 Docker SDK 在启动容器时配置以下约束：

| 安全维度 | 限制措施 | 说明 |
|----------|---------|------|
| **网络隔离** | `network_disabled=True` | 完全禁止容器访问外网，防止代码注入后对外通信 |
| **用户权限** | `user="nobody"` | 以非 root 用户运行 |
| **Capabilities** | `cap_drop=["ALL"]` | 丢弃所有 Linux capabilities |
| **提权防护** | `no_new_privileges=True` | 禁止进程通过 setuid/setgid 等机制提权 |
| **内存限制** | `mem_limit="512m"` | 限制容器最大内存使用 |
| **CPU 限制** | `cpu_quota=50000, cpu_period=100000` | 限制最多使用 50% 单核 CPU |
| **超时控制** | `timeout_seconds=30` | 超时后 Docker 强制 kill 容器 |
| **工作区隔离** | 每次执行使用临时目录，执行后清理 | 防止文件残留和相互干扰 |

### Prompt 注入防护

- 检测并拒绝包含已知注入模式的题目描述
- 限制题目描述长度上限为 3000 字符
- 日志输出前脱敏处理（API Key、Token 等敏感信息）

---

## 补充设计：Prompt 工程策略

### Agent Prompt 架构

每个 Agent 使用结构化的 System Prompt + 动态 Context 注入：

```
┌─────────────────────────────────────────┐
│           System Prompt (固定)           │
│  - Agent 角色定义                        │
│  - 输出格式约束 (JSON Schema)             │
│  - 行为规则与边界                         │
├─────────────────────────────────────────┤
│           Problem Context (动态)          │
│  - 算法题目原文                          │
│  - Analyzer 的题目分析（Coder 使用）      │
├─────────────────────────────────────────┤
│           Feedback Context (动态)         │
│  - 上一轮测试失败详情                    │
│  - Diagnose 的修改建议                   │
│  - 本轮约束条件                          │
└─────────────────────────────────────────┘
```

### 各 Agent Prompt 设计要点

**Analyzer Agent：**
- 输入：算法题目原文
- 输出：`{"algorithm_type": "DP", "time_complexity_target": "O(n)", "space_complexity_target": "O(n)", "edge_cases": ["空数组", "单元素"], "approach_summary": "使用动态规划...", "example_input_output": [{"input": "...", "output": "..."}]}`
- 关键约束：不确定时必须标记 `"uncertain": true`；优先识别为常见算法模式

**Coder Agent：**
- 输入：题目分析 + 测试失败反馈（如有）
- 输出：`{"solution_code": "def solve(nums, target):\n    ...", "explanation": "...", "time_complexity": "O(n)", "space_complexity": "O(n)"}`
- 关键约束：代码必须是独立可执行的 Python 函数；必须包含类型注解；必须处理边界条件；禁止使用外部依赖

**Tester Agent：**
- 输入：solution_code + test_file
- 输出：`{"exit_code": 0, "passed_count": 5, "failed_count": 2, "error_category": "WrongAnswer", "failed_test_names": ["test_edge_case_empty"], "error_details": "..."}`
- 关键约束：区分"测试框架本身出错"和"业务逻辑错误"；超时单独归类

**Diagnose Agent：**
- 输入：TestResult + 失败历史 + retry_count
- 输出：`{"error_category": "WrongAnswer", "root_cause": "未处理空数组输入", "suggested_fix": "在函数开头增加 if not nums: return [] 判断", "should_retry": true, "new_constraints": ["必须检查输入为空的情况"]}`
- 关键约束：重试 > 2 次时必须缩小修改范围；给出具体的代码修改建议而非泛泛的"检查边界条件"

---

## 补充设计：API 与集成接口

### REST API 设计

系统对外暴露 RESTful API，供外部工具和前端调用：

**`POST /api/v1/problems`** — 创建并启动解题任务
- 请求体：`problem_description`（必填）、`test_file`（必填）、`max_retries`（默认 3）、`model`（默认 `"gpt-4o"`）
- 响应：`201 Created`，返回 `task_id` 和初始 `status`

**`GET /api/v1/problems/{task_id}`** — 查询任务状态与结果
- 响应体包含：`task_id`、`status`、`problem_analysis`、`solution_code`、`test_result`、`agent_messages`、`total_tokens`、`total_cost`

**`POST /api/v1/benchmarks`** — 批量运行评测

### CLI 接口设计

提供命令行工具 `algosolver`，覆盖三种核心使用场景：

**单题解答 (`algosolver solve`)：**
- 参数：`--problem`（题目描述文件路径）、`--test`（测试文件路径）、`--max-retries`（默认 3）、`--model`（模型选择）、`--output`（输出目录）
- 行为：同步执行完整的分析 → 编码 → 测试 → 反思闭环，结果写入指定输出目录

**批量评测 (`algosolver benchmark`)：**
- 参数：`--dataset`（数据集 JSONL 路径）、`--baseline`（基线类型）、`--parallel`（并行任务数）、`--output`（结果输出目录）
- 行为：批量运行数据集中的所有题目，生成评测报告和可视化图表

---

## 补充设计：可扩展性设计

### 多语言支持扩展（预留）

```
                     ┌──────────────────┐
                     │  LanguagePlugin  │  (Abstract Base Class)
                     │  + generate_code()│
                     │  + get_test_cmd()│
                     │  + get_image()   │
                     └──────┬───────────┘
            ┌───────────────┼───────────────┐
     ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
     │PythonPlugin │ │ CppPlugin   │ │ JavaPlugin  │
     │(当前实现)    │ │(后续扩展)   │ │(后续扩展)   │
     └─────────────┘ └─────────────┘ └─────────────┘
```

- 第一阶段：Python（当前实现）
- 第二阶段：C++（g++ 编译 + 沙箱执行）
- 第三阶段：Java（javac 编译 + JUnit 测试）

### 模型提供商适配

通过统一的 `LLMProvider` 抽象层解耦 Agent 逻辑与具体模型 API：
- 抽象接口：`chat(messages, tools, schema)` + `count_tokens(text)`
- 具体实现：`OpenAIProvider`、`DeepSeekProvider`、`ZhipuProvider`、`OllamaProvider`
- 切换模型只需修改配置中的 provider 名称，Agent 代码无需改动

---

## 补充设计：测试策略

### 测试金字塔

```
           ┌──────┐
           │ E2E  │  ~8 个：完整题目 → 解答 → 验证流程
           ├──────┤
           │ 集成  │  ~20 个：模块间交互 (Agent + Sandbox + Graph)
           ├──────────┤
           │   单元测试  │  ~80+ 个：每个模块的独立测试
           └──────────┘
```

### 测试目录结构

```
tests/
├── unit/
│   ├── test_sandbox.py              # Docker 沙箱执行
│   ├── test_state_models.py         # Pydantic 模型验证
│   ├── test_prompts.py              # Prompt 模板构建
│   ├── test_error_classifier.py     # 错误分类逻辑
│   └── test_providers.py            # LLM Provider 适配
├── integration/
│   ├── test_graph_flow.py           # LangGraph 流程 (Mock LLM)
│   ├── test_sandbox_pytest.py       # 沙箱内 pytest 执行
│   └── test_agent_pipeline.py       # Analyzer → Coder → Test 链路
├── e2e/
│   ├── test_two_sum.py              # 端到端：Two Sum
│   ├── test_fibonacci.py            # 端到端：斐波那契
│   ├── test_valid_parentheses.py    # 端到端：有效括号
│   ├── test_binary_search.py        # 端到端：二分搜索
│   └── test_retry_on_failure.py     # 端到端：失败重试
├── conftest.py                       # 共享 fixture
└── fixtures/
    ├── problems/                     # 测试用题目
    │   ├── two_sum.md
    │   ├── fibonacci.md
    │   └── binary_search.md
    └── mock_llm_responses/           # Mock LLM 返回
```

### 覆盖率目标

| 模块 | 目标覆盖率 | 备注 |
|------|-----------|------|
| `agents/` | ≥ 75% | LLM 调用层 Mock，Prompt 构建层实测 |
| `graph/` | ≥ 80% | Mock LLM 后测试流程 |
| `sandbox/` | ≥ 70% | 依赖 Docker，部分测试需标记 `@pytest.mark.docker` |
| `evaluation/` | ≥ 80% | 核心指标计算逻辑 |
| `llm/` | ≥ 75% | Provider 适配和 Token 计数 |

---

## 补充设计：评估指标体系

### 核心指标定义

| 指标 | 公式 | 说明 |
|------|------|------|
| **解题成功率 (Solve Rate)** | `solved_tasks / total_tasks` | 所有测试用例通过视为成功 |
| **平均解题耗时 (MTTS)** | `sum(solve_duration) / total_tasks` | 包含 LLM 调用 + 测试时间 |
| **Token 效率** | `total_tokens / solved_tasks` | 每次成功解题的平均 Token 消耗 |
| **首次通过率** | `first_try_passes / total_tasks` | 无需重试即通过的比例 |
| **平均重试次数** | `sum(retries) / total_tasks` | 仅统计最终成功的题目 |
| **按难度分层成功率** | `solved_by_difficulty / total_by_difficulty` | Easy / Medium / Hard 分别统计 |
| **按类型分层成功率** | `solved_by_type / total_by_type` | DP / 贪心 / 图论等分别统计 |

### 对比基线

```
实验组 A: 本系统 (LangGraph Multi-Agent + Reflexion)
实验组 B: Single Agent Zero-shot (一次 LLM 调用生成解答)
实验组 C: Single Agent + Retry (单 Agent 可重试 3 次)
实验组 D: Multi-Agent without Reflexion (去掉诊断回路)
```

---

## 第 1 周：需求冻结与工程骨架

### 本周目标

完成项目定位、模块边界和代码仓库基础结构，确保项目从"代码自动修复"完全转向"算法题目解答"方向。

### 主要任务

- 明确系统输入输出：
  - 输入：算法题目描述（Markdown/文本）、pytest 测试文件
  - 输出：题目分析、解答代码、测试结果、解题报告
- 重组项目目录：
  - 将 `src/mse/` 重命名为 `src/algosolver/`
  - 保留 `src/algosolver/sandbox/`（复用现有 Docker 沙箱实现）
  - 新增 `src/algosolver/agents/`（Analyzer、Coder、Tester、Diagnose）
  - 新增 `src/algosolver/graph/`（LangGraph 图定义和 State）
  - 新增 `src/algosolver/llm/`（LLM Provider 抽象层）
  - 新增 `src/algosolver/cli/`（命令行入口）
  - 清理不再需要的目录（`retrieval/`、`patching/`）
- 初始化配置文件：
  - 填充 `pyproject.toml`（依赖、lint、type check 配置）
  - 填充 `requirements.txt`
  - 更新 `.env.example`
- 定义核心数据模型：
  - `ProblemSpec`：题目描述 + 测试文件
  - `ProblemAnalysis`：算法类型、复杂度目标、边界条件
  - `TestResult`：测试执行结果
  - `AlgoState`：全局状态

### 技术指导

- `AlgoState` 是全系统关键，应包含六个维度的字段（见上文 State 设计）
- 所有 Agent 的输入输出都优先使用 Pydantic 模型
- 先支持 Python 解答，不追求多语言
- 沙箱模块已有完整实现（`src/mse/sandbox/sandbox.py`），直接迁移复用

### 验收标准

- 项目能通过 `pytest` 跑通基础测试
- `AlgoState`、核心 Pydantic 模型和目录结构完成
- README 和 Plan.md 反映新的项目定位
- 沙箱模块迁移完成且测试通过

---

## 第 2 周：LangGraph 多 Agent 主流程

### 本周目标

完成 Analyzer、Coder、Tester、Diagnose 四个 Agent 的最小可运行图，让一道算法题能按固定路径流转。

### 主要任务

- 实现 LLM Provider 抽象层：
  - `LLMProvider` 基类（`chat()` + `count_tokens()`）
  - `OpenAIProvider` 实现
  - `DeepSeekProvider` 实现（兼容 OpenAI SDK）
- 实现 Agent 基类和 Prompt 模板：
  - `BaseAgent`：统一的消息构建、LLM 调用、Structured Output 解析
  - 四个 Agent 各自的 System Prompt 模板
- 实现 LangGraph 节点：
  - `analyzer_node`：解析题目，输出结构化 ProblemAnalysis
  - `coder_node`：根据分析和反馈生成 Python 代码
  - `tester_node`：将代码注入沙箱，运行 pytest，收集结果
  - `diagnose_node`：分析失败日志，给出修改建议
- 实现条件边：
  - 测试全部通过：进入 `END`
  - 测试失败且 `retry_count < max_retries`：进入 `diagnose_node` → `coder_node`
  - 达到重试上限：进入 `fail_node` → `END`

### 技术指导

- 图结构：
```text
START
  → analyzer_node
  → coder_node
  → tester_node
  → route_by_test_result
       → END (全部通过)
       → diagnose_node → coder_node (失败可重试)
       → fail_node → END (超过上限)
```

- 本周可以使用 Mock LLM 或规则逻辑，使图先稳定运行
- 条件边建议用枚举状态：`SUCCESS` / `NEED_RETRY` / `FAILED`

### 验收标准

- 能用一个示例题目触发完整 LangGraph 流程
- 能打印每个 Agent 的消息、节点跳转路径和最终状态
- 至少有 3 个图流程测试：成功、失败后重试成功、超过重试上限失败

---

## 第 3 周：沙箱集成与测试闭环

### 本周目标

完成沙箱模块与 Tester Agent 的深度集成，确保代码能在安全环境中正确执行并分类错误。

### 主要任务

- 沙箱模块迁移与增强：
  - 将现有 `Sandbox` 类从 `src/mse/sandbox/` 迁移到 `src/algosolver/sandbox/`
  - 增强错误分类能力（WrongAnswer / TLE / RuntimeError / SyntaxError / ImportError）
  - 添加 pytest 输出解析，提取每个测试用例的通过/失败状态
  - 设置合理的超时和内存限制（算法题执行通常 < 5s）
- 实现 Tester Agent 的测试编排逻辑：
  - 将 solution_code 写入临时 Python 文件
  - 与 test_file 一起注入沙箱
  - 运行 pytest -v
  - 解析结果并结构化返回
- 建立最小示例题目：
  - Two Sum（Easy）
  - Fibonacci（Easy）
  - Valid Parentheses（Easy）
  - 每道题包含 problem.md + test_solution.py

### 技术指导

- 超时限制建议按题目难度分级：Easy 5s、Medium 10s、Hard 30s
- 错误分类先用规则实现（正则匹配 stderr），再交给 LLM 总结
- 沙箱日志要完整保留 stdout 和 stderr，方便调试

### 验收标准

- 沙箱能正确执行 Python 解答代码并运行 pytest
- 能正确分类：全部通过、答案错误、超时、运行时错误、语法错误
- 3 道示例题目的测试用例覆盖正常和边界情况

---

## 第 4 周：最小闭环打通

### 本周目标

完成"分析 → 编码 → 测试 → 反思 → 重试"的完整闭环，系统能自主解决简单算法题目。

### 主要任务

- 打通 LangGraph 全流程：
  - Analyzer 分析题目 → Coder 生成代码 → Tester 沙箱执行 → 全部通过则输出
  - 若失败 → Diagnose 分析 → Coder 基于反馈重新编码 → Tester 再次测试
- 实现最佳代码保留机制：
  - 每次测试后比较通过用例数
  - 保留通过最多的代码版本
- 实现 CLI 入口：
  - `algosolver solve` 命令
  - 支持从文件读取题目和测试
- 建立端到端测试：
  - Two Sum：首次失败 → 诊断 → 修正 → 成功
  - 验证完整闭环可运行

### 技术指导

- 第 4 周不追求复杂题目，重点是闭环稳定
- 日志记录要完整：节点轨迹、Agent 输出、每轮代码、测试日志
- 每轮迭代的代码快照保存到 runs/ 目录

### 验收标准

- 第 4 周结束时必须可以运行一个端到端 Demo
- Demo 能展示至少一次测试失败后的自动修复重试
- Two Sum 题目能在 3 次重试内通过所有测试用例
- 具备基础日志和运行记录持久化

---

## 第 5 周：Reflexion 增强与错误诊断优化

### 本周目标

让解题系统从"能跑"变成"更准"，重点解决错误诊断粗糙和无效重试问题。

### 主要任务

- 增强 Diagnose Agent：
  - 支持更细粒度的错误分类：IndexError、KeyError、TypeError、AttributeError 等
  - 将错误位置与代码行号关联，提供精确的修改定位
  - 分析超时原因：是否时间复杂度不达标（O(n²) vs O(n)）
- 增强 Coder Agent：
  - 在重试时必须引用上一轮 Diagnose 的修改建议
  - 重试 > 2 次时强制缩小修改范围（只改特定函数/逻辑块）
  - 支持"部分正确"反馈：告知哪些用例已通过、哪些未通过
- 增加状态控制：
  - `seen_error_signatures`：错误签名去重
  - `code_history`：每轮代码快照
  - `best_solution` + `best_passed_count`：最佳代码追踪
- 实现死循环检测：
  - 同一错误签名连续出现 2 次 → 触发 Analyzer 重新规划
  - 连续 3 次 retry 无进展 → 终止并输出最佳尝试

### 技术指导

- 诊断时给 Coder 具体的约束，而不是泛泛的"修复 bug"：
  - "test_edge_case_empty 失败，需要在函数开头增加空数组判断"
  - "test_large_input 超时，当前 O(n²) 需优化到 O(n log n)"
- 错误分类层次：
  - Level 1（规则）：SyntaxError、ImportError、Timeout
  - Level 2（LLM）：WrongAnswer 的根因分析、TLE 的复杂度分析

### 验收标准

- 至少 5 道 Easy/Medium 题目能自动跑完全流程
- 错误诊断能精确定位到代码行和原因
- 失败报告能清晰说明尝试过的修改和最终失败原因

---

## 第 6 周：算法题目集构建与批量评测

### 本周目标

构建本地算法题目评测集，并用数据证明多 Agent 闭环相比单 Agent 的优势。

### 主要任务

- 构建 20-30 道算法题目：
  - Easy（8-10 道）：数组、字符串、哈希表、简单数学
  - Medium（7-10 道）：DP 入门、二叉树遍历、图 BFS/DFS、二分搜索变体
  - Hard（5-7 道）：复杂 DP、单调栈/队列、高级图论
- 为每道题定义：
  - `problem.md`：题目描述（中文 or 英文）
  - `test_solution.py`：pytest 测试文件（含正常用例和边界用例）
  - `solution.py`：参考答案（用于验证测试正确性）
  - `difficulty`、`algorithm_type` 标签
- 实现 Baseline：
  - Single Agent Zero-shot：一次 LLM 调用
  - Single Agent + Retry：同一 Agent 最多 3 次重试
  - Multi-Agent without Reflexion：去掉 Diagnose 回路
  - 本系统：完整 LangGraph + Reflexion
- 实现指标计算和报告生成：
  - 解题成功率、平均重试次数、Token 消耗
  - 按难度分层统计
  - 按算法类型分层统计

### 技术指导

- 题目优先选 LeetCode 经典题，保证测试用例质量
- 每道题的测试用例应包含：普通用例、边界用例（空输入、单元素、极值）、性能用例（大数据量）
- 用 `evaluation/runner.py` 批量执行，输出 CSV 或 JSONL
- 用 matplotlib 生成对比图表

### 验收标准

- 至少完成 20 道题目及其测试用例
- 能一键运行对比实验
- 生成包含关键指标的评测报告和可视化图表
- 多 Agent 系统相比单 Agent Zero-shot 有显著提升

---

## 第 7 周：工程打磨与质量保障

### 本周目标

完成代码质量、测试覆盖率和文档的全面提升。

### 主要任务

- 工程质量：
  - 补齐单元测试，覆盖率达到 75%+
  - 补充异常处理和日志
  - 完善 ruff + mypy + pre-commit 配置
  - CI/CD 流水线（GitHub Actions）：lint → type check → test
- 文档交付：
  - README 最终完善（架构图、使用教程、Demo 截图）
  - API 参考文档
  - 评测报告（Benchmark Report）
  - 贡献指南
- 运行记录与复盘：
  - 每次运行自动保存完整轨迹（state.json、代码快照、测试日志、报告）
  - 支持运行记录回放和对比分析

### 技术指导

- `pyproject.toml` 配置要点：
  - 项目元数据：`name="algosolver"`, `version="0.1.0"`, `requires-python=">=3.11"`
  - 核心依赖：langgraph、langchain、openai、docker、pydantic、instructor、pytest、python-dotenv
  - dev 依赖：ruff、mypy、pre-commit、pytest-cov
  - ruff 规则：E/F/I/N/W/UP/B/C4/SIM
  - mypy：strict=true
- pytest markers：`docker`（需要 Docker）、`slow`（耗时 > 10s）、`llm`（调用真实 LLM）

### 验收标准

- 单元测试覆盖率 ≥ 75%
- README 可指导新用户从零启动
- 端到端 Demo 稳定可运行

---

## 第 8 周：面试材料与最终交付

### 本周目标

将项目整理成可展示、可复现、可写进简历的最终版本。

### 主要任务

- 展示材料：
  - 录制或准备一次完整 Demo 流程（Two Sum 失败→修正→成功）
  - 准备 3 个代表性案例：简单题一次通过、中等题失败后重试成功、困难题的解题过程
  - 准备简历项目描述（STAR 原则）
- 面试讲解稿：
  - 系统架构设计决策
  - 为什么用 LangGraph？
  - 如何避免死循环？
  - Reflexion 回路如何工作？
  - 沙箱安全如何保证？
  - 评测结果和量化指标
- 最终复盘：
  - 总结系统优势（成功案例）
  - 总结失败案例和原因分析
  - 总结未来优化方向

### 验收标准

- 项目可以按 README 从零启动
- 端到端 Demo 稳定可运行
- 评测报告包含真实数据和图表
- 简历描述能量化体现解题成功率、Token 成本、重试次数
- 面试高频问题准备充分

---

## 关键风险与应对

| 风险 | 表现 | 应对方案 |
| --- | --- | --- |
| LLM 输出不可解析 | JSON 格式错误、代码不可执行 | 使用 Pydantic/Instructor 约束输出；失败后让模型只修复格式 |
| Coder 生成错误算法 | 解题思路根本错误，修改无法收敛 | Analyzer 重新规划；超过 2 次无效重试后触发重分析 |
| 沙箱不稳定 | Docker 环境差异导致测试结果不一致 | 固定基础镜像；记录完整执行日志 |
| Agent 死循环 | 多轮修改无进展 | 错误签名去重；最大重试次数；最佳代码保留 |
| 超时处理不当 | 暴力解法在大量数据上超时 | Diagnose 检测复杂度问题；引导 Coder 优化算法 |
| 评测不公平 | Baseline 与本系统条件不一致 | 固定模型、温度、题目、测试用例和超时配置 |
| 成本过高 | 多 Agent + 重试导致 Token 消耗大 | 使用 DeepSeek 等低成本模型；设置 max_retries 和 budget 上限 |

---

## 推荐里程碑

- Week 2 Milestone：LangGraph 多 Agent 流程跑通（含 Mock LLM）。
- Week 4 Milestone：最小解题闭环完成，Two Sum 能自动解答通过。
- Week 6 Milestone：题目集和批量评测完成，多基线对比数据出炉。
- Week 8 Milestone：文档、面试材料和最终交付完成。

---

## 最小可行版本范围

如果时间紧张，优先保证以下能力：

1. Python 算法题解答（单函数型题目）。
2. LangGraph 四 Agent 闭环（Analyzer → Coder → Tester → Diagnose）。
3. Docker 沙箱 pytest 执行（已有实现）。
4. 测试失败驱动重试（Reflexion）。
5. 最佳代码保留机制。
6. 10 道以上题目及其测试用例。
7. 多基线对比评测。
8. 一份可展示的 CLI Demo 和运行报告。

这些能力足以支撑项目作为简历亮点，并能应对大部分技术追问。
