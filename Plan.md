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

---

## 补充设计：系统架构深度解析

### 整体分层架构

系统采用经典的四层架构，每层职责清晰、接口明确，便于独立测试和替换：

```
┌──────────────────────────────────────────────────────────────────┐
│                     Presentation Layer                           │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐  │
│  │   Streamlit/Gradio   │  │   CLI (click/typer)              │  │
│  │   Dashboard          │  │   mse-cli run --issue "..."      │  │
│  └──────────┬───────────┘  └───────────────┬──────────────────┘  │
│             └──────────────────────────────┘                     │
│                         │ HTTP/WebSocket                         │
├─────────────────────────┼────────────────────────────────────────┤
│                     Application Layer                            │
│  ┌──────────────────────┴──────────────────────────────────┐    │
│  │              LangGraph StateGraph Engine                  │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────┐   │    │
│  │  │Product  │  │Retriever│  │ Coder   │  │ Tester    │   │    │
│  │  │Agent    │─>│Node     │─>│Agent    │─>│Agent      │   │    │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────┬─────┘   │    │
│  │                      ┌────────────────────────┘         │    │
│  │                      │  Reflexion Loop                   │    │
│  │                      │  (Diagnose → Coder → Test)        │    │
│  │                      └────────────────────────────────── │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│                     Domain / Service Layer                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Codebase RAG │  │ Diff/Patch   │  │ Docker Sandbox        │   │
│  │ - AST Parser │  │ Engine       │  │ - Container Lifecycle │   │
│  │ - Chroma Vec │  │ - Unified    │  │ - Executor            │   │
│  │ - Reranker   │  │   Diff       │  │ - Result Parser       │   │
│  │ - Call Graph │  │ - S&R Blocks │  │ - Resource Limits     │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│  ┌──────────────┐  ┌──────────────────────────────────────┐     │
│  │ Observability│  │ Evaluation Engine                     │     │
│  │ - Traces     │  │ - Benchmark Runner                    │     │
│  │ - Metrics    │  │ - Metric Calculator                   │     │
│  │ - Cost Track │  │ - Report Generator                    │     │
│  └──────────────┘  └──────────────────────────────────────┘     │
│                                                                   │
├──────────────────────────────────────────────────────────────────┤
│                     Infrastructure Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ LLM      │  │ Vector   │  │ Docker   │  │ File System  │    │
│  │ Provider │  │ Store    │  │ Daemon   │  │ (Snapshot)    │    │
│  │ (OpenAI/ │  │ (Chroma) │  │          │  │              │    │
│  │ DeepSeek)│  │          │  │          │  │              │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### LangGraph StateGraph 详细设计

#### 全局 State 定义 (ThreadState)

`ThreadState` 是全系统的核心数据结构，使用 Pydantic BaseModel 定义，LangGraph 在每个节点间传递并增量更新。State 按职责分为七个维度：

**1. 任务定义维度：**
- `issue`：用户输入的 Issue 原文
- `repo_path`：目标代码仓库路径
- `test_command`：测试命令（默认 `pytest`）

**2. 上下文维度：**
- `retrieved_context`：检索到的代码块列表（`CodeChunk` 对象，包含 file_path、symbol_name、symbol_type、start_line、end_line、signature、content、imports、docstring、callers、callees）
- `project_outline`：项目大纲摘要
- `dependency_graph`：调用依赖图

**3. 规划维度：**
- `task_breakdown`：Product Agent 拆解的子任务列表
- `modification_targets`：允许修改的目标文件列表

**4. 修改轨迹维度：**
- `patches`：当前轮的补丁列表（`PatchBlock` 对象，包含 id、file_path、original、replacement、strategy（unified_diff / search_replace）、source_agent、applied、error）
- `patch_history`：每轮 patches 的历史快照
- `last_stable_snapshot`：最后一个稳定版本的文件系统备份路径

**5. 测试与诊断维度：**
- `test_result`：当前测试结果（`TestResult` 对象，包含 exit_code、stdout、stderr、duration_seconds、passed、failed、errors、error_details）
- `test_history`：历次测试结果
- `failure_reason`：失败原因描述
- `error_category`：错误分类（SyntaxError / ImportError / AssertionError / TimeoutExpired / PatchApplyError 等）

**6. 流程控制维度：**
- `retry_count`：当前重试次数
- `max_retries`：最大重试次数（默认 3）
- `seen_errors`：已见过的错误签名集合，用于死循环检测
- `status`：当前状态（init → planning → coding → testing → success / failed）
- `termination_reason`：终止原因

**7. 可观测维度：**
- `agent_messages`：所有 Agent 的对话记录（`AgentMessage` 对象，包含 agent、timestamp、role、content、token_usage）
- `node_trajectory`：节点访问轨迹
- `total_tokens` / `total_cost`：累计 Token 消耗和成本
- `start_time` / `end_time`：任务起止时间

#### 节点职责与接口契约

每个节点遵循统一的接口契约：

- **Node 函数**：接收 `ThreadState`，返回 `dict`（需要增量更新的字段），不修改传入的 state 对象
- **条件边函数**：接收 `ThreadState`，返回字符串字面量（`"success"` / `"need_retry"` / `"max_retries_exceeded"`），决定下一步跳转

**节点详细设计：**

| 节点 | 输入依赖 | 输出字段 | 失败策略 |
|------|---------|---------|---------|
| `product_node` | `issue`, `project_outline` | `task_breakdown`, `modification_targets` | 让 LLM 重试 2 次，仍失败则终止 |
| `retrieve_context_node` | `issue`, `task_breakdown`, `repo_path` | `retrieved_context`, `dependency_graph` | 降级到全文件扫描 |
| `coder_node` | `task_breakdown`, `retrieved_context`, `failure_reason` | `patches` (新增) | 让 LLM 修复格式，最多 2 次 |
| `patch_node` | `patches`, `repo_path` | `patches[*].applied`, `patches[*].error` | 回退未应用的 patch |
| `tester_node` | `repo_path`, `test_command` | `test_result` | 超时 kill；沙箱异常则标记 `error_category="SandboxError"` |
| `diagnose_node` | `test_result`, `patches`, `retry_count` | `failure_reason`, `error_category` | 规则引擎 + LLM 兜底 |
| `fail_node` | 全量 state | `termination_reason`, 最终报告 | 生成人类可读的失败报告 |

### 图结构定义（含条件路由）

使用 LangGraph 的 `StateGraph` 构建工作流图，通过 `add_node` 注册 7 个节点，`add_edge` 定义固定流转边，`add_conditional_edges` 在 `tester` 节点后实现三分支路由：

**固定边（顺序执行）：**
`product` → `retrieve_context` → `coder` → `patch` → `tester`

**条件边（tester 之后的三分支路由）：**
- 测试通过（`exit_code == 0`）→ `END`（成功终止）
- 测试失败且未超重试上限 → `diagnose` → `coder`（Reflexion 重试回路）
- 测试失败且超过重试上限 → `fail` → `END`（失败终止）

**入口点：** `product` 节点

### 条件路由逻辑

路由决策基于 `ThreadState` 中的测试结果和重试状态：

1. **成功判定**：若 `test_result.exit_code == 0`，返回 `"success"`，流程进入 END
2. **死循环检测**：对当前错误生成签名（error_signature），若该签名已在 `seen_errors` 集合中，说明是重复错误，加速 `retry_count` 计数（额外 +1）
3. **上限判定**：若 `retry_count >= max_retries`，返回 `"max_retries_exceeded"`，进入失败总结节点
4. **默认重试**：以上条件均不满足时返回 `"need_retry"`，触发 Diagnose → Coder 回路

设计要点：死循环检测不只看重试次数，还看错误是否"重复出现"——同一个错误反复出现比不同错误更有害，因此重复错误会加速触发终止，避免浪费 Token。

---

## 补充设计：安全模型

### 威胁模型

系统面临的主要安全威胁：

| 威胁 | 攻击面 | 风险等级 |
|------|--------|---------|
| 恶意代码执行 | Docker 沙箱内的用户代码 | 高 |
| Prompt 注入 | 用户输入的 Issue 描述 | 中 |
| 敏感信息泄露 | LLM 上下文、日志输出 | 中 |
| 资源耗尽 | 无限循环、超大文件 | 中 |
| 供应链攻击 | 依赖包、基础镜像 | 低 |

### Docker 沙箱安全策略

沙箱采用多层安全限制，通过 Docker SDK 在启动容器时配置以下约束：

| 安全维度 | 限制措施 | 说明 |
|----------|---------|------|
| **网络隔离** | `network_disabled=True` | 完全禁止容器访问外网，防止代码注入后对外通信 |
| **用户权限** | `user="nobody"` | 以非 root 用户运行，降低权限提升风险 |
| **Capabilities** | `cap_drop=["ALL"]` | 丢弃所有 Linux capabilities |
| **提权防护** | `no_new_privileges=True` | 禁止进程通过 setuid/setgid 等机制提权 |
| **内存限制** | `mem_limit="512m"` | 限制容器最大内存使用 |
| **CPU 限制** | `cpu_quota=50000, cpu_period=100000` | 限制最多使用 50% 单核 CPU |
| **超时控制** | `timeout_seconds=60` | 超时后 Docker 强制 kill 容器 |
| **临时目录** | `tmpfs={"/tmp": "size=64m"}` | 临时目录大小限制，防止磁盘写满 |
| **工作区隔离** | 只读挂载 + Copy-on-Write | 目标仓库以只读方式挂载（`mode="ro"`），修改在容器可写层进行 |
| **环境变量** | `PYTHONDONTWRITEBYTECODE=1` | 禁止生成 `.pyc` 文件，减少写入副作用 |
| **基础镜像** | 固定预装镜像 `mse-sandbox:latest` | 预装 pytest 和相关工具，避免每次构建

### Prompt 注入防护

**输入清洗策略：**
- 检测并拒绝包含已知注入模式的 Issue（正则匹配 `"ignore previous instructions"`、`"<|...|>"` 特殊 token、`"system prompt:"` 等模式），匹配后直接抛出异常
- 限制 Issue 长度上限为 2000 字符，超出部分截断
- 对特殊字符进行转义处理

**日志脱敏策略：**
- 扫描所有输出文本，使用正则替换敏感信息：
  - OpenAI API Key 格式（`sk-...`）→ `***API_KEY***`
  - Bearer Token → `Bearer ***TOKEN***`
  - 密码赋值（`password="xxx"`）→ `password=***`
- Dashboard 展示前必须经过脱敏处理，防止 `.env` 和 API Key 泄露

### 文件系统安全

- **修改范围控制**：Coder Agent 只能修改 `retrieved_context` 中命中的文件，超出范围需 Product Agent 审批
- **快照与回退**：每轮修改前使用 `shutil.copytree` 创建快照，失败后自动恢复
- **路径遍历防护**：所有文件操作使用 `os.path.realpath` 校验，拒绝 `../` 越权

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
│           Task Context (动态)             │
│  - 用户 Issue                            │
│  - Product Agent 的任务拆解               │
├─────────────────────────────────────────┤
│           Code Context (动态)             │
│  - 检索到的代码片段 (Top-K)               │
│  - 调用依赖 (Call Graph)                 │
│  - 项目大纲                              │
├─────────────────────────────────────────┤
│           History Context (动态)          │
│  - 上一轮失败原因                         │
│  - 已尝试的修改                          │
│  - 本轮约束条件                          │
└─────────────────────────────────────────┘
```

### 各 Agent Prompt 设计要点

**Product Agent：**
- 输入：Issue 原文 + 项目大纲
- 输出：`{"task_breakdown": ["子任务1", ...], "modification_targets": ["文件路径"], "priority": "high|medium|low"}`
- 关键约束：只做规划不做代码修改；不确定时标记 `"uncertain": true` 并说明

**Coder Agent：**
- 输入：子任务 + 代码上下文 + 失败原因（如有）
- 输出：`{"patches": [{"file_path": "...", "original": "...", "replacement": "...", "reason": "..."}]}`
- 关键约束：使用 Search/Replace 精确匹配；禁止重写整个文件；每处修改必须说明原因

**Tester Agent：**
- 输入：测试日志 (stdout + stderr) + exit code
- 输出：`{"error_category": "AssertionError", "failed_tests": ["test_xxx"], "root_cause": "...", "severity": "blocker|minor"}`
- 关键约束：区分"测试框架本身出错"和"业务逻辑错误"

**Diagnose Agent：**
- 输入：TestResult + 失败历史 + retry_count
- 输出：`{"should_retry": true, "new_constraints": ["只修改 X 函数"], "suggested_approach": "..."}`
- 关键约束：重试 > 2 次时必须缩小修改范围

### Context Budget 管理

```
┌────────────────────────────────────────────┐
│ 总预算: ~8000 tokens (模型上下文窗口的合理子集) │
├────────────────────────────────────────────┤
│ System Prompt         │ ~800 tokens        │
│ Issue + Task          │ ~500 tokens        │
│ Project Outline       │ ~1000 tokens       │
│ Retrieved Code (Top-K)│ ~4000 tokens       │
│ Dependency Info       │ ~700 tokens        │
│ History + Constraints │ ~1000 tokens       │
├────────────────────────────────────────────┤
│ 预留 (Response)       │ ~4000 tokens       │
└────────────────────────────────────────────┘
```

---

## 补充设计：API 与集成接口

### REST API 设计

系统对外暴露 RESTful API 和 WebSocket 端点，供 Dashboard 和外部工具调用：

**`POST /api/v1/tasks`** — 创建并启动修复任务
- 请求体：`issue`（必填）、`repo_path`（必填）、`test_command`（默认 `"pytest"`）、`max_retries`（默认 3）、`model`（默认 `"gpt-4o"`）
- 响应：`201 Created`，返回 `task_id` 和初始 `status`

**`GET /api/v1/tasks/{task_id}`** — 查询任务状态与结果
- 响应体包含：`task_id`、`status`（init / running / success / failed）、`node_trajectory`（节点访问轨迹）、`patches`（所有补丁记录）、`test_result`（最终测试结果）、`agent_messages`（Agent 对话记录）、`total_tokens`、`total_cost`

**`GET /api/v1/tasks/{task_id}/stream`** — WebSocket 端点，实时推送任务执行状态，供 Dashboard 动态展示图状态变化

**`POST /api/v1/benchmarks`** — 批量运行 Benchmark
- 请求体：`dataset_path`（数据集路径）、`baseline`（single_agent / multi_agent / both）、`parallel`（并行数，默认 1）

### CLI 接口设计

提供命令行工具 `mse-cli`，覆盖三种核心使用场景：

**单次修复 (`mse-cli run`)：**
- 参数：`--issue`（Issue 描述）、`--repo`（目标仓库路径）、`--test`（测试命令，默认 `"pytest tests/ -v"`）、`--max-retries`（最大重试次数，默认 3）、`--model`（模型选择）、`--output`（输出目录）
- 行为：同步执行完整的 Issue → 修复 → 测试闭环，结果写入指定输出目录

**批量 Benchmark (`mse-cli benchmark`)：**
- 参数：`--dataset`（数据集 JSONL 路径）、`--baseline`（基线类型，single_agent / multi_agent / both）、`--parallel`（并行任务数，默认 4）、`--output`（结果输出目录）
- 行为：批量运行数据集中的所有任务，生成 `benchmark_results.csv` 和可视化图表

**启动 Dashboard (`mse-cli dashboard`)：**
- 参数：`--host`（绑定地址，默认 `0.0.0.0`）、`--port`（端口，默认 8501）、`--runs-dir`（历史运行记录目录）
- 行为：启动 Streamlit Web 服务，提供可视化交互界面

---

## 补充设计：可扩展性设计

### 多语言支持扩展

```
                     ┌──────────────────┐
                     │  LanguagePlugin  │  (Abstract Base Class)
                     │  + parse_ast()   │
                     │  + extract_deps()│
                     │  + get_test_cmd()│
                     └──────┬───────────┘
            ┌───────────────┼───────────────┐
     ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
     │PythonPlugin │ │ JSPlugin    │ │ RustPlugin  │
     │(ast module) │ │(tree-sitter)│ │(tree-sitter)│
     └─────────────┘ └─────────────┘ └─────────────┘
```

- 第一阶段：Python（内置 `ast` 模块）
- 第二阶段：JavaScript/TypeScript（Tree-sitter）
- 第三阶段：Rust/Go（Tree-sitter + cargo/go test 集成）

### Agent 类型扩展

系统通过插件注册机制支持新增 Agent 类型：

- 定义 `AgentPlugin` 抽象基类，要求实现 `get_system_prompt()`（返回 Agent 的系统提示词）和 `execute(state: ThreadState) -> dict`（执行逻辑并返回 State 增量更新）
- 通过 `agent_registry.register(name, instance)` 注册新 Agent
- 在 LangGraph 图中通过 `workflow.add_node(name, agent_registry.get(name).execute)` 动态插入
- 示例扩展方向：Security Auditor（安全审计 Agent）、Refactor Agent（重构 Agent）、Documentation Agent（文档生成 Agent）

### 模型提供商适配

通过统一的 `LLMProvider` 抽象层解耦 Agent 逻辑与具体模型 API：

- 抽象接口定义两个核心方法：`chat(messages, tools, schema)` → 统一聊天接口（支持 Function Calling 和 Structured Output）；`count_tokens(text)` → Token 计数
- 具体实现：`OpenAIProvider`（GPT-4o 系列）、`DeepSeekProvider`（兼容 OpenAI SDK）、`OllamaProvider`（本地部署模型）
- 切换模型只需修改配置中的 provider 名称，Agent 代码无需任何改动
- 后续可扩展：Claude Provider、Gemini Provider、vLLM Provider

---

## 补充设计：测试策略

### 测试金字塔

```
           ┌──────┐
           │ E2E  │  ~10 个：完整 Issue → 修复 → 验证流程
           ├──────┤
           │ 集成  │  ~30 个：模块间交互 (Agent + RAG + Sandbox)
           ├──────────┤
           │   单元测试  │  ~100+ 个：每个模块的独立测试
           └──────────┘
```

### 测试目录结构

```
tests/
├── unit/
│   ├── test_ast_parser.py          # AST 解析
│   ├── test_retrieval.py           # 检索模块
│   ├── test_diff_engine.py         # Diff/Patch 引擎
│   ├── test_sandbox_executor.py    # Docker 沙箱 (需 Docker 环境)
│   ├── test_prompt_builder.py      # Prompt 构造
│   ├── test_state_models.py        # Pydantic 模型验证
│   └── test_sanitizer.py           # 输入清洗
├── integration/
│   ├── test_graph_flow.py          # LangGraph 流程 (Mock LLM)
│   ├── test_rag_pipeline.py        # 完整检索链路
│   ├── test_patch_apply_flow.py    # 补丁应用流程
│   └── test_sandbox_pytest.py      # 沙箱内 pytest 执行
├── e2e/
│   ├── test_fix_syntax_error.py    # 端到端：语法错误修复
│   ├── test_fix_import_error.py    # 端到端：导入错误修复
│   ├── test_fix_logic_bug.py       # 端到端：逻辑错误修复
│   └── test_retry_on_failure.py    # 端到端：失败重试
├── conftest.py                     # 共享 fixture
└── fixtures/
    ├── demo_projects/              # 测试用最小项目
    │   ├── simple_bug/             # 含单个语法错误的项目
    │   ├── cross_file_bug/         # 含跨文件错误的项目
    │   └── logic_bug/              # 含逻辑错误的项目
    └── mock_llm_responses/         # Mock LLM 返回
```

### 覆盖率目标与排除

| 模块 | 目标覆盖率 | 备注 |
|------|-----------|------|
| `retrieval/` | ≥ 85% | 核心逻辑，易测试 |
| `patching/` | ≥ 85% | 纯字符串处理 |
| `sandbox/` | ≥ 60% | 依赖 Docker，部分测试需标记 `@pytest.mark.docker` |
| `graph/` | ≥ 80% | Mock LLM 后测试流程 |
| `agents/` | ≥ 70% | LLM 调用层 Mock，Prompt 构建层实测 |
| `evaluation/` | ≥ 75% | 核心指标计算逻辑 |
| Dashboard | ≥ 40% | UI 组件不强制高覆盖，逻辑层独立测试 |

---

## 补充设计：CI/CD 与 DevOps

### 项目配置完整性

**pyproject.toml 配置要点：**

- **项目元数据**：`name="mse-system"`、`version="0.1.0"`、`requires-python=">=3.11"`
- **核心依赖**：langgraph（Agent 编排）、langchain（LLM 抽象）、openai（模型调用）、chromadb（向量存储）、docker（沙箱执行）、pydantic（数据模型）、instructor（结构化输出）、streamlit（Dashboard）、tree-sitter（代码解析）、pytest + pytest-cov + pytest-asyncio（测试框架）、python-dotenv（配置管理）
- **可选依赖分组**：
  - `dev`：ruff（代码检查）、mypy（类型检查，strict 模式）、pre-commit（提交钩子）
  - `dashboard`：streamlit
  - `deepseek`：openai SDK（DeepSeek 兼容 OpenAI 接口）
- **代码质量工具配置**：
  - ruff：line-length=100，py311 目标版本，启用 E/F/I/N/W/UP/B/C4/SIM 规则
  - mypy：strict=true 严格模式
- **pytest 配置**：`testpaths=["tests"]`，自定义 markers：`docker`（需要 Docker 环境）、`slow`（耗时 > 10s）、`llm`（调用真实 LLM API，会产生费用）

### Docker 部署配置

**Dockerfile 设计：**
- 基础镜像：`python:3.11-slim`（轻量级）
- 工作目录：`/app`
- 分层构建：先复制依赖文件并安装（利用 Docker 缓存层），再复制源码
- 启动命令：Streamlit Dashboard，监听 8501 端口

**docker-compose.yml 服务编排（两个服务）：**

| 服务 | 镜像 | 端口 | 挂载 | 说明 |
|------|------|------|------|------|
| `mse-system` | 本地构建 | `8501:8501` | `./runs`（运行记录）、`./examples`（只读）、`/var/run/docker.sock`（沙箱需要） | Dashboard 主服务，需挂载 Docker socket 以管理沙箱容器 |
| `chroma` | `chromadb/chroma:latest` | `8001:8000` | `./chroma_data`（向量数据持久化） | 向量数据库，独立部署 |

关键设计决策：`mse-system` 需要挂载宿主机的 `/var/run/docker.sock`，这是因为沙箱执行模块需要通过 Docker SDK 在宿主机上启动隔离容器（DinD 模式在资源隔离上不如直接操作宿主机 Docker）。

---

## 补充设计：评估指标体系

### 核心指标定义

| 指标 | 公式 | 说明 |
|------|------|------|
| **修复成功率 (Fix Rate)** | `passed_tasks / total_tasks` | 测试全部通过视为成功 |
| **平均修复耗时 (MTTR)** | `sum(fix_duration) / total_tasks` | 包含 LLM 调用 + 测试时间 |
| **Token 效率 (Token Efficiency)** | `total_tokens / passed_tasks` | 每次成功修复的平均 Token 消耗 |
| **首次通过率 (First-Pass Rate)** | `first_try_passes / total_tasks` | 无需重试即修复的比例 |
| **平均重试次数 (Avg Retries)** | `sum(retries) / total_tasks` | 仅统计最终成功的任务 |
| **Patch 应用成功率 (Patch Apply Rate)** | `applied_patches / total_patches` | LLM 输出的补丁能被成功应用的比例 |
| **检索精度 (Retrieval Precision)** | `relevant_chunks / retrieved_chunks` | 检索到的代码块与修复实际相关的比例 |
| **退化率 (Regression Rate)** | `new_failures / total_tasks` | 修复 A 导致 B 失败的比例 |

### 对比基线

```
实验组 A: 本系统 (LangGraph Multi-Agent + Reflexion)
实验组 B: Single Agent Zero-shot (一次 LLM 调用生成修复)
实验组 C: Single Agent + Retry (单 Agent 可重试 3 次)
实验组 D: Multi-Agent without Reflexion (去掉诊断回路)
```

### 结果可视化模板

Benchmark 报告应包含：
1. 总体指标对比表（A vs B vs C vs D）
2. 按难度分层的修复成功率柱状图
3. Token 消耗 vs 修复成功率的散点图
4. 重试次数分布直方图
5. 各阶段耗时 Stacked Bar Chart（检索 / LLM 生成 / 沙箱测试）
6. 典型案例的时间线图（展示 Reflexion 过程）

---

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
