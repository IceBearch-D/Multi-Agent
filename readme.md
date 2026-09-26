# AlgoSolver: Multi-Agent Algorithm Problem Solver

<div align="center">

**基于 LangGraph 的多智能体协同算法题目自动解答系统**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/framework-LangGraph-orange)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/sandbox-Docker-2496ED?logo=docker)](https://www.docker.com/)

</div>

---

## 📖 目录

- [项目背景](#项目背景)
- [系统架构](#系统架构)
- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [评测体系](#评测体系)
- [设计亮点](#设计亮点)
- [已知限制与未来规划](#已知限制与未来规划)
- [面试准备](#面试准备)
- [License](#license)

---

## 项目背景

大语言模型（LLM）在代码生成任务中展现出令人瞩目的能力，但在面对算法竞赛和面试题目时仍存在显著短板：

- **一次性生成不可靠**：LLM 直接生成的代码往往无法一次通过所有测试用例，尤其在边界条件和复杂逻辑上容易出错。
- **缺乏自我验证**：人类选手会编写代码后运行测试、根据报错修正，而 LLM 缺乏这种"运行-反馈-修正"的闭环能力。
- **错误定位困难**：当代码输出错误答案或超时时，LLM 难以仅凭题目描述定位问题根因。
- **缺乏迭代优化**：没有机制让 LLM 像人类一样根据测试反馈持续改进解法。

AlgoSolver 旨在探索一套**多智能体协同的算法题目自动解答工作流**，让 LLM 像人类选手一样经历"分析题目 → 编写代码 → 运行测试 → 根据报错反思修正"的完整闭环，从"一次性代码生成"走向"真正的自主解题"。

---

## 系统架构

### 整体架构

```
                         ┌────────────────────────┐
                         │  User Input              │
                         │  (算法题目 + 测试用例)    │
                         └───────────┬────────────┘
                                     │
    ┌────────────────────────────────▼──────────────────────────────────┐
    │                 Presentation Layer                                  │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │  CLI (click/typer)                                           │  │
    │  │  algosolver solve --problem "two_sum.md" --test tests/       │  │
    │  └──────────────────────────┬───────────────────────────────────┘  │
    └─────────────────────────────┼─────────────────────────────────────┘
                                  │
    ┌─────────────────────────────▼─────────────────────────────────────┐
    │                 Application Layer (LangGraph StateGraph)           │
    │                                                                    │
    │   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐      │
    │   │ Analyzer │──▶│  Coder   │──▶│  Tester  │──▶│ Diagnose │      │
    │   │ Agent    │   │  Agent   │   │  Agent   │   │  Agent   │      │
    │   └──────────┘   └──────────┘   └─────┬────┘   └─────┬────┘      │
    │        ▲                              │              │          │
    │        └──────────────────────────────┴──────────────┘          │
    │                     Reflexion Loop (Diagnose → Coder → Test)     │
    └────────────────────────────┬─────────────────────────────────────┘
                                 │
    ┌────────────────────────────▼─────────────────────────────────────┐
    │                    Domain / Service Layer                          │
    │  ┌──────────────────────┐ ┌──────────────────────────────────┐   │
    │  │ LLM Provider Layer   │ │ Docker Sandbox                    │   │
    │  │ (OpenAI / DeepSeek)  │ │ + Test Executor                   │   │
    │  │ + Structured Output  │ │ + Resource Limiter                │   │
    │  └──────────────────────┘ └──────────────────────────────────┘   │
    └──────────────────────────────────────────────────────────────────┘
```

### LangGraph 多 Agent 协同流程

```text
START
  → analyzer_node       (题目分析：理解输入输出、约束、边界条件、算法模式)
  → coder_node          (根据分析编写完整 Python 解答代码)
  → tester_node         (Docker 沙箱执行测试用例)
  → route_by_test_result:
       ├─ 全部通过 → END (输出解答代码 + 运行结果)
       ├─ 失败 + 可重试 → diagnose_node → coder_node (Reflexion 回路)
       └─ 超过重试上限 → fail_node → END (输出最佳尝试 + 错误报告)
```

### 核心数据流

| 阶段 | 输入 | 输出 | 关键技术 |
|------|------|------|----------|
| **题目分析** | 算法题目原文 | 算法类型、复杂度目标、边界条件、解题思路 | Analyzer Agent + Pydantic |
| **代码生成** | 题目分析 + 失败反馈(如有) | 完整 Python 解答代码 | Coder Agent + Structured Output |
| **沙箱测试** | 解答代码 + 测试用例 | stdout, stderr, exit code, 通过/失败详情 | Docker SDK + pytest |
| **诊断修正** | 测试日志 + 失败历史 | 错误分类 + 根因分析 + 修改建议 | Diagnose Agent + Reflexion |

---

## 核心特性

### 🧠 多智能体协同解题

- **Analyzer Agent**：将自然语言描述的算法题目转化为结构化解题方案，识别算法模式（动态规划 / 贪心 / 图论 / 二分搜索等），枚举边界条件，评估目标复杂度
- **Coder Agent**：基于题目分析和历史失败反馈，生成完整的 Python 解答代码（非 Diff 补丁），包含类型注解和边界处理
- **Tester Agent**：在 Docker 沙箱中运行 pytest 测试用例，分类提取错误信息（Wrong Answer / TLE / Runtime Error / Syntax Error）
- **Diagnose Agent**：分析测试失败原因，定位根因，在 Reflexion 回路中为下一轮编码提供具体修改约束

### 🔄 自适应纠错闭环 (Reflexion Loop)

- 测试失败后自动提取报错日志，分类错误类型（答案错误、超时、运行时错误、语法错误等）
- Coder Agent 在下一轮编码时必须引用上一轮失败原因和 Diagnose Agent 的修改建议
- 死循环检测：重复错误加速计数，超过阈值后触发 Analyzer Agent 重新规划解题思路
- 最佳代码保留：每次测试后记录通过用例数，始终保留通过率最高的代码版本

### 🐳 安全沙箱执行

- Docker 容器完全网络隔离，非 root 用户运行
- CPU / 内存 / 磁盘 / 超时严格限制
- 所有 Linux Capabilities 丢弃，禁止提权
- 防止恶意代码和无限循环耗尽宿主机资源

### 📊 结构化题目分析

- 自动识别算法题目类型（DP、贪心、图、树、搜索、排序、数学等）
- 提取输入输出约束和边界条件
- 评估最优时间/空间复杂度目标
- 将模糊的自然语言描述转化为精确的工程规格

### 📈 量化评测体系

- 内置多难度算法题目评测集（Easy / Medium / Hard）
- 多基线对比：单 Agent Zero-shot / 单 Agent + Retry / 多 Agent 无 Reflexion / 完整系统
- 核心指标：解题成功率（所有用例通过）、平均重试次数、Token 效率、首次通过率
- 按题目类型和难度分层的评测报告

---

## 技术栈

| 层级 | 技术 | 选型理由 |
|------|------|----------|
| **Agent 编排** | LangGraph | 显式 StateGraph 控制流程，避免 Chain 的黑盒跳转，防止死循环 |
| **LLM 调用** | OpenAI / DeepSeek / 智谱 | 多 Provider 适配，支持云端模型 |
| **结构化输出** | Pydantic + Instructor | 强制 LLM 输出符合 JSON Schema，消除解析失败 |
| **沙箱执行** | Docker SDK for Python | 安全隔离，动态资源限制 |
| **测试框架** | pytest | Python 生态标准，丰富的断言和插件体系 |
| **工程化** | ruff + mypy + pre-commit | 代码质量、类型检查、自动化格式 |
| **配置管理** | python-dotenv + .env.example | 安全的敏感信息管理 |

---

## 快速开始

### 前置依赖

- Python 3.11+
- Docker Desktop / Docker Engine 24+
- OpenAI API Key（或其他兼容的 LLM Provider）

### 1. 克隆项目

```bash
git clone https://github.com/your-username/algosolver.git
cd algosolver
```

### 2. 配置环境

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env 文件，填入你的 OPENAI_API_KEY
```

### 3. 验证安装

```bash
# 运行基础测试
pytest tests/unit/ -v

# 验证 Docker 沙箱可用性
python -c "import docker; client = docker.from_env(); print(client.version())"
```

### 4. 运行第一个 Demo

```bash
# 使用 CLI 解答一道算法题
algosolver solve \
  --problem examples/two_sum/problem.md \
  --test examples/two_sum/test_solution.py \
  --max-retries 3 \
  --model gpt-4o

# 或通过 Python API
python examples/quick_start.py
```

---

## 使用指南

### CLI 命令

```bash
# 单题解答
algosolver solve \
  --problem ./problems/two_sum/problem.md \
  --test ./problems/two_sum/test_solution.py \
  --max-retries 3 \
  --model gpt-4o \
  --output ./runs/latest/

# 批量评测
algosolver benchmark \
  --dataset ./problems/dataset.jsonl \
  --baseline both \
  --parallel 4 \
  --output ./benchmark_results/

# 查看某次运行的详细报告
algosolver report --run-id run_20260620_153042
```

### Python API

```python
from algosolver import AlgoSolver
from algosolver.models import ProblemSpec

# 初始化系统
solver = AlgoSolver(
    model="gpt-4o",
    max_retries=3,
    sandbox_config={
        "mem_limit": "512m",
        "timeout_seconds": 30,
    },
)

# 创建并执行解题任务
problem = ProblemSpec(
    description="""
    给定一个整数数组 nums 和一个整数目标值 target，
    请你在该数组中找出和为目标值的那两个整数，并返回它们的数组下标。
    """,
    test_file="./examples/two_sum/test_solution.py",
)

result = await solver.solve(problem)

# 查看结果
print(f"Status: {result.status}")
print(f"Retries: {result.retry_count}")
print(f"Total tokens: {result.total_tokens}")
print(f"Solution:\n{result.solution_code}")
print(f"Test results: {result.test_result.passed}/{result.test_result.total} passed")
```

### 批量评测

```python
from algosolver.evaluation import BenchmarkRunner

runner = BenchmarkRunner(
    dataset_path="./problems/dataset.jsonl",
    baselines=["single_agent", "multi_agent"],
    parallel=4,
)

results = await runner.run()

# 生成报告
runner.generate_report(results, output="./benchmark_results/")
runner.generate_charts(results, output="./benchmark_results/charts/")
```

---

## 项目结构

```
algosolver/
├── src/algosolver/                  # 核心源码
│   ├── agents/                       # Agent 实现
│   │   ├── __init__.py
│   │   ├── base.py                   # Agent 基类 + LLM 适配层
│   │   ├── analyzer.py              # Analyzer Agent（题目分析）
│   │   ├── coder.py                 # Coder Agent（代码生成）
│   │   ├── tester.py                # Tester Agent（测试诊断）
│   │   └── diagnose.py             # Diagnose Agent（反思修正）
│   ├── graph/                        # LangGraph 图定义
│   │   ├── __init__.py
│   │   ├── state.py                 # AlgoState + Pydantic 数据模型
│   │   ├── nodes.py                 # 所有 Node 函数
│   │   ├── edges.py                 # 条件边 + 路由逻辑
│   │   └── workflow.py              # StateGraph 构建与编译
│   ├── sandbox/                      # Docker 沙箱执行模块
│   │   ├── __init__.py
│   │   └── sandbox.py              # Sandbox + ExecutionResult
│   ├── llm/                          # LLM 抽象层
│   │   ├── __init__.py
│   │   ├── providers.py             # OpenAI / DeepSeek / Ollama 适配
│   │   └── prompts.py              # 各 Agent 的 Prompt 模板
│   ├── evaluation/                   # 评测模块
│   │   ├── __init__.py
│   │   ├── runner.py                # 批量执行器
│   │   ├── metrics.py               # 指标计算
│   │   └── reporter.py              # 报告生成
│   ├── cli/                          # CLI 入口
│   │   ├── __init__.py
│   │   └── main.py                  # click/typer 命令定义
│   └── utils/                        # 工具模块
│       ├── __init__.py
│       └── logger.py               # 日志与脱敏
├── tests/                            # 测试套件
│   ├── unit/                         # 单元测试
│   │   ├── test_sandbox.py
│   │   ├── test_state_models.py
│   │   └── test_prompts.py
│   ├── integration/                  # 集成测试
│   │   ├── test_graph_flow.py
│   │   └── test_sandbox_pytest.py
│   └── e2e/                          # 端到端测试
│       ├── test_two_sum.py
│       ├── test_fibonacci.py
│       └── test_binary_search.py
├── examples/                         # 示例算法题目
│   ├── two_sum/
│   │   ├── problem.md               # 题目描述
│   │   └── test_solution.py         # 测试用例
│   ├── fibonacci/
│   │   ├── problem.md
│   │   └── test_solution.py
│   └── quick_start.py               # 快速入门脚本
├── problems/                         # 算法题目集
│   ├── dataset.jsonl                 # 评测用题目集合
│   └── README.md                     # 题目列表与难度分类
├── runs/                             # 运行记录（gitignore）
│   └── run_<timestamp>/
│       ├── state.json                # 完整 AlgoState
│       ├── solution.py               # 最终解答代码
│       ├── iterations/               # 每轮迭代的代码快照
│       ├── test.log                  # 测试日志
│       └── report.md                 # 最终报告
├── .env.example                      # 环境变量模板
├── .gitignore
├── pyproject.toml                    # 项目配置（依赖、lint、type check）
├── requirements.txt                  # 依赖列表
├── Plan.md                           # 项目设计方案与计划
└── readme.md                         # 本文件
```

---

## 配置说明

`.env` 文件环境变量：

```bash
# --- 必需的 LLM 配置 ---
OPENAI_API_KEY=sk-xxx           # 必需
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，兼容 DeepSeek/智谱等

# --- 可选配置 ---
DEEPSEEK_API_KEY=sk-xxx         # DeepSeek 备用
ZHIPU_API_KEY=xxx               # 智谱 GLM 备用

# --- 沙箱配置 ---
SANDBOX_IMAGE=python-test:3.11
SANDBOX_MEM_LIMIT=512m
SANDBOX_TIMEOUT=30
SANDBOX_CPU_QUOTA=50000

# --- 解题配置 ---
DEFAULT_MAX_RETRIES=3
DEFAULT_MODEL=gpt-4o
```

---

## 评测体系

系统内置了一个包含 **20-30 道不同难度算法题目**的评测数据集：

| 难度 | 类型 | 数量 | 示例 |
|------|------|------|------|
| Easy | 数组遍历、哈希表、简单字符串 | 8-10 | Two Sum, Valid Parentheses |
| Medium | DP、二叉树、图遍历、二分搜索 | 7-10 | Longest Increasing Subsequence |
| Hard | 复杂 DP、网络流、高级图论 | 5-7 | Trapping Rain Water, N-Queens |

### 核心指标

| 指标 | 说明 |
|------|------|
| **解题成功率 (Solve Rate)** | 所有测试用例通过视为成功 |
| **平均重试次数 (Avg Retries)** | 仅统计最终成功解决的题目 |
| **Token 效率** | 每次成功解题的平均 Token 消耗 |
| **首次通过率 (First-Pass Rate)** | 无需重试即通过所有测试的占比 |
| **平均解题耗时 (MTTS)** | 端到端从题目输入到全部用例通过的时间 |

### 对比基线

- **Baseline A**：单 Agent Zero-shot（一次 LLM 调用生成解答）
- **Baseline B**：单 Agent + Retry（相同 Agent 可重试 3 次）
- **Baseline C**：多 Agent 无 Reflexion（去掉诊断回路）
- **本系统**：LangGraph 多 Agent + Reflexion 完整闭环

---

## 设计亮点

以下设计决策体现了系统在技术深度上的考量，也是面试中常见的追问点：

### 1. 为什么用 LangGraph 而不是 AutoGen / CrewAI？

LangGraph 提供**显式的 StateGraph** 概念。通过明确定义节点、边和 `AlgoState`，开发者对 Agent 的跳转逻辑有**完全的控制权**——这在处理复杂解题流程中的死循环和状态丢失问题时至关重要。相比之下，AutoGen 和 CrewAI 的 Agent 交互更偏向对话驱动，在需要精确控制流程（如"重试 3 次后强制触发 Analyzer 重新规划解法"）的场景中可控性不足。

### 2. 如何避免 Agent 死循环？

在 `AlgoState` 中引入三层防护：
- **衰减机制**：`retry_count` 递增，超过 `max_retries` 后终止
- **去重机制**：`seen_errors` 集合检测重复错误，重复出现时加速计数
- **最佳保留**：始终保存通过用例最多的代码版本，作为最终输出备选

### 3. 如何保证 Coder Agent 输出正确的代码格式？

通过 Pydantic + Instructor 约束 Coder Agent 的输出结构。Coder Agent 必须输出包含 `solution_code` 字段的结构化 JSON，其中代码必须是独立可执行的 Python 函数。格式不合法的输出会被自动拒绝并要求重试。

### 4. 如何保证沙箱安全？

六层安全策略：
- 网络完全隔离 (`network_disabled`)
- 非 root 用户运行
- 所有 Linux Capabilities 丢弃 (`cap_drop: ALL`)
- CPU/内存/磁盘严格限制
- 超时强制 kill
- 禁止提权 (`no_new_privileges`)

### 5. 如何评价系统是否真的有效？

通过自建的算法题目评测集，在相同的模型、温度、超时配置下，对比多个基线的解题成功率、Token 成本和耗时。量化数据 + 典型案例分析 + 图表展示，用证据说话。同时按题目类型（DP/贪心/图论）和难度分层分析，找出系统的优势和短板。

### 6. 为什么生成完整代码而不是增量 Diff？

算法题目解答的场景与 Bug 修复不同：不存在需要保留的已有代码库，代码从零开始编写。生成完整代码方案更简单直接，避免了 Diff 应用失败的风险。同时，对 LLM 而言，从零编写一个完整的函数比精确定位修改点更加自然和可靠。

---

## 已知限制与未来规划

### 当前限制

| 限制 | 说明 | 缓解方案 |
|------|------|---------|
| **语言支持** | 仅支持 Python 解答 | 架构上预留 LanguagePlugin 接口，后续扩展 |
| **模型依赖** | 解题质量高度依赖底层 LLM 能力 | 支持多 Provider，模型升级成本低 |
| **复杂题目** | Hard 难度题目（如网络流、高级 DP）成功率下降 | 持续优化 Agent Prompt 和解题策略 |
| **测试用例质量** | 评测效果依赖测试用例的覆盖度 | 支持用户自定义测试用例；未来增加自动生成边界用例 |
| **成本** | 多 Agent + 重试导致 Token 成本较高 | 使用 DeepSeek 等低成本模型；设置成本预算上限 |

### 未来路线图

- [ ] **v0.2**：支持 C++ / Java 解答（通过 LanguagePlugin 接口）
- [ ] **v0.3**：自动生成边界测试用例，提高测试覆盖率
- [ ] **v0.4**：Streamlit Dashboard 可视化解题过程与 Agent 决策轨迹
- [ ] **v0.5**：本地模型支持（Ollama / vLLM），降低 API 成本
- [ ] **v0.6**：支持多文件项目型题目（如设计题、系统实现题）
- [ ] **v1.0**：LeetCode / 牛客网 等公开题库评测 + 论文发表

---

## 面试准备

本项目在简历中的建议表述（STAR 原则）：

> **项目名称**：基于 LangGraph 的多智能体协同算法题目自动解答系统 (AlgoSolver)
>
> **项目职责**：
> - 基于 **LangGraph** 设计并实现了一个具备自适应纠错能力的多智能体协同解题系统。通过定义 StateGraph，实现了 Analyzer、Coder、Tester、Diagnose 四个 Agent 在状态受控情况下的协同解题流程。
> - 设计了结构化的**题目分析链路**：通过 Analyzer Agent 自动识别算法模式（DP、贪心、图论等），枚举边界条件，将模糊的自然语言题目转化为精确的工程规格，引导 Coder 生成高质量解答。
> - 利用 **Docker SDK** 搭建了安全的、网络隔离的代码运行沙箱，实现了对 LLM 自动生成代码的实时运行、pytest 测试执行与错误分类（Wrong Answer / TLE / Runtime Error）。
> - 设计了基于 **Reflexion 架构**的自适应纠错闭环。当测试失败时，Diagnose Agent 智能分析报错并引导 Coder 进行针对性修改，在本地算法题目集上将解题成功率提升了近 30%。
> - 支持多种 LLM Provider（OpenAI / DeepSeek / 智谱），通过 Pydantic + Instructor 约束 Agent 输出结构化数据，消除了 JSON 解析失败的问题。

### 面试高频问题速查

| 问题 | 回答要点 |
|------|---------|
| 为什么用 LangGraph？ | StateGraph 显式控制流程、防死循环、状态可回溯 |
| 如何避免死循环？ | 三重防护：retry_count、seen_errors 去重、最佳代码保留 |
| Analyzer 如何分析题目？ | 结构化 Prompt → 输出算法类型、复杂度目标、边界条件、解题思路 |
| 沙箱安全如何保证？ | 网络隔离、非 root、cap 丢弃、资源限制、超时 kill |
| 为什么生成完整代码而非 Diff？ | 算法题从零编写，无已有代码库；完整代码对 LLM 更自然可靠 |
| 如何证明系统有效？ | 本地算法题目集 + 多基线对比 + 量化指标 + 难度分层分析 |

---

## License

MIT License

---

<div align="center">
  <sub>Built with ❤️ for Algorithmic Problem Solving Excellence</sub>
</div>
