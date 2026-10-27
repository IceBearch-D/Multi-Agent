"""mse 包：Multi-Agent Solver Engine（多智能体解题引擎）。

这是一个多智能体算法题自动求解系统的核心包。它把一道算法题的求解流程
拆成多个智能体角色（Analyzer -> Coder -> Tester -> Diagnose），并使用
状态图编排引擎将其连接成闭环流水线。

子包结构：
  - mse.llm       统一 LLM 客户端（OpenAI 兼容 + Ollama 兜底）
  - mse.sandbox   代码沙箱（本地子进程实现，无需 Docker）
  - mse.agents    四种角色的智能体定义（分析器/编码器/测试器/诊断器）

主要入口：
  - mse.orchestrator.solve_problem()  对一道题运行完整流水线
"""