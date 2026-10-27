"""LLM 包：统一大模型访问接口（多模型 + 降级）。

对外暴露 LLMProvider / LLMError / load_env / _Msg / ModelConfig / build_models，
调用方只需 `from mse.llm import LLMProvider` 即可使用。
"""
# 从 provider 模块重导出资底层能力，作为本包的公开 API
from mse.llm.provider import (
    LLMProvider,     # 统一客户端：多模型 + 降级
    LLMError,        # LLM 请求错误类型
    ModelConfig,     # 单个模型端点配置（name/base_url/api_key/kind）
    build_models,    # 从环境变量构建按能力排序的模型列表
    load_env,        # 解析 .env 文件
    _Msg,            # 极简消息包装（兼容 response.content）
)

# __all__ 声明本包对外暴露的符号，供 `from mse.llm import *` 使用
__all__ = ["LLMProvider", "LLMError", "ModelConfig", "build_models", "load_env", "_Msg"]