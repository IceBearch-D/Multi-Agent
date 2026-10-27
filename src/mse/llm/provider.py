"""统一 LLM Provider：多模型 + 能力降级（OpenAI 兼容 / Ollama 原生）。

【核心设计：模型降级策略】
  系统支持配置多个大模型，并按照"能力从强到弱"排序。每次调用 LLM 时：
    1. 从"最强且未被禁用"的模型开始尝试；
    2. 若某模型调用失败（重试耗尽 / 认证失败 / 连接失败等），立即把该模型加入
       本次运行的**黑名单**（进程级 `_disabled` 集合），此后整个运行过程不再使用它；
    3. 自动降级到下一个较弱的模型，依次类推；
    4. 若所有模型都被禁用，则抛出 LLMError。

  典型顺序（可在 .env 中调整）：
    GLM-4.7  >  GLM-4.6V  >  GLM-4.5-Air  >  deepseek-r1:8b  >  qwen3.5:4b

【多 Provider 支持】
  每个模型属于一个"端点"（服务），端点类型：
    - openai：OpenAI 兼容接口（POST /chat/completions），如智谱 GLM；
    - ollama：Ollama 原生接口（POST /api/chat），如本地 deepseek-r1 / qwen。

【环境变量（.env）】
  GLM_BASE_URL / GLM_API_KEY / GLM_MODELS（逗号分隔，云端最强）
  OLLAMA_BASE_URL / OLLAMA_MODELS（逗号分隔，本地较弱）
  兼容旧配置：LLM_BASE_URL / LLM_API_KEY / LLM_MODEL（单一模型）

【网络健壮性（针对 WSL/代理环境）】
- 默认 urllib 不识别 no_proxy 的 `10.*`/`127.*` 通配符，会把私有 IP 请求误发到
  代理（127.0.0.1:7897）导致 502。这里强制 **直连优先**：
  - 先使用 ProxyHandler({}) 的直连 opener（绕过所有代理）；
  - 直连抛连接类异常时，回退到「系统默认代理」opener。

【能力】
- 失败重试 + 指数退避（应对 502 / 超时等瞬断）
- health() 探测所有端点，只要任一可用即返回 True
- 暴露 invoke() / ainvoke() 以兼容 agent_manager 对 `response.content` 的约定
"""
from __future__ import annotations  # 延迟注解解析（可空注解等向后兼容）

import json          # 序列化/反序列化 JSON 请求与响应
import os            # 环境变量处理
import time          # 指数退避中的睡眠
import urllib.request  # 发起 HTTP 请求（标准库）
from mse.ledger import record as _ledger  # 台账：记录模型选择/降级等事件
import urllib.error    # HTTPError 等网络异常
from dataclasses import dataclass  # 用 dataclass 表示模型配置
from typing import Any, Optional  # 类型标注


# ---------------- 网络 opener：直连优先，代理兜底 ----------------
# 直连 opener：显式清空代理配置，强制不经过任何代理
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
# 默认 opener：尊重系统 http_proxy/https_proxy 环境变量
_PROXY_OPENER = urllib.request.build_opener()


def _open(req: urllib.request.Request, timeout: Optional[int] = None):
    """统一的网络打开函数：先走直连，失败再走系统代理。

    参数:
      req:     已构造好的 Request 对象
      timeout: 超时秒数（None 则用默认）
    返回:
      响应对象；如果直连与代理都失败，抛出最后一个异常。
    """
    last_err = None
    for opener in (_DIRECT_OPENER, _PROXY_OPENER):
        try:
            return opener.open(req, timeout=timeout)  # 哪个 opener 成功用哪个
        except Exception as e:  # 连接失败/超时等 -> 尝试下一个 opener
            last_err = e
    raise last_err  # type: ignore[misc]  # 都失败则抛出最后一次的错误


class LLMError(RuntimeError):
    """本项目自定义的 LLM 请求错误类型（便于上层区分业务异常）。"""
    pass


class _Msg:
    """极简的消息包装类：只为兼容 `response.content` 的访问约定。

    真实模型中 response 往往是一个完整的 Message 对象；这里只需要 .content 字段。
    """
    def __init__(self, content: str):
        self.content = content  # 消息文本内容


# ---------------- 模型配置 ----------------
@dataclass
class ModelConfig:
    """描述"一个可用的模型端点"。

    属性:
      name:    模型名（如 "GLM-4.7"、"deepseek-r1:8b"），也是降级时展示/去重的标识
      base_url:服务根地址。OpenAI 兼容为含 /v1 的地址；Ollama 为不含 /v1 的地址
      api_key: 鉴权 Key（Ollama 可为空）
      kind:    端点类型："openai" | "ollama"
    """
    name: str
    base_url: str
    api_key: str = ""
    kind: str = "openai"  # "openai" | "ollama"

    @property
    def ollama_base(self) -> str:
        """返回 Ollama 原生 API 的根地址（去掉末尾 /v1 后缀）。"""
        b = self.base_url.rstrip("/")
        return b[:-3] if b.endswith("/v1") else b


def _split_env(value: Optional[str]) -> list:
    """把逗号/空格分隔的环境变量字符串拆成去空列表（如 "a,b , c" -> ["a","b","c"]）。"""
    if not value:
        return []
    return [v.strip() for v in value.replace("，", ",").split(",") if v.strip()]


def build_models() -> list[ModelConfig]:
    """从环境变量构建"能力从强到弱"的模型列表。

    读取顺序（越靠前越强，优先尝试）：
      1. GLM 组：GLM_BASE_URL / GLM_API_KEY / GLM_MODELS（OpenAI 兼容）
      2. Ollama 组：OLLAMA_BASE_URL / OLLAMA_MODELS（Ollama 原生）
      3. 兼容旧配置：LLM_BASE_URL / LLM_API_KEY / LLM_MODEL（单一模型）

    返回按优先级排列的 ModelConfig 列表。
    """
    models: list[ModelConfig] = []

    # 1) 云端 GLM（能力最强，排最前）
    glm_base = os.getenv("GLM_BASE_URL")
    glm_key = os.getenv("GLM_API_KEY", "")
    for name in _split_env(os.getenv("GLM_MODELS")):
        if glm_base:  # 有地址才加入，避免无效配置
            models.append(ModelConfig(name=name, base_url=glm_base, api_key=glm_key, kind="openai"))

    # 2) 本地 Ollama（能力较弱，排后面）
    ollama_base = os.getenv("OLLAMA_BASE_URL")
    for name in _split_env(os.getenv("OLLAMA_MODELS")):
        if ollama_base:
            models.append(ModelConfig(name=name, base_url=ollama_base, kind="ollama"))

    # 3) 兼容旧的单一模型配置（上面都没配时才用）
    if not models:
        base = os.getenv("LLM_BASE_URL", "http://10.0.131.251:9005/v1")
        key = os.getenv("LLM_API_KEY", "")
        m = os.getenv("LLM_MODEL") or None
        models = [ModelConfig(name=m or "default", base_url=base, api_key=key, kind="openai")]

    return models


def load_env(path: str = ".env") -> None:
    """手动解析 .env 文件并写入 os.environ（避免引入 python-dotenv 依赖）。

    参数:
      path: .env 文件路径；若默认路径不存在会自动回退到项目根目录下的 .env。
    规则:
      - 忽略空行、以 # 开头的注释行；
      - 用第一个 = 分隔键值；
      - 已存在的环境变量不会被覆盖（setdefault 语义）。
    """
    from pathlib import Path  # 延迟导入以保持统一导入风格

    p = Path(path)
    if not p.exists():  # 指定路径不存在则回退到项目根目录（当前文件向上 3 级）
        p = Path(__file__).resolve().parents[3] / ".env"
    if not p.exists():  # 仍然不存在则静默跳过
        return
    for line in p.read_text(encoding="utf-8").splitlines():  # 逐行解析
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:  # 跳过无效行
            continue
        k, v = line.split("=", 1)  # 只分割第一个 "="
        os.environ.setdefault(k.strip(), v.strip())  # 不覆盖已有值


class LLMProvider:
    """统一的 LLM 客户端：多模型 + 能力降级。

    主要公共接口：
      - chat(messages, temperature, max_tokens) -> str  直接返回文本（自动降级）
      - invoke/ainvoke(...) -> _Msg（兼容 agent）。
      - health() -> bool                                健康检查（任一端点可用即 True）
      - disabled_models -> set[str]                     本次运行已禁用的模型名集合
    """

    def __init__(
        self,
        models: Optional[list[ModelConfig]] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 5,
        temperature: float = 0.2,
    ):
        # 显式传 models 则直接使用；否则若给了 base_url/model 则构造单一模型；
        # 都未传则从环境变量构建多模型列表
        if models is not None:
            self.models = list(models)
        elif base_url or model:
            self.models = [ModelConfig(
                name=model or self._discover_single(base_url or "") or "default",
                base_url=base_url or "http://10.0.131.251:9005/v1",
                api_key=api_key or "",
                kind="openai",
            )]
        else:
            self.models = build_models()

        # 空列表兜底：至少保留一个"default"模型，避免后续索引越界
        if not self.models:
            self.models = [ModelConfig(name="default", base_url="http://127.0.0.1:11434", kind="ollama")]

        self.timeout = timeout              # HTTP 请求超时（秒）
        self.max_retries = max_retries      # 单个模型请求的最大重试次数
        self.temperature = temperature      # 默认采样温度
        self._disabled: set[str] = set()    # 本次运行中已判定不可用的模型名黑名单

    # ---------------- 降级相关 ----------------
    @property
    def disabled_models(self) -> set[str]:
        """本次运行已禁用的模型名集合（只读视图）。"""
        return set(self._disabled)

    @property
    def model(self) -> str:
        """当前生效的模型名：最强的、未被禁用的模型；全部被禁用则返回 "default"。"""
        for cfg in self.models:
            if cfg.name not in self._disabled:
                return cfg.name
        return "default"

    def _pick(self) -> Optional[ModelConfig]:
        """按能力顺序选择"第一个未被禁用的模型"，全部被禁用则返回 None。"""
        for cfg in self.models:
            if cfg.name not in self._disabled:
                return cfg
        return None

    def _disable(self, cfg: ModelConfig, reason: str) -> None:
        """把指定模型加入本次运行的黑名单，并打印降级日志。"""
        self._disabled.add(cfg.name)  # 加入黑名单，本次运行不再使用
        remain = [c.name for c in self.models if c.name not in self._disabled]  # 剩余可用模型
        # 【台账】记录模型降级：被禁用的模型、原因、剩余候选模型
        _ledger("model.disable", model=cfg.name, kind=cfg.kind, reason=reason, remaining=remain)
        if remain:
            print(f"[LLM] 模型 {cfg.name} 不可用（{reason}），本次运行不再使用。"
                  f"降级到：{remain[0]}", flush=True)
        else:
            print(f"[LLM] 模型 {cfg.name} 不可用（{reason}），且已无其它可用模型。", flush=True)

    # ---------------- 底层 HTTP ----------------
    def _request(
        self,
        cfg: ModelConfig,
        method: str,
        path: str,
        payload: Optional[dict] = None,
        timeout: Optional[int] = None,
    ):
        """对指定模型端点发起带重试与指数退避的 HTTP 请求。

        参数:
          cfg:     目标模型配置（决定 base_url 与 api_key）
          method:  HTTP 方法（GET/POST 等）
          path:    相对 base_url 的路径（如 /models、/chat/completions）
          payload: 请求体字典（POST 时提供）
          timeout: 覆盖默认超时
        返回:
          (status_code, body_text)
        抛错:
          LLMError：重试耗尽可能仍失败，或遇到非可重试的 4xx 错误
        """
        url = cfg.base_url.rstrip("/") + path          # 组装完整 URL
        # 需要 body 才序列化 JSON；否则为 None（GET 不带 body）
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(                 # 构造请求对象
            url,
            data=data,
            method=method,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.api_key or 'sk-noauth'}",
            },
        )
        last_err = None
        # 重试循环：attempt = 1..max_retries（应对 502/超时等瞬断）
        for attempt in range(1, self.max_retries + 1):
            try:
                with _open(req, timeout=timeout or self.timeout) as resp:  # 发送请求
                    # 读取响应体，解码失败时用 replace 兜底
                    return resp.status, resp.read().decode("utf-8", "replace")
            except urllib.error.HTTPError as e:
                # HTTP 错误：提取 body 前 500 字符，作为错误信息的一部分
                body = e.read().decode("utf-8", "replace")[:500]
                last_err = f"HTTP {e.code}: {body}"
                # 5xx / 429 属于瞬时错误 -> 可重试；其它 4xx（如 401/404）直接抛错
                if e.code in (500, 502, 503, 504, 429):
                    pass
                else:
                    raise LLMError(last_err)
            except Exception as e:
                # 连接类异常（超时、拒绝等）
                last_err = f"{type(e).__name__}: {e}"
            # 指数退避：1s, 2s, 4s, 8s...，封顶 30s
            wait = min(2 ** attempt, 30)
            print(f"[LLM] 第{attempt}次请求失败（{last_err}），{wait}s 后重试…", flush=True)
            time.sleep(wait)
        raise LLMError(f"LLM 请求在 {self.max_retries} 次重试后仍失败: {last_err}")

    def _discover_single(self, base_url: str) -> Optional[str]:
        """在单一模型模式下，若未显式指定模型名，尝试从 /v1/models 自动发现一个。"""
        try:
            cfg = ModelConfig(name="__discover__", base_url=base_url, api_key="", kind="openai")
            _, raw = self._request(cfg, "GET", "/models", timeout=10)
            data = json.loads(raw)
            models = data.get("data") or []
            if models:
                return models[0].get("id")
        except Exception:
            pass
        return None

    # ---- OpenAI 兼容聊天 ----
    def _chat_openai(self, cfg: ModelConfig, messages, temperature, max_tokens) -> str:
        """OpenAI 兼容的 chat completion 调用：POST /chat/completions → 取 content。"""
        payload = {
            "model": cfg.name,          # 模型名
            "messages": messages,       # 对话历史（list of {role, content}）
            "temperature": temperature,  # 采样温度
            "max_tokens": max_tokens,   # 最大生成长度
        }
        _, raw = self._request(cfg, "POST", "/chat/completions", payload)  # 发起请求
        out = json.loads(raw)                     # 解析响应
        return out["choices"][0]["message"]["content"]  # 取第一条消息的 content

    # ---- Ollama 原生聊天 ----
    def _chat_ollama(self, cfg: ModelConfig, messages, temperature, max_tokens) -> str:
        """Ollama 原生 /api/chat 调用。"""
        payload = {
            "model": cfg.name,               # 模型名（如 deepseek-r1:8b）
            "messages": messages,            # Ollama 同样使用标准 messages 结构
            "stream": False,                 # 非流式
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        url = cfg.ollama_base + "/api/chat"  # 原生端点（根地址去掉 /v1）
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST", headers={"Content-Type": "application/json"}
        )
        with _open(req, timeout=self.timeout) as resp:
            out = json.loads(resp.read().decode("utf-8", "replace"))
        return out["message"]["content"]  # 取 message.content

    def _chat_one(self, cfg: ModelConfig, messages, temperature, max_tokens) -> str:
        """对单个模型发起一次聊天（按端点类型分派）。"""
        if cfg.kind == "ollama":
            return self._chat_ollama(cfg, messages, temperature, max_tokens)
        return self._chat_openai(cfg, messages, temperature, max_tokens)

    def chat(self, messages, *, temperature=None, max_tokens=2048, **kw) -> str:
        """统一的聊天入口，带**模型降级**：

        按能力顺序从强到弱尝试每个未被禁用的模型：
          - 尝试成功 -> 直接返回文本；
          - 尝试失败（重试耗尽 / 4xx）-> 该模型加入本次运行黑名单，继续尝试下一个；
          - 全部失败 -> 抛 LLMError。

        参数:
          messages:    [{role, content}, ...] 聊天消息列表
          temperature: 温度（不传则用构造时的默认值）
          max_tokens:  最大生成 token 数
        返回:
          模型回复的纯文本（content）
        """
        temperature = self.temperature if temperature is None else temperature  # 默认温度
        errors: list[str] = []  # 记录每个失败模型的原因（最终异常信息用）
        while True:
            cfg = self._pick()  # 取当前最强的未禁用模型
            if cfg is None:     # 全部被禁用 -> 终止
                break
            t0 = time.time()  # 计时开始（记录单次模型调用耗时）
            try:
                text = self._chat_one(cfg, messages, temperature, max_tokens)  # 尝试该模型
                # 【台账】记录本次调用使用的模型 + 耗时 + 返回 token 数
                _ledger("model.call_success", model=cfg.name, kind=cfg.kind,
                        duration_s=round(time.time() - t0, 3),
                        output_chars=len(text), messages=messages)
                return text
            except LLMError as e:
                self._disable(cfg, str(e))  # 该模型不可用 -> 黑名单 + 降级日志
                errors.append(f"{cfg.name}: {e}")
            except Exception as e:           # 其它未预期异常同样降级
                self._disable(cfg, f"{type(e).__name__}: {e}")
                errors.append(f"{cfg.name}: {type(e).__name__}: {e}")
        raise LLMError("所有 LLM 模型均不可用: " + " | ".join(errors) if errors else "无可用模型")

    def invoke(self, messages, **kw) -> _Msg:
        """同步调用接口（兼容 LangChain 式约定）：返回带 .content 的对象。"""
        return _Msg(self.chat(messages, **kw))

    async def ainvoke(self, messages, **kw) -> _Msg:
        """异步调用接口：当前实现为同步方式的简单封装，返回 _Msg 对象。"""
        return _Msg(self.chat(messages, **kw))

    def health(self, timeout: int = 6) -> bool:
        """健康检查：依次探测**每个未被禁用**的模型端点。

        规则：
          - 任一端点返回状态码 < 500（或 4xx）即视为整体可用（还有备用模型）；
          - 若存在可用的端点则立即返回 True；
          - 全部端点不可达（5xx / 无法连接）才返回 False。
        注意：健康检查不会把模型加入黑名单（黑名单只在真实调用失败时生效）。
        """
        for cfg in self.models:  # 遍历所有模型（含已禁用——禁用也可能想恢复？此处仅为启动探测）
            if cfg.name in self._disabled:  # 本次运行已禁用的模型不再探测
                continue
            # 优先探测 OpenAI 兼容端点；Ollama 模型探测其原生端点
            paths = ["/models", "/health", "/"] if cfg.kind == "openai" else ["/api/tags", "/"]
            for path in paths:
                url = cfg.base_url.rstrip("/") + path if cfg.kind == "openai" \
                    else cfg.ollama_base + path
                req = urllib.request.Request(
                    url, method="GET",
                    headers={"Authorization": f"Bearer {cfg.api_key or 'sk-noauth'}"},
                )
                try:
                    with _open(req, timeout=timeout) as resp:
                        if resp.status < 500:  # 非 5xx 就算可达
                            # 【台账】记录该模型健康检查探测成功
                            _ledger("model.health_ok", model=cfg.name, path=f"{cfg.base_url}{path}",
                                    status=resp.status)
                            return True
                except urllib.error.HTTPError as e:
                    if e.code < 500:  # HTTP 错误但 < 500（如 404）也算可达
                        # 【台账】记录该模型健康检查探测成功（出现 4xx 仍视为在线服务）
                        _ledger("model.health_ok", model=cfg.name, path=f"{cfg.base_url}{path}",
                                status=e.code)
                        return True
                except Exception:
                    pass  # 连接类错误 -> 继续尝试下一个路径 / 下一个模型
        # 【台账】记录所有模型健康检查均失败的端点状态
        _ledger("model.health_all_failed", models=[c.name for c in self.models])
        return False  # 全部失败 -> 不可用