"""
LangChain + ChatGPT Codex 订阅 Demo
====================================
前提：本机已有 ~/.codex/auth.json（运行过 `codex login`）
不依赖任何第三方包，只需要：
  pip install langchain-core httpx

用法：
  python langchain_codex_demo.py
"""

import base64
import json
import os
import pathlib
import time
from typing import Any, Iterator, List, Optional

import httpx
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

CODEX_ENDPOINT = "https://chatgpt.com/backend-api/codex/responses"
OAUTH_TOKEN_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"  # Codex CLI 公开 client_id

# Codex 后端会校验 instructions 必须匹配官方 system prompt
# 这里用精简版，实际项目可从 codex-rs 仓库拉完整版
CODEX_SYSTEM_PROMPT = (
    "You are Codex, based on GPT-5. You are running as a coding agent in the "
    "Codex CLI on a user's computer. You help users write, understand, debug, "
    "and reason about code."
)


# ──────────────────────────────────────────────
# Token 管理
# ──────────────────────────────────────────────

def _find_auth_json() -> pathlib.Path:
    """按优先级查找 auth.json"""
    candidates = [
        os.environ.get("CODEX_HOME"),
        os.environ.get("CHATGPT_LOCAL_HOME"),
        "~/.codex",
        "~/.hermes",
        "~/.chatgpt-local",
    ]
    for base in candidates:
        if not base:
            continue
        p = pathlib.Path(base).expanduser() / "auth.json"
        if p.exists():
            return p
    raise FileNotFoundError(
        "找不到 auth.json，请先运行 `codex login` 完成 ChatGPT 授权"
    )


def _jwt_account_id(token: str) -> str:
    """从 JWT payload 里提取 chatgpt_account_id"""
    try:
        payload_b64 = token.split(".")[1]
        # base64url 补齐
        padding = 4 - len(payload_b64) % 4
        payload_b64 += "=" * (padding % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload_b64))
        return claims["https://api.openai.com/auth"]["chatgpt_account_id"]
    except Exception:
        return ""


def _load_tokens(auth_path: pathlib.Path) -> dict:
    return json.loads(auth_path.read_text())["tokens"]


def _refresh_tokens(auth_path: pathlib.Path) -> dict:
    """用 refresh_token 换新 access_token，并写回文件"""
    raw = json.loads(auth_path.read_text())
    rt = raw["tokens"]["refresh_token"]
    resp = httpx.post(
        OAUTH_TOKEN_URL,
        json={
            "grant_type": "refresh_token",
            "refresh_token": rt,
            "client_id": CLIENT_ID,
            "scope": "openid profile email offline_access",
        },
        timeout=30,
    )
    resp.raise_for_status()
    new = resp.json()
    raw["tokens"]["access_token"] = new["access_token"]
    raw["tokens"]["id_token"] = new.get("id_token", raw["tokens"].get("id_token", ""))
    raw["tokens"]["refresh_token"] = new.get("refresh_token", rt)
    raw["last_refresh"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    auth_path.write_text(json.dumps(raw, indent=2))
    print("🔄 Token 已刷新并写回", auth_path)
    return raw["tokens"]


# ──────────────────────────────────────────────
# 消息格式转换（LangChain → Codex Responses API）
# ──────────────────────────────────────────────

def _lc_messages_to_codex_input(messages: List[BaseMessage]) -> list:
    """把 LangChain messages 转成 Codex Responses API 的 input 格式"""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            # system message 通过 instructions 字段传，这里跳过
            continue
        is_user = isinstance(msg, HumanMessage)
        role = "user" if is_user else "assistant"
        content_type = "input_text" if is_user else "output_text"
        result.append({
            "role": role,
            "content": [{"type": content_type, "text": str(msg.content)}],
        })
    return result


def _extract_system(messages: List[BaseMessage]) -> Optional[str]:
    for msg in messages:
        if isinstance(msg, SystemMessage):
            return str(msg.content)
    return None


# ──────────────────────────────────────────────
# ChatCodex：LangChain BaseChatModel 实现
# ──────────────────────────────────────────────

class ChatCodex(BaseChatModel):
    """
    使用 ChatGPT Plus/Pro 订阅（Codex OAuth）的 LangChain Chat Model。
    读取本机 ~/.codex/auth.json，不需要 OpenAI API Key。

    示例：
        from langchain_core.messages import HumanMessage
        llm = ChatCodex(model="gpt-5.3-codex")
        print(llm.invoke([HumanMessage(content="写个冒泡排序")]).content)
    """

    model: str = "gpt-5.3-codex"
    temperature: float = 1.0
    max_tokens: int = 4096
    timeout: int = 120
    # 每次请求前是否尝试刷新 token（建议开启）
    auto_refresh: bool = True

    # 内部状态（不暴露给用户）
    _auth_path: pathlib.Path = None
    _tokens: dict = None

    def _ensure_tokens(self) -> dict:
        if self._auth_path is None:
            self._auth_path = _find_auth_json()
        if self._tokens is None:
            self._tokens = _load_tokens(self._auth_path)
        return self._tokens

    def _get_headers(self) -> dict:
        tokens = self._ensure_tokens()
        access_token = tokens["access_token"]
        account_id = tokens.get("account_id") or _jwt_account_id(access_token)
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",          # 必须，否则 403
            "ChatGPT-Account-Id": account_id,
            "User-Agent": "codex-cli/0.125.0",
            "Accept": "text/event-stream",
        }

    def _build_body(
        self,
        messages: List[BaseMessage],
        stream: bool = False,
        system_override: Optional[str] = None,
    ) -> dict:
        system_msg = system_override or _extract_system(messages) or CODEX_SYSTEM_PROMPT
        # chatgpt.com/backend-api/codex/responses 是 Codex CLI 的内部流式端点。
        # 它要求 stream=true，且不接受公开 Responses API 的 temperature/max_tokens 字段。
        return {
            "model": self.model,
            "instructions": system_msg,
            "input": _lc_messages_to_codex_input(messages),
            "store": False,
            "stream": True,
        }

    def _stream_events_with_retry(self, messages):
        """流式请求，遇到 401 自动刷新 token 重试一次"""
        for attempt in range(2):
            headers = self._get_headers()
            body = self._build_body(messages, stream=True)
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", CODEX_ENDPOINT, headers=headers, json=body) as resp:
                    if resp.status_code == 401 and attempt == 0 and self.auto_refresh:
                        print("⚠️  Token 过期，尝试刷新...")
                        self._tokens = _refresh_tokens(self._auth_path)
                        continue
                    if resp.status_code >= 400:
                        detail = resp.read().decode("utf-8", errors="replace")
                        raise httpx.HTTPStatusError(
                            f"{resp.status_code} error from Codex endpoint: {detail}",
                            request=resp.request,
                            response=resp,
                        )
                    yield from self._parse_sse_events(resp)
                    return
        raise RuntimeError("Token 刷新后仍然 401，请重新运行 `codex login`")

    def _collect_stream_response(self, messages) -> tuple[str, dict, str]:
        """消费 SSE，拼成 LangChain invoke 需要的一次性响应。"""
        text = ""
        usage = {}
        response_id = ""
        for event in self._stream_events_with_retry(messages):
            event_type = event.get("type")
            if event_type == "response.output_text.delta":
                text += event.get("delta", "")
            elif event_type == "response.completed":
                response = event.get("response", {})
                usage = response.get("usage", {}) or {}
                response_id = response.get("id", "") or response_id
            elif event_type == "response.created":
                response_id = event.get("response", {}).get("id", "") or response_id
        return text, usage, response_id

    # ── 非流式 ──────────────────────────────────

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        text, usage, response_id = self._collect_stream_response(messages)
        message = AIMessage(
            content=text,
            response_metadata={
                "model": self.model,
                "usage": usage,
                "id": response_id,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    # ── 流式 ─────────────────────────────────────

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for event in self._stream_events_with_retry(messages):
            if event.get("type") == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    chunk = AIMessageChunk(content=delta)
                    yield ChatGenerationChunk(message=chunk)

    def _parse_sse_events(self, resp) -> Iterator[dict]:
        """解析 Codex SSE 流，返回每条 data JSON。"""
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            raw = line[len("data: "):]
            if raw.strip() == "[DONE]":
                break
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue

    @property
    def _llm_type(self) -> str:
        return "chatgpt-codex-oauth"


# ──────────────────────────────────────────────
# Demo 入口
# ──────────────────────────────────────────────

def demo_basic():
    """基础对话"""
    print("\n" + "="*50)
    print("📌 Demo 1: 基础对话")
    print("="*50)
    llm = ChatCodex(model="gpt-5.3-codex")
    response = llm.invoke([HumanMessage(content="用 Python 写一个快速排序，加注释")])
    print(response.content)


def demo_streaming():
    """流式输出"""
    print("\n" + "="*50)
    print("📌 Demo 2: 流式输出")
    print("="*50)
    llm = ChatCodex(model="gpt-5.3-codex")
    print("回答：", end="", flush=True)
    for chunk in llm.stream([HumanMessage(content="解释一下什么是 LangChain，100字以内")]):
        print(chunk.content, end="", flush=True)
    print()


def demo_multi_turn():
    """多轮对话"""
    print("\n" + "="*50)
    print("📌 Demo 3: 多轮对话")
    print("="*50)
    llm = ChatCodex(model="gpt-5.3-codex")
    messages = [
        SystemMessage(content="你是一个 Python 专家，回答简洁"),
        HumanMessage(content="列表推导式是什么？"),
    ]
    r1 = llm.invoke(messages)
    print("助手:", r1.content)

    # 继续对话
    messages += [r1, HumanMessage(content="给一个实际例子")]
    r2 = llm.invoke(messages)
    print("助手:", r2.content)


def demo_langchain_chain():
    """组成 LangChain 链"""
    print("\n" + "="*50)
    print("📌 Demo 4: LangChain Chain（需要 pip install langchain-core）")
    print("="*50)
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        llm = ChatCodex(model="gpt-5.3-codex", max_tokens=512)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是代码审查专家，用中文给出简洁反馈"),
            ("human", "审查这段代码：\n{code}"),
        ])
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"code": "def add(a,b):\n    return a+b"})
        print(result)
    except ImportError:
        print("跳过（需要 pip install langchain-core）")


if __name__ == "__main__":
    print("🚀 LangChain + Codex OAuth Demo")
    print("使用的 auth 文件:", _find_auth_json())

    # 选择要跑的 demo
    demo_basic()
    # demo_streaming()   # 取消注释启用
    # demo_multi_turn()  # 取消注释启用
    # demo_langchain_chain()  # 取消注释启用
