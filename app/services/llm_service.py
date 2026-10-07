"""DeepSeek LLM 服务（TDD §7）。

接入方式：OpenAI 兼容协议（openai SDK 指向 DeepSeek base_url）。

双模型路由（TDD §7.2）：
    reasoning → deepseek-v4-pro  （T=0.2，thinking=on，复杂业务分析：访前准备 / Evaluation）
    chat      → deepseek-v4-flash（T=0.4，thinking=off，高频交互：情景演练）

超时/重试（TDD §7.3）：read=120s；SDK 隐式重试关闭，现有外部调用边界共用总尝试次数与有上限退避。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from openai import OpenAI

from app.config import Settings, get_settings
from app.services.external_calls import ExternalCalls, ExternalCallError, check_cancelled, check_deadline, timeout_for, bound_timeout

logger = logging.getLogger(__name__)

MODEL_ROLES = ("chat", "reasoning")

# ```json ... ``` 围栏（LLM 常见包裹），宽松匹配首尾围栏
_CODE_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


class LLMServiceError(RuntimeError):
    """LLM 配置或调用错误。"""


class LLMService:
    """DeepSeek 调用封装（OpenAI 兼容协议）。"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        api_key = self.settings.llm.api.api_key
        if not api_key:
            raise LLMServiceError(
                "DEEPSEEK_API_KEY 未配置：config.yaml 中 ${DEEPSEEK_API_KEY} "
                "未能从环境变量 / .env 解析，请检查 .env 文件"
            )
        self.calls = ExternalCalls(self.settings)
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.settings.llm.api.base_url,
            timeout=timeout_for(self.settings, self.settings.llm.timeout.read),
            max_retries=0,  # One retry owner, shared with embedding/rerank.
        )

    def model_name(self, model_role: str) -> str:
        """模型角色 → 模型名映射。"""
        if model_role not in MODEL_ROLES:
            raise ValueError(f"未知模型角色: {model_role!r}，可选 {MODEL_ROLES}")
        if model_role == "reasoning":
            return self.settings.llm.models.reasoning.name
        return self.settings.llm.models.chat.name

    def chat(
        self,
        prompt: str,
        *,
        model_role: str = "chat",
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        on_delta: Optional[callable] = None,
        timeout: Optional[float] = None,
    ) -> str:
        """调用 DeepSeek 生成回复。

        Args:
            prompt: 用户提示词。
            model_role: "chat"（高频交互）或 "reasoning"（复杂分析）。
            system: 可选系统提示词。
            temperature: 缺省使用 config.yaml 中该角色默认值（chat=0.4 / reasoning=0.2）。
            max_tokens: 缺省使用 config.yaml 中该角色默认值。
            on_delta: 可选增量回调（流式）。提供时走 stream=True 通道，
                每收到一段文本即回调一次；回调异常只记日志不影响生成，
                返回值仍为完整文本（调用方无需自行拼接）。

        thinking 通过 DeepSeek extra_body 传递；reasoning_effort 在思考模式启用时传递。
        思考模式下 DeepSeek 不使用 temperature；保留该字段供非思考模式使用。
        """
        if model_role not in MODEL_ROLES:
            raise ValueError(f"未知模型角色: {model_role!r}，可选 {MODEL_ROLES}")
        model_cfg = getattr(self.settings.llm.models, model_role)

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        request = dict(
            model=model_cfg.name,
            messages=messages,
            temperature=model_cfg.temperature if temperature is None else temperature,
            max_tokens=model_cfg.max_tokens if max_tokens is None else max_tokens,
            extra_body={"thinking": {"type": "enabled" if model_cfg.thinking.enabled else "disabled"}},
        )
        if model_cfg.thinking.enabled and model_cfg.reasoning_effort:
            request["reasoning_effort"] = model_cfg.reasoning_effort
        if timeout is not None:
            import httpx
            request["timeout"] = httpx.Timeout(timeout, connect=min(timeout, self.settings.llm.timeout.connect))
        if on_delta is None:
            return self._chat_once(request, model_role, model_cfg.name)
        return self._chat_streaming(request, model_role, model_cfg.name, on_delta)

    def _attempt_request(self, request):
        attempt = dict(request)
        attempt["timeout"] = bound_timeout(request.get("timeout") or timeout_for(
            self.settings, self.settings.llm.timeout.read))
        return attempt

    def _chat_once(self, request: dict, model_role: str, model_name: str) -> str:
        """非流式调用（原路径）。"""
        try:
            response = self.calls.run("llm", lambda: self._client.chat.completions.create(**self._attempt_request(request)))
        except ExternalCallError as exc:
            raise LLMServiceError(str(exc)) from None
        if response.usage:
            logger.info(
                "deepseek call: role=%s model=%s prompt_tokens=%s completion_tokens=%s",
                model_role, model_name,
                response.usage.prompt_tokens, response.usage.completion_tokens,
            )
        return response.choices[0].message.content or ""

    def _chat_streaming(
        self, request: dict, model_role: str, model_name: str, on_delta: callable,
    ) -> str:
        """流式调用：逐段回调 on_delta，返回完整文本。"""
        pieces: list[str] = []
        emitted = False
        usage = None
        def once():
            nonlocal emitted, usage
            stream = self._client.chat.completions.create(**self._attempt_request(request), stream=True)
            try:
                for chunk in stream:
                    check_cancelled()
                    check_deadline()
                    if getattr(chunk, "usage", None) is not None:
                        from app.services.external_calls import usage_of
                        usage = usage_of(chunk)
                    if not getattr(chunk, "choices", None):
                        continue
                    piece = getattr(chunk.choices[0].delta, "content", None) or ""
                    if not piece:
                        continue
                    pieces.append(piece)
                    emitted = True
                    try:
                        on_delta(piece)
                    except Exception:
                        logger.warning("on_delta callback failure")
                return "".join(pieces)
            finally:
                if hasattr(stream, "close"):
                    stream.close()
        try:
            return self.calls.run("llm_stream", once, emitted=lambda: emitted, usage=lambda _: usage)
        except ExternalCallError as exc:
            raise LLMServiceError(str(exc)) from None

    def chat_json_once(self, prompt: str, *, model_role="reasoning", system=None):
        """不做JSON修复；评估管线统一拥有唯一一次schema修复预算。"""
        return self._loads_json(self.chat(prompt, model_role=model_role, system=system))

    def chat_json(
        self,
        prompt: str,
        *,
        model_role: str = "reasoning",
        system: Optional[str] = None,
        on_delta: Optional[callable] = None,
    ) -> dict:
        """调用 LLM 并解析 JSON 输出（Agent 节点结构化抽取统一入口）。

        on_delta：流式回调（收到的是**原始 JSON 文本**增量，调用方自行
        增量抽取所需字段）。流式已发出内容后解析失败立即报错，不重新请求或叠加输出。

        容错：自动剥离 markdown 代码围栏；解析失败时把错误信息追加到
        提示词重试 1 次，仍失败抛 LLMServiceError。
        """
        emitted = False
        def tracked_delta(piece):
            nonlocal emitted
            emitted = True
            on_delta(piece)
        text = self.chat(prompt, model_role=model_role, system=system,
                         on_delta=tracked_delta if on_delta is not None else None)
        try:
            return self._loads_json(text)
        except (ValueError, TypeError) as exc:
            logger.warning("chat_json response invalid; emitted=%s", emitted)
        check_cancelled()
        if emitted:
            raise LLMServiceError("LLM 流式输出无法解析为 JSON；已发送内容，不自动重试")
        retry = (
            f"{prompt}\n\n---\n你上一次的输出不是合法 JSON（错误：解析失败），"
            f"请仅输出一个合法的 JSON 对象，不要包含任何其他文字或代码围栏。"
        )
        text = self.chat(retry, model_role=model_role, system=system)
        try:
            return self._loads_json(text)
        except (ValueError, TypeError) as exc:
            raise LLMServiceError(f"LLM 输出无法解析为 JSON: {exc}") from exc

    @staticmethod
    def _loads_json(text: str) -> dict:
        """剥围栏后 json.loads，要求结果为 dict。"""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = _CODE_FENCE.sub("", cleaned).strip()
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError(f"期望 JSON 对象，得到 {type(data).__name__}")
        return data

    def list_models(self) -> list[str]:
        """列出账号可用模型（用于连通性 / 模型名校验）。"""
        return sorted(m.id for m in self.calls.run("llm_models", self._client.models.list).data)
