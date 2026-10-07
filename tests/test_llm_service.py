"""LLMService 单元测试（使用 stub 客户端，不发起真实网络请求）。"""
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.services.llm_service import LLMService, LLMServiceError


def make_settings() -> Settings:
    return Settings(_env_file=None)


class TestInit:

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        with pytest.raises(LLMServiceError, match="DEEPSEEK_API_KEY"):
            LLMService(make_settings())

    def test_client_initialized_with_key(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-key")
        svc = LLMService(make_settings())
        assert str(svc._client.base_url).rstrip("/") == "https://api.deepseek.com"


class TestModelRouting:

    def test_role_to_model_mapping(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-key")
        svc = LLMService(make_settings())
        assert svc.model_name("chat") == "deepseek-v4-flash"
        assert svc.model_name("reasoning") == "deepseek-v4-pro"

    def test_unknown_role_raises(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-key")
        svc = LLMService(make_settings())
        with pytest.raises(ValueError):
            svc.model_name("v4")


class _StubCompletions:
    def __init__(self, reply: str):
        self.replies = [reply] if isinstance(reply, str) else list(reply)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.replies.pop(0) if self.replies else ""
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )


class TestChatStubbed:
    """用 stub 替换底层客户端，验证 chat() 的参数装配逻辑。"""

    def _make(self, monkeypatch, reply="收到") -> LLMService:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-key")
        svc = LLMService(make_settings())
        svc._client = SimpleNamespace(
            chat=SimpleNamespace(completions=_StubCompletions(reply))
        )
        return svc

    def test_returns_content(self, monkeypatch):
        svc = self._make(monkeypatch, reply="好的")
        assert svc.chat("你好") == "好的"

    def test_default_uses_chat_model_and_config_temperature(self, monkeypatch):
        svc = self._make(monkeypatch)
        svc.chat("你好")
        call = svc._client.chat.completions.calls[0]
        assert call["model"] == "deepseek-v4-flash"
        assert call["temperature"] == 0.4
        assert call["max_tokens"] == 4096
        assert call["extra_body"] == {"thinking": {"type": "disabled"}}
        assert "reasoning_effort" not in call

    def test_reasoning_role_routing(self, monkeypatch):
        svc = self._make(monkeypatch)
        svc.chat("分析", model_role="reasoning")
        call = svc._client.chat.completions.calls[0]
        assert call["model"] == "deepseek-v4-pro"
        assert call["temperature"] == 0.2
        assert call["max_tokens"] == 20000
        assert call["extra_body"] == {"thinking": {"type": "enabled"}}
        assert call["reasoning_effort"] == "high"

    def test_explicit_overrides(self, monkeypatch):
        svc = self._make(monkeypatch)
        svc.chat("你好", temperature=0.9, max_tokens=128)
        call = svc._client.chat.completions.calls[0]
        assert call["temperature"] == 0.9
        assert call["max_tokens"] == 128

    def test_system_message_prepended(self, monkeypatch):
        svc = self._make(monkeypatch)
        svc.chat("你好", system="你是保理行业销售教练")
        call = svc._client.chat.completions.calls[0]
        assert call["messages"][0] == {"role": "system", "content": "你是保理行业销售教练"}
        assert call["messages"][-1] == {"role": "user", "content": "你好"}

    def test_unknown_role_raises_before_call(self, monkeypatch):
        svc = self._make(monkeypatch)
        with pytest.raises(ValueError):
            svc.chat("你好", model_role="unknown")
        assert svc._client.chat.completions.calls == []


class TestChatJsonStubbed:
    """chat_json：围栏剥离 / 重试 / 失败抛错。"""

    def _make(self, monkeypatch, replies) -> LLMService:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-key")
        svc = LLMService(make_settings())
        svc._client = SimpleNamespace(
            chat=SimpleNamespace(completions=_StubCompletions(replies))
        )
        return svc

    def test_plain_json(self, monkeypatch):
        svc = self._make(monkeypatch, '{"company": "华建建设"}')
        assert svc.chat_json("抽取") == {"company": "华建建设"}

    def test_code_fence_stripped(self, monkeypatch):
        svc = self._make(monkeypatch, '```json\n{"a": 1}\n```')
        assert svc.chat_json("抽取") == {"a": 1}

    def test_retry_then_success(self, monkeypatch):
        svc = self._make(monkeypatch, ["不是 JSON", '{"a": 1}'])
        assert svc.chat_json("抽取") == {"a": 1}
        assert len(svc._client.chat.completions.calls) == 2
        assert "合法 JSON" in svc._client.chat.completions.calls[1]["messages"][-1]["content"]

    def test_non_dict_rejected(self, monkeypatch):
        svc = self._make(monkeypatch, '["列表"]')
        with pytest.raises(LLMServiceError, match="无法解析为 JSON"):
            svc.chat_json("抽取")

    def test_both_attempts_fail_raises(self, monkeypatch):
        svc = self._make(monkeypatch, ["坏", "还是坏"])
        with pytest.raises(LLMServiceError):
            svc.chat_json("抽取")
        assert len(svc._client.chat.completions.calls) == 2

    def test_uses_reasoning_defaults(self, monkeypatch):
        svc = self._make(monkeypatch, "{}")
        svc.chat_json("分析")
        call = svc._client.chat.completions.calls[0]
        assert call["model"] == "deepseek-v4-pro"
        assert call["temperature"] == 0.2


class _StubStreamCompletions:
    """响应队列：每次 create 弹出一个响应；响应为分片列表。

    stream=True 逐片产出（末尾补一个空增量，应被跳过）；
    否则拼接为整段返回。
    """

    def __init__(self, responses):
        self.responses = [
            list(r) if isinstance(r, (list, tuple)) else [r] for r in responses
        ]
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        chunks = self.responses.pop(0) if self.responses else [""]
        if not kwargs.get("stream"):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content="".join(chunks)))],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=5),
            )

        def _chunks():
            for p in chunks:
                yield SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content=p))])
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=None))])      # 空增量应被跳过

        return _chunks()


class TestChatStreaming:
    """on_delta 流式路径（stream=True 分片回调）。"""

    def _make(self, monkeypatch, responses) -> LLMService:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-fake-key")
        svc = LLMService(make_settings())
        svc._client = SimpleNamespace(
            chat=SimpleNamespace(completions=_StubStreamCompletions(responses))
        )
        return svc

    def test_deltas_in_order_and_full_text(self, monkeypatch):
        svc = self._make(monkeypatch, [["这个我", "暂时不方便", "说。"]])
        got: list[str] = []
        text = svc.chat("你好", on_delta=got.append)
        assert got == ["这个我", "暂时不方便", "说。"]
        assert text == "这个我暂时不方便说。"
        assert svc._client.chat.completions.calls[0]["stream"] is True

    def test_on_delta_exception_swallowed(self, monkeypatch):
        svc = self._make(monkeypatch, [["a", "b"]])

        def boom(piece):
            raise RuntimeError("回调挂了")

        assert svc.chat("你好", on_delta=boom) == "ab"     # 回调故障不影响生成

    def test_no_on_delta_keeps_non_stream_path(self, monkeypatch):
        svc = self._make(monkeypatch, ["整段"])
        assert svc.chat("你好") == "整段"
        assert "stream" not in svc._client.chat.completions.calls[0]

    def test_chat_json_retry_not_streamed_twice(self, monkeypatch):
        svc = self._make(monkeypatch, ["不是 JSON", '{"a": 1}'])
        seen: list[str] = []
        with pytest.raises(LLMServiceError, match="不自动重试"):
            svc.chat_json("抽取", on_delta=seen.append)
        assert seen == ["不是 JSON"]                        # 重试那次不再上屏
        streams = [c for c in svc._client.chat.completions.calls if c.get("stream")]
        assert len(streams) == 1
        assert len(svc._client.chat.completions.calls) == 1


def test_thinking_and_effort_reach_actual_sdk_http_body(monkeypatch):
    import json
    import httpx
    from openai import OpenAI
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-key")
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "offline", "object": "chat.completion", "created": 0,
            "model": "offline", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]})
    service = LLMService(make_settings())
    service._client.close()
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        service._client = OpenAI(api_key="offline-key", base_url="https://offline.invalid", http_client=client, max_retries=0)
        assert service.chat("analyse", model_role="reasoning") == "ok"
        assert service.chat("reply", on_delta=None) == "ok"
    assert requests[0]["thinking"] == {"type": "enabled"}
    assert requests[0]["reasoning_effort"] == "high"
    assert requests[1]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in requests[1]
