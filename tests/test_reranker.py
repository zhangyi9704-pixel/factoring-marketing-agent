"""Reranker 单元测试（DashScope gte-rerank-v2 + passthrough/降级，不触网）。"""
import httpx

from app.config import Settings
from app.rag.reranker import Reranker

import json


def make_items(n):
    return [{"chunk_id": f"c{i}", "content": f"文档{i}"} for i in range(n)]


def make_settings(**overrides) -> Settings:
    settings = Settings(_env_file=None)
    rerank = settings.rag.retrieval.rerank
    rerank.min_score = 0.1  # 测试固定阈值，不耦合生产校准值
    rerank.provider = overrides.pop("provider", "dashscope")
    rerank.api_key = overrides.pop("api_key", "sk-test")
    for key, value in overrides.items():
        setattr(rerank, key, value)
    return settings


def dashscope_response(scores):        # [(index, relevance_score), ...]
    return {
        "output": {
            "results": [
                {"index": index, "relevance_score": score} for index, score in scores
            ],
        },
    }


class RequestRecorder:
    """捕获请求体并返回预设响应（httpx.MockTransport handler）。"""

    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status_code, json=self.payload)


def make_reranker(settings, handler) -> Reranker:
    return Reranker(settings, http_client=httpx.Client(
        transport=httpx.MockTransport(handler),
    ))


# =====================================================
# DashScope gte-rerank-v2
# =====================================================

class TestDashscopeRerank:

    def test_reorders_and_overwrites_scores(self):
        recorder = RequestRecorder(dashscope_response([(2, 0.9), (0, 0.5), (1, 0.2)]))
        reranker = make_reranker(make_settings(), recorder)

        kept = reranker.rerank("保理融资比例", make_items(3))

        assert [i["chunk_id"] for i in kept] == ["c2", "c0", "c1"]  # 分数降序
        assert [i["score"] for i in kept] == [0.9, 0.5, 0.2]        # RRF 分被覆写

    def test_min_score_filters_weak_items(self):
        recorder = RequestRecorder(dashscope_response([(0, 0.9), (1, 0.05)]))
        reranker = make_reranker(make_settings(min_score=0.1), recorder)

        kept = reranker.rerank("q", make_items(2))
        assert [i["chunk_id"] for i in kept] == ["c0"]              # 0.05 < 0.1 剔除

    def test_all_below_min_score_returns_empty(self):
        recorder = RequestRecorder(dashscope_response([(0, 0.01), (1, 0.02)]))
        reranker = make_reranker(make_settings(), recorder)

        assert reranker.rerank("q", make_items(2)) == []             # 检索缺失语义

    def test_truncates_to_top_n(self):
        recorder = RequestRecorder(dashscope_response(
            [(i, 0.9 - i * 0.1) for i in range(8)],
        ))
        reranker = make_reranker(make_settings(), recorder)

        assert len(reranker.rerank("q", make_items(8))) == 5         # config top_n=5
        assert len(reranker.rerank("q", make_items(8), top_n=2)) == 2

    def test_request_payload_format(self):
        recorder = RequestRecorder(dashscope_response([(0, 0.9)]))
        reranker = make_reranker(make_settings(), recorder)
        reranker.rerank("反向保理", make_items(2))

        request = recorder.requests[0]
        assert request.headers["Authorization"] == "Bearer sk-test"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "qwen3.7-text-rerank"
        assert body["input"]["query"] == "反向保理"
        assert body["input"]["documents"] == ["文档0", "文档1"]
        assert body["parameters"]["return_documents"] is False

    def test_invalid_index_entries_skipped(self):
        # 索引越界 / 字段缺失的条目跳过，其余照常参与
        payload = {"output": {"results": [
            {"index": 9, "relevance_score": 0.99},     # 越界
            {"relevance_score": 0.5},                  # 缺 index
            {"index": 1, "relevance_score": 0.4},
        ]}}
        reranker = make_reranker(make_settings(), RequestRecorder(payload))
        assert [i["chunk_id"] for i in reranker.rerank("q", make_items(2))] == ["c1"]


# =====================================================
# 降级（故障不中断检索链路）
# =====================================================

class TestDashscopeDegradation:

    def test_non_200_degrades_to_passthrough(self):
        recorder = RequestRecorder({}, status_code=429)
        reranker = make_reranker(make_settings(), recorder)

        kept = reranker.rerank("q", make_items(7))
        assert [i["chunk_id"] for i in kept] == [f"c{i}" for i in range(5)]  # 透传截断

    def test_malformed_response_degrades(self):
        reranker = make_reranker(make_settings(), RequestRecorder({"unexpected": True}))
        assert len(reranker.rerank("q", make_items(3))) == 3        # 无 top_n 截断上限内

    def test_missing_api_key_degrades(self):
        reranker = make_reranker(make_settings(api_key=None), RequestRecorder({}))
        assert len(reranker.rerank("q", make_items(3))) == 3

    def test_network_error_degrades(self):
        def boom(request):
            raise httpx.ConnectError("connection refused")

        reranker = make_reranker(make_settings(), boom)
        kept = reranker.rerank("q", make_items(6))
        assert [i["chunk_id"] for i in kept] == [f"c{i}" for i in range(5)]


# =====================================================
# passthrough / 开关
# =====================================================

class TestPassthroughRerank:

    def test_passthrough_truncates_only(self):
        reranker = Reranker(make_settings(provider="passthrough", api_key=None))
        kept = reranker.rerank("query", make_items(10))
        assert [item["chunk_id"] for item in kept] == [f"c{i}" for i in range(5)]

    def test_disabled_returns_all(self):
        settings = make_settings(enabled=False)
        reranker = Reranker(settings)
        assert len(reranker.rerank("q", make_items(9))) == 9

    def test_empty_input_returns_empty(self):
        reranker = Reranker(make_settings(provider="passthrough", api_key=None))
        assert reranker.rerank("q", []) == []
