"""EmbeddingService 单元测试（注入假模型 / MockTransport，不触网、不加载真实权重）。"""
import json
import math

import httpx
import numpy as np
import pytest

from app.config import Settings
from app.rag.embedder import EmbeddingError, EmbeddingService, lexical_sparse_texts

DIM = 1024  # 与 config.yaml rag.embedding.dimension 一致


class FakeModel:
    """模拟 BGEM3FlagModel.encode 的返回契约。"""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.calls: list[list[str]] = []

    def encode(self, texts, return_dense, return_sparse, return_colbert_vecs):
        assert return_dense and return_sparse and not return_colbert_vecs
        self.calls.append(list(texts))
        dense = np.array([[1.0] + [0.0] * (self.dim - 1) for _ in texts], dtype=np.float32)
        lexical = [{str(10 + i): 0.5} for i in range(len(texts))]
        return {"dense_vecs": dense, "lexical_weights": lexical}


class ZeroModel(FakeModel):
    def encode(self, texts, **kw):
        super().encode(texts, **kw)
        return {
            "dense_vecs": np.zeros((len(texts), self.dim), dtype=np.float32),
            "lexical_weights": [{} for _ in texts],
        }


def make_service(model, batch_size=16):
    settings = Settings(_env_file=None)
    settings.rag.embedding.provider = "local"
    return EmbeddingService(settings, model=model, batch_size=batch_size)


class TestEmbed:

    def test_empty_input(self):
        dense, sparse = make_service(FakeModel()).embed([])
        assert dense == [] and sparse == []

    def test_shape_normalize_and_sparse_int_keys(self):
        dense, sparse = make_service(FakeModel()).embed(["保理", "应收账款"])
        assert len(dense) == 2 and all(len(v) == DIM for v in dense)
        assert abs(np.linalg.norm(dense[0]) - 1.0) < 1e-5   # L2 归一化
        assert all(isinstance(k, int) for s in sparse for k in s)
        assert sparse[0] == {10: 0.5} and sparse[1] == {11: 0.5}

    def test_blank_text_replaced_with_space(self):
        fake = FakeModel()
        make_service(fake).embed(["", "   "])
        assert fake.calls == [[" ", " "]]  # 空白文本不传给模型

    def test_batching(self):
        fake = FakeModel()
        make_service(fake, batch_size=2).embed(["a", "b", "c", "d", "e"])
        assert [len(c) for c in fake.calls] == [2, 2, 1]

    def test_zero_vector_no_nan(self):
        dense, sparse = make_service(ZeroModel()).embed(["x"])
        assert all(v == 0.0 for v in dense[0])               # 除零保护
        assert not np.isnan(np.array(dense)).any()

    def test_dimension_mismatch_raises(self):
        with pytest.raises(EmbeddingError, match="维度"):
            make_service(FakeModel(dim=999)).embed(["x"])

    def test_constructor_does_not_load_model(self):
        service = EmbeddingService(Settings(_env_file=None))
        assert service._model is None          # 懒加载：构造不触发
        assert service.dimension == DIM


# =====================================================
# http provider（TDD §11.4 方案 A：独立 embedding 服务）
# =====================================================

def make_http_settings(url: str = "http://embed:8100") -> Settings:
    settings = Settings(_env_file=None)
    settings.rag.embedding.provider = "http"
    settings.rag.embedding.url = url
    return settings


def ok_response(request) -> httpx.Response:
    n = len(json.loads(request.content)["texts"])
    return httpx.Response(200, json={
        "dense": [[0.1] * DIM for _ in range(n)],
        "sparse": [{"5": 0.5, "7": 0.0} for _ in range(n)],
    })


def make_http_service(handler, batch_size=16, url="http://embed:8100") -> EmbeddingService:
    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5)
    return EmbeddingService(make_http_settings(url), http_client=client, batch_size=batch_size)


class TestHttpProvider:

    def test_embed_via_http(self):
        dense, sparse = make_http_service(ok_response).embed(["保理", ""])
        assert len(dense) == 2 and len(dense[0]) == DIM
        assert sparse[0] == {5: 0.5}           # str key → int，零权重剔除

    def test_batching(self):
        calls = []

        def handler(request):
            calls.append(len(json.loads(request.content)["texts"]))
            return ok_response(request)

        make_http_service(handler, batch_size=2).embed(["a", "b", "c", "d", "e"])
        assert calls == [2, 2, 1]

    def test_url_joins_embed_path_and_blank_replaced(self):
        seen = []

        def handler(request):
            seen.append((str(request.url), json.loads(request.content)["texts"]))
            return ok_response(request)

        make_http_service(handler, url="http://embed:8100/").embed(["", "  "])
        assert seen[0][0] == "http://embed:8100/embed"
        assert seen[0][1] == [" ", " "]

    def test_http_error_status(self):
        service = make_http_service(lambda r: httpx.Response(500, text="boom"))
        with pytest.raises(EmbeddingError, match="500"):
            service.embed(["x"])

    def test_connection_error(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        with pytest.raises(EmbeddingError, match="请求失败"):
            make_http_service(handler).embed(["x"])

    def test_dimension_mismatch(self):
        def handler(request):
            return httpx.Response(200, json={"dense": [[0.1] * 3], "sparse": [{}]})

        with pytest.raises(EmbeddingError, match="维度"):
            make_http_service(handler).embed(["x"])

    def test_missing_url(self):
        settings = make_http_settings(url=None)
        service = EmbeddingService(
            settings,
            http_client=httpx.Client(transport=httpx.MockTransport(ok_response)),
        )
        with pytest.raises(EmbeddingError, match="未配置"):
            service.embed(["x"])

    def test_unknown_provider(self):
        settings = Settings(_env_file=None)
        settings.rag.embedding.provider = "xxx"
        with pytest.raises(EmbeddingError, match="provider"):
            EmbeddingService(settings).embed(["x"])


# =====================================================
# dashscope provider（阿里云托管，OpenAI 兼容协议）
# =====================================================

def make_dashscope_settings() -> Settings:
    settings = Settings(_env_file=None)
    settings.rag.embedding.provider = "dashscope"
    settings.rag.embedding.api_key = "sk-test"
    settings.rag.embedding.model = "text-embedding-v4"
    settings.rag.embedding.base_url = "https://dashscope.example/v1"
    return settings


def ok_openai_response(request) -> httpx.Response:
    n = len(json.loads(request.content)["input"])
    return httpx.Response(200, json={
        "data": [{"index": i, "embedding": [0.1] * DIM} for i in range(n)],
        "usage": {"total_tokens": 1},
    })


def make_dashscope_service(handler, batch_size=16) -> EmbeddingService:
    client = httpx.Client(transport=httpx.MockTransport(handler), timeout=5)
    return EmbeddingService(
        make_dashscope_settings(), http_client=client, batch_size=batch_size
    )


class TestDashscopeProvider:

    def test_dense_from_api_and_sparse_from_lexical(self):
        dense, sparse = make_dashscope_service(ok_openai_response).embed(
            ["正向保理", "应收账款融资"]
        )
        assert len(dense) == 2 and len(dense[0]) == DIM
        assert sparse == lexical_sparse_texts(["正向保理", "应收账款融资"])
        assert sparse[0]                                    # 非空

    def test_url_auth_and_model(self):
        seen = []

        def handler(request):
            seen.append((
                str(request.url),
                request.headers.get("Authorization"),
                json.loads(request.content)["model"],
            ))
            return ok_openai_response(request)

        make_dashscope_service(handler).embed(["x"])
        assert seen[0] == (
            "https://dashscope.example/v1/embeddings",
            "Bearer sk-test",
            "text-embedding-v4",
        )

    def test_batch_capped_at_ten(self):
        calls = []

        def handler(request):
            calls.append(len(json.loads(request.content)["input"]))
            return ok_openai_response(request)

        make_dashscope_service(handler, batch_size=32).embed(["t"] * 12)
        assert calls == [10, 2]

    def test_default_dimension_not_sent(self):
        body = {}

        def handler(request):
            body.update(json.loads(request.content))
            return ok_openai_response(request)

        make_dashscope_service(handler).embed(["x"])
        assert body["dimensions"] == DIM                 # 显式固定集合维度

    def test_non_default_dimension_sent(self):
        body = {}

        def handler(request):
            payload = json.loads(request.content)
            body.update(payload)
            dim = payload.get("dimensions", DIM)
            return httpx.Response(200, json={
                "data": [{"index": 0, "embedding": [0.1] * dim}], "usage": {}
            })

        service = make_dashscope_service(handler)
        service.dimension = 768
        service.embed(["x"])
        assert body.get("dimensions") == 768

    def test_response_sorted_by_index(self):
        def handler(request):
            n = len(json.loads(request.content)["input"])
            data = [
                {"index": n - 1 - i, "embedding": [float(i)] + [0.0] * (DIM - 1)}
                for i in range(n)
            ]
            return httpx.Response(200, json={"data": data, "usage": {}})

        dense, _ = make_dashscope_service(handler).embed(["a", "b", "c"])
        assert [row[0] for row in dense] == [2.0, 1.0, 0.0]

    def test_dimension_mismatch(self):
        def handler(request):
            return httpx.Response(200, json={
                "data": [{"index": 0, "embedding": [0.1] * 3}], "usage": {}
            })

        with pytest.raises(EmbeddingError, match="维度"):
            make_dashscope_service(handler).embed(["x"])

    def test_missing_api_key(self):
        settings = make_dashscope_settings()
        settings.rag.embedding.api_key = None
        service = EmbeddingService(
            settings,
            http_client=httpx.Client(transport=httpx.MockTransport(ok_openai_response)),
        )
        with pytest.raises(EmbeddingError, match="api_key"):
            service.embed(["x"])

    def test_quota_error_surfaced(self):
        def handler(request):
            return httpx.Response(400, json={
                "error": {"message": "Free quota exhausted.", "code": "AllocationQuota.FreeTierOnly"}
            })

        with pytest.raises(EmbeddingError, match="quota"):
            make_dashscope_service(handler).embed(["x"])


# =====================================================
# 词法稀疏编码（dashscope 模式的 sparse 来源）
# =====================================================

class TestLexicalSparse:

    def test_deterministic_and_bounded_index(self):
        a = lexical_sparse_texts(["应收账款转让给保理商"])
        b = lexical_sparse_texts(["应收账款转让给保理商"])
        assert a == b
        assert all(0 <= k < 65536 for k in a[0])

    def test_sublinear_tf(self):
        # 重复 bigram 权重 = 1 + ln(count)，次线性增长
        vec = lexical_sparse_texts(["钱钱钱"])[0]        # bigram "钱钱" ×2
        assert len(vec) == 1
        assert list(vec.values())[0] == pytest.approx(1.0 + math.log(2))

    def test_empty_and_blank(self):
        assert lexical_sparse_texts(["", "   "]) == [{}, {}]

    def test_distinct_texts_distinct_vectors(self):
        a, b = lexical_sparse_texts(["正向保理产品", "商业保理准入"])
        assert a != b

    def test_query_and_doc_share_encoder(self):
        # 检索端与入库端使用同一函数 → 同一 bigram 命中同一维度
        doc_vec = lexical_sparse_texts(["应收账款转让通知债务人"])[0]
        query_vec = lexical_sparse_texts(["应收账款转让"])[0]
        shared = set(doc_vec) & set(query_vec)
        assert shared                                # query bigram 均在 doc 中命中
