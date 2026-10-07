"""Cloud imports/requests must work without optional local model packages."""
import builtins
import importlib
import sys

import httpx
import pytest

from app.config import Settings
from app.rag.embedder import EmbeddingError, EmbeddingService
from app.rag.reranker import Reranker


@pytest.fixture
def no_local_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "FlagEmbedding"}:
            raise ImportError("Optional local package deliberately unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_dashscope_embedding_and_rerank_without_local_packages(no_local_imports):
    settings = Settings(_env_file=None)
    settings.rag.embedding.provider = "dashscope"
    settings.rag.embedding.api_key = "unit-fake-key"
    settings.rag.retrieval.rerank.provider = "dashscope"
    settings.rag.retrieval.rerank.api_key = "unit-fake-key"

    def response(request):
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.] * 1024}]})
        return httpx.Response(200, json={"output": {"results": [{"index": 0, "relevance_score": .9}]}})

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        dense, sparse = EmbeddingService(settings, http_client=client).embed(["保理"])
        assert len(dense[0]) == 1024 and sparse[0]
        assert Reranker(settings, http_client=client).rerank("保理", [{"content": "保理产品"}])
    assert "torch" not in sys.modules and "FlagEmbedding" not in sys.modules


def test_explicit_local_mode_reports_missing_optional_dependency(no_local_imports):
    settings = Settings(_env_file=None)
    settings.rag.embedding.provider = "local"
    with pytest.raises(EmbeddingError, match="optional FlagEmbedding/torch"):
        EmbeddingService(settings).embed(["保理"])


def test_legacy_server_import_does_not_load_model(no_local_imports):
    server = importlib.import_module("scripts.embed_server")
    assert server.MODEL is None
    assert "torch" not in sys.modules and "FlagEmbedding" not in sys.modules
