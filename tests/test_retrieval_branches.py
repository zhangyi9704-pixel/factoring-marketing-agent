"""三分支与排序边界；全部使用注入组件，不依赖外部服务。"""
from copy import deepcopy
import pytest
from app.config import Settings
from app.rag.retriever import HybridRetriever
from app.rag import revision
from app.rag.embedder import lexical_sparse_texts


class Embedder:
    def embed(self, texts):
        return [[0.1]], [{1: 1.0}]


class Store:
    def __init__(self, rows, parents=()):
        self.rows, self.parents, self.calls = rows, parents, []

    def hybrid_search(self, dense, sparse, top_k, **kwargs):
        self.calls.append(kwargs["filter_expr"])
        return deepcopy(self.rows)

    def get_parents(self, ids):
        return self.parents


class Ranker:
    def __init__(self, scores=None, degraded=False):
        self.scores, self.degraded, self.seen = scores or {}, degraded, []

    def rerank_with_status(self, query, items):
        self.seen = [i.chunk_id for i in items]
        if not self.degraded:
            for item in items:
                item.rerank_score = self.scores.get(item.chunk_id, .9)
                item.score = item.rerank_score
            items = sorted([i for i in items if i.score >= .1], key=lambda i: i.score, reverse=True)[:5]
        return items, self.degraded


def row(name="a", **kwargs):
    return dict(chunk_id=name, parent_id="p"+name, chunk_type="child", source="脱敏制度",
                doc_name="示例.md", priority=1, content="可核实的依据", score=.03,
                doc_id=name, version="v1", industry="医药", status="active", topic="确权", **kwargs)


def build(rows, **kwargs):
    settings = Settings(_env_file=None)
    settings.rag.retrieval.cache_ttl = 0
    settings.rag.retrieval.fallback_min_score = None
    return HybridRetriever(settings, embedder=Embedder(), store=Store(rows, kwargs.pop("parents", ())),
                           reranker=kwargs.pop("ranker", Ranker()))


def conflicted(name, version):
    value = row(name)
    value.update(status="conflicted", doc_id="same", version=version)
    return value


@pytest.mark.parametrize("rows,mode", [([row()], "rag"), ([], "missing"),
    ([conflicted("a", "v1"), conflicted("b", "v2")], "conflict")])
def test_three_modes(rows, mode):
    result = build(rows).retrieve("付款人确权")
    data = result.to_dict()
    assert data["mode"] == mode and "items" in data and "escalation_id" in data
    if mode != "rag":
        assert data["escalation_id"].startswith("esc_") and data["message"]
    if mode == "conflict":
        assert {i["version"] for i in data["items"]} == {"v1", "v2"}
        assert all(i["source"] and i["summary"] for i in data["items"])


def test_conflicted_marker_cannot_be_cleared_by_top_k_or_threshold():
    rows = [conflicted("a", "v1"), conflicted("b", "v2")]
    single = build(rows).retrieve("确权", top_k=1)
    assert single.mode == "conflict" and single.reason == "conflict_evidence_incomplete"
    assert single.action == "human_review" and single.escalation_id
    assert build(rows, ranker=Ranker({"b": .01})).retrieve("确权").mode == "conflict"
    rows[1]["version"] = "v1"
    assert build(rows).retrieve("确权").mode == "conflict"  # Catalog marker still requires human resolution.
    rows[1].update(version="v2", topic="不同主题")
    assert build(rows).retrieve("确权").mode == "conflict"


def test_out_of_scope_is_missing_without_external_call():
    retriever = build([row()])
    result = retriever.retrieve("帮我预测明天天气")
    assert result.mode == "missing" and result.reason == "out_of_scope"
    assert not retriever.store.calls


def test_industry_all_archived_and_no_default_archived():
    rows = [row("medical"), row("global"), row("infra"), row("old")]
    rows[1]["industry"] = "all"
    rows[2]["industry"] = "基建"
    rows[3]["status"] = "archived"
    retriever = build(rows)
    assert {i.doc_id for i in retriever.retrieve("确权", industry="医药").items} == {"medical", "global"}
    assert 'industry in ["all", "医药"]' in retriever.store.calls[-1]
    assert "old" not in {i.doc_id for i in retriever.retrieve("确权").items}
    assert 'status in ["active", "conflicted"]' in retriever.store.calls[-1]


def test_rerank_before_dedup_parent_version_matches():
    a, b = row("a"), row("b")
    a.update(parent_id="p", doc_id="same", version="v1")
    b.update(parent_id="p", doc_id="same", version="v2")
    ranker = Ranker()
    retriever = build([a, b], ranker=ranker, parents=[dict(chunk_id="p", doc_id="same", version="v1", content="版本1全文")])
    result = retriever.retrieve("确权")
    assert ranker.seen == ["a", "b"]
    assert len(result.items) == 2
    assert result.items[0].parent_content == "版本1全文" and result.items[1].parent_content is None
    assert result.items[0].retrieval_score == .03 and result.items[0].rerank_score == .9


def test_degraded_threshold_is_separate_and_uncalibrated_fails_closed():
    retriever = build([row()], ranker=Ranker(degraded=True))
    assert retriever.retrieve("确权").reason == "rerank_unavailable"
    retriever.settings.rag.retrieval.fallback_min_score = .025
    result = retriever.retrieve("确权")
    assert result.mode == "rag" and result.rerank_degraded
    assert result.items[0].rerank_score is None
    retriever.settings.rag.retrieval.fallback_min_score = .031
    assert retriever.retrieve("确权").mode == "missing"


def test_cache_invalidated_across_publish_and_instances(tmp_path, monkeypatch):
    monkeypatch.setattr(revision, "PATH", tmp_path / "revision")
    retriever = build([row()])
    retriever.settings.rag.retrieval.cache_ttl = 30
    assert retriever.retrieve("确权").mode == "rag"
    retriever.store.rows = []
    assert retriever.retrieve("确权").mode == "rag"  # 同一提交代号缓存
    revision.invalidate()
    assert retriever.retrieve("确权").mode == "missing"


def test_sparse_bigram_fixed_id_known_vector():
    import zlib
    vector = lexical_sparse_texts(["保理保理"])[0]
    assert zlib.crc32("保理".encode()) % 65536 in vector
    assert lexical_sparse_texts(["保理保理"])[0] == vector


def test_sparse_stable_across_python_processes():
    import subprocess
    import sys
    import os
    script = "import json; from app.rag.embedder import lexical_sparse_texts; print(json.dumps(lexical_sparse_texts(['保理医药'])[0], sort_keys=True))"
    first = subprocess.check_output([sys.executable, "-c", script], env=dict(os.environ, PYTHONHASHSEED="1"))
    second = subprocess.check_output([sys.executable, "-c", script], env=dict(os.environ, PYTHONHASHSEED="2"))
    assert first == second
