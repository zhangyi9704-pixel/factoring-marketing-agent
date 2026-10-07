"""HybridRetriever 编排单元测试（假 embedder / store / reranker，不触网）。"""
import logging

from app.config import Settings
from app.rag.retriever import HybridRetriever, RetrievalResult
from app.rag.reranker import Reranker


class FakeEmbedder:
    def __init__(self):
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[0.1] * 8], [{5: 0.7}]


class FakeStore:
    def __init__(self, hits, parents=None):
        self.hits = hits
        self.parents = parents or []
        self.search_calls: list[tuple] = []
        self.parent_ids: list[str] = []

    def hybrid_search(self, dense, sparse, top_k, **kwargs):
        self.search_calls.append((dense, sparse, top_k))
        return self.hits

    def get_parents(self, parent_ids):
        self.parent_ids = list(parent_ids)
        return self.parents


def hit(chunk_id, parent_id, source="product", priority=2, score=0.02,
        content=None) -> dict:
    return {
        "chunk_id": chunk_id, "parent_id": parent_id, "chunk_type": "child",
        "source": source, "doc_name": f"{source}.md", "priority": priority,
        "content": content or f"{chunk_id} 内容", "score": score,
    }


def parent_row(chunk_id, content) -> dict:
    return {
        "chunk_id": chunk_id, "parent_id": "", "chunk_type": "parent",
        "source": "product", "doc_name": "product.md", "priority": 2,
        "content": content,
    }


def make_retriever(embedder, store, settings=None, reranker=None):
    settings = settings or Settings(_env_file=None)
    settings.rag.retrieval.fallback_min_score = 0.0
    if reranker is None:
        reranker = Reranker(settings)
        reranker.provider = "passthrough"   # 单测不触网（dashscope 留给 test_reranker）
    return HybridRetriever(
        settings, embedder=embedder, store=store,
        reranker=reranker,
    )


class TestRetrieve:

    def test_query_embedded_then_searched(self):
        embedder, store = FakeEmbedder(), FakeStore([hit("c1", "p1")])
        make_retriever(embedder, store).retrieve("正向保理的融资比例")

        assert embedder.calls == [["正向保理的融资比例"]]
        dense, sparse, top_k = store.search_calls[0]
        assert dense == [0.1] * 8 and sparse == {5: 0.7}
        assert top_k == 10                       # config rag.retrieval.top_k

    def test_same_parent_children_deduped_best_score_kept(self):
        hits = [
            hit("c1", "p1", score=0.02),          # 同 parent 低分在前
            hit("c2", "p1", score=0.03),          # 同 parent 高分
            hit("c3", "p2", score=0.01),
        ]
        store = FakeStore(hits, parents=[parent_row("p1", "P1 全文"),
                                         parent_row("p2", "P2 全文")])
        result = make_retriever(FakeEmbedder(), store).retrieve("q")

        assert sorted(store.parent_ids) == ["p1", "p2"]
        assert len(result.items) == 2
        p1_item = next(i for i in result.items if i.parent_id == "p1")
        assert p1_item.content == "c2 内容"        # 保留分数最高的 child
        assert p1_item.parent_content == "P1 全文"  # 回溯 parent 全文
        assert [i.score for i in result.items] == [0.03, 0.01]  # 分数降序

    def test_parent_missing_tolerated(self):
        store = FakeStore([hit("c1", "p1")], parents=[])   # 回溯无结果
        result = make_retriever(FakeEmbedder(), store).retrieve("q")
        assert result.items[0].parent_content is None

    def test_rerank_truncates_to_top_n(self):
        hits = [hit(f"c{i}", f"p{i}", score=0.02 - i * 0.001) for i in range(7)]
        result = make_retriever(FakeEmbedder(), FakeStore(hits)).retrieve("q")
        assert len(result.items) == 5             # config rerank.top_n = 5
        assert [i.chunk_id for i in result.items] == ["c0", "c1", "c2", "c3", "c4"]

    def test_priority_sort_and_reference_only_marking(self):
        hits = [
            hit("c1", "p1", source="product", priority=2, score=0.05),
            hit("c2", "p2", source="industry", priority=3, score=0.04),
            hit("c3", "p3", source="compliance", priority=1, score=0.03),
        ]
        result = make_retriever(FakeEmbedder(), FakeStore(hits)).retrieve("q")

        # conflict.order: compliance(1) > product(2) > industry(3)
        assert [i.source for i in result.items] == ["compliance", "product", "industry"]
        flags = {i.source: i.reference_only for i in result.items}
        assert flags == {"compliance": False, "product": True, "industry": True}
        assert result.items[1].to_dict()["note"] == "仅供参考"
        assert "note" not in result.items[0].to_dict()

    def test_single_source_no_reference_mark(self):
        hits = [hit("c1", "p1", source="industry", priority=3, score=0.05),
                hit("c2", "p2", source="industry", priority=3, score=0.04)]
        result = make_retriever(FakeEmbedder(), FakeStore(hits)).retrieve("q")
        assert all(not i.reference_only for i in result.items)

    def test_unknown_source_sorted_last_and_marked(self):
        hits = [hit("c1", "p1", source="industry", score=0.05),
                hit("c2", "p2", source="other", score=0.04)]
        result = make_retriever(FakeEmbedder(), FakeStore(hits)).retrieve("q")
        assert [i.source for i in result.items] == ["industry", "other"]
        assert result.items[1].reference_only is True

    def test_missing_result_human_review_branch(self, caplog):
        settings = Settings(_env_file=None)
        store = FakeStore([])
        with caplog.at_level(logging.WARNING, logger="app.rag.retriever"):
            result = make_retriever(FakeEmbedder(), store, settings).retrieve("冷门问题")

        assert isinstance(result, RetrievalResult)
        assert result.missing is True and result.items == []
        assert result.action == settings.knowledge.missing.action == "human_review"
        assert result.message == settings.knowledge.missing.message
        assert "冷门问题" not in caplog.text            # 待补充知识条目留痕
        data = result.to_dict()
        assert data["missing"] is True and data["items"] == []
        assert data["action"] == "human_review"

    def test_rerank_all_filtered_treated_as_missing(self, caplog):
        # 精排全量低于 min_score → 与检索空结果同走 knowledge.missing 分支（§9.1）
        class AllFilteredReranker:
            def rerank(self, query, items, top_n=None):
                return []

        settings = Settings(_env_file=None)
        store = FakeStore([hit("c1", "p1")])
        with caplog.at_level(logging.WARNING, logger="app.rag.retriever"):
            result = make_retriever(
                FakeEmbedder(), store, settings, reranker=AllFilteredReranker(),
            ).retrieve("边缘问题")

        assert result.missing is True and result.items == []
        assert result.action == "human_review"
        assert "未命中" in caplog.text

    def test_top_k_override(self):
        store = FakeStore([hit("c1", "p1")])
        make_retriever(FakeEmbedder(), store).retrieve("q", top_k=3)
        assert store.search_calls[0][2] == 3
