"""VectorStore 单元测试（不依赖真实 Milvus 服务）+ 写入路径集成测试。"""
import pytest

from app.config import Settings
from app.rag.milvus_client import VectorStore, VectorStoreError


def make_settings() -> Settings:
    return Settings(_env_file=None)


class TestBuildSchema:
    """验证 Schema 与 TDD §11.2 字段设计一致。"""

    def test_field_names_and_order(self):
        schema = VectorStore.build_schema(1024)
        names = [f.name for f in schema.fields]
        assert names == [
            "id", "chunk_id", "parent_id", "chunk_type", "source",
            "doc_name", "priority", "content", "doc_id", "category", "industry", "version",
            "effective_date", "status", "topic", "chunk_index", "embedding", "sparse_vector",
        ]

    def test_primary_key_auto_id(self):
        schema = VectorStore.build_schema()
        by_name = {f.name: f for f in schema.fields}
        assert by_name["id"].is_primary is True
        assert by_name["id"].auto_id is False

    def test_dense_vector_dimension(self):
        schema = VectorStore.build_schema(1024)
        by_name = {f.name: f for f in schema.fields}
        assert by_name["embedding"].params["dim"] == 1024

    def test_dimension_from_config(self):
        # config.yaml: rag.embedding.dimension = 1024
        s = make_settings()
        assert VectorStore.build_schema(s.rag.embedding.dimension).fields[-2].params["dim"] == 1024


class TestUri:

    def test_uri_from_settings(self):
        store = VectorStore(make_settings())
        # config.yaml 中 rag.vector_database.host=localhost, port=19530
        assert store.uri == "http://127.0.0.1:19530"

    def test_uri_explicit_override(self):
        store = VectorStore(make_settings(), uri="http://8.133.229.0:19530")
        assert store.uri == "http://8.133.229.0:19530"

    def test_collection_name_from_settings(self):
        store = VectorStore(make_settings())
        assert store.collection_name == "factoring_knowledge_v2"


# =====================================================
# 写入接口（假 client，不触网）
# =====================================================

class FakeClient:
    def __init__(self):
        self.inserted: list[tuple[str, list[dict]]] = []
        self.deleted: list[str] = []
        self.flushed = 0
        self.loaded = 0
        self.queries: list[str] = []
        self.query_rows: list[dict] = []
        self.hybrid_calls: list[tuple] = []
        self.hybrid_results: list[list[dict]] = [[]]

    def insert(self, collection_name, data):
        self.inserted.append((collection_name, data))
        return {"insert_count": len(data)}

    def delete(self, collection_name, filter):
        self.deleted.append(filter)

    def flush(self, collection_name):
        self.flushed += 1

    def load_collection(self, collection_name):
        self.loaded += 1

    def query(self, collection_name, filter, output_fields):
        self.queries.append(filter)
        if output_fields == ["count(*)"]:
            return [{"count(*)": 3}]
        return list(self.query_rows)

    def hybrid_search(self, collection_name, reqs, ranker, limit, output_fields):
        self.hybrid_calls.append((collection_name, reqs, ranker, limit, output_fields))
        return self.hybrid_results


def make_row(chunk_id="c1", doc="a.md", **overrides) -> dict:
    row = {
        "chunk_id": chunk_id, "parent_id": "", "chunk_type": "child",
        "source": "product", "doc_name": doc, "priority": 2, "content": "文本内容",
        "doc_id": "KB-TEST", "category": "product", "industry": "all", "version": "v1",
        "effective_date": "2026-09-05", "status": "active", "topic": "保理", "chunk_index": 0,
        "embedding": [0.1, 0.2], "sparse_vector": {"5": 0.5},
    }
    row.update(overrides)
    return row


def make_store_with_fake_client() -> tuple[VectorStore, FakeClient]:
    fake = FakeClient()
    store = VectorStore(make_settings())
    store._client = fake
    return store, fake


class TestInsert:

    def test_insert_converts_types_and_flushes(self):
        store, fake = make_store_with_fake_client()
        assert store.insert([make_row(), make_row("c2")]) == 2
        collection, rows = fake.inserted[0]
        assert collection == "factoring_knowledge_v2" and len(rows) == 2
        assert rows[0]["embedding"] == [0.1, 0.2]       # float
        assert rows[0]["sparse_vector"] == {5: 0.5}     # str key → int
        assert fake.flushed == 1

    def test_insert_empty_list_noop(self):
        store, fake = make_store_with_fake_client()
        assert store.insert([]) == 0
        assert fake.inserted == [] and fake.flushed == 0

    def test_insert_missing_field_raises(self):
        store, fake = make_store_with_fake_client()
        row = make_row()
        del row["embedding"]
        with pytest.raises(VectorStoreError, match="缺少字段"):
            store.insert([row])
        assert fake.inserted == []                      # 校验失败不触达 client


class TestDeleteByDoc:

    def test_filter_expr(self):
        store, fake = make_store_with_fake_client()
        store.delete_by_doc("a.md")
        assert fake.deleted == ['doc_name == "a.md"']
        assert fake.flushed == 1

    def test_quotes_escaped(self):
        store, fake = make_store_with_fake_client()
        store.delete_by_doc('恶"名.md')
        assert fake.deleted == ['doc_name == "恶\\"名.md"']


class TestCount:

    def test_count_loads_then_queries(self):
        store, fake = make_store_with_fake_client()
        assert store.count() == 3
        assert fake.loaded == 1
        assert fake.queries == ['chunk_id != ""']

    def test_count_with_expr(self):
        store, fake = make_store_with_fake_client()
        store.count('source == "product"')
        assert fake.queries == ['source == "product"']


# =====================================================
# 在线检索（hybrid_search / get_parents，假 client）
# =====================================================

def make_hit(chunk_id="c1", parent_id="p1", source="product", priority=2,
             score=0.0164, content="child 文本") -> dict:
    return {
        "chunk_id": chunk_id, "parent_id": parent_id, "chunk_type": "child",
        "source": source, "doc_name": "a.md", "priority": priority,
        "content": content, "distance": score,
    }


class TestHybridSearch:

    def test_two_requests_rrf_child_filter(self):
        store, fake = make_store_with_fake_client()
        fake.hybrid_results = [[make_hit()]]
        items = store.hybrid_search([0.1] * 4, {3: 0.5}, 10)

        name, reqs, ranker, limit, output = fake.hybrid_calls[0]
        assert name == "factoring_knowledge_v2" and limit == 10
        assert [r.anns_field for r in reqs] == ["embedding", "sparse_vector"]
        assert all(r.expr == VectorStore.CHILD_FILTER for r in reqs)
        assert all(r.limit == 20 for r in reqs)  # 双路各20，RRF融合top10
        assert reqs[0].data == [[0.1] * 4] and reqs[1].data == [{3: 0.5}]
        from pymilvus import RRFRanker
        assert isinstance(ranker, RRFRanker)
        assert ranker._k == 60                       # config rrf_k（无公开属性）
        assert output == list(VectorStore.SCALAR_FIELDS)
        assert items[0]["chunk_id"] == "c1"
        assert items[0]["score"] == 0.0164           # distance → score

    def test_rrf_k_from_config_override(self):
        store, fake = make_store_with_fake_client()
        store.hybrid_search([0.1] * 4, {3: 0.5}, 5, rrf_k=100)
        _, _, ranker, limit, _ = fake.hybrid_calls[0]
        assert ranker._k == 100 and limit == 5

    def test_empty_sparse_degrades_to_dense_only(self):
        store, fake = make_store_with_fake_client()
        store.hybrid_search([0.1] * 4, {}, 10)
        _, reqs, _, _, _ = fake.hybrid_calls[0]
        assert [r.anns_field for r in reqs] == ["embedding"]

    def test_empty_result(self):
        store, fake = make_store_with_fake_client()
        assert store.hybrid_search([0.1] * 4, {3: 0.5}, 10) == []

    def test_error_wrapped(self):
        store, fake = make_store_with_fake_client()

        def boom(**kwargs):
            raise RuntimeError("connect broken")

        fake.hybrid_search = boom
        with pytest.raises(VectorStoreError, match="混合检索失败"):
            store.hybrid_search([0.1] * 4, {3: 0.5}, 10)


class TestGetParents:

    def test_filter_dedupes_and_sorts_ids(self):
        store, fake = make_store_with_fake_client()
        fake.query_rows = [{"chunk_id": "p1", "content": "parent 全文"}]
        rows = store.get_parents(["p2", "p1", "p2", ""])
        assert fake.queries == ['chunk_id in ["p1", "p2"]']
        assert rows == [{"chunk_id": "p1", "content": "parent 全文"}]
        assert fake.loaded == 1                       # 查询前先 load

    def test_empty_ids_noop(self):
        store, fake = make_store_with_fake_client()
        assert store.get_parents(["", ""]) == []
        assert fake.queries == [] and fake.loaded == 0

    def test_quotes_escaped(self):
        store, fake = make_store_with_fake_client()
        store.get_parents(['恶"名'])
        assert fake.queries == ['chunk_id in ["恶\\"名"]']


class TestInsertIntegration:
    """真实 Milvus 写入/计数/删除回环（不加载模型，用假向量）。"""

    @pytest.mark.integration
    def test_insert_count_delete_roundtrip(self):
        store = VectorStore()  # get_settings()：.env 指向远程 Milvus
        store.ensure_collection()
        from uuid import uuid4
        run_id = uuid4().hex
        doc = "__test_vs_" + run_id + ".md"
        dim = store.dimension
        rows = [
            make_row(chunk_id=f"__tc_{run_id}_{i}__", doc=doc,
                     embedding=[0.1] * dim, sparse_vector={i + 1: 0.5})
            for i in range(3)
        ]
        try:
            assert store.insert(rows) == 3
            assert store.count(f'doc_name == "{doc}"') == 3
        finally:
            store.delete_by_doc(doc)
        assert store.count(f'doc_name == "{doc}"') == 0
