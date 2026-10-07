from app.agents.coach.briefing import _sources
from app.config import Settings
from app.rag.retriever import HybridRetriever
from app.rag.milvus_client import VectorStore
from test_ingest import Embedder, HEADER
from app.rag.loader import parse_metadata


def test_retrieval_fields_industry_filter_and_briefing_versions():
    _, meta = parse_metadata(HEADER.format(version="v1", status="conflicted") + "正文", "样例.md")
    hits = [dict(meta, chunk_id=f"c{i}", parent_id=f"p{i}", chunk_type="child",
                 doc_name="same.md", priority=2, content="正文", score=0.8, version=f"v{i}")
            for i in (1, 2)]

    class Store:
        def hybrid_search(self, dense, sparse, top_k, *, filter_expr):
            assert 'industry in ["all", "基建"]' in filter_expr
            assert 'status in ["active", "conflicted"]' in filter_expr
            return hits

        def get_parents(self, ids):
            return [dict(h, chunk_id=h["parent_id"], chunk_type="parent") for h in hits]

    class Reranker:
        def rerank(self, query, items):
            return items

    retriever = HybridRetriever(Settings(_env_file=None), embedder=Embedder(), store=Store(), reranker=Reranker())
    result = retriever.retrieve("保理", industry="基建").to_dict()
    assert len(result["items"]) == 2
    for hit in result["items"]:
        for key, value in meta.items():
            if key != "version":
                assert hit[key] == value
        assert hit["parent_content"] == "正文"
    sources = _sources(result["items"] + result["items"])
    assert {s["version"] for s in sources} == {"v1", "v2"}
    assert all(s["effective_date"] == "2026-09-05" for s in sources)


def test_milvus_nested_entity_metadata_echo():
    store = VectorStore(Settings(_env_file=None))
    _, meta = parse_metadata(HEADER.format(version="v1", status="active") + "正文", "样例.md")

    class Client:
        def hybrid_search(self, **kwargs):
            assert all(field in kwargs["output_fields"] for field in meta)
            return [[{"entity": dict(meta, chunk_id="c1", content="正文"), "distance": 0.8}]]

    store._client = Client()
    result = store.hybrid_search([0.1] * 4, {1: 0.5}, 5)
    assert all(result[0][key] == value for key, value in meta.items())
