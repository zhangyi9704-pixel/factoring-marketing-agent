"""在线检索真实链路集成测试（远程 Milvus + dashscope embedding）。

依赖已入库的M2产品知识KB-PD-003费率条目（旧手册兼容）。
source是来源说明，category才是知识类别。

注：知识库非空时 ANN 检索总会返回最近邻（RRF 融合分无相关性语义，
精排占位亦无阈值），空结果（human_review）分支无法用真实 query 触发，
该分支由 tests/test_retriever.py 单测覆盖；待真实 reranker 接入后
补相关性阈值 E2E。
"""
import pytest

from app.rag.retriever import HybridRetriever


@pytest.mark.integration
class TestRetrieveIntegration:

    def test_retrieve_hits_ingested_product_doc(self):
        retriever = HybridRetriever()      # get_settings()：.env → dashscope + 远程 Milvus
        result = retriever.retrieve("保理费率由哪些部分构成，哪些因素影响费率？")

        assert result.missing is False
        assert result.items, "知识库应命中产品费率条目"
        # 四来源知识库：按 conflict 优先级排序（compliance > product > industry > process），
        # 存在更高优先级来源时低优先级条目标注仅供参考
        ranks = {"compliance": 1, "product": 2, "industry": 3, "process": 4}
        order = [ranks[i.category or i.source] for i in result.items]
        assert order == sorted(order)
        product_hits = [i for i in result.items if (i.category or i.source) == "product" and (i.doc_id == "KB-PD-003" or i.doc_name == "正向保理产品手册.md")]
        assert product_hits, "应命中产品费率内容（旧手册或当前KB-PD-003）"
        item = product_hits[0]
        assert (item.category or item.source) == "product"
        assert item.chunk_type == "child"
        assert item.parent_content                 # child 命中回溯 parent 全文
        assert "费率" in item.parent_content
        if item.doc_id:
            assert item.source and item.version
        assert item.score > 0
        if len({i.category or i.source for i in result.items}) > 1:
            assert all(i.reference_only for i in result.items
                       if ranks[i.category or i.source] > min(order))

    def test_queries_collapse_children_per_parent(self):
        # 每文档仅 1 个 parent（3-4 child）；命中去重后每文档至多一条，
        # 且 parent 全文长度 ≥ 命中 child 片段
        retriever = HybridRetriever()
        # Dedup requires a known ingested positive query. The old vague query
        # is below the calibrated threshold; that limitation is separately reported.
        result = retriever.retrieve("保理费率由哪些部分构成，哪些因素影响费率？")
        assert result.missing is False
        assert result.items
        for item in result.items:
            assert item.parent_content is None or \
                len(item.parent_content) >= len(item.content)
        docs = [i.doc_name for i in result.items]
        assert len(docs) == len(set(docs))         # 同文档多 child 已去重
