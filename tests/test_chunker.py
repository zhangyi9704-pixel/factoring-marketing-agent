"""ParentChildChunker 单元测试（纯函数，不触网）。"""
import pytest

from app.config import Settings
from app.rag.chunker import ParentChildChunker, _SENTENCE_SPLIT

CHUNK_FIELDS = {"chunk_id", "parent_id", "chunk_type", "source",
                "doc_name", "priority", "content"}
SENTENCE = "保理业务可以把应收账款提前变现。"  # 15 字符，含句末边界


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_SPLIT.split(text.replace("\r\n", "\n")) if s.strip()]


class TestBasic:

    def test_short_text_one_parent_one_child(self):
        chunks = ParentChildChunker().split(
            "短文本。", doc_name="a.md", source="product", priority=2
        )
        parents = [c for c in chunks if c["chunk_type"] == "parent"]
        children = [c for c in chunks if c["chunk_type"] == "child"]
        assert len(parents) == 1 and len(children) == 1
        assert children[0]["parent_id"] == parents[0]["chunk_id"]
        assert parents[0]["parent_id"] == ""
        assert set(chunks[0]) == CHUNK_FIELDS
        assert chunks[0]["doc_name"] == "a.md"
        assert chunks[0]["source"] == "product"
        assert chunks[0]["priority"] == 2

    def test_size_limits_respected(self):
        text = SENTENCE * 200  # 3000 字符
        chunks = ParentChildChunker(1200, 300, 50).split(
            text, doc_name="b.md", source="industry", priority=3
        )
        assert len(chunks) > 2
        for c in chunks:
            limit = 1200 if c["chunk_type"] == "parent" else 300
            assert len(c["content"]) <= limit, c["chunk_type"]

    def test_children_link_to_existing_parent(self):
        chunks = ParentChildChunker(200, 60, 10).split(
            SENTENCE * 50, doc_name="c.md", source="compliance", priority=1
        )
        parent_ids = {c["chunk_id"] for c in chunks if c["chunk_type"] == "parent"}
        children = [c for c in chunks if c["chunk_type"] == "child"]
        assert len(parent_ids) > 1 and children
        for c in children:
            assert c["parent_id"] in parent_ids

    def test_children_cover_parent_sentences(self):
        chunks = ParentChildChunker(200, 60, 10).split(
            SENTENCE * 30, doc_name="d.md", source="product", priority=2
        )
        parents = [c for c in chunks if c["chunk_type"] == "parent"]
        for p in parents:
            own = "".join(c["content"] for c in chunks if c["parent_id"] == p["chunk_id"])
            for s in split_sentences(p["content"]):
                assert s in own, f"句子未被 child 覆盖: {s!r}"

    def test_overlap_tail_carried_to_next_child(self):
        chunks = ParentChildChunker(400, 100, 10).split(
            SENTENCE * 40, doc_name="e.md", source="industry", priority=3
        )
        # chunks 列表本身即文档顺序；只校验同一 parent 内的相邻 child
        first_parent = next(c for c in chunks if c["chunk_type"] == "parent")["chunk_id"]
        children = [c for c in chunks if c["parent_id"] == first_parent]
        assert len(children) > 2
        for prev, nxt in zip(children, children[1:]):
            assert nxt["content"].startswith(prev["content"][-10:])

    def test_deterministic_chunk_ids(self):
        text = SENTENCE * 60
        kwargs = dict(doc_name="f.md", source="product", priority=2)
        ids1 = [c["chunk_id"] for c in ParentChildChunker(300, 80, 20).split(text, **kwargs)]
        ids2 = [c["chunk_id"] for c in ParentChildChunker(300, 80, 20).split(text, **kwargs)]
        assert ids1 == ids2 and len(set(ids1)) == len(ids1)  # 确定且唯一

    def test_different_doc_name_different_ids(self):
        kwargs = dict(source="product", priority=2)
        a = ParentChildChunker().split("同一段文本。", doc_name="a.md", **kwargs)
        b = ParentChildChunker().split("同一段文本。", doc_name="b.md", **kwargs)
        assert a[0]["chunk_id"] != b[0]["chunk_id"]

    def test_long_unpunctuated_text_hard_split(self):
        text = "无标点连续文字内容" * 400  # 3600 字符，无句末边界
        chunks = ParentChildChunker(200, 80, 10).split(
            text, doc_name="g.md", source="product", priority=2
        )
        parents = [c for c in chunks if c["chunk_type"] == "parent"]
        assert len(parents) >= 18
        for c in chunks:
            limit = 200 if c["chunk_type"] == "parent" else 80
            assert len(c["content"]) <= limit

    def test_invalid_parameters(self):
        with pytest.raises(ValueError):
            ParentChildChunker(overlap=-1)
        with pytest.raises(ValueError):
            ParentChildChunker(child_size=50, overlap=50)   # overlap 必须 < child_size
        with pytest.raises(ValueError):
            ParentChildChunker(parent_size=100, child_size=200)

    def test_from_settings_reads_config(self):
        chunker = ParentChildChunker.from_settings(Settings(_env_file=None))
        assert (chunker.parent_size, chunker.child_size, chunker.overlap) == (1200, 300, 50)
