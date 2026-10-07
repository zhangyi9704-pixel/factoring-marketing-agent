"""版本化知识入库：真实 SQLAlchemy 事务 + 可注入 Milvus 故障。"""
from contextlib import contextmanager
from copy import deepcopy

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.knowledge_migration import migrate_knowledge_metadata
from app.models import KnowledgeDoc
from app.rag.ingest import KnowledgeIngestor
from app.rag.milvus_client import VectorStore

HEADER = """---
doc_id: KB-TEST-001
category: product
industry: 基建
source: SRC-01《验收来源》
version: {version}
effective_date: 2026-09-05
status: {status}
topic: 保理基础
---
"""

class Embedder:
    def embed(self, texts):
        return [[0.1] * 4 for t in texts], [{1: 0.5} for t in texts]

class MemoryStore:
    def __init__(self):
        self.rows = {}
        self.fail = None
        self.validator = VectorStore(Settings(_env_file=None))
        self.validator.dimension = 4

    def validate_chunks(self, rows):
        self.validator.validate_chunks(rows)

    def snapshot_doc(self, doc_id):
        return deepcopy([r for r in self.rows.values() if r["doc_id"] == doc_id])

    def upsert(self, rows):
        for row in rows:
            self.rows[row["id"]] = deepcopy(row)
            if self.fail == "partial":
                self.fail = None
                raise RuntimeError("partial write")
        return len(rows)

    def cleanup_version(self, doc_id, version, keep_ids):
        self.rows = {k: r for k, r in self.rows.items() if not (
            r["doc_id"] == doc_id and r["version"] == version and k not in keep_ids)}
        if self.fail == "cleanup":
            self.fail = None
            raise RuntimeError("cleanup failed")

    def restore_doc(self, doc_id, version, rows):
        self.upsert(rows)
        self.cleanup_version(doc_id, version, [r["id"] for r in rows if r["version"] == version])

@pytest.fixture
def kb(tmp_path):
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("""CREATE TABLE t_knowledge_doc (
            id INTEGER PRIMARY KEY AUTOINCREMENT, source VARCHAR(16) NOT NULL,
            doc_name VARCHAR(256) NOT NULL, priority INTEGER NOT NULL,
            chunk_cnt INTEGER, status VARCHAR(16) NOT NULL, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"""))
    migrate_knowledge_metadata(engine)
    @contextmanager
    def sf():
        with Session(engine) as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
    store = MemoryStore()
    ingestor = KnowledgeIngestor(Settings(_env_file=None), embedder=Embedder(), store=store, session_factory=sf)
    path = tmp_path / "条目.md"
    return ingestor, store, sf, path, engine


def ingest(kb, *, version="v1", status="active", body="保理业务。" * 400):
    ingestor, _, _, path, _ = kb
    path.write_text(HEADER.format(version=version, status=status) + body, encoding="utf-8")
    return ingestor.ingest_file(path, source="product", priority=2)


def test_fields_persisted_parent_body_and_backtrace(kb):
    result = ingest(kb)
    assert result.status == "indexed", result.error
    _, store, sf, _, _ = kb
    with sf() as session:
        row = session.scalars(select(KnowledgeDoc)).one()
        assert (row.doc_id, row.version, row.effective_date, row.status, row.industry, row.category, row.topic, row.source) == (
            "KB-TEST-001", "v1", "2026-09-05", "active", "基建", "product", "保理基础", "SRC-01《验收来源》")
    parents = [r for r in store.rows.values() if r["chunk_type"] == "parent"]
    assert "".join(p["content"] for p in parents) == "保理业务。" * 400
    for child in [r for r in store.rows.values() if r["chunk_type"] == "child"]:
        assert store.rows[child["parent_id"]]["version"] == child["version"]


def test_same_version_no_duplicates_and_shortening_cleans_stale(kb):
    assert ingest(kb).status == "indexed"
    original_ids = set(kb[1].rows)
    assert ingest(kb).status == "indexed"
    assert set(kb[1].rows) == original_ids
    assert ingest(kb, body="缩短后的保理正文。").status == "indexed"
    assert len(kb[1].rows) == 2
    with kb[2]() as session:
        assert len(list(session.scalars(select(KnowledgeDoc)))) == 1


def test_different_versions_preserve_body_archive_old_active(kb):
    assert ingest(kb).status == "indexed"
    before = deepcopy(kb[1].rows)
    assert ingest(kb, version="v2", body="新版。").status == "indexed"
    for key, row in before.items():
        assert kb[1].rows[key] == dict(row, status="archived")
    with kb[2]() as session:
        assert {r.version: r.status for r in session.scalars(select(KnowledgeDoc))} == {"v1": "archived", "v2": "active"}


def test_conflicted_versions_coexist(kb):
    assert ingest(kb, status="conflicted").status == "indexed"
    assert ingest(kb, version="v2", status="conflicted").status == "indexed"
    assert {r["version"] for r in kb[1].rows.values()} == {"v1", "v2"}
    assert all(r["status"] == "conflicted" for r in kb[1].rows.values())


@pytest.mark.parametrize("failure", ["partial", "cleanup", "embedding", "metadata", "commit"])
@pytest.mark.parametrize("version", ["v1", "v2"])
def test_failure_retry_preserves_previous_data(kb, failure, version):
    assert ingest(kb).status == "indexed"
    before = deepcopy(kb[1].rows)
    ingestor = kb[0]
    original_sf = ingestor.session_factory
    original_embedder = ingestor.embedder
    if failure == "embedding":
        class BadEmbedder:
            def embed(self, texts):
                return [[0.1] * 4], [{1: 0.5}]
        ingestor.embedder = BadEmbedder()
    elif failure in {"metadata", "commit"}:
        @contextmanager
        def failing_sf():
            with original_sf() as session:
                def fail():
                    raise RuntimeError("mysql failure")
                if failure == "metadata":
                    session.flush = fail
                else:
                    session.commit = fail
                yield session
        ingestor.session_factory = failing_sf
    else:
        kb[1].fail = failure
    assert ingest(kb, version=version, body="修改内容。" * 400).status == "failed"
    assert kb[1].rows == before
    with original_sf() as session:
        row = session.scalars(select(KnowledgeDoc)).one()
        assert row.status == "active" and row.version == "v1"
    ingestor.session_factory = original_sf
    ingestor.embedder = original_embedder
    assert ingest(kb, version=version, body="修改内容。" * 400).status == "indexed"


def test_migration_idempotent_and_keeps_old_rows(kb):
    assert migrate_knowledge_metadata(kb[4]) == []
    assert ingest(kb).status == "indexed"
    assert migrate_knowledge_metadata(kb[4]) == []
    with kb[2]() as session:
        assert session.scalars(select(KnowledgeDoc)).one().doc_id == "KB-TEST-001"


def test_invalid_metadata_no_writes(kb):
    kb[3].write_text("没有元数据", encoding="utf-8")
    result = kb[0].ingest_file(kb[3], source="product", priority=2)
    assert result.status == "failed" and kb[3].name in result.error
    assert kb[1].rows == {}


@pytest.mark.integration
def test_ingest_end_to_end_real_stores(tmp_path):
    import uuid
    from sqlalchemy import delete
    doc_id = "KB-VERIFY-" + uuid.uuid4().hex
    path = tmp_path / "__test_kb__.md"
    path.write_text(HEADER.format(version="v1", status="active").replace("KB-TEST-001", doc_id) + "保理验证正文。" * 150, encoding="utf-8")
    ingestor = KnowledgeIngestor()
    ingestor.store.ensure_collection()
    try:
        result = ingestor.ingest_file(path, source="product", priority=2)
        assert result.status == "indexed", "真实入库失败"
        rows = ingestor.store.snapshot_doc(doc_id)
        assert len(rows) == result.chunk_cnt
        assert ingestor.ingest_file(path, source="product", priority=2).status == "indexed"
        assert len(ingestor.store.snapshot_doc(doc_id)) == len(rows)
        with ingestor.session_factory() as session:
            row = session.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id == doc_id)).one()
            assert row.version == "v1" and row.industry == "基建" and row.status == "active"
    finally:
        ingestor.store.cleanup_version(doc_id, "v1", [])
        with ingestor.session_factory() as session:
            session.execute(delete(KnowledgeDoc).where(KnowledgeDoc.doc_id == doc_id))


@pytest.mark.parametrize("failure", ["dimension", "nan", "length"])
def test_validate_every_slice_before_mutating(kb, failure):
    assert ingest(kb).status == "indexed"
    before = deepcopy(kb[1].rows)
    class CorruptEmbedder:
        def embed(self, texts):
            dense, sparse = Embedder().embed(texts)
            if failure == "dimension":
                dense[-1] = [0.1]
            elif failure == "nan":
                dense[-1][0] = float("nan")
            return dense, sparse
    kb[0].embedder = CorruptEmbedder()
    if failure == "length":
        kb[3].write_text(HEADER.format(version="v1", status="active").replace("SRC-01《验收来源》", "来源" * 300) + "正文", encoding="utf-8")
        result = kb[0].ingest_file(kb[3], source="product", priority=2)
    else:
        result = ingest(kb)
    assert result.status == "failed"
    assert kb[1].rows == before
