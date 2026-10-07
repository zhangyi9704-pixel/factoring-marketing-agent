"""知识条目解析、版本化入库与失败补偿（技术文档 §7.1）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ContextManager, Optional

import hashlib
import json
import uuid
from sqlalchemy import select, text as sql_text
from sqlalchemy.orm import Session

from app.config import PROJECT_ROOT, Settings, get_settings
from app.db.mysql import get_mysql_database
from app.models import KnowledgeDoc, KnowledgeIngestionOperation
from app.rag.integrity import document_hash, scalar_hash, vector_hash
from app.rag.chunker import ParentChildChunker
from app.rag.embedder import EmbeddingService
from app.rag.loader import KnowledgeLoader
from app.rag.milvus_client import VectorStore

STATUS_INDEXED = "indexed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


@dataclass
class DocResult:
    """单文档入库结果。"""

    doc_name: str
    source: str
    status: str
    chunk_cnt: int = 0
    error: Optional[str] = None


@dataclass
class IngestSummary:
    """一次入库任务的汇总。"""

    results: list[DocResult] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(r.status == status for r in self.results)

    def __str__(self) -> str:  # 便于脚本直接打印
        return (
            f"indexed={self.count(STATUS_INDEXED)} "
            f"failed={self.count(STATUS_FAILED)} "
            f"skipped={self.count(STATUS_SKIPPED)}"
        )


class KnowledgeIngestor:
    """入库流水线编排器。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        loader: Optional[KnowledgeLoader] = None,
        chunker: Optional[ParentChildChunker] = None,
        embedder: Optional[Any] = None,
        store: Optional[VectorStore] = None,
        session_factory: Optional[Callable[[], ContextManager]] = None,
        project_root: Optional[Path] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.loader = loader or KnowledgeLoader()
        self.chunker = chunker or ParentChildChunker.from_settings(self.settings)
        self.embedder = embedder or EmbeddingService(self.settings)
        self.store = store or VectorStore(self.settings)
        self.session_factory = session_factory or get_mysql_database().session_scope
        self.project_root = Path(project_root) if project_root else PROJECT_ROOT

    # -------------------------------------------------
    # 入库入口
    # -------------------------------------------------

    def ingest_all(self) -> IngestSummary:
        """按 config knowledge.sources 全量入库。"""
        summary = IngestSummary()
        for source in sorted(self.settings.knowledge.sources):
            summary.results.extend(self.ingest_source(source))
        return summary

    def ingest_source(self, source: str) -> list[DocResult]:
        """入库单个来源目录（product / industry / compliance）。"""
        try:
            source_cfg = self.settings.knowledge.sources[source]
        except KeyError:
            return [DocResult(doc_name=f"<source:{source}>", source=source,
                              status=STATUS_SKIPPED, error="config 未定义该来源")]

        directory = (self.project_root / source_cfg.path).resolve()
        if not directory.is_dir():
            return [DocResult(doc_name=f"<dir:{source}>", source=source,
                              status=STATUS_SKIPPED, error=f"目录不存在: {directory}")]

        results: list[DocResult] = []
        files = sorted(
            p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in self.loader.SUPPORTED_SUFFIXES
        )
        for path in files:
            results.append(
                self.ingest_file(path, source=source, priority=source_cfg.priority)
            )
        return results

    def ingest_file(self, path: Path, *, source: str, priority: int, staged=False, already_locked=False) -> DocResult:
        """先完整校验，版本 upsert 后清理；失败补偿，不改旧有效版元数据。"""
        path = Path(path)
        snapshot = None
        metadata = None
        mutated = False
        locked_connection = None
        lock_name = None
        operation_id = uuid.uuid4().hex
        commit_attempted = False
        commit_engine = None
        try:
            body, metadata = self.loader.load_document(path)
            if staged:
                metadata["status"] = "staged"
            if metadata["category"] != source:
                raise ValueError(f"{path.name}: category 与入库目录不一致")
            chunks = self.chunker.split(body, doc_name=path.name, source=metadata["source"], priority=priority)
            # 用全局切片序号（包括父块）生成同版本稳定主键，改文件名不改身份。
            ids = {c["chunk_id"]: hashlib.md5(json.dumps(
                [metadata["doc_id"], metadata["version"], i], ensure_ascii=False
            ).encode()).hexdigest() for i, c in enumerate(chunks)}
            for i, ch in enumerate(chunks):
                ch.update(metadata)
                ch["parent_id"] = ids.get(ch["parent_id"], "")
                ch["chunk_id"] = ch["id"] = ids[ch["chunk_id"]]
                ch["chunk_index"] = i
            dense, sparse = self.embedder.embed([c["content"] for c in chunks])
            if len(dense) != len(chunks) or len(sparse) != len(chunks):
                raise ValueError("embedding 行数与全部切片不一致")
            for ch, d, sp in zip(chunks, dense, sparse):
                ch.update(embedding=d, sparse_vector=sp)
            self.store.validate_chunks(chunks)  # 所有字段/长度/向量/父子引用，写入前完成
            # 一个 doc_id 的发布跨进程串行；独立连接持锁直到事务提交或失败补偿结束。
            with self.session_factory() as session:
                commit_engine = session.get_bind()
                connection = session.get_bind().connect()
                if connection.dialect.name == "mysql" and not already_locked:
                    lock_name = "kb:" + hashlib.sha256(metadata["doc_id"].encode()).hexdigest()[:60]
                    locked_connection = connection
                    if connection.execute(sql_text("SELECT GET_LOCK(:name, 10)"), {"name": lock_name}).scalar() != 1:
                        raise RuntimeError("知识发布锁等待超时")
                else:
                    connection.close()
                snapshot = self.store.snapshot_doc(metadata["doc_id"])
                from app.rag.revision import invalidate
                invalidate()
                mutated = True
                inserted = self.store.upsert(chunks)
                if inserted != len(chunks):
                    raise RuntimeError("Milvus upsert 行数不一致")
                self.store.cleanup_version(metadata["doc_id"], metadata["version"], list(ids.values()))
                expected_vector_hash = vector_hash(chunks)
                written = [row for row in self.store.snapshot_doc(metadata["doc_id"])
                           if row["version"] == metadata["version"]]
                if len(written) != inserted or vector_hash(written) != expected_vector_hash:
                    raise RuntimeError("Milvus written vectors failed verification")
                # active 发布后归档旧 active；conflicted 两版保留，不能静默裁决。
                if metadata["status"] == "active":
                    old_active = [dict(row, status="archived") for row in snapshot
                                  if row["version"] != metadata["version"] and row["status"] == "active"]
                    if old_active:
                        self.store.upsert(old_active)
                rows = list(session.scalars(select(KnowledgeDoc).where(
                    KnowledgeDoc.doc_id == metadata["doc_id"])))
                current = [r for r in rows if r.version == metadata["version"]]
                if len(current) > 1:
                    raise RuntimeError("同一 doc_id/version 存在重复元数据")
                row = current[0] if current else KnowledgeDoc()
                for key, value in metadata.items():
                    setattr(row, key, value)
                row.doc_name, row.priority, row.chunk_cnt = path.name, priority, inserted
                row.content_snapshot = body
                row.content_hash = document_hash(metadata, body)
                row.chunk_manifest = {ch["chunk_id"]: scalar_hash(ch) for ch in chunks}
                row.ingestion_operation_id = operation_id
                session.add(row)
                session.add(KnowledgeIngestionOperation(id=operation_id, doc_id=metadata["doc_id"],
                    version=metadata["version"], input_hash=row.content_hash, vector_hash=expected_vector_hash,
                    metadata_json=dict(metadata), chunk_cnt=inserted))
                for old in rows:
                    if old.version != row.version and old.status == "active" and row.status == "active":
                        old.status = "archived"
                session.flush()
                commit_attempted = True
            return DocResult(path.name, metadata["source"], STATUS_INDEXED, chunk_cnt=inserted)
        except Exception as exc:
            error = f"{path.name}: {type(exc).__name__}: {exc}"
            if commit_attempted:
                try:
                    # Independent read-only session: the original failed unit of
                    # work or its acknowledgement cannot establish commit state.
                    with Session(commit_engine) as check:
                        committed = check.get(KnowledgeIngestionOperation, operation_id)
                        if committed is not None:
                            current = check.scalars(select(KnowledgeDoc).where(
                                KnowledgeDoc.doc_id == committed.doc_id, KnowledgeDoc.version == committed.version)).one()
                            actual = [row for row in self.store.snapshot_doc(committed.doc_id)
                                      if row["version"] == committed.version]
                            if (committed.input_hash == document_hash(metadata, body)
                                and current.content_hash == committed.input_hash and current.ingestion_operation_id == operation_id
                                and all(getattr(current, key) == value for key, value in metadata.items())
                                and document_hash(metadata, current.content_snapshot) == committed.input_hash
                                and len(actual) == committed.chunk_cnt
                                and vector_hash(actual) == committed.vector_hash):
                                return DocResult(path.name, metadata["source"], STATUS_INDEXED, chunk_cnt=committed.chunk_cnt)
                            # A committed operation must never be undone because its
                            # vector verification failed; retain it for reconciliation.
                            return DocResult(path.name, source, STATUS_FAILED, error="committed_ingestion_requires_reconciliation")
                except Exception:
                    return DocResult(path.name, source, STATUS_FAILED, error="ingestion_outcome_unknown")
            if mutated and snapshot is not None:
                try:
                    self.store.restore_doc(metadata["doc_id"], metadata["version"], snapshot)
                except Exception as rollback_exc:
                    error += f"; compensation_failed:{type(rollback_exc).__name__}"
            return DocResult(path.name, source, STATUS_FAILED, error=error)
        finally:
            try:
                if mutated:
                    try:
                        invalidate()
                    except OSError:
                        pass  # Catalog content hashes also invalidate committed caches.
            finally:
                if locked_connection is not None:
                    try:
                        locked_connection.execute(sql_text("SELECT RELEASE_LOCK(:name)"), {"name": lock_name})
                    except Exception:
                        locked_connection.invalidate()
                    finally:
                        locked_connection.close()
