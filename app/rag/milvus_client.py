"""Milvus 向量库封装（TDD §11.2 / §11.3）。

Collection: factoring_knowledge
    - 稠密向量 FLOAT_VECTOR(1024) + HNSW 索引（M=16, efConstruction=200, IP）
    - 稀疏向量 SPARSE_FLOAT_VECTOR（BM25）+ SPARSE_INVERTED_INDEX（IP）
    - 标量字段：chunk_id / parent_id / chunk_type / source / doc_name /
      priority / content（引用溯源与知识优先级排序）

本模块实现：连接管理 / 版本探测 / Collection 创建与校验 / 向量写入 /
按文档删除（幂等重入库）/ hybrid_search（dense+sparse → RRF，§11.3）/
get_parents（child 命中后回溯 parent 上下文）。
"""
from __future__ import annotations

from typing import Any, Optional
import math
from app.rag.loader import METADATA_FIELDS

from app.config import Settings, get_settings


class VectorStoreError(RuntimeError):
    """Milvus 连接或操作错误。"""


class VectorStore:
    """Milvus Collection 管理封装（MilvusClient 接口）。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        uri: Optional[str] = None,
    ) -> None:
        self.settings = settings or get_settings()
        vd = self.settings.rag.vector_database
        self.uri = uri or f"http://{vd.host}:{vd.port}"
        self.collection_name = vd.collection
        self.dimension = self.settings.rag.embedding.dimension
        self._client = None

    # -------------------------------------------------
    # 连接管理
    # -------------------------------------------------

    def connect(self):
        """建立连接（幂等）。"""
        if self._client is None:
            from pymilvus import MilvusClient

            self._client = MilvusClient(uri=self.uri)
        return self._client

    @property
    def client(self):
        if self._client is None:
            self.connect()
        return self._client

    def ping(self) -> str:
        """连通性检查，返回服务端版本号。"""
        try:
            return self.client.get_server_version()
        except Exception as exc:
            raise VectorStoreError(f"Milvus 连接失败 ({self.uri}): {exc}") from exc

    # -------------------------------------------------
    # Schema（纯函数，可离线单测）
    # -------------------------------------------------

    @staticmethod
    def build_schema(dimension: int = 1024):
        """按 TDD §11.2 构建 Collection Schema。"""
        from pymilvus import CollectionSchema, DataType, FieldSchema

        fields = [
            FieldSchema("id", DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False),
            FieldSchema("chunk_id", DataType.VARCHAR, max_length=64),
            FieldSchema("parent_id", DataType.VARCHAR, max_length=64),
            FieldSchema("chunk_type", DataType.VARCHAR, max_length=8),  # parent / child
            FieldSchema("source", DataType.VARCHAR, max_length=512),     # product / industry / compliance
            FieldSchema("doc_name", DataType.VARCHAR, max_length=256),
            FieldSchema("priority", DataType.INT8),                     # compliance=1 > product=2 > industry=3
            FieldSchema("content", DataType.VARCHAR, max_length=8192),
            *[FieldSchema(name, DataType.VARCHAR, max_length=length) for name, length in
              {"doc_id": 64, "category": 16, "industry": 16, "version": 64,
               "effective_date": 10, "status": 16, "topic": 256}.items()],
            FieldSchema("chunk_index", DataType.INT32),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema("sparse_vector", DataType.SPARSE_FLOAT_VECTOR),
        ]
        return CollectionSchema(
            fields=fields,
            description="保理知识库 factoring_knowledge（TDD §11.2）",
        )

    # -------------------------------------------------
    # Collection 管理
    # -------------------------------------------------

    def ensure_collection(self) -> str:
        """创建 / 校验 Collection。

        Returns:
            "created"：本次新建；"exists"：已存在。
        """
        if self.client.has_collection(self.collection_name):
            from scripts.verify_connections import validate_schema
            validate_schema(self.client.describe_collection(self.collection_name), self.build_schema(self.dimension))
            return "exists"

        schema = self.build_schema(self.dimension)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Strong",
        )
        return "created"

    # -------------------------------------------------
    # 读写接口（TDD §11.1 入库）
    # -------------------------------------------------

    SCALAR_FIELDS = (
        "chunk_id", "parent_id", "chunk_type", "source",
        "doc_name", "priority", "content", "chunk_index", *(f for f in METADATA_FIELDS if f != "source"),
    )

    def validate_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            raise VectorStoreError("没有切片")
        ids = {c["chunk_id"] for c in chunks}
        if len(ids) != len(chunks):
            raise VectorStoreError("切片主键重复")
        fields = {f.name: f for f in self.build_schema(self.dimension).fields}
        for ch in chunks:
            missing = [f for f in (*self.SCALAR_FIELDS, "id", "embedding", "sparse_vector") if f not in ch]
            if missing:
                raise VectorStoreError(f"chunk 缺少字段 {missing}")
            if ch["id"] != ch["chunk_id"]:
                raise VectorStoreError("id/chunk_id 不一致")
            for name, field in fields.items():
                limit = field.params.get("max_length")
                if limit and (not isinstance(ch[name], str) or len(ch[name].encode("utf-8")) > limit):
                    raise VectorStoreError(f"{name} 类型或 UTF-8 长度错误")
            vector = ch["embedding"]
            if len(vector) != self.dimension or not all(math.isfinite(float(x)) for x in vector):
                raise VectorStoreError("dense 维度或有限值错误")
            sparse = ch["sparse_vector"]
            if not isinstance(sparse, dict) or not all(int(k) >= 0 and int(k) < 2**32 and math.isfinite(float(v)) for k, v in sparse.items()):
                raise VectorStoreError("sparse 格式错误")
            if not isinstance(ch["chunk_index"], int) or ch["chunk_index"] < 0:
                raise VectorStoreError("chunk_index 错误")
            if ch["chunk_type"] not in {"parent", "child"}:
                raise VectorStoreError("chunk_type 错误")
            if ch["chunk_type"] == "child":
                parent = next((p for p in chunks if p["chunk_id"] == ch["parent_id"]), None)
                if not parent or parent["chunk_type"] != "parent" or any(parent[k] != ch[k] for k in METADATA_FIELDS):
                    raise VectorStoreError("子块缺少同版父块")

    def upsert(self, chunks: list[dict[str, Any]]) -> int:
        if not chunks:
            return 0
        self.validate_chunks(chunks)
        result = self.client.upsert(collection_name=self.collection_name, data=chunks)
        self.client.flush(collection_name=self.collection_name)
        return int(result["upsert_count"])

    def snapshot_doc(self, doc_id: str) -> list[dict[str, Any]]:
        self.load()
        iterator = self.client.query_iterator(collection_name=self.collection_name,
            filter=f'doc_id == "{self._escape(doc_id)}"', output_fields=["*"], batch_size=500,
            consistency_level="Strong")
        rows = []
        try:
            while batch := iterator.next():
                rows.extend(batch)
        finally:
            iterator.close()
        return rows

    def cleanup_version(self, doc_id: str, version: str, keep_ids: list[str]) -> None:
        # delete 仅作用于指定版本，且必须在全部 upsert 完成后调用。
        import json
        expr = f'doc_id == "{self._escape(doc_id)}" and version == "{self._escape(version)}"'
        if keep_ids:
            expr += " and id not in " + json.dumps(keep_ids)
        self.client.delete(collection_name=self.collection_name, filter=expr)
        self.client.flush(collection_name=self.collection_name)

    def restore_doc(self, doc_id: str, version: str, rows: list[dict[str, Any]]) -> None:
        # 快照包含多个版本，upsert 原行后只清理本次版本新增行。
        if rows:
            result = self.client.upsert(collection_name=self.collection_name, data=rows)
            if result["upsert_count"] != len(rows):
                raise VectorStoreError("快照恢复数量不一致")
            self.client.flush(collection_name=self.collection_name)
        self.cleanup_version(doc_id, version, [r["id"] for r in rows if r["version"] == version])

    def insert(self, chunks: list[dict[str, Any]]) -> int:
        """写入 chunk 行（须含标量字段 + embedding + sparse_vector）。

        Returns:
            实际写入行数。
        """
        if not chunks:
            return 0
        rows = []
        for ch in chunks:
            missing = [f for f in (*self.SCALAR_FIELDS, "embedding", "sparse_vector")
                       if f not in ch]
            if missing:
                raise VectorStoreError(
                    f"chunk 缺少字段 {missing}: {ch.get('chunk_id', '?')}"
                )
            rows.append({
                "id": ch["chunk_id"],
                **{f: ch[f] for f in self.SCALAR_FIELDS},
                "embedding": [float(x) for x in ch["embedding"]],
                "sparse_vector": {int(k): float(v) for k, v in ch["sparse_vector"].items()},
            })
        try:
            self.client.insert(collection_name=self.collection_name, data=rows)
            self.client.flush(collection_name=self.collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Milvus 写入失败: {exc}") from exc
        return len(rows)

    def delete_by_doc(self, doc_name: str) -> None:
        """按文档名删除该文档全部 chunk（重入库前清理，幂等）。"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                filter=f'doc_name == "{self._escape(doc_name)}"',
            )
            self.client.flush(collection_name=self.collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Milvus 删除失败 ({doc_name}): {exc}") from exc

    def load(self) -> None:
        """加载 Collection 到内存（查询/检索前需已加载，服务端幂等）。"""
        try:
            self.client.load_collection(self.collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Milvus 加载 Collection 失败: {exc}") from exc

    def count(self, expr: str = 'chunk_id != ""') -> int:
        """按表达式统计行数（默认全表）。"""
        try:
            self.load()
            res = self.client.query(
                collection_name=self.collection_name,
                filter=expr,
                output_fields=["count(*)"],
            )
            return int(res[0]["count(*)"])
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Milvus 计数失败 ({expr}): {exc}") from exc

    # -------------------------------------------------
    # 在线检索（§11.3）
    # -------------------------------------------------

    # 在线检索仅 child 参与（parent 行向量不参与，命中后回溯取全文）
    CHILD_FILTER = 'chunk_type == "child" and status in ["active", "conflicted"]'

    @staticmethod
    def _escape(value: str) -> str:
        """过滤表达式字符串转义（反斜杠 + 双引号）。"""
        return value.replace("\\", "\\\\").replace('"', '\\"')

    def hybrid_search(
        self,
        dense: list[float],
        sparse: dict[int, float],
        top_k: int,
        *,
        rrf_k: Optional[int] = None,
        filter_expr: str = CHILD_FILTER,
    ) -> list[dict[str, Any]]:
        """混合检索：dense（HNSW/IP）+ sparse（SPARSE_INVERTED_INDEX/IP）
        双路召回 → 服务端 RRF 融合（k=60）→ top_k。

        Args:
            dense / sparse: 查询向量（与入库同编码器）。
            top_k: 融合后返回条数（config rag.retrieval.top_k）。
            rrf_k: RRF 常数（缺省取 config rag.retrieval.hybrid.rrf_k）。
            filter_expr: 标量过滤（缺省仅 child 参与）。

        Returns:
            命中列表（标量字段 + score=RRF 融合分，按分数降序）。
        """
        from pymilvus import AnnSearchRequest, RRFRanker

        if rrf_k is None:
            rrf_k = self.settings.rag.retrieval.hybrid.rrf_k
        reqs = [
            AnnSearchRequest(
                data=[dense], anns_field="embedding",
                param={"metric_type": "IP"},
                limit=max(top_k, self.settings.rag.retrieval.recall_k), expr=filter_expr,
            ),
        ]
        if sparse:  # 空 sparse（如空 query 无词法信号）退化为 dense 单路
            reqs.append(
                AnnSearchRequest(
                    data=[sparse], anns_field="sparse_vector",
                    param={"metric_type": "IP", "params": {}},
                    limit=max(top_k, self.settings.rag.retrieval.recall_k), expr=filter_expr,
                )
            )
        try:
            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=reqs,
                ranker=RRFRanker(k=rrf_k),
                limit=top_k,
                output_fields=list(self.SCALAR_FIELDS),
            )
        except Exception as exc:
            raise VectorStoreError(f"Milvus 混合检索失败: {exc}") from exc
        hits = results[0] if results else []
        items = []
        for hit in hits:
            entity = hit.get("entity", hit)
            item = {f: entity.get(f) for f in self.SCALAR_FIELDS}
            item["score"] = hit.get("distance")
            items.append(item)
        return items

    def get_parents(self, parent_ids: list[str]) -> list[dict[str, Any]]:
        """按 chunk_id 批量取 parent 行（child 命中后回溯上下文）。

        Args:
            parent_ids: 命中 child 的 parent_id 列表（自动去重、去空）。

        Returns:
            parent 行列表（含 SCALAR_FIELDS，无向量字段）。
        """
        ids = sorted({pid for pid in parent_ids if pid})
        if not ids:
            return []
        quoted = ", ".join(f'"{self._escape(i)}"' for i in ids)
        try:
            self.load()
            return list(self.client.query(
                collection_name=self.collection_name,
                filter=f"chunk_id in [{quoted}]",
                output_fields=list(self.SCALAR_FIELDS),
            ))
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Milvus parent 回溯失败: {exc}") from exc
