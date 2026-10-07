"""三分支检索：双路 RRF → 精排/阈值 → 冲突 → 同版本父块回溯。"""
from __future__ import annotations

import logging
import copy
import time
import re
from app.rag.revision import revision
from app.services.escalation import register_escalation
from dataclasses import dataclass
from typing import Any, Optional

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

REFERENCE_ONLY_NOTE = "仅供参考"


@dataclass
class RetrievalItem:
    """单条检索结果（child 命中 + 回溯的 parent 上下文）。"""

    chunk_id: str
    parent_id: str
    chunk_type: str
    source: str
    doc_name: str
    priority: int
    content: str                  # 命中 child 片段
    score: float                  # RRF 融合分（降序）
    parent_content: Optional[str] = None   # 回溯的 parent 全文（缺失为 None）
    reference_only: bool = False  # 存在更高优先级来源时标注"仅供参考"
    doc_id: Optional[str] = None
    category: Optional[str] = None
    industry: Optional[str] = None
    version: Optional[str] = None
    effective_date: Optional[str] = None
    status: Optional[str] = None
    topic: Optional[str] = None
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        data = {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "source": self.source,
            "doc_name": self.doc_name,
            "priority": self.priority,
            "content": self.content,
            "score": self.score,
            "retrieval_score": self.retrieval_score if self.retrieval_score is not None else self.score,
            "rerank_score": self.rerank_score,
            "summary": self.content[:300],
            "parent_content": self.parent_content,
            "reference_only": self.reference_only,
        }
        if self.reference_only:
            data["note"] = REFERENCE_ONLY_NOTE
        for key in ("doc_id", "category", "industry", "version", "effective_date", "status", "topic"):
            data[key] = getattr(self, key)
        return data


@dataclass
class RetrievalResult:
    """检索结果（空结果时 missing=True，携带 human_review 动作与话术）。"""

    query: str
    items: list[RetrievalItem]
    missing: bool = False
    action: Optional[str] = None    # knowledge.missing.action（human_review）
    message: Optional[str] = None   # knowledge.missing.message

    mode: str = "rag"
    reason: Optional[str] = None
    escalation_id: Optional[str] = None
    rerank_degraded: bool = False

    def __post_init__(self):
        if self.missing:
            self.mode = "missing"
        if self.mode not in {"rag", "conflict", "missing"}:
            raise ValueError("unknown retrieval mode")

    def to_dict(self) -> dict[str, Any]:
        return dict(mode=self.mode, reason=self.reason, items=[i.to_dict() for i in self.items],
                    escalation_id=self.escalation_id, rerank_degraded=self.rerank_degraded,
                    message=self.message, action=self.action, query=self.query, missing=self.missing)


class HybridRetriever:
    """混合检索 + 重排 + 优先级排序封装（组件全部可注入）。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        embedder: Any = None,
        store: Any = None,
        reranker: Any = None,
        visibility: Any = None,
    ) -> None:
        self.settings = settings or get_settings()
        if embedder is None:
            from app.rag.embedder import EmbeddingService
            embedder = EmbeddingService(self.settings)
        if store is None:
            from app.rag.milvus_client import VectorStore
            store = VectorStore(self.settings)
        if reranker is None:
            from app.rag.reranker import Reranker
            reranker = Reranker(self.settings)
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        if visibility is None:
            from app.rag.milvus_client import VectorStore
            if isinstance(store, VectorStore):
                from app.rag.visibility import KnowledgeVisibility
                visibility = KnowledgeVisibility()
        self.visibility = visibility
        self._cache = {}

    # -------------------------------------------------
    # 对外入口
    # -------------------------------------------------

    def retrieve(self, query: str, top_k: Optional[int] = None, *, industry: Optional[str] = None) -> RetrievalResult:
        """行业过滤用于双路召回；冲突只看经阈值过滤后的 top-K。"""
        if industry is not None and industry not in {"all", "基建", "医药", "电商"}:
            raise ValueError("industry 枚举错误")
        if self.out_of_scope(query):
            return self._missing_result(query, "out_of_scope")
        cfg = self.settings.rag.retrieval
        top_k = top_k or cfg.top_k
        stamp = revision()
        try:
            catalog = self.visibility.catalog() if self.visibility else None
        except Exception:
            return self._missing_result(query,"catalog_unavailable")  # Fail closed.
        key = (query, industry, top_k, stamp, catalog[2] if catalog else None, cfg.rerank.model, cfg.rerank.min_score,
               cfg.fallback_min_score, cfg.rerank.enabled)
        cached = self._cache.get(key)
        if cached and time.monotonic() - cached[0] < cfg.cache_ttl:
            return copy.deepcopy(cached[1])
        if len(self._cache) >= 128:
            self._cache.clear()
        expr = 'chunk_type == "child" and status in ["active", "conflicted"]'
        if catalog is not None:
            expr = 'chunk_type == "child" and ' + self.visibility.expression(catalog)
        if industry is not None:
            expr += f' and industry in ["all", "{industry}"]'
        try:
            dense, sparse = self.embedder.embed([query])
            hits = self.store.hybrid_search(dense[0], sparse[0], top_k, filter_expr=expr)
            if self.visibility:
                catalog = self.visibility.catalog()
                hits = self.visibility.filter(hits,catalog)
            # 防御过滤；真实 Milvus 已过滤，旧资产的空元数据只用于兼容历史测试。
            hits = [h for h in hits if h.get("status") != "archived" and
                    (industry is None or not h.get("industry") or h["industry"] in {industry, "all"})][:top_k]
            items = [self._to_item(h) for h in hits]
            if hasattr(self.reranker, "rerank_with_status"):
                items, degraded = self.reranker.rerank_with_status(query, items)
            else:
                items, degraded = self.reranker.rerank(query, items), False
            if degraded:
                # RRF 与精排分尺度不同。缺少已校准召回阈值就不提供依据。
                threshold = cfg.fallback_min_score
                items = [] if threshold is None else [i for i in items if i.retrieval_score >= threshold]
            if not items:
                result = self._missing_result(query, "rerank_unavailable" if degraded else "insufficient_evidence")
                result.rerank_degraded = degraded
            else:
                conflict_topics = self.conflict_topics(items)
                # 检测先于父块合并；同 doc_id 的两个版本不丢失。
                mode = "conflict" if conflict_topics else "rag"
                selected = [i for i in items if i.topic in conflict_topics] if conflict_topics else items
                incomplete = bool(conflict_topics) and any(len({(i.doc_id, i.version) for i in selected
                    if i.topic == topic}) < 2 for topic in conflict_topics)
                selected = self._parents(selected,catalog)
                result = RetrievalResult(query, selected if conflict_topics else self._sort_by_priority(selected), mode=mode,
                    reason="conflict_evidence_incomplete" if incomplete else None,
                    message=("知识已标记冲突，但本次未召回完整双方依据；请人工核对全部版本后答复。" if incomplete
                             else "知识存在冲突，请并列核实双方来源，待人工确认后答复。") if conflict_topics else None,
                    escalation_id=register_escalation("conflict") if conflict_topics else None,
                    action="human_review" if conflict_topics else None, rerank_degraded=degraded)
        except Exception as exc:
            logger.warning("retrieval_degraded exception=%s", type(exc).__name__)
            result = self._missing_result(query, "retrieval_unavailable")
        try:
            latest = self.visibility.catalog() if self.visibility else None
            if latest and catalog[2] != latest[2]:
                return self._missing_result(query, "knowledge_changed_during_retrieval")
            if stamp == revision() and (not latest or key[4] == latest[2]):
                self._cache[key] = (time.monotonic(), copy.deepcopy(result))
        except Exception:
            return self._missing_result(query, "catalog_unavailable")
        return result

    @staticmethod
    def conflict_topics(items):
        """A conflicted catalog marker cannot become evidence by truncating peers."""
        return {item.topic for item in items if item.status == "conflicted"}

    @staticmethod
    def out_of_scope(query: str) -> bool:
        # 明确不支持的主题；非关键词命中不等于超范围，仍交检索阈值处理。
        return bool(re.search(r"天气|天气预报|股票买卖|荐股|彩票|星座|做菜|菜谱|量子纠缠|世界杯", query))

    def _missing_result(self, query: str, reason="insufficient_evidence") -> RetrievalResult:
        logger.warning("知识库未命中 reason=%s", reason)
        return RetrievalResult(query, [], missing=True, action="human_review",
            message=self.settings.knowledge.missing.message, reason=reason,
            escalation_id=register_escalation("out_of_scope" if reason == "out_of_scope" else "missing"))

    @staticmethod
    def _to_item(hit):
        keys = ("doc_id", "category", "industry", "version", "effective_date", "status", "topic")
        return RetrievalItem(chunk_id=hit["chunk_id"], parent_id=hit.get("parent_id") or "",
            chunk_type=hit.get("chunk_type", "child"), source=hit["source"], doc_name=hit["doc_name"],
            priority=int(hit.get("priority") or 0), content=hit["content"],
            score=float(hit.get("score") or 0), retrieval_score=float(hit.get("score") or 0),
            **{k: hit.get(k) for k in keys})

    def _parents(self, items, catalog=None):
        best = {}
        for i in items:
            identity = (i.parent_id or i.chunk_id, i.doc_id, i.version)
            if identity not in best or i.score > best[identity].score:
                best[identity] = i
        ids = list(dict.fromkeys(i.parent_id for i in best.values() if i.parent_id))
        parents = self.store.get_parents(ids) if ids else []
        if catalog is not None:
            parents = self.visibility.filter(parents,catalog)
        for i in best.values():
            parent = next((p for p in parents if p["chunk_id"] == i.parent_id and
                           p.get("doc_id") == i.doc_id and p.get("version") == i.version and
                           p.get("status") != "archived"), None)
            i.parent_content = parent.get("content") if parent else None
        return sorted(best.values(), key=lambda i: i.score, reverse=True)

    def _merge_parent_context(self, hits):
        return self._parents([self._to_item(h) for h in hits])

    def _sort_by_priority(self, items: list[RetrievalItem]) -> list[RetrievalItem]:
        """按 knowledge.conflict.order 稳定排序；低优先级条目标注仅供参考。

        稳定排序保证同优先级内保持 RRF 分数序；仅当结果中存在更高
        优先级来源时才标注（单一来源不标）。
        """
        order = self.settings.knowledge.conflict.order
        rank = {source: i for i, source in enumerate(order)}
        fallback = len(order)

        def rank_of(item: RetrievalItem) -> int:
            return rank.get(item.category or item.source, fallback)

        items = sorted(items, key=rank_of)
        if items:
            top_rank = min(rank_of(item) for item in items)
            for item in items:
                item.reference_only = rank_of(item) > top_rank
        return items
