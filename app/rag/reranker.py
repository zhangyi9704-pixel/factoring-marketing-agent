"""精排器（技术设计文档（新）§9.1：阿里云 DashScope gte-rerank-v2）。

    RRF 粗排 top10 → gte-rerank-v2 精排 → 覆写分数 relevance_score(0-1)
    降序重排 → min_score 过滤 → 截断 top5

线上 API（与 embedding 复用 ${ALIYUN_API_KEY}）：
    POST https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank
    {"model": "gte-rerank-v2",
     "input": {"query": "...", "documents": ["...", ...]},   # ≤500 条
     "parameters": {"return_documents": false}}
    → {"output": {"results": [{"index": 0, "relevance_score": 0.87}, ...]}}

容错（继承检索链路降级模式）：rerank.enabled=false、api_key 缺失、
请求失败/非 200、响应畸形时一律降级为透传 + 截断（上游 RRF 分数序
即相关性序），warning 留痕不中断检索。全部条目低于 min_score 时返回
空列表，由 HybridRetriever 转 knowledge.missing（human_review）分支。
"""
from __future__ import annotations

import logging
import math
from typing import Any, Optional

from app.config import Settings, get_settings
from app.services.external_calls import ExternalCalls, timeout_for

logger = logging.getLogger(__name__)

DASHSCOPE_RERANK_URL = (
    "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
)
MAX_DOCUMENTS = 500                    # DashScope 单请求文档上限


def _content(item: Any) -> str:
    """取参与精排的文本（RetrievalItem.content；dict 兼容 content 字段）。"""
    if isinstance(item, dict):
        return str(item.get("content") or "")
    return str(getattr(item, "content", "") or "")


def _set_score(item: Any, score: float) -> None:
    """覆写相关性分（RRF 分 → gte relevance_score，0-1 归一）。"""
    if isinstance(item, dict):
        item.setdefault("retrieval_score", item.get("score", 0.0))
        item["rerank_score"] = score
        item["score"] = score
    else:
        item.retrieval_score = item.retrieval_score if item.retrieval_score is not None else item.score
        item.rerank_score = score
        item.score = score


def _score(item: Any) -> float:
    if isinstance(item, dict):
        return float(item.get("score") or 0.0)
    return float(getattr(item, "score", 0.0) or 0.0)


class Reranker:
    """精排封装（provider 路由：dashscope / passthrough）。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        http_client: Any = None,       # 测试注入 httpx.Client（MockTransport）
    ) -> None:
        self.settings = settings or get_settings()
        rerank_cfg = self.settings.rag.retrieval.rerank
        self.model = rerank_cfg.model
        self.enabled = rerank_cfg.enabled
        self.provider = rerank_cfg.provider
        self.min_score = rerank_cfg.min_score
        self.api_key = rerank_cfg.api_key
        self.base_url = rerank_cfg.base_url or DASHSCOPE_RERANK_URL
        self.timeout = rerank_cfg.timeout
        self.calls = ExternalCalls(self.settings)
        self._http_client = http_client

    # -------------------------------------------------
    # 对外契约
    # -------------------------------------------------

    def rerank(
        self,
        query: str,
        items: list[Any],
        top_n: Optional[int] = None,
    ) -> list[Any]:
        """按与 query 的相关度重排并截断。

        dashscope 模式：覆写分数降序重排 + min_score 过滤 + 截断 top_n；
        passthrough / 降级：不改变顺序仅截断（缺省取 config
        rag.retrieval.rerank.top_n，§11.3 为 top10 → top5）。
        """
        return self.rerank_with_status(query, items, top_n)[0]

    def rerank_with_status(self, query, items, top_n=None):
        """返回 (items, degraded)，状态属于本次调用，支持并发。"""
        top_n = top_n or self.settings.rag.retrieval.rerank.top_n
        if not items:
            return [], False
        if not self.enabled:
            return list(items), True
        if self.provider != "dashscope" or not self.api_key:
            return list(items[:top_n]), True
        return self._rerank_dashscope(query, items, top_n)

    # -------------------------------------------------
    # DashScope gte-rerank-v2
    # -------------------------------------------------

    def _client(self):
        if self._http_client is None:
            import httpx
            self._http_client = httpx.Client(timeout=timeout_for(self.settings, self.timeout))
        return self._http_client

    def _rerank_dashscope(
        self, query: str, items: list[Any], top_n: int,
    ) -> list[Any]:
        """线上精排：全量低于 min_score → 空列表（检索缺失语义）。"""
        capped = items[:MAX_DOCUMENTS]
        try:
            payload = self.calls.post(
                self._client(),
                self.base_url, operation="rerank",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": {
                        "query": query,
                        "documents": [_content(item) for item in capped],
                    },
                    "parameters": {"return_documents": False},
                },
            )
            results = (payload.get("output") or {}).get("results") or []
        except Exception as exc:                     # 网络/解析故障留痕降级
            logger.warning("rerank 请求失败，降级为透传截断: %s", type(exc).__name__)
            return list(items[:top_n]), True

        if not results:
            logger.warning("rerank 响应无结果，降级为透传截断")
            return list(items[:top_n]), True

        scored: list[tuple[float, Any]] = []
        seen = set()
        for entry in results:
            try:
                index = int(entry["index"])
                relevance = float(entry["relevance_score"])
            except (KeyError, TypeError, ValueError):
                continue                          # 单条畸形跳过
            if 0 <= index < len(capped) and index not in seen and math.isfinite(relevance) and 0 <= relevance <= 1:
                seen.add(index)
                _set_score(capped[index], relevance)
                scored.append((relevance, capped[index]))
        if not scored:
            logger.warning("rerank 结果索引全部无效，降级为透传截断")
            return list(items[:top_n]), True

        scored.sort(key=lambda pair: pair[0], reverse=True)   # 分数降序
        ranked = [item for _, item in scored if _score(item) >= self.min_score]
        if len(ranked) < len(scored):
            logger.info(
                "rerank min_score=%.3f 过滤 %d/%d 条",
                self.min_score, len(scored) - len(ranked), len(scored),
            )
        return ranked[:top_n], False


__all__ = ["Reranker", "DASHSCOPE_RERANK_URL"]
