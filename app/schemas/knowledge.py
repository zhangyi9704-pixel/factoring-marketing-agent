"""知识库注入响应模型（TDD §15：POST /knowledge/ingest）。"""
from __future__ import annotations

from typing import Optional, Literal

from pydantic import BaseModel, Field, ConfigDict


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    industry: Optional[Literal["all", "基建", "医药", "电商"]] = None
    top_k: int = Field(default=10, ge=1, le=20)


class SearchResponse(BaseModel):
    mode: Literal["rag", "conflict", "missing"]
    reason: Optional[str] = None
    items: list[dict]
    escalation_id: Optional[str] = None
    message: Optional[str] = None
    rerank_degraded: bool = False


class IngestDocResult(BaseModel):
    """单文档入库结果（回流 KnowledgeIngestor 的 DocResult）。"""

    doc_name: str
    source: str
    status: str                      # indexed / failed / skipped
    chunk_cnt: int = 0
    error: Optional[str] = None
