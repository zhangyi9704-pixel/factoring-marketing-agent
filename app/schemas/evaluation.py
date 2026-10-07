"""复盘评估响应模型（TDD §15：GET /sessions/{id}/evaluation）。

机会诊断（§13.2）/ 能力评分（§13.3）/ 证据（§13.4）/ 遗漏问题与下一步行动
（§13.5）的动态结构以 dict 承载（内容由评估模块生成，字段随维度变化），
内部模型输出及辅导字段由evaluation/contracts.py严格校验；这里保留旧报告兼容骨架。
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationReport(BaseModel):
    """复盘评估报告（回放用）。"""

    session_id: str
    opportunity: dict = Field(default_factory=dict, description="机会诊断（score/level/reasons/dimension_scores）")
    sales_score: dict = Field(default_factory=dict, description="四维能力评分（total/dimension_scores/reasons）")
    missing_questions: list[dict] = Field(default_factory=list, description="遗漏问题（item/impact/suggestion）")
    next_actions: list[dict] = Field(default_factory=list, description="下一步行动（action/priority/note）")
    evidence: list[dict] = Field(default_factory=list, description="评分证据链（dimension/turn/quote_sales/quote_customer/comment）")
    total_score: float = 0.0
    generated_at: str = ""
    markdown: str = Field(default="", description="Markdown 版复盘报告（§20 Phase 4）")
    compliance_events: list[dict] = Field(default_factory=list)
    needs_human_review: bool = False
    coaching: list[dict] = Field(default_factory=list)
    sources: list[dict] = Field(default_factory=list)
    schema_version: str = 'legacy'
    degraded: bool = False
    snapshot_unavailable: bool = False
