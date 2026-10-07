"""用户成绩画像响应模型（TDD §20 Phase 4 主管视角 `GET /users/{id}/scores`）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class UserScoreReport(BaseModel):
    """主管看板：跨会话聚合该销售的成绩画像 + 五段式辅导板块。"""

    user_id: int
    total_sessions: int = 0
    evaluated_sessions: int = 0
    avg_opportunity: float = 0.0                     # 机会诊断均分
    avg_sales_score: float = 0.0                     # 能力评分均分
    level_distribution: dict[str, int] = Field(default_factory=dict)
    dimension_averages: dict[str, float] = Field(default_factory=dict)
    latest_sessions: list[dict] = Field(default_factory=list)
    # 五段式辅导（§20）：opportunity/review/evidence/missing/improvement
    coaching: dict = Field(default_factory=dict)
