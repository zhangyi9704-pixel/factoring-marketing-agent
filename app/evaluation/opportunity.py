"""机会诊断（TDD §13.2，5 维加权评分 + 证据门控，非独立 Agent）。

LLM（reasoning）针对 5 维打分（交易真实性 25% / 应收质量 25% / 付款来源 20% /
融资需求 15% / 风险识别 15%），再按权重加权（保留 0-100）；每个维度经证据
门控——无对话证据则贡献记 0（PRD #3 / §13.4）。最终按阈值映射机会等级：
≥high(80) 高潜机会 / ≥observe(60) 观察 / <observe 暂缓。
"""
from __future__ import annotations

from typing import Any, Optional

from app.evaluation.common import (
    DIM_LABELS,
    build_dim_specs,
    build_scoring_prompt,
    map_level,
    weighted_gated_score,
)


class OpportunityDiagnoser:
    """机会诊断器（LLM reasoning + 加权门控）。"""

    def __init__(self, llm, settings) -> None:
        self.llm = llm
        self.levels = settings.evaluation.levels
        self.weights = {
            k: v.weight for k, v in settings.evaluation.modules.opportunity_diagnosis.dimensions.items()
        }

    def analyze(self, conversation: list[dict], case: dict, evidence: dict[str, list]) -> dict:
        """输出 {"score","level","reasons","dimension_scores","errors"}。"""
        dims = build_dim_specs(self.weights)
        prompt = build_scoring_prompt(
            task_name="机会诊断",
            task_guidance="你是保理业务的风控/客户经理，评估这一单保理业务的机会成熟度。"
                          "评分越高代表越适合推进保理融资；需严格基于已验证对话，不推断隐藏案例事实。",
            dims=dims,
            conversation=conversation,
            case=case,
        )
        from app.evaluation.contracts import score_with_validation
        llm_scores, evidence, errors = score_with_validation(self.llm, prompt, self.weights, conversation, evidence)
        total, dimension_results = weighted_gated_score(
            llm_scores, self.weights, evidence,
        )
        level = map_level(total, high=self.levels.high, observe=self.levels.observe)
        return {
            "score": total,
            "level": level,
            "reasons": self._reasons(dimension_results),
            "dimension_scores": dimension_results,
            "errors": errors,
        }

    @staticmethod
    def _reasons(dimension_results: dict[str, dict]) -> list[str]:
        reasons: list[str] = []
        for key, res in dimension_results.items():
            label = DIM_LABELS.get(key, key)
            if res.get("evidence_missing"):
                reasons.append(f"{label}：对话中未实质涉足（证据缺失），按 0 计")
            elif res.get("rationale"):
                reasons.append(f"{label}：{res['rationale']}")
        return reasons
