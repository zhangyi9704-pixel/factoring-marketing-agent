"""销售能力评分（TDD §13.3，Rubric 四维各占 25% + 证据门控）。

LLM（reasoning）针对四维打分：需求挖掘 / 业务理解 / 风险意识 / 合规意识，
各占 25%。每维经证据门控——无对话证据则贡献记 0（PRD #3 / §13.4），
其中合规维度的证据来自 evidence 抽取的合规表现（无违禁即正样本）。
"""
from __future__ import annotations

from typing import Any

from app.evaluation.common import (
    DIM_LABELS,
    build_dim_specs,
    build_scoring_prompt,
    weighted_gated_score,
)


class SalesScorer:
    """销售能力评分器（LLM reasoning rubic + 加权门控）。"""

    def __init__(self, llm, settings) -> None:
        self.llm = llm
        self.weights = {
            k: v.weight for k, v in settings.evaluation.modules.sales_score.rubric.items()
        }
        self.warn_penalty = settings.evaluation.warn_penalty

    def score(self, conversation: list[dict], case: dict, evidence: dict[str, list]) -> dict:
        """输出 {"total","dimension_scores","reasons","errors"}。"""
        dims = build_dim_specs(self.weights)
        prompt = build_scoring_prompt(
            task_name="销售能力评分",
            task_guidance="你是带教型销售主管，评估这名销售在本次演练中的专业能力表现，"
                          "评分越高代表能力越强；需严格基于对话内容。",
            dims=dims,
            conversation=conversation,
            case=case,
        )
        from app.evaluation.contracts import score_with_validation
        audits = case.get('_compliance_events', [])
        blocked = any(a.get('action') == 'block' or (a.get('details') or {}).get('severity') == 'block' for a in audits)
        blocked = blocked or any(e.get("compliance_severity") == "block" for e in evidence.get("compliance", []))
        llm_scores, evidence, errors = score_with_validation(self.llm, prompt, self.weights, conversation, evidence)
        if blocked:
            llm_scores["compliance"] = {"score": 0, "rationale": "独立审计确认严重违规拦截，按CMP-05/06规则合规维度计0分。"}
        else:
            warnings = {(a.get('turn'), a.get('rule'), a.get('raw_text')) for a in audits
                if (a.get('details') or {}).get('severity', 'warn') == 'warn' and a.get('action') in {'log', 'warn'}}
            if 'compliance' in llm_scores:
                llm_scores['compliance']['score'] = max(0, llm_scores['compliance']['score'] - len(warnings) * self.warn_penalty)
                if warnings:
                    llm_scores['compliance']['rationale'] += f'；{len(warnings)}条warn各扣{self.warn_penalty}分。'
        total, dimension_results = weighted_gated_score(
            llm_scores, self.weights, evidence,
        )
        if blocked:
            dimension_results['compliance']['rationale'] = '独立合规审计确认block，合规维度计0；原话只在审计展示，不充当送达对话证据。'
        return {
            "total": total,
            "dimension_scores": dimension_results,
            "reasons": self._reasons(dimension_results),
            "errors": errors,
        }

    @staticmethod
    def _reasons(dimension_results: dict[str, dict]) -> list[str]:
        reasons: list[str] = []
        for key, res in dimension_results.items():
            label = DIM_LABELS.get(key, key)
            if res.get("evidence_missing"):
                reasons.append(f"{label}：未实质讨论（证据缺失），按 0 计")
            elif res.get("rationale"):
                reasons.append(f"{label}：{res['rationale']}")
        return reasons
