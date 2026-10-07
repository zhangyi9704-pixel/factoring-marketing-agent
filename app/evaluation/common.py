"""评估共享工具（TDD §13：加权门控 / 机会等级映射 / 评分提示词）。

机会诊断（§13.2）与能力评分（§13.3）共用：
- 维度加权总分 + 证据门控（PRD #3 / §13.4：无证据的评分项按 0 处理）；
- 等级阈值映射（§13.2：≥high 高潜机会 / ≥observe 观察 / <observe 暂缓）。

维度中文标签为证据 comment 与评分文案的统一来源（本模块单点维护）。
"""
from __future__ import annotations

import json
from typing import Any, Optional

# 维度英文键 → 中文标签（证据 comment / 评分文案用）
DIM_LABELS: dict[str, str] = {
    "transaction_authenticity": "交易真实性",
    "receivable_quality": "应收质量",
    "payer_confirmation": "付款来源",
    "financing_need": "融资需求",
    "risk_identification": "风险识别",
    "requirement_discovery": "需求挖掘",
    "business_understanding": "业务理解",
    "risk_awareness": "风险意识",
    "compliance": "合规意识",
}

# 维度中文定义（评分提示词中的判定要点，TDD §13.2 / §13.3）
DIM_GUIDELINES: dict[str, str] = {
    "transaction_authenticity": "交易是否真实存在（合同/中标/结算/回款线索）",
    "receivable_quality": "应收是否具备可保理条件（账期/对账/验收/计量）",
    "payer_confirmation": "最终付款来源是否明确、可增信",
    "financing_need": "企业资金需求是否真实、迫切、可量化",
    "risk_identification": "业务/合规风险是否被识别（回款逾期/质押/涉诉）",
    "requirement_discovery": "是否主动挖掘资金需求、痛点与金额",
    "business_understanding": "是否沿交易→应收→付款→融资主线展开",
    "risk_awareness": "是否主动询问验收、对账、回款、质押等风险项",
    "compliance": "是否避免违规承诺固定额度/审批/放款/利率",
}


class DimensionSpec:
    """单个评分维度的提示词规格（key/label/weight/guideline）。"""

    __slots__ = ("key", "label", "weight", "guideline")

    def __init__(self, key: str, weight: int, guideline: Optional[str] = None) -> None:
        self.key = key
        self.label = DIM_LABELS.get(key, key)
        self.weight = weight
        self.guideline = guideline or DIM_GUIDELINES.get(key, "")


def build_dim_specs(weights: dict[str, int], order: Optional[list[str]] = None) -> list[DimensionSpec]:
    """按 order 顺序构造维度规格（order 缺省用 weights 的插入序）。"""
    seq = order or list(weights.keys())
    return [DimensionSpec(k, weights.get(k, 0)) for k in seq]


def weighted_gated_score(
    llm_scores: dict[str, dict],
    weights: dict[str, int],
    evidence: dict[str, list],
    order: Optional[list[str]] = None,
) -> tuple[float, dict[str, dict]]:
    """加权总分 + 证据门控。

    每个维度：LLM 给出 0-100 score；若该维度无对话证据则贡献记 0（按证据缺失处理）。
    返回 (加权总分_0-100, {维度: {score, weight, gated_score, evidence_missing, rationale}})。
    """
    seq = order or list(weights.keys())
    total_weight = sum(weights.get(d, 0) for d in seq) or 1
    weighted_sum = 0.0
    dimension_results: dict[str, dict] = {}
    for d in seq:
        weight = weights.get(d, 0)
        raw = llm_scores.get(d) or {}
        llm_score = raw.get("score")
        import math
        llm_score = float(llm_score) if type(llm_score) in (int, float) and math.isfinite(llm_score) and 0 <= llm_score <= 100 else 0.0
        has_evidence = bool(evidence.get(d))
        gated_score = llm_score if has_evidence else 0.0
        weighted_sum += gated_score * weight
        dimension_results[d] = {
            "score": round(gated_score, 2),
            "weight": weight,
            "gated_score": round(gated_score, 2),
            "evidence_missing": not has_evidence,
            "rationale": raw.get("rationale", "") if has_evidence else "无有效销售对话证据，按0计。",
            "evidence_refs": [{"turn_id": e["turn"], "original_text": e["quote_sales"]} for e in evidence.get(d, [])],
        }
    total = round(weighted_sum / total_weight, 2)
    return total, dimension_results


def map_level(score: float, *, high: float, observe: float) -> str:
    """机会等级映射（TDD §13.2）。"""
    if score >= high:
        return "高潜机会"
    if score >= observe:
        return "观察"
    return "暂缓"


def build_transcript(conversation: list[dict]) -> str:
    """多轮对话转为可读脚本（评分提示词用）。"""
    return "\n".join(
        f"第{m.get('turn')}轮 [{'销售' if m.get('role') == 'sales' else '客户'}] "
        f"{m.get('content', '')}"
        for m in (conversation or [])
    ) or "（无对话记录）"


def build_case_facts(case: dict) -> str:
    """Agent 全量 Case（含 unknown_info / hidden_risks，内部标记）转 JSON 注入。"""
    return json.dumps(case or {}, ensure_ascii=False, indent=2)


def build_scoring_prompt(
    *,
    task_name: str,
    task_guidance: str,
    dims: list[DimensionSpec],
    conversation: list[dict],
    case: dict,
) -> str:
    """评分提示词：维度定义 + 全量案例 + 完整对话 → JSON 输出。"""
    dim_text = "\n".join(
        f"- {d.key} / {d.label}（权重 {d.weight}）：{d.guideline}" for d in dims
    ) or "（无维度）"
    keys = [d.key for d in dims]
    return f"""{task_guidance}

【任务】{task_name}

【评分维度】（按权重加权）
{dim_text}

【行业】
{case.get('industry', '')}

【完整对话记录】
{build_transcript(conversation)}

请输出全部维度英文键{keys}，仅依据真实对话对各维度在0-100评分，不推断客户未说出的事实。每维evidence须引用同轮销售完整原话；没有证据输出空列表且score=0。
仅输出一个合法 JSON 对象，不要 markdown 围栏或任何其他文字：按如下键给出各维度
{{
  "dimensions": {{
    "{keys[0]}": {{"score": 80, "rationale": "依据一句话", "evidence": [{{"turn_id": 1, "original_text": "同轮销售完整原话", "speaker": "sales"}}]}}
  }}
}}"""
