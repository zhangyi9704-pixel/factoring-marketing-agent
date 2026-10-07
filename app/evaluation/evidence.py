"""评分证据抽取（TDD §13.4，纯规则：从对话按 turn 抽原话）。

PRD #3 / §13.4：「每个子项评分必须引用原始对话（轮次 + 双方原话 + 评价）」。

本模块按维度关键词扫描对话，抽取命中轮次的销售/客户原话作为证据；
无命中 → 空列表（调用方据此把该评分项按 0 处理并标注"证据缺失"）。

compliance 维度特殊：合规证据来自"是否出现违禁承诺"——无违禁给一条
正样本（引用最近销售原话 + 全程合规说明）；有违禁引用违禁轮次与命中片段。
"""
from __future__ import annotations

from typing import Any, Optional

from app.evaluation.common import DIM_LABELS


class EvidenceExtractor:
    """从对话抽取各评分维度的证据（纯规则，无 LLM）。

    证据项结构：{"dimension", "turn", "quote_sales", "quote_customer", "comment"}。
    """

    def __init__(self, settings) -> None:
        self.topics: dict[str, list[str]] = (
            getattr(settings, "evaluation", None).modules.evidence.topics
            if getattr(settings, "evaluation", None) else {}
        ) or {}

    def extract(
        self,
        conversation: list[dict],
        *,
        guardrail_hits: Optional[list[dict]] = None,
        case: Optional[dict] = None,
    ) -> dict[str, list[dict]]:
        """返回 {dimension: [证据项|空]}，供评分模块做证据门控。"""
        conversation = conversation or []
        guardrail_hits = guardrail_hits or []
        result: dict[str, list[dict]] = {}
        for dimension, keywords in self.topics.items():
            if dimension == "compliance":
                result[dimension] = self._compliance_evidence(conversation, guardrail_hits)
            else:
                result[dimension] = self._topic_evidence(conversation, keywords, dimension)
        return result

    # -------------------------------------------------
    # 内部
    # -------------------------------------------------

    def _topic_evidence(self, conversation: list[dict], keywords: list[str], dimension: str) -> list[dict]:
        """取关键词命中数最多的 turn 为证据；无命中返回空。"""
        best_turn: Optional[int] = None
        best_count = 0
        for turn in _distinct_turns(conversation):
            if not _quotes_at_turn(conversation, turn)["quote_sales"]:
                continue
            text = (_quotes_at_turn(conversation, turn)['quote_sales']
                if dimension in {'requirement_discovery', 'business_understanding', 'risk_awareness'}
                else _turn_text(conversation, turn))
            count = sum(1 for kw in keywords if kw in text)
            if count > best_count:
                best_count, best_turn = count, turn
        if best_turn is None:
            return []
        label = DIM_LABELS.get(dimension, dimension)
        quotes = _quotes_at_turn(conversation, best_turn)
        return [{
            "dimension": dimension,
            "turn": best_turn,
            "quote_sales": quotes["quote_sales"],
            "quote_customer": quotes["quote_customer"],
            "comment": f"{label}：第{best_turn}轮有实质讨论（命中{best_count}个关键词）",
        }]

    def _compliance_evidence(self, conversation: list[dict], guardrail_hits: list[dict]) -> list[dict]:
        """compliance 证据：无违禁 → 正样本；有违禁 → 引用违禁轮次。"""
        if guardrail_hits:
            hit = next((h for h in guardrail_hits if h.get("severity") == "block"), guardrail_hits[0])
            turn = int(hit.get("turn") or 0)
            snippet = hit.get("snippet") or hit.get("raw_text") or ""
            quotes = _quotes_at_turn(conversation, turn)
            if not quotes['quote_sales'] or snippet not in quotes['quote_sales']:
                # 被拦截原话仅属独立审计，不能冒充某轮送达销售证据。
                return []
            return [{
                "dimension": "compliance",
                "turn": turn,
                "quote_sales": quotes["quote_sales"],
                "quote_customer": quotes["quote_customer"],
                "comment": f"触发违禁规则：{hit.get('rule', 'unknown')}",
                "compliance_severity": hit.get("severity", "warn"),
            }]
        # 全程合规：引用最近一次销售原话作为正样本
        sales_turns = [t for t in _distinct_turns(conversation)
                       if _quotes_at_turn(conversation, t)["quote_sales"]]
        turn = sales_turns[-1] if sales_turns else 0
        quotes = _quotes_at_turn(conversation, turn)
        if not quotes['quote_sales']:
            return []
        return [{
            "dimension": "compliance",
            "turn": turn,
            "quote_sales": quotes["quote_sales"],
            "quote_customer": quotes["quote_customer"],
            "comment": "全程未出现违禁承诺（合规意识良好）",
        }]


# -------------------------------------------------
# 对话工具
# -------------------------------------------------

def _distinct_turns(conversation: list[dict]) -> list[int]:
    seen: list[int] = []
    for m in conversation:
        turn = m.get("turn")
        if turn is not None and turn not in seen:
            seen.append(turn)
    return seen


def _turn_text(conversation: list[dict], turn: int) -> str:
    return "\n".join(
        str(m.get("content", "")) for m in conversation if m.get("turn") == turn
    )


def _quotes_at_turn(conversation: list[dict], turn: int) -> dict[str, str]:
    sales = next((m.get("content", "") for m in conversation
                  if m.get("turn") == turn and m.get("role") == "sales"), "")
    customer = next((m.get("content", "") for m in conversation
                     if m.get("turn") == turn and m.get("role") == "customer"), "")
    return {"quote_sales": sales or "", "quote_customer": customer or ""}
