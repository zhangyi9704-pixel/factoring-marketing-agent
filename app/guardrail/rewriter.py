"""系统输出侧合规改写（TDD §14 action=rewrite）。

系统输出（报告 / 建议 / 复盘话术）先经 GuardrailService.detect 检测违禁表述；
**命中才**调用 LLM 改写为合规表述（"以审批结果为准 / 以合同约定为准"式），
并把命中动作 / 原文交给调用方落 `t_compliance_log`（action=rewrite）。
多数报告不命中 → 零 LLM 调用。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.guardrail.detector import GuardrailHit, GuardrailService

logger = logging.getLogger(__name__)

REWRITE_SYSTEM = (
    "你是合规的保理从业人员。请把下列销售话术改写为合规表述。铁律："
    "不得承诺固定额度、审批通过、放款时间或固定利率；涉及额度/审批/放款/利率时"
    "统一改为‘以审批结果为准’、‘以合同约定为准’等表述；"
    "只输出改写后的合规话术，不要解释，不要 JSON。"
)


@dataclass(frozen=True)
class RewriteResult:
    """一次改写结果。violations 为改写前的命中（供留痕）；remaining 为改写后仍残留的命中。"""

    original: str
    rewritten: str
    violations: list[GuardrailHit]
    remaining: list[GuardrailHit]


class GuardrailRewriter:
    """系统输出侧文本改写器（检测 + 命中 LLM 改写 + 复检）。"""

    def __init__(self, guardrail: GuardrailService, llm=None) -> None:
        self.guardrail = guardrail
        self.llm = llm

    def rewrite(self, text: str) -> RewriteResult:
        """改写一段报告/建议文本。"""
        first_hits = self.guardrail.detect(text)
        if not first_hits:
            return RewriteResult(text, text, violations=[], remaining=[])
        rewritten = self._llm_rewrite(text, first_hits)
        remaining = self.guardrail.detect(rewritten) or []
        return RewriteResult(text, rewritten, violations=first_hits, remaining=remaining)

    def _llm_rewrite(self, text: str, hits: list[GuardrailHit]) -> str:
        if self.llm is None:
            logger.warning("无 LLM 可改写，保留原文: %s", text)
            return text
        rules = "、".join(sorted({h.rule for h in hits}))
        prompt = (
            f"下列话语命中违禁表述（规则：{rules}）：\n"
            f"原文：{text}\n\n"
            "请改写为合规表述，保留原意与语气，输出改写后内容。"
        )
        try:
            result = self.llm.chat(prompt, model_role="chat", system=REWRITE_SYSTEM).strip()
            return result or text
        except Exception as exc:
            logger.warning("合规改写失败，保留原文: %s", exc)
            return text
