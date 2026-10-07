"""遗漏问题检测（TDD §13.5，行业 Checklist 规则比对，非 LLM 判断）。

按案例所处行业从 config 取 Checklist 条目；若该条目的关键词均未在对话中
出现（无论是销售主动发问，还是客户主动透露），判为"遗漏"，输出
遗漏 / 影响 / 建议 三段式。纯规则、可确定性测试。
"""
from __future__ import annotations

from typing import Any


class MissingQuestionAnalyzer:
    """行业 Checklist 比对器（纯规则，无 LLM）。"""

    def __init__(self, settings) -> None:
        self.industry_checklist = settings.evaluation.modules.missing_question.industry_checklist

    def analyze(self, conversation: list[dict], case: dict) -> list[dict]:
        industry = (case or {}).get("industry", "") or ""
        checklist = self.industry_checklist.get(industry, [])
        transcript = _transcript_text(conversation)
        missing: list[dict] = []
        for item in checklist:
            if not any(kw in transcript for kw in item.keywords):
                missing.append({
                    "item": item.item,
                    "impact": item.impact,
                    "suggestion": item.suggestion,
                })
        return missing


def _transcript_text(conversation: list[dict]) -> str:
    return "\n".join(
        str(m.get("content", "")) for m in (conversation or [])
    )
