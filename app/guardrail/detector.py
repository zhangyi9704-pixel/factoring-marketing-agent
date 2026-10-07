"""违禁话术检测（TDD §14：规则检测——正则 + 关键词）。

检测范围（§14）：
    - 情景演练中**销售侧话术**（评估合规意识）——roleplay 每轮调用；
    - 系统输出（报告/建议）——Phase 4 Evaluation 接入。

config.guardrail.rules.forbidden 的 4 条规则（保证融资额度 / 保证审批通过 /
保证放款时间 / 承诺固定利率）是规则**名**，实际匹配用下方 PATTERNS 的
语义化正则（"保证……额度" / "审批……肯定能过" 等变体均命中）；
未知规则名回退为子串匹配。命中后动作由调用方决定（销售侧留痕计数，
累计 ≥3 升级 human_review；系统输出侧 action=rewrite 走 LLM 改写）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Pattern

from app.config import GuardrailSettings, Settings, get_settings

# 每条违禁规则 → 匹配正则列表（任一命中即违规；(?s) 跨句段近距共现）
PATTERNS: dict[str, list[Pattern[str]]] = {
    "保证融资额度": [
        re.compile(r"(保证|承诺|放心)[^。！？\n]{0,12}(额度|金额|规模|批下来|批出来|融到)"),
        re.compile(r"(额度|金额|规模)[^。！？\n]{0,10}(保证|没问题|肯定|稳)"),
    ],
    "保证审批通过": [
        re.compile(r"(保证|承诺|放心)[^。！？\n]{0,12}(审批|审核)[^。！？\n]{0,8}(通过|通过率|没问题)?"),
        re.compile(r"(审批|审核)[^。！？\n]{0,8}(肯定|一定|保证|包过|稳过|没问题)"),
    ],
    "保证放款时间": [
        re.compile(r"(保证|承诺|放心)[^。！？\n]{0,12}(放款|到账|下款|打款)"),
        re.compile(r"(放款|到账|下款)[^。！？\n]{0,10}(保证|肯定|一定|没问题|最快|之内)"),
        re.compile(r"(?<!不)(肯定|一定)[^。！？\n]{0,8}(放款|到账|下款|打款)"),
        re.compile(r"(最快|之内|当天)[^。！？\n]{0,8}(放款|到账|下款)"),
    ],
    "承诺固定利率": [
        re.compile(r"(承诺|保证|给到|做到)[^。！？\n]{0,10}(固定)?利率"),
        re.compile(r"利率[^。！？\n]{0,8}(固定不变|保证|承诺|最低|绝对)"),
    ],
}

# 合规表述特征（命中任一 PATTERNS 但含这些短语时豁免——以审批结果为准类话术）
EXEMPT_PATTERNS: list[Pattern[str]] = [
    re.compile(r"(以|按)?(最终)?(审批|审核)结果为准"),
    re.compile(r"利率?以(合同|最终|审批)?(约定|为准)"),
    re.compile(r"(无法|不能|没法)(保证|承诺)"),
]


@dataclass(frozen=True)
class GuardrailHit:
    """一次违禁命中（写入 t_conversation_message.guardrail_hit / t_compliance_log）。"""

    rule: str          # 违禁规则名（config.guardrail.rules.forbidden）
    snippet: str       # 命中片段（引用定位，最长 60 字）
    action: str        # 记录的动作：销售侧为 "log"（留痕计数），系统输出侧为 "rewrite"
    severity: str = "warn"
    compliant_response: str = ""
    details: dict = field(default_factory=dict)


class GuardrailService:
    """违禁话术规则检测器（纯规则，无 LLM 调用，每轮同步执行）。"""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self.enabled: bool = self.settings.guardrail.enabled
        self.forbidden: list[str] = self.settings.guardrail.rules.forbidden
        self.escalate_threshold: int = 3   # 同会话累计 ≥3 → human_review（TDD §14）
        self._block_rules = None

    def detect_input(self, text: str) -> list[GuardrailHit]:
        """3.2前置门禁：既有warn规则 + 已确定的CMP-05/06词法block；三层引擎待3.4。"""
        hits = self.detect(text)
        if not self.enabled or not text:
            return hits
        if self._block_rules is None:
            from app.guardrail.rules import load_compliance_rules
            self._block_rules = tuple(r for r in load_compliance_rules().rules if r.severity == "block")
        for rule in self._block_rules:
            patterns = rule.patterns
            if patterns.context_keywords and not any(k in text for k in patterns.context_keywords):
                continue
            matches = [re.search(re.escape(k), text) for k in patterns.keywords]
            matches += [pattern.search(text) for pattern in patterns.compiled_regex()]
            match = next((m for m in matches if m), None)
            if match:
                hits.append(GuardrailHit(rule.rule_id, match.group(0)[:60], "block", "block", rule.compliant_response))
        return hits

    def detect(self, text: str) -> list[GuardrailHit]:
        """检测一段话术，返回全部命中（无命中返回空列表）。

        disabled 或空文本直接放行；命中后仍检查豁免短语
        （"具体额度以审批结果为准" 属于合规表述，不误伤）。
        """
        if not self.enabled or not text:
            return []
        hits: list[GuardrailHit] = []
        for rule in self.forbidden:
            for pattern in PATTERNS.get(rule, [re.compile(re.escape(rule))]):
                match = pattern.search(text)
                if not match:
                    continue
                if any(ex.search(text) for ex in EXEMPT_PATTERNS):
                    continue
                snippet = match.group(0)
                hits.append(GuardrailHit(
                    rule=rule,
                    snippet=snippet if len(snippet) <= 60 else snippet[:60],
                    action="log",
                ))
                break                              # 同规则只记一次
        return hits

    def should_escalate(self, cumulative_hit_count: int) -> bool:
        """同会话累计命中是否达到 human_review 升级阈值。"""
        return self.enabled and cumulative_hit_count >= self.escalate_threshold


def get_guardrail_settings(settings: Settings) -> GuardrailSettings:
    """从 Settings 提取 guardrail 段（供节点工厂透传）。"""
    return settings.guardrail
