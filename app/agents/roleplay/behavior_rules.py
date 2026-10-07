"""历史 DSL 兼容工具（coach_graph 已不调用；仅保留旧解析测试）。

behavior_rules DSL 解析与执行（TDD §10.4 state_judge 的规则引擎）。

DSL 形如（cases/*.yaml behavior_rules 字段原文）::

    turn==2 -> 透露付款方为市交通集团
    turn>=4 -> 透露……
    asked_about_对账 -> 视提问质量决定是否透露对账状态
    asked_about_质押 -> 仅在销售主动提供融资方案后透露已质押部分应收

条件（左侧）是简单确定性判断——轮次比较 / 销售输入关键词命中，
按附录 A.1 不交给 LLM；action 文本（含"若销售已询问……""仅在……后"
等限定语）作为释放指令注入 customer_reply 提示词，由 chat 模型把握
释放时机。未释放的指令进入 pending_directives 跨轮保留，直至被释放。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_RULE_PATTERN = re.compile(
    r"^\s*(?:(turn)\s*(==|>=|>)\s*(\d+)|(asked_about)_(\S+?))\s*->\s*(.+?)\s*$"
)


@dataclass(frozen=True)
class BehaviorRule:
    """单条 behavior_rule（不可变，可安全跨轮复用）。"""

    raw: str
    kind: str                     # "turn" / "asked"
    op: str                       # "==" / ">=" / ">"（asked 规则为空串）
    turn: Optional[int]           # turn 规则的目标轮次
    keyword: Optional[str]        # asked 规则的匹配关键词（如 "对账"）
    action: str                   # 右侧释放指令全文

    def satisfied(self, turn: int, sales_text: str, *, cumulative_text: str = "") -> bool:
        """判断条件是否满足。

        Args:
            turn: 本轮轮次（第 N 轮销售发言为 N）。
            sales_text: 本轮销售输入。
            cumulative_text: 本轮 + 历史销售输入拼接（asked_about 曾问过即算）。
        """
        if self.kind == "turn":
            if self.turn is None:
                return False
            if self.op == "==":
                return turn == self.turn
            if self.op == ">=":
                return turn >= self.turn
            return turn > self.turn
        keyword = self.keyword or ""
        return bool(keyword) and (keyword in sales_text or keyword in cumulative_text)


def parse_rules(rules: list[str]) -> list[BehaviorRule]:
    """解析 behavior_rules 列表；无法解析的条目告警跳过（不影响其余规则）。"""
    parsed: list[BehaviorRule] = []
    for raw in rules or []:
        match = _RULE_PATTERN.match(raw or "")
        if not match:
            logger.warning("behavior_rule 无法解析，已跳过: %r", raw)
            continue
        if match.group(1):                          # turn 条件
            parsed.append(BehaviorRule(
                raw=raw, kind="turn", op=match.group(2),
                turn=int(match.group(3)), keyword=None, action=match.group(6),
            ))
        else:                                       # asked_about_ 关键词
            parsed.append(BehaviorRule(
                raw=raw, kind="asked", op="",
                turn=None, keyword=match.group(5), action=match.group(6),
            ))
    return parsed


def evaluate_rules(
    rules: list[BehaviorRule],
    *,
    turn: int,
    sales_text: str,
    cumulative_text: str = "",
) -> list[str]:
    """返回本轮条件满足的 action 文本（按规则顺序，去重）。"""
    actions: list[str] = []
    for rule in rules:
        if rule.action not in actions and rule.satisfied(
            turn, sales_text, cumulative_text=cumulative_text,
        ):
            actions.append(rule.action)
    return actions


def sales_history_text(conversation: list[dict]) -> str:
    """拼接全部销售发言（asked_about_ 的累积匹配语料）。"""
    return "\n".join(
        m.get("content", "") for m in (conversation or [])
        if m.get("role") == "sales"
    )
