"""事实卡解锁；状态仅含 JSON 类型，可随 coach_graph checkpoint 保存。"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def card_directive(card: dict) -> str:
    """只为已获准卡片展开事实和行为要求，不让模型决定解锁。"""
    action = "披露事实"
    if card["level"] == "sensitive":
        action = "按卡片语义提出异议、回避或反问；不得补充未获准事实"
    return f"[{card['id']}] {action}：{card['fact']}；表达提示：{card['hint']}"


def allowed_card_texts(case: dict, ids: list[str]) -> list[str]:
    return [card_directive(c) for c in case.get("fact_cards", []) if c["id"] in ids]


def unlock_fact_cards(state: dict) -> dict:
    """每个命中维度每轮只加一；解锁不等于模型已经释放。"""
    cards = (state.get("case") or {}).get("fact_cards") or []
    counts = dict(state.get("dim_asked_count") or {})
    unlocked = list(state.get("unlocked_ids") or [])
    fired = list(state.get("sensitive_fired") or [])
    sales = state.get("sales_input", "").casefold()
    hits = {c["id"] for c in cards
            if any(t and t.casefold() in sales for t in c["triggers"])}
    for dim in {c["dim"] for c in cards if c["id"] in hits}:
        counts[dim] = counts.get(dim, 0) + 1
    newly = []
    for card in cards:
        cid = card["id"]
        eligible = card["level"] == "disclosed" or (
            cid in hits and (not card["progressive"] or counts.get(card["dim"], 0) >= 2)
        )
        if not eligible or cid in unlocked:
            continue
        unlocked.append(cid)
        newly.append(cid)
        if card["level"] == "sensitive" and cid not in fired:
            fired.append(cid)
    released = set(state.get("released_info") or [])
    # 未知项、旧 DSL 文本、未解锁卡均不能通过 pending 绕过解锁。
    carried = [cid for cid in state.get("pending_directives") or []
               if cid in unlocked and cid not in released
               and any(c["id"] == cid for c in cards)]
    whitelist = list(dict.fromkeys(cid for cid in newly + carried if cid not in released))
    return {"dim_asked_count": counts, "unlocked_ids": unlocked,
            "sensitive_fired": fired, "release_whitelist": whitelist,
            "sensitive_directives": [card_directive(c) for c in cards
                                     if c["level"] == "sensitive" and c["id"] in whitelist]}


def validate_revealed(declared, whitelist: list[str]) -> tuple[list[str], int]:
    """硬不变量；只记录数量，不把模型文本或内部事实写入日志。"""
    if not isinstance(declared, list):
        logger.warning("fact_cards: revealed_outside_whitelist count=1")
        return [], 1
    valid = []
    rejected = 0
    for item in declared:
        if isinstance(item, str) and item in whitelist:
            if item not in valid:
                valid.append(item)
        else:
            rejected += 1
    if rejected:
        logger.warning("fact_cards: revealed_outside_whitelist count=%d", rejected)
    return valid, rejected
