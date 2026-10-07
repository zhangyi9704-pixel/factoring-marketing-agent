"""roleplay 节点：customer_reply（TDD §10.4，chat 模型 + fact_constraint）。

记忆窗口：仅将最近 window*2 条消息（一轮双向）注入提示词；
LLM 失败降级为保守回复（不释放任何信息）+ errors 留痕。
revealed 硬约束：白名单外的"释放申报"一律剔除（fact_constraint）。

流式（custom stream）：模型原始增量只缓冲，不发送给客户端；
完整校验和最多一次强约束重生成后，通过customer_delta发送最终安全回复。
图外调用writer为None，不影响主流程。
"""
from __future__ import annotations

import logging
import re

from app.prompts.roleplay_prompt import CUSTOMER_SYSTEM_PROMPT, build_reply_prompt
from app.engine.fact_cards import validate_revealed
from app.engine.leak_guard import check_reply, SAFE_REPLY, REGEN_CONSTRAINT

logger = logging.getLogger(__name__)

DEGRADED_REPLY = "这个我暂时不方便说，你先说说你们这边的方案吧。"
_VALID_MOODS = ("neutral", "positive", "impatient", "angry")

_REPLY_KEY = re.compile(r'"reply"\s*:\s*"')
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


def _stream_writer():
    """LangGraph custom stream writer；图外调用（RuntimeError）返回 None。"""
    try:
        from langgraph.config import get_stream_writer
        return get_stream_writer()
    except Exception:
        return None


def _reply_so_far(buffer: str) -> str:
    """从流式累计的原始 JSON 文本中抽取 reply 字段当前值。

    键尚未出现 → ""；值未闭合 → 已流出的部分（含常见转义还原）；
    值已闭合 → 完整值。追加字符只会让结果单调变长，调用方据此取后缀。
    """
    match = _REPLY_KEY.search(buffer)
    if not match:
        return ""
    rest = buffer[match.end():]
    out: list[str] = []
    i = 0
    while i < len(rest):
        ch = rest[i]
        if ch == "\\" and i + 1 < len(rest):
            out.append(_ESCAPES.get(rest[i + 1], "\\" + rest[i + 1]))
            i += 2
            continue
        if ch == '"':                                     # 值闭合
            break
        out.append(ch)
        i += 1
    return "".join(out)


BLOCK_REPLY = "该话术已拦截，请使用合规表述继续。"


def make_customer_reply_node(llm, *, window: int = 20, model_role: str = "chat", guardrail=None, session_service=None):
    """先完整校验，最多强约束重生成一次，再发送安全文本。"""
    def customer_reply(state: dict) -> dict:
        if state.get("finished"):
            return {"customer_reply": ""}
        writer = _stream_writer()

        def emit(reply):
            if writer is not None:
                try:
                    writer({"kind": "customer_delta", "text": reply})
                except Exception:
                    logger.warning("customer_delta channel unavailable")

        if state.get("input_blocked"):
            emit(BLOCK_REPLY)
            return {"customer_reply": BLOCK_REPLY, "revealed_this_turn": []}
        prompt = build_reply_prompt(
            case=state.get("case") or {},
            conversation_window=(state.get("conversation") or [])[-(window * 2):],
            sales_input=state.get("sales_input", ""),
            whitelist=state.get("release_whitelist") or [],
            released_info=state.get("released_info") or [],
        )
        errors = list(state.get("errors") or [])
        for attempt in range(2):
            try:
                data = llm.chat_json(
                    prompt + ("\n\n" + REGEN_CONSTRAINT if attempt else ""),
                    model_role=model_role, system=CUSTOMER_SYSTEM_PROMPT,
                    **({"on_delta": lambda piece: None} if writer is not None else {}),
                )
                if not isinstance(data, dict):
                    raise ValueError("invalid_reply_json")
            except Exception as exc:
                errors.append("customer_reply: " + type(exc).__name__)
                logger.warning("customer_reply: %s", type(exc).__name__)
                reply = SAFE_REPLY if attempt else DEGRADED_REPLY
                emit(reply)
                return {"customer_reply": reply, "revealed_this_turn": [],
                        "mood": "neutral", "errors": errors}
            reply = str(data.get("reply") or "").strip() or SAFE_REPLY
            output_filtered = False
            if guardrail is not None and hasattr(guardrail, 'protect_tree') and guardrail.enabled:
                # 完整输出先合规改写复检，再做未解锁指纹校验，最后发送delta。
                protected = {'reply': reply}
                checked = guardrail.protect_tree(protected, session_id=state.get('session_id', ''), session_service=session_service)
                # 标准合规应答面向销售，不能借改写让模拟客户变成销售或假称新事实。
                output_filtered = bool(checked and (checked.hits or checked.l2_degraded))
                reply = SAFE_REPLY if output_filtered else protected['reply']
            revealed, rejected = validate_revealed(data.get("revealed", []), state.get("release_whitelist") or [])
            if output_filtered:
                revealed = []  # 安全回退没有披露事实，不能消耗pending指令。
            if rejected:
                errors.append(f"customer_reply: revealed_outside_whitelist count={rejected}")
            result = check_reply(state, reply)
            if result.safe:
                emit(reply)
                return {"customer_reply": reply, "revealed_this_turn": revealed,
                        "mood": data.get("mood") if data.get("mood") in _VALID_MOODS else "neutral",
                        "errors": errors}
            # 只记录分类，既不记录泄漏指纹，也不把上一份危险回复带入重生成。
            logger.warning("leak_guard: rejected attempt=%d reasons=%s", attempt + 1, ",".join(result.reasons))
            errors.append("leak_guard: " + ("regenerate" if attempt == 0 else "safe_fallback"))
            errors.append("leak_guard_reason: " + ",".join(result.reasons))
        emit(SAFE_REPLY)
        return {"customer_reply": SAFE_REPLY, "revealed_this_turn": [],
                "mood": "neutral", "errors": errors}
    return customer_reply
