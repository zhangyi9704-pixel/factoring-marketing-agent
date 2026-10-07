"""roleplay 节点：load_case / init_customer_state（TDD §10.4）。

load_case 从 CaseService 取 agent 全量视角 Case State，并顺带产出
public_case（销售视角，技术设计文档（新）§6.1——briefing 节点输入专用）；
init_customer_state 生成客户开场白（turn 0）并初始化循环状态，
幂等（initialized 标记防重复执行，checkpoint 恢复安全）。
"""
from __future__ import annotations

import logging
import re
from app.services.session_service import SessionServiceError

from app.prompts.roleplay_prompt import OPENING_SYSTEM_PROMPT, build_opening_prompt
from app.schemas.case import to_public_case, assert_case_text_safe, CaseLeakError
from app.services.case_service import CaseService
from app.engine.fact_cards import unlock_fact_cards
from app.engine.leak_guard import check_reply, SAFE_REPLY
from app.services.external_calls import request_deadline, check_deadline

logger = logging.getLogger(__name__)

DEGRADED_OPENING = "你好，请先简单介绍一下你们的业务。"


def make_load_case_node(case_service: CaseService):
    """load_case：case_id → 完整 Case State（agent 视角，§10.3 all）。"""

    def load_case(state: dict) -> dict:
        case_id = state.get("case_id", "")
        try:
            case = case_service.get_case(case_id)
        except Exception as exc:                       # 基础设施故障留痕
            logger.exception("load_case 失败: %s", case_id)
            return {
                "errors": [*state.get("errors", []), f"load_case: {exc}"],
                "finished": True, "finish_reason": "error",
            }
        if case is None:
            return {
                "errors": [*state.get("errors", []), f"案例不存在: {case_id}"],
                "finished": True, "finish_reason": "error",
            }
        return {
            "case": case.model_dump(),
            "public_case": to_public_case(case).model_dump(),   # 销售视角（§7.3）
        }

    return load_case


def _only_customer_questions(opening, hits):
    """Only exempt a lexical warning when every matching clause is a question.

    This output-role exception requires a successful negative L2 decision. It
    does not change the sales-input severity policy or exempt a promise followed
    by a question/disclaimer.
    """
    clauses = re.findall(r"[^。！？?\n]+[。！？?\n]?", opening)
    return all(any(h.snippet in clause for clause in clauses) and
        all(clause.rstrip().endswith(("?", "？")) for clause in clauses if h.snippet in clause)
        for h in hits)


def make_init_customer_state_node(llm, session_service=None, *, guardrail=None, timeout_seconds=120):
    """init_customer_state：开场白 + 循环初始状态（turn=0，已释放=空）。"""

    def _init_customer_state(state: dict) -> dict:
        if state.get("initialized") or state.get("finished"):
            return {}                                  # checkpoint 恢复幂等
        case = state.get("case") or {}
        errors = list(state.get("errors", []))
        review_reasons = []
        review_id = state.get("compliance_escalation_id")
        try:
            if case.get("opening_line"):
                opening = case["opening_line"]
            else:
                check_deadline()
                opening = llm.chat(
                    build_opening_prompt(case),
                    model_role="chat", system=OPENING_SYSTEM_PROMPT,
                ).strip()
        except Exception as exc:
            error_text = type(exc).__name__
            logger.warning("开场白生成失败，使用降级文案: %s", error_text)
            errors.append(f"init_customer_state: {error_text}")
            opening = DEGRADED_OPENING
            review_reasons.append("opening_generation_unavailable")

        # Do not self-whitelist opening_line, and never send a private rejected
        # opening to another cloud model or to the delivered conversation.
        fact_state = {"case": case, "release_whitelist": [], "released_info": [], "unlocked_ids": []}
        try:
            assert_case_text_safe(case, opening)
            if not check_reply(fact_state, opening).safe:
                raise CaseLeakError("opening_fact_rejected")
        except CaseLeakError:
            opening = SAFE_REPLY
            errors.append("init_customer_state: opening_fact_rejected")
            review_reasons.append("opening_fact_rejected")

        if not review_reasons and guardrail is not None and getattr(guardrail, "enabled", False):
            scan = guardrail.scan(opening) if hasattr(guardrail, "scan") else None
            hits = scan.hits if scan else guardrail.detect_input(opening)
            degraded = bool(scan and scan.l2_degraded)
            bad = any(h.severity == "block" or h.details.get("l2") for h in hits)
            lexical_question = bool(scan and not degraded and _only_customer_questions(opening, hits))
            if degraded or bad or (hits and not lexical_question):
                if session_service and hits:
                    for hit in hits:
                        details = {**hit.details, "phase": "customer_opening"}
                        session_service.record_compliance(state["session_id"], 0, hit.rule,
                            opening, action="rewrite", details=details)
                opening = SAFE_REPLY
                reason = "opening_compliance_unavailable" if degraded else "opening_compliance_rejected"
                errors.append("init_customer_state: " + reason)
                review_reasons.append(reason)

        if review_reasons and session_service:
            if hasattr(session_service, "mark_human_review"):
                session_service.mark_human_review(state["session_id"])
            if hasattr(session_service, "register_escalation") and "user_id" in state:
                review_id = session_service.register_escalation("compliance", "客户开场验证未完成，请人工确认。",
                    user_id=state["user_id"], session_id=state["session_id"], severity="warn",
                    context={"turn": 0, "phase": "customer_opening", "reasons": review_reasons})
        conversation = [{"role": "customer", "turn": 0, "content": opening}]
        if session_service:
            try:
                session_service.record_message(
                    state["session_id"], 0, "customer", opening,
                )
            except Exception as exc:
                raise SessionServiceError("opening_write_failed") from None
        facts = unlock_fact_cards({"case": case})
        return {
            **facts,
            "initialized": True, "turn": 0, "conversation": conversation,
            "mood": "neutral", "pending_directives": facts["release_whitelist"], "released_info": [],
            "guardrail_hit_count": 0, "needs_human_review": bool(state.get("needs_human_review")) or bool(review_reasons),
            "compliance_escalation_id": review_id,
            "errors": errors,
        }

    def init_customer_state(state: dict) -> dict:
        with request_deadline(timeout_seconds):
            return _init_customer_state(state)

    return init_customer_state
