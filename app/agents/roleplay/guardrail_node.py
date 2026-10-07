"""roleplay 节点：guardrail / update_state（TDD §10.4 / §14）。

guardrail：每轮检测**销售侧话术**（评估合规意识），命中写
t_compliance_log；同会话累计 ≥3 次置 needs_human_review（HITL 队列）。

update_state：双方向消息落库（customer 行带 revealed_info、sales 行带
guardrail_hit），轮次 +1，未释放白名单转入 pending_directives 跨轮保留，
达到 max_turns 触发循环出口（§8.1）。
"""
from __future__ import annotations

import logging
from app.services.session_service import SessionServiceError

from app.guardrail.detector import GuardrailService

logger = logging.getLogger(__name__)


def make_guardrail_node(guardrail: GuardrailService, session_service=None):
    """消费前置合规检查结果并留痕；未预检时调用注入的检测器。"""

    def guardrail_node(state: dict) -> dict:
        if state.get("finished"):
            return {}
        if state.get("guardrail_checked"):
            payload = list(state.get("guardrail_hits") or [])
        else:
            payload = [dict(rule=h.rule, snippet=h.snippet, action=h.action, severity=h.severity,
                            compliant_response=h.compliant_response, details=getattr(h,"details",{})) for h in guardrail.detect_input(state.get("sales_input", ""))]
        count = state.get("guardrail_hit_count", 0) + sum(h.get("severity", "warn") == "warn" for h in payload)
        needs_review = bool(state.get("needs_human_review")) or any(h.get("severity") == "block" for h in payload) or guardrail.should_escalate(count)
        escalation_id = state.get("compliance_escalation_id")
        if session_service and payload:
            turn = state.get("turn", 0) + 1
            for hit in payload:
                if hit.get("details"):
                    hit["details"]["phase"] = "sales_input"
                try:
                    registered = session_service.record_compliance(
                        state["session_id"], turn, hit["rule"],
                        state.get("audit_input") or state.get("sales_input", ""),
                        action=hit.get("action", "log"),
                        **({"details": hit["details"]} if hit.get("details") else {}),
                    )
                    if registered:
                        escalation_id = registered
                except Exception:
                    raise SessionServiceError("compliance_write_failed") from None
            if needs_review and hasattr(session_service, "mark_human_review"):
                session_service.mark_human_review(state["session_id"])
        if needs_review and not escalation_id:
            from app.services.escalation import register_escalation
            if session_service and hasattr(session_service, "latest_compliance_pending"):
                escalation_id = session_service.latest_compliance_pending(state["session_id"])
            else:
                escalation_id = register_escalation("compliance")
        return {
            "guardrail_hits": payload,
            "guardrail_hit_count": count,
            "needs_human_review": needs_review,
            "compliance_escalation_id": escalation_id,
        }

    return guardrail_node


def make_update_state_node(*, max_turns: int = 20, session_service=None):
    """update_state：状态推进 + 逐轮落库 + 循环出口判定。"""

    def update_state(state: dict) -> dict:
        finished_pre = bool(state.get("finished"))          # [结束] 控制令：不虚增轮次
        turn = state.get("turn", 0) if finished_pre else state.get("turn", 0) + 1
        finished = finished_pre
        finish_reason = state.get("finish_reason", "")
        errors = list(state.get("errors", []))

        blocked = bool(state.get("input_blocked"))
        if blocked and not finished_pre:
            turn = state.get("turn", 0)
            conversation = state.get("conversation") or []
            revealed = state.get("released_info") or []
            pending = state.get("pending_directives") or []
        elif not finished:
            conversation = list(state.get("conversation") or [])
            conversation.append({"role": "sales", "turn": turn,
                                 "content": state.get("sales_input", "")})
            conversation.append({"role": "customer", "turn": turn,
                                 "content": state.get("customer_reply", "")})
            revealed = list(state.get("released_info") or [])
            revealed += [d for d in state.get("revealed_this_turn") or []
                         if d not in revealed]
            pending = [d for d in state.get("release_whitelist") or []
                       if d not in (state.get("revealed_this_turn") or [])]
            if turn >= max_turns:
                finished, finish_reason = True, "max_turns"
        else:
            # [结束] 控制令：不追加对话，仅收敛会话
            conversation = state.get("conversation") or []
            revealed = state.get("released_info") or []
            pending = state.get("pending_directives") or []

        if session_service:
            try:
                if hasattr(session_service, 'persist_turn'):
                    session_service.persist_turn(dict(state, finish_reason=finish_reason), turn, finished=finished, blocked=blocked)
                elif not blocked and (not finished or finish_reason == "max_turns"):
                    session_service.record_message(
                        state["session_id"], turn, "sales", state.get("sales_input", ""),
                        guardrail_hit=state.get("guardrail_hits") or None,
                    )
                    session_service.record_message(
                        state["session_id"], turn, "customer", state.get("customer_reply", ""),
                        revealed_info=state.get("revealed_this_turn") or None,
                    )
                if not hasattr(session_service, 'persist_turn'):
                    session_service.touch_session(state["session_id"], turn, finished=finished)
            except Exception as exc:
                raise SessionServiceError("conversation_write_failed") from None

        return {
            "turn": turn, "conversation": conversation, "released_info": revealed,
            "pending_directives": pending, "finished": finished,
            "finish_reason": finish_reason, "errors": errors,
            "audit_input": "", "sales_input": "", "guardrail_checked": False,
        }

    return update_state
