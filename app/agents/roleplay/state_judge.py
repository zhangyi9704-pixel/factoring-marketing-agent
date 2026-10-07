"""roleplay 节点：sales_input / state_judge（TDD §10.4）。

sales_input 通过 LangGraph interrupt() 挂起等待销售发言（HITL，
API 层用 Command(resume=content) 续跑）；[结束] 控制令直接标记
finished（§8.1 循环出口之一）。

state_judge 先执行销售侧合规检查（包含 L2 语义复核），再以纯规则
执行事实卡解锁；轮次比较与关键词解锁不交给 LLM。合并本轮新触发
与历史未释放，得到本轮释放白名单。
"""
from __future__ import annotations

from langgraph.types import interrupt

from app.engine.fact_cards import unlock_fact_cards

END_TOKEN = "[结束]"
_INTERRUPT_PAYLOAD = {"status": "waiting_sales_input"}


def sales_input(state: dict) -> dict:
    """挂起等待销售发言；返回值即 Command(resume=...) 注入的内容。"""
    text = str(interrupt(_INTERRUPT_PAYLOAD) or "").strip()
    if text == END_TOKEN:
        return {"sales_input": "", "finished": True, "finish_reason": "sales_end"}
    return {"sales_input": text}


def make_state_judge_node(guardrail=None):
    """合规前置门禁通过后，以纯规则计算事实卡释放白名单。"""

    def state_judge(state: dict) -> dict:
        if state.get("finished"):
            return {}
        if guardrail is None:
            from app.guardrail.detector import GuardrailService
            detector = GuardrailService()
        else:
            detector = guardrail
        raw = state.get("sales_input", "")
        scan = detector.scan(raw) if hasattr(detector, "scan") else None
        hits = scan.hits if scan else detector.detect_input(raw)
        payload = [dict(rule=h.rule, snippet=h.snippet, action=h.action,
                        severity=h.severity, compliant_response=h.compliant_response, details=h.details) for h in hits]
        blocked = any(h.severity == "block" for h in hits)
        preflight = {"guardrail_checked": True, "guardrail_hits": payload,
                     "input_blocked": blocked, "audit_input": raw if hits else ""}
        if scan:
            preflight["compliance_status"] = dict(l2_degraded=scan.l2_degraded,
                l1_elapsed_ms=scan.l1_elapsed_ms, l2_elapsed_ms=scan.l2_elapsed_ms)
        if blocked:
            return {**preflight, "sales_input": "", "release_whitelist": [], "sensitive_directives": [],
                    "blocked_turns": state.get("blocked_turns", 0) + 1}
        return {**unlock_fact_cards(state), **preflight}

    return state_judge
