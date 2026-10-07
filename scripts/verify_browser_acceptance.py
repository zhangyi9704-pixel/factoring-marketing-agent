"""只读核对浏览器创建的本步会话；不 invoke、不重放、不调用云模型。"""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.agents.roleplay.agent import RoleplayAgent
from app.services.session_service import SessionService
from app.models import Evaluation
from app.engine.leak_guard import check_reply
from app.prompts.roleplay_prompt import build_reply_prompt

TOKEN = "BLOCK_UI_34"
TARGET = Path("docs/validation/3.4-browser-session.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--phase", choices=["first", "before_block", "blocked", "final"], required=True)
    args = parser.parse_args()
    service = SessionService()
    row = service.get_session(args.session)
    assert row and row["user_id"] == 1001344 and row["case_id"] == "PHARMA_001"
    agent = RoleplayAgent()
    state = agent.graph.get_state(agent._config(args.session)).values
    messages = service.list_messages(args.session)
    audits = service.list_compliance(args.session)
    prior = json.loads(TARGET.read_text(encoding="utf-8")) if TARGET.exists() else {}
    assert not prior or prior["session_id"] == args.session
    snapshot = {key: state.get(key) for key in ("turn", "dim_asked_count", "unlocked_ids", "blocked_turns",
        "needs_human_review", "finished", "errors", "compliance_status")}
    prior.update(session_id=args.session, user_id=row["user_id"], case_id=row["case_id"],
                 conversation=messages, compliance_events=audits, db_needs_human_review=row["needs_human_review"])
    prior[args.phase] = snapshot
    assert TOKEN not in json.dumps(messages, ensure_ascii=False)
    prompt = build_reply_prompt(case=state["case"], conversation_window=state["conversation"], sales_input="继续交流",
        whitelist=state["release_whitelist"], released_info=state["released_info"])
    assert TOKEN not in prompt
    if args.phase == "first":
        private = {c["id"] for c in state["case"]["fact_cards"] if c["level"] != "disclosed"}
        assert state["turn"] == 1 and not private.intersection(state["unlocked_ids"])
        assert check_reply(state, state["customer_reply"]).safe
    elif args.phase == "before_block":
        assert state["turn"] == 2
        assert any(a["rule"] == "CMP-01" and a["action"] == "log" for a in audits)
        assert any(m["role"] == "sales" and "当天肯定到账" in m["content"] for m in messages)
    elif args.phase == "blocked":
        before = prior["before_block"]
        assert state["turn"] == before["turn"] and state["blocked_turns"] == 1
        assert state["dim_asked_count"] == before["dim_asked_count"] and state["unlocked_ids"] == before["unlocked_ids"]
        assert state["needs_human_review"] and row["needs_human_review"]
        assert any(a["rule"] == "CMP-05" and a["action"] == "block" and TOKEN in a["raw_text"] for a in audits)
        assert len(messages) == 5
    else:
        assert state["turn"] == 3 and state["finished"] and state["blocked_turns"] == 1 and len(messages) == 7
        assert state["unlocked_ids"] == prior["blocked"]["unlocked_ids"] and state["needs_human_review"]
        assert row["needs_human_review"]
        assert any(m["role"] == "sales" and "当天肯定到账" in m["content"] for m in messages)
        warn = next(a for a in audits if a["rule"] == "CMP-01" and a["action"] == "log")
        block = next(a for a in audits if a["rule"] == "CMP-05" and a["action"] == "block" and TOKEN in a["raw_text"])
        assert warn["details"]["severity"] == "warn"
        assert block["details"]["l1"] and block["details"]["l2"] and not block["details"]["l2_degraded"]
        assert all(a["details"]["l1_elapsed_ms"] >= 0 for a in (warn, block))
        with service.session_factory() as db:
            report = db.query(Evaluation).filter(Evaluation.session_id == args.session).one()
            prior["sales_score"] = report.sales_score
            prior["opportunity"] = report.opportunity
            assert report.sales_score["dimension_scores"]["compliance"]["gated_score"] == 0
        prior["checks"] = dict(warn_delivered=True, block_excluded_from_history=True, block_excluded_from_counts=True,
            independent_audit_retained=True, customer_does_not_know_blocked_token=True,
            fresh_process_same_unlock_state=True, review_flag_persistent=True, block_compliance_score_zero=True)
        prior["checks"].update(block_double_hit_verified=True, l1_actual_timing_recorded=True)
    TARGET.parent.mkdir(exist_ok=True)
    TARGET.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] phase={args.phase}; turn={state['turn']}; messages={len(messages)}; audits={len(audits)}; review={state.get('needs_human_review')}")


if __name__ == "__main__":
    main()
