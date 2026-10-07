"""3.2真实验收：未问到事实不披露、block不入史、独立审计与新进程恢复。

PowerShell项目根目录：.\.venv\Scripts\python.exe -u scripts\verify_leak_guard.py
仅打印脱敏指标；不重跑业务图，保留本次测试会话。
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.roleplay.agent import RoleplayAgent
from app.engine.leak_guard import check_reply
from app.models import ComplianceLog
from app.services.session_service import SessionService
from app.prompts.roleplay_prompt import build_reply_prompt

TOKEN = "BLOCKSECRET83"
BLOCKED_TEXT = "付款方 " + TOKEN + " 单据可以想办法"


def values(agent, sid):
    return agent.graph.get_state(agent._config(sid)).values


def resumed_process(sid):
    agent = RoleplayAgent()
    before = values(agent, sid)
    assert before["turn"] == 1 and before["blocked_turns"] == 1
    assert TOKEN not in json.dumps(before["conversation"], ensure_ascii=False)
    agent.send_message(sid, "你记得有哪个暗号吗？不清楚就说不清楚。")
    after = values(agent, sid)
    assert after["unlocked_ids"] == before["unlocked_ids"]
    assert after["dim_asked_count"] == before["dim_asked_count"]
    assert after["turn"] == 2 and TOKEN not in after["customer_reply"]
    assert check_reply(after, after["customer_reply"]).safe
    assert not any(e.startswith("customer_reply:") for e in after.get("errors", []))
    prompt = build_reply_prompt(case=after["case"], conversation_window=after["conversation"],
                                sales_input="继续交流", whitelist=after["release_whitelist"],
                                released_info=after["released_info"])
    assert TOKEN not in prompt
    print("PASS: new process / same unlocked state / no blocked text in memory or prompt", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-existing")
    args = parser.parse_args()
    if args.resume_existing:
        resumed_process(args.resume_existing)
        return
    agent = RoleplayAgent()
    created = agent.start_session("PHARMA_001", 1001)
    sid = created["session_id"]
    print("session=" + sid + "; mode=real cloud/MySQL/Redis", flush=True)
    agent.send_message(sid, "您好，先认识一下，您希望这次交流如何安排？")
    first = values(agent, sid)
    assert first["turn"] == 1
    private = {c["id"] for c in first["case"]["fact_cards"] if c["level"] != "disclosed"}
    assert not private.intersection(first["unlocked_ids"])
    assert check_reply(first, first["customer_reply"]).safe
    assert not any(e.startswith("customer_reply:") for e in first.get("errors", []))
    print("PASS: unasked private facts / fingerprints absent", flush=True)
    blocked = agent.send_message(sid, BLOCKED_TEXT)
    state = values(agent, sid)
    assert blocked["input_blocked"] and state["turn"] == 1
    assert state["dim_asked_count"] == first["dim_asked_count"]
    assert state["unlocked_ids"] == first["unlocked_ids"]
    assert state["conversation"] == first["conversation"]
    assert len(SessionService().list_messages(sid)) == 3
    with SessionService().session_factory() as session:
        audits = session.query(ComplianceLog).filter(ComplianceLog.session_id == sid).all()
        assert any(a.rule == "CMP-05" and a.action == "block" and a.raw_text == BLOCKED_TEXT for a in audits)
    print("PASS: blocked turn excluded / count unchanged / independent audit present", flush=True)
    # 真正的新Python进程，从现有interrupt续跑下一条正常消息，不重放任何已执行轮次。
    result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--resume-existing", sid],
                            check=False)
    assert result.returncode == 0
    final = agent.finish_session(sid)
    assert final["finished"] and final["turn"] == 2
    messages = SessionService().list_messages(sid)
    assert len(messages) == 5
    assert all(TOKEN not in m["content"] for m in messages)
    evidence = dict(date="2026-10-06", session_id=sid, case_id="PHARMA_001", unasked_leak_check="passed",
                    blocked_history="excluded", blocked_count="unchanged", independent_audit="preserved",
                    fresh_process_resume="passed", unlocked_state="consistent", blocked_memory="absent",
                    completed_turns=2, blocked_turns=1, mysql_messages=5, finished=True)
    path = Path(__file__).resolve().parents[1] / "docs/validation/3.2-leak-guard-live.json"
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PASS: finished / turns=2 / blocked=1 / MySQL 5 messages", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAIL: " + type(exc).__name__ + "; no graph retry", file=sys.stderr, flush=True)
        sys.exit(1)
