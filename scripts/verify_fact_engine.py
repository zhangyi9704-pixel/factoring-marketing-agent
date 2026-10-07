"""本地真实云模型 / Redis checkpoint / MySQL 三轮演练；不重跑业务图。

PowerShell（项目根目录）：.\.venv\Scripts\python.exe scripts\verify_fact_engine.py
仅打印卡片 ID、计数和验收结果；保留验收会话，不删除业务记录。
"""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.roleplay.agent import RoleplayAgent
from app.services.session_service import SessionService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="真实YAML和图，外部组件用替身；不调用云、不写真实库")
    args = parser.parse_args()
    recorder = None
    if args.offline:
        from langgraph.checkpoint.memory import InMemorySaver
        from app.config import Settings
        from app.services.case_service import CaseService
        from tests.test_roleplay_nodes import FakeCaseService, FakeLLM, FakeRetriever, FakeSessionService
        case = next(c for c in CaseService().load_yaml_cases() if c.case_id == "PHARMA_001")
        target_fact = next(c.fact for c in case.fact_cards if c.id == "PHA-06")
        recorder = FakeSessionService()
        llm = FakeLLM(replies=[{"reply": "您好，先介绍一下吧。", "revealed": [], "mood": "neutral"},
                               {"reply": "结算细节需要再核实。", "revealed": [], "mood": "neutral"},
                               {"reply": target_fact, "revealed": ["PHA-06"], "mood": "neutral"}])
        agent = RoleplayAgent(Settings(_env_file=None), llm=llm, case_service=FakeCaseService(case),
                              retriever=FakeRetriever(), session_service=recorder, checkpointer=InMemorySaver())
    else:
        agent = RoleplayAgent()
    created = agent.start_session("PHARMA_001", 1001)
    sid = created["session_id"]
    print(f"session={sid}; mode={'offline graph + YAML / fakes' if args.offline else 'real cloud/MySQL/Redis'}")
    target = "PHA-06"
    inputs = ["您好，先认识一下，您希望这次交流如何安排？",
              "医院客户的账期通常多久？",
              "再追问一下账期，具体结算方式如何，银票和商票占比怎样？"]
    for index, text in enumerate(inputs, 1):
        agent.send_message(sid, text)
        state = agent.graph.get_state(agent._config(sid)).values
        assert not state.get("errors"), "node error/degraded; not a live pass"
        private = {c["id"] for c in state["case"]["fact_cards"] if c["level"] != "disclosed"}
        revealed = set(state.get("revealed_this_turn") or [])
        assert revealed <= set(state["release_whitelist"])
        if index == 1:
            assert not (private & set(state["unlocked_ids"]))
            assert not (private & revealed)
        elif index == 2:
            assert state["dim_asked_count"]["账期结算"] == 1
            assert target not in state["release_whitelist"]
            assert target not in revealed
        else:
            assert state["dim_asked_count"]["账期结算"] == 2
            assert target in state["release_whitelist"]
            assert target in revealed, "unlocked but real model did not declare disclosure"
        print(f"turn={index}; target_allowed={target in state['release_whitelist']}; "
              f"target_revealed={target in revealed}; whitelist_subset=PASS")
    final = agent.finish_session(sid)
    assert final["finished"] and final["turn"] == 3
    messages = ([dict(turn=m[0], role=m[1], revealed_info=m[3]) for m in recorder.messages]
                if recorder else SessionService().list_messages(sid))
    assert len(messages) == 7
    customer = [m for m in messages if m["role"] == "customer" and m["turn"] == 3]
    assert target in customer[0]["revealed_info"]
    print("PASS: 不问不披露 / 首问未解锁 / 二次追问披露 / interrupt / 结束不增轮 / 7条消息")
    if args.offline:
        print("External cloud / Redis / MySQL: NOT VERIFIED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}; no automatic graph retry", file=sys.stderr)
        sys.exit(1)
