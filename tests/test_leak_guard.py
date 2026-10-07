"""3.2：数字/专名/编号、一次重生成、缓冲流式及拦截后的客户记忆。"""
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.engine.leak_guard import SAFE_REPLY, check_reply, fingerprints
from app.engine.fact_cards import unlock_fact_cards
from app.agents.roleplay.customer_reply import make_customer_reply_node, BLOCK_REPLY
from tests.test_roleplay_graph import make_case, make_graph, start, CONFIG
from tests.test_roleplay_nodes import FakeLLM, FakeSessionService


def fixture_state():
    case = make_case()
    c = case.fact_cards[0]
    c.fact = "付款方是秘密星海医院，账款777万，结算T+7。"
    c.triggers = ["核查医院"]
    return dict(case=case.model_dump(), sales_input="你好", conversation=[],
                release_whitelist=[], released_info=[], unlocked_ids=[], errors=[])


@pytest.mark.parametrize("reply,reason", [("秘密星海医院", "locked_name"), ("金额777万。", "locked_number"),
                                        ("结算T+7", "locked_code"), ("融资999万", "unsupported_number")])
def test_unlocked_fingerprints_and_invented_numbers(reply, reason):
    assert reason in check_reply(fixture_state(), reply).reasons


def test_money_normalization_and_unit_boundaries():
    assert fingerprints("0.5亿元")["number"] == fingerprints("5000万")["number"]
    assert fingerprints("30天")["number"] != fingerprints("130天")["number"]
    assert fingerprints("Ｔ＋７")["code"] == {"t+7"}
    assert fingerprints("AB123")["code"] == {"ab123"}


def test_overlapping_disclosed_number_does_not_unlock_private_name():
    state = fixture_state()
    case = state["case"]
    case["fact_cards"][1].update(level="disclosed", fact="应收777万，账期30天")
    case["fact_cards"][0]["fact"] += "账期30天"
    assert check_reply(state, "应收777万，账期30天").safe
    assert not check_reply(state, "秘密星海医院，应收777万").safe


def test_whitelist_and_released_replies_are_allowed():
    state = fixture_state()
    state.update(unlocked_ids=["INF-01"], release_whitelist=["INF-01"])
    fact = state["case"]["fact_cards"][0]["fact"]
    assert check_reply(state, fact).safe
    state.update(release_whitelist=[], released_info=["INF-01"])
    assert check_reply(state, fact).safe


def test_regenerate_once_without_injecting_unsafe_answer(caplog):
    llm = FakeLLM(replies=[{"reply": "秘密星海医院欠款777万", "revealed": []},
                           {"reply": SAFE_REPLY, "revealed": []}])
    result = make_customer_reply_node(llm)(fixture_state())
    assert result["customer_reply"] == SAFE_REPLY
    assert len(llm.json_prompts) == 2
    assert "强约束重生成" in llm.json_prompts[1]
    assert "秘密星海医院" not in llm.json_prompts[1] and "777万" not in llm.json_prompts[1]
    assert "秘密星海医院" not in caplog.text


def test_second_leak_falls_back_without_release():
    llm = FakeLLM(replies=[{"reply": "编造999万", "revealed": []}] * 2)
    result = make_customer_reply_node(llm)(fixture_state())
    assert len(llm.json_prompts) == 2
    assert result["customer_reply"] == SAFE_REPLY and result["revealed_this_turn"] == []
    assert "leak_guard: safe_fallback" in result["errors"]
    assert any(error.startswith("leak_guard_reason: ") for error in result["errors"])


def test_regeneration_service_failure_has_no_third_attempt():
    llm = FakeLLM(replies=[{"reply": "秘密星海医院", "revealed": []}, RuntimeError("private error")])
    result = make_customer_reply_node(llm)(fixture_state())
    assert len(llm.json_prompts) == 2 and result["customer_reply"] == SAFE_REPLY
    assert result["revealed_this_turn"] == []
    assert "private error" not in str(result["errors"])


def test_stream_only_receives_final_safe_text(monkeypatch):
    events = []
    monkeypatch.setattr("app.agents.roleplay.customer_reply._stream_writer", lambda: events.append)
    llm = FakeLLM(replies=[{"reply": "777万", "revealed": []}] * 2)
    make_customer_reply_node(llm)(fixture_state())
    assert events == [{"kind": "customer_delta", "text": SAFE_REPLY}]


def test_block_is_never_delivered_counted_or_summarized_and_audit_keeps_evidence():
    saver = InMemorySaver()
    case = make_case()
    case.fact_cards[0].progressive = True
    llm = FakeLLM()
    recorder = FakeSessionService()
    graph = make_graph(llm, case=case, checkpointer=saver, session_service=recorder)
    start(graph)
    first = graph.invoke(Command(resume="付款方是谁"), CONFIG)
    assert first["dim_asked_count"]["付款人"] == 1
    calls = len(llm.json_prompts)
    forbidden = "付款方SECRET_BLOCK_829 单据可以想办法"
    denied = graph.invoke(Command(resume=forbidden), CONFIG)
    assert denied["input_blocked"] and denied["customer_reply"] == BLOCK_REPLY
    assert denied["turn"] == 1 and denied["blocked_turns"] == 1
    assert denied["dim_asked_count"]["付款人"] == 1
    assert len(llm.json_prompts) == calls
    assert denied["conversation"] == first["conversation"]
    assert denied["audit_input"] == "" and denied["sales_input"] == ""
    assert recorder.compliance[-1][1:] == ("CMP-05", forbidden)
    assert len(recorder.messages) == 3
    # 新图实例从同一checkpoint恢复，客户下一轮不知道被拦截的原话。
    fresh = FakeLLM()
    resumed = make_graph(fresh, case=case, checkpointer=saver, session_service=recorder)
    next_state = resumed.invoke(Command(resume="你好，继续聊"), CONFIG)
    assert "SECRET_BLOCK_829" not in fresh.json_prompts[-1]
    summary = "\n".join(m["content"] for m in next_state["conversation"])
    assert forbidden not in summary
    assert next_state["dim_asked_count"]["付款人"] == 1
    assert len(recorder.compliance) == 1
    assert next_state["input_blocked"] is False


def test_warn_is_delivered_and_block_context_gate():
    from app.guardrail.detector import GuardrailService
    from app.config import Settings
    guard = GuardrailService(Settings(_env_file=None))
    assert not guard.detect_input("我们再想办法沟通一下")
    assert any(h.severity == "block" for h in guard.detect_input("不用通知付款人"))
    graph = make_graph(FakeLLM())
    start(graph)
    state = graph.invoke(Command(resume="我保证融资额度"), CONFIG)
    assert state["input_blocked"] is False and state["turn"] == 1
    assert state["conversation"][1]["content"] == "我保证融资额度"


def test_public_block_event_contains_response_but_no_raw_evidence():
    from app.agents.roleplay.agent import RoleplayAgent
    from app.schemas.session import TurnResponse
    graph = make_graph(FakeLLM())
    start(graph)
    state = graph.invoke(Command(resume="SECRET_BLOCK_71 单据可以想办法"), CONFIG)
    public = TurnResponse(**RoleplayAgent._turn_result(state)).model_dump()
    assert public["input_blocked"]
    assert public["guardrail_events"][0]["severity"] == "block"
    assert public["guardrail_events"][0]["compliant_response"]
    assert "SECRET_BLOCK_71" not in str(public)
    assert "snippet" not in str(public) and "audit_input" not in str(public)


def test_audit_failure_does_not_write_blocked_conversation():
    class FailingAudit(FakeSessionService):
        def record_compliance(self, *args, **kwargs):
            raise RuntimeError("audit unavailable")
    from app.services.session_service import SessionServiceError
    recorder = FailingAudit()
    graph = make_graph(FakeLLM(), session_service=recorder)
    start(graph)
    with pytest.raises(SessionServiceError, match="compliance_write_failed"):
        graph.invoke(Command(resume="单据可以想办法"), CONFIG)
    assert len(recorder.messages) == 1
