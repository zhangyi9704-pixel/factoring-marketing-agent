"""单 Agent 教练图全流程测试（InMemorySaver checkpoint + interrupt/Command 续跑）。

覆盖循环语义与 checkpoint 验收（中断后可恢复续跑）：
briefing（访前简报，§7）→ interrupt 挂起 → Command(resume) 注入销售发言
→ 一轮六节点 → 再挂起；[结束] / max_turns 两个出口；
checkpoint 跨 graph 实例恢复。
"""
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from app.config import Settings
from app.graphs.coach_graph import build_coach_graph
from app.guardrail.detector import GuardrailService
from tests.test_case_schema import make_case as legacy_case
from tests.test_fact_engine import card
from app.schemas.case import FactCard
from tests.test_roleplay_nodes import FakeCaseService, FakeLLM, FakeRetriever, FakeSessionService


def make_case():
    case = legacy_case()
    case.fact_cards = [FactCard(**card())] + [
        FactCard(**card(f"INF-{i:02}", triggers=[f"特殊触发{i}"], fact=f"测试事实{i}"))
        for i in range(2, 17)
    ]
    case.version = "test-v1"
    case.source = "unit fixture"
    case.opening_line = "最近垫资压力很大啊。"
    case.persona = {"性格": "谨慎"}
    case.payer_info = {"type": "企业"}
    return case


CONFIG = {"configurable": {"thread_id": "sess_graph_test"}}


def make_graph(llm, *, session_service=None, max_turns=None, checkpointer=None,
               retriever=None, case=None):
    settings = Settings(_env_file=None)
    if max_turns is not None:
        settings.agents.roleplay.max_turns = max_turns
    return build_coach_graph(
        llm,
        case_service=FakeCaseService(case or make_case()),
        guardrail=GuardrailService(settings),
        settings=settings,
        session_service=session_service or FakeSessionService(),
        retriever=retriever or FakeRetriever(),     # 单测不触 Milvus
        checkpointer=checkpointer or InMemorySaver(),
    )


def start(graph, config=CONFIG):
    return graph.invoke(
        {"session_id": "sess_graph_test", "user_id": 1, "case_id": "INFRA_001"}, config,
    )


class TestLoopBasics:

    def test_start_runs_briefing_then_interrupt_with_opening(self):
        graph = make_graph(FakeLLM(openings=["最近垫资压力很大啊。"]))
        result = start(graph)
        assert result["initialized"] is True and result["turn"] == 0
        assert result["conversation"] == [
            {"role": "customer", "turn": 0, "content": "最近垫资压力很大啊。"}]
        assert result["briefing"]["industry"] == "基建"          # 简报先于开场白产出
        assert result["briefing"]["case_id"] == "INFRA_001"
        assert graph.get_state(CONFIG).next == ("sales_input",)   # 挂起等待销售

    def test_one_turn_via_command_resume(self):
        graph = make_graph(FakeLLM())
        start(graph)
        result = graph.invoke(Command(resume="你们回款压力大吗"), CONFIG)
        assert result["turn"] == 1 and result["customer_reply"]
        assert [m["turn"] for m in result["conversation"]] == [0, 1, 1]
        assert [m["role"] for m in result["conversation"]] == ["customer", "sales", "customer"]
        assert graph.get_state(CONFIG).next == ("sales_input",)   # 一轮后继续挂起

    def test_asked_card_pending_released_next_turn(self):
        # 提问解锁后，未释放卡片跨轮保留
        llm = FakeLLM(replies=[
            {"reply": "嗯。", "revealed": [], "mood": "neutral"},
            {"reply": "付款方是市交通集团。", "revealed": ["INF-01"], "mood": "neutral"},
        ])
        graph = make_graph(llm)
        start(graph)
        graph.invoke(Command(resume="咱们项目付款方是谁"), CONFIG)
        result = graph.invoke(Command(resume="账期多久"), CONFIG)
        assert result["released_info"] == ["INF-01"]
        assert result["pending_directives"] == []

    def test_unreleased_directive_carried_to_pending(self):
        llm = FakeLLM(replies=[{"reply": "嗯。", "revealed": [], "mood": "neutral"}] * 2)
        graph = make_graph(llm)
        start(graph)
        graph.invoke(Command(resume="付款方是谁"), CONFIG)
        result = graph.invoke(Command(resume="再聊"), CONFIG)     # 第2轮未释放
        assert result["pending_directives"] == ["INF-01"]
        assert result["released_info"] == []

    def test_guardrail_hit_visible_in_state(self):
        graph = make_graph(FakeLLM())
        start(graph)
        result = graph.invoke(Command(resume="我保证额度肯定批下来"), CONFIG)
        assert [h["rule"] for h in result["guardrail_hits"]] == ["保证融资额度"]
        assert result["guardrail_hit_count"] == 1

    def test_messages_persisted_each_turn(self):
        recorder = FakeSessionService()
        graph = make_graph(FakeLLM(), session_service=recorder)
        start(graph)
        graph.invoke(Command(resume="你们资金紧张吗"), CONFIG)
        # turn0 开场 + 第1轮双向
        assert [(m[0], m[1]) for m in recorder.messages] == [
            (0, "customer"), (1, "sales"), (1, "customer")]
        assert recorder.touches[-1] == {"turn": 1, "finished": False}


class TestExits:

    def test_end_token_finishes(self):
        graph = make_graph(FakeLLM())
        start(graph)
        result = graph.invoke(Command(resume="[结束]"), CONFIG)
        assert result["finished"] is True
        assert result["finish_reason"] == "sales_end"
        assert graph.get_state(CONFIG).next == ()                 # 到达 END

    def test_max_turns_finishes(self):
        recorder = FakeSessionService()
        graph = make_graph(FakeLLM(), session_service=recorder, max_turns=1)
        start(graph)
        result = graph.invoke(Command(resume="你好"), CONFIG)
        assert result["finished"] is True and result["finish_reason"] == "max_turns"
        assert result["turn"] == 1
        assert recorder.touches[-1] == {"turn": 1, "finished": True}


class TestCheckpointResume:
    """中断及新图实例恢复验收。"""

    def test_progressive_count_survives_new_graph(self):
        fixture = make_case()
        fixture.fact_cards[0].progressive = True
        saver = InMemorySaver()
        graph1 = make_graph(FakeLLM(), checkpointer=saver, case=fixture)
        start(graph1)
        first = graph1.invoke(Command(resume="付款方是谁"), CONFIG)
        assert first["dim_asked_count"] == {"付款人": 1}
        assert first["release_whitelist"] == []
        graph2 = make_graph(FakeLLM(replies=[{"reply": "付款方是市交通集团。",
                                            "revealed": ["INF-01"], "mood": "neutral"}]), checkpointer=saver, case=fixture)
        second = graph2.invoke(Command(resume="再问一下业主是谁"), CONFIG)
        assert second["dim_asked_count"] == {"付款人": 2}
        assert second["released_info"] == ["INF-01"]
        assert second["pending_directives"] == []
        assert graph2.get_state(CONFIG).next == ("sales_input",)


    def test_state_survives_new_graph_instance(self):
        checkpointer = InMemorySaver()
        graph1 = make_graph(FakeLLM(), checkpointer=checkpointer)
        start(graph1)
        # 模拟重启：同一 checkpointer、全新 graph / LLM 实例
        graph2 = make_graph(FakeLLM(), checkpointer=checkpointer)
        snapshot = graph2.get_state(CONFIG)
        assert snapshot.next == ("sales_input",)
        assert snapshot.values["conversation"][0]["turn"] == 0     # 开场还在
        result = graph2.invoke(Command(resume="恢复后继续聊"), CONFIG)
        assert result["turn"] == 1                                  # 从 checkpoint 续跑

    def test_thread_isolation(self):
        other = {"configurable": {"thread_id": "sess_other"}}
        graph = make_graph(FakeLLM())
        start(graph)
        start(graph, config=other)                                  # 另一会话互不干扰
        assert graph.invoke(Command(resume="A的话"), CONFIG)["turn"] == 1
        assert graph.get_state(other).values["turn"] == 0
