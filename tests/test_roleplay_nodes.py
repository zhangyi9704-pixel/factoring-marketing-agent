"""roleplay 节点单元测试（FakeLLM / FakeSession 注入，不触图不触库）。

测试替身：FakeLLM 区分 chat（开场白）/ chat_json（briefing 简报与
客户回复 JSON，按 system 提示词路由）两个调用面。
"""
import pytest

from app.agents.roleplay.case_loader import (
    make_init_customer_state_node,
    make_load_case_node,
)
from app.agents.roleplay.customer_reply import DEGRADED_REPLY, make_customer_reply_node
from app.agents.roleplay.guardrail_node import make_guardrail_node, make_update_state_node
from app.agents.roleplay.state_judge import make_state_judge_node
from app.config import Settings
from app.guardrail.detector import GuardrailService
from tests.test_case_schema import make_case

FAKE_BRIEFING_LLM_OUTPUT = {
    "visit_goals": ["确认交易背景真实性", "确认付款来源"],
    "priority_questions": [
        {"question": "项目业主是哪家？", "purpose": "确认付款主体",
         "risk_value": "回款来源", "order_rationale": "风险第一"},
    ],
    "possible_objections": [
        {"objection": "我们不需要融资", "response_hint": "从垫资压力切入"},
    ],
    "non_goals": ["不涉及具体利率"],
    "risk_warnings": [
        {"risk": "应收账款已质押", "basis": "公开信息未提及质押解除"},
    ],
}


class FakeLLM:
    """chat 返回开场白序列；chat_json 按 system 路由（briefing / 客户回复）。

    briefing 调用（system 含"访前简报"）返回固定简报 JSON，不消耗
    replies 队列；客户回复调用弹出 replies（Exception 则抛出）。
    """

    def __init__(self, openings=None, replies=None, briefing=None):
        self.openings = list(openings or ["您好，我们最近资金压力确实比较大。"])
        self.replies = list(replies or [])
        self.briefing = briefing if briefing is not None else dict(
            FAKE_BRIEFING_LLM_OUTPUT,
        )
        self.chat_prompts: list[str] = []
        self.json_prompts: list[str] = []

    def chat(self, prompt, *, model_role="chat", system=None, **kw):
        self.chat_prompts.append(prompt)
        return self.openings.pop(0) if self.openings else "默认开场白"

    def chat_json(self, prompt, *, model_role="reasoning", system=None, on_delta=None):
        self.json_prompts.append(prompt)
        if system and "访前简报" in system:
            return self.briefing
        item = self.replies.pop(0) if self.replies else {
            "reply": "嗯，你先说说你们的方案。", "revealed": [], "mood": "neutral",
        }
        if isinstance(item, Exception):
            raise item
        if on_delta is not None and isinstance(item, dict) and item.get("reply"):
            # 模拟流式：把 reply 的 JSON 文本分两段喂给回调（含围栏外的裸 JSON）
            import json as _json
            raw = _json.dumps({"reply": item["reply"], "revealed": [],
                               "mood": item.get("mood", "neutral")}, ensure_ascii=False)
            half = len(raw) // 2
            on_delta(raw[:half])
            on_delta(raw[half:])
        return item


class FakeRetriever:
    """检索替身：返回固定命中（RetrievalResult 语义），可注入异常。"""

    def __init__(self, items=None, missing=False, error=None):
        self.items = items or []
        self.missing = missing
        self.error = error
        self.queries: list[str] = []

    def retrieve(self, query, *, industry=None):
        self.queries.append(query)
        if self.error is not None:
            raise self.error

        class _Result:
            pass

        result = _Result()
        result.missing = self.missing
        result.items = self.items
        result.message = "知识库未命中，请转人工审核" if self.missing else None
        return result


class FakeCaseService:
    def __init__(self, case=None):
        self.case = case

    def get_case(self, case_id):
        return self.case

    def get_public_case(self, case_id):
        from app.schemas.case import to_public_case
        return to_public_case(self.case) if self.case else None


class FakeSessionService:
    def __init__(self):
        self.created: list[tuple] = []
        self.messages: list[tuple] = []
        self.compliance: list[tuple] = []
        self.touches: list[dict] = []
        self.briefings: dict[str, dict] = {}

    def create_session(self, session_id, user_id, case_id, *, phase="roleplay"):
        self.created.append((session_id, user_id, case_id))

    def record_message(self, session_id, turn, role, content,
                       *, revealed_info=None, guardrail_hit=None):
        self.messages.append((turn, role, content, revealed_info, guardrail_hit))

    def record_compliance(self, session_id, turn, rule, raw_text, *, action="log"):
        self.compliance.append((turn, rule, raw_text))

    def touch_session(self, session_id, turn_count, *, finished=False, phase=None):
        self.touches.append({"turn": turn_count, "finished": finished})

    def save_briefing(self, session_id, briefing):
        self.briefings[session_id] = briefing

    def get_briefing(self, session_id):
        return self.briefings.get(session_id)


def base_state(**overrides) -> dict:
    state = {
        "session_id": "sess_test", "user_id": 1, "case_id": "INFRA_001",
        "case": make_case().model_dump(), "initialized": True, "turn": 0,
        "conversation": [{"role": "customer", "turn": 0, "content": "开场白"}],
        "sales_input": "你们资金压力大吗？", "release_whitelist": [],
        "pending_directives": [], "released_info": [], "errors": [],
    }
    state.update(overrides)
    return state


# =====================================================
# load_case / init_customer_state
# =====================================================

class TestLoadCase:

    def test_load_ok(self):
        node = make_load_case_node(FakeCaseService(make_case()))
        result = node({"case_id": "INFRA_001"})
        assert result["case"]["company"] == "华建建设有限公司"
        assert result["public_case"]["company"] == "华建建设有限公司"  # 双视角同产
        assert "unknown_info" not in result["public_case"]             # 内部字段隔离

    def test_case_missing_marks_error(self):
        node = make_load_case_node(FakeCaseService(None))
        result = node({"case_id": "NOPE"})
        assert result["finished"] is True and result["finish_reason"] == "error"
        assert any("案例不存在" in e for e in result["errors"])


class TestInitCustomerState:

    def test_opening_generated_and_recorded(self):
        llm = FakeLLM(openings=["张经理你好，最近垫资压力不小啊。"])
        recorder = FakeSessionService()
        node = make_init_customer_state_node(llm, recorder)
        result = node(base_state(initialized=False, conversation=[]))
        assert result["initialized"] is True and result["turn"] == 0
        assert result["conversation"][0]["content"] == "张经理你好，最近垫资压力不小啊。"
        assert recorder.messages[0][:3] == (0, "customer", "张经理你好，最近垫资压力不小啊。")

    def test_llm_failure_degrades(self):
        class Boom:
            def chat(self, *a, **kw):
                raise RuntimeError("LLM 挂了")

        node = make_init_customer_state_node(Boom(), None)
        result = node(base_state(initialized=False, conversation=[]))
        assert "开场白生成失败" not in result["conversation"][0]["content"]
        assert result["conversation"][0]["content"]       # 有降级文案
        assert any("init_customer_state" in e for e in result["errors"])

    def test_idempotent_on_resume(self):
        node = make_init_customer_state_node(FakeLLM(), None)
        assert node(base_state(initialized=True)) == {}


# =====================================================
# state_judge（纯规则）
# =====================================================

class TestStateJudge:
    def test_fact_cards_replace_turn_dsl(self):
        from tests.test_fact_engine import card
        state = base_state(turn=1)
        state["case"]["fact_cards"] = [card()]
        assert make_state_judge_node()(state)["release_whitelist"] == []
        state["sales_input"] = "付款方是谁"
        assert make_state_judge_node()(state)["release_whitelist"] == ["INF-01"]

    def test_finished_skips(self):
        assert make_state_judge_node()(base_state(finished=True)) == {}


# =====================================================
# customer_reply
# =====================================================

class TestCustomerReply:

    def test_reply_parsed(self):
        llm = FakeLLM(replies=[{"reply": "还行，就是回款慢。", "revealed": [],
                                "mood": "impatient"}])
        node = make_customer_reply_node(llm)
        result = node(base_state())
        assert result["customer_reply"] == "还行，就是回款慢。"
        assert result["mood"] == "impatient" and result["revealed_this_turn"] == []

    def test_revealed_outside_whitelist_rejected(self):
        llm = FakeLLM(replies=[{
            "reply": "付款方是市交通集团。",
            "revealed": ["透露付款方", "白名单外的编造条目"],
            "mood": "neutral",
        }])
        node = make_customer_reply_node(llm)
        result = node(base_state(release_whitelist=["透露付款方"]))
        assert result["revealed_this_turn"] == ["透露付款方"]
        assert "customer_reply: revealed_outside_whitelist count=1" in result["errors"]

    def test_llm_failure_degraded_no_release(self):
        llm = FakeLLM(replies=[RuntimeError("超时")])
        node = make_customer_reply_node(llm)
        result = node(base_state(release_whitelist=["透露付款方"]))
        assert result["customer_reply"] == DEGRADED_REPLY
        assert result["revealed_this_turn"] == []
        assert any("customer_reply" in e for e in result["errors"])

    def test_invalid_mood_defaults_neutral(self):
        llm = FakeLLM(replies=[{"reply": "嗯。", "revealed": [], "mood": "暴怒"}])
        node = make_customer_reply_node(llm)
        assert node(base_state())["mood"] == "neutral"

    def test_memory_window_trims_prompt(self):
        long_history = [
            {"role": "sales" if i % 2 else "customer", "turn": i // 2 + 1,
             "content": f"消息{i}"}
            for i in range(40)
        ]
        llm = FakeLLM()
        node = make_customer_reply_node(llm, window=3)    # 最近 3 轮 = 6 条
        node(base_state(conversation=long_history))
        assert "消息33" not in llm.json_prompts[0]        # 窗口外不进提示词
        assert "消息34" in llm.json_prompts[0]

    def test_fact_constraint_in_system(self):
        llm = FakeLLM()
        node = make_customer_reply_node(llm)
        node(base_state())
        # system prompt 在 chat 层传递，这里验证提示词含白名单与销售发言
        assert "本轮可释放信息" in llm.json_prompts[0]
        assert "你们资金压力大吗？" in llm.json_prompts[0]


# =====================================================
# guardrail / update_state
# =====================================================

class TestGuardrailNode:

    def test_violation_recorded(self):
        recorder = FakeSessionService()
        node = make_guardrail_node(GuardrailService(Settings(_env_file=None)), recorder)
        result = node(base_state(sales_input="我保证额度肯定批下来"))
        assert [h["rule"] for h in result["guardrail_hits"]] == ["保证融资额度"]
        assert result["guardrail_hit_count"] == 1
        assert recorder.compliance and recorder.compliance[0][1] == "保证融资额度"

    def test_cumulative_escalation(self):
        node = make_guardrail_node(GuardrailService(Settings(_env_file=None)), None)
        result = node(base_state(sales_input="保证审批通过，保证放款时间", guardrail_hit_count=1))
        assert result["needs_human_review"] is True      # 1 + 2 = 3 → 升级
        assert result["guardrail_hit_count"] == 3

    def test_clean_input(self):
        node = make_guardrail_node(GuardrailService(Settings(_env_file=None)), None)
        result = node(base_state(sales_input="额度以审批结果为准"))
        assert result["guardrail_hits"] == [] and result["needs_human_review"] is False


class TestUpdateState:

    def make_node(self, recorder, max_turns=20):
        return make_update_state_node(max_turns=max_turns, session_service=recorder)

    def test_turn_advances_and_persists(self):
        recorder = FakeSessionService()
        node = self.make_node(recorder)
        result = node(base_state(
            customer_reply="回款慢，垫资多。", revealed_this_turn=["透露付款方"],
            release_whitelist=["透露付款方"], guardrail_hits=[],
        ))
        assert result["turn"] == 1 and result["finished"] is False
        assert [m["role"] for m in result["conversation"][1:]] == ["sales", "customer"]
        assert result["released_info"] == ["透露付款方"]
        assert result["pending_directives"] == []        # 已释放的不留 pending
        roles = [(m[1]) for m in recorder.messages]
        assert roles == ["sales", "customer"]
        assert recorder.messages[0][4] is None           # sales 行 guardrail_hit=None
        assert recorder.touches[-1] == {"turn": 1, "finished": False}

    def test_unreleased_moves_to_pending(self):
        recorder = FakeSessionService()
        node = self.make_node(recorder)
        result = node(base_state(
            customer_reply="嗯。", revealed_this_turn=[],
            release_whitelist=["透露付款方"],
        ))
        assert result["pending_directives"] == ["透露付款方"]

    def test_max_turns_finishes(self):
        node = self.make_node(FakeSessionService(), max_turns=2)
        result = node(base_state(turn=1, customer_reply="嗯。"))
        assert result["finished"] is True and result["finish_reason"] == "max_turns"

    def test_end_token_finalizes_without_new_messages(self):
        recorder = FakeSessionService()
        node = self.make_node(recorder)
        result = node(base_state(
            turn=3, finished=True, finish_reason="sales_end",
            conversation=[{"role": "customer", "turn": 0, "content": "开场"}],
        ))
        assert result["turn"] == 3                        # 不虚增
        assert len(result["conversation"]) == 1           # 不追加消息
        assert recorder.messages == []
        assert recorder.touches[-1] == {"turn": 3, "finished": True}


# =====================================================
# 流式：_reply_so_far 增量抽取 + 图外不传 on_delta
# =====================================================

class TestReplySoFar:

    def test_key_not_started_returns_empty(self):
        from app.agents.roleplay.customer_reply import _reply_so_far
        assert _reply_so_far('{"revealed": []') == ""

    def test_partial_value_returns_prefix(self):
        from app.agents.roleplay.customer_reply import _reply_so_far
        assert _reply_so_far('{"reply": "我们账期大概 18') == "我们账期大概 18"

    def test_closed_value_returns_full(self):
        from app.agents.roleplay.customer_reply import _reply_so_far
        assert _reply_so_far('{"reply": "180 天", "mood": "neutral"}') == "180 天"

    def test_common_escapes_restored(self):
        from app.agents.roleplay.customer_reply import _reply_so_far
        raw = '{"reply": "第一行\n第二行 \\"引号\\" \\ 反斜杠"}'
        assert _reply_so_far(raw) == '第一行\n第二行 "引号" \ 反斜杠'

    def test_key_after_other_fields(self):
        from app.agents.roleplay.customer_reply import _reply_so_far
        assert _reply_so_far('{"mood":"neutral","reply":"靠后"}') == "靠后"

    def test_monotonic_growth_only(self):
        from app.agents.roleplay.customer_reply import _reply_so_far
        # 追加字符只会让结果变长（节点据此取新增后缀）
        steps = ['{"rep', '{"reply": "账', '{"reply": "账期180', '{"reply": "账期180天"}']
        values = [_reply_so_far(s) for s in steps]
        assert values == ["", "账", "账期180", "账期180天"]


class TestCustomerReplyStreaming:

    def test_no_on_delta_outside_graph_context(self):
        """图外直调节点（无 runnable 配置）不向 LLM 传 on_delta。"""
        captured = {}

        class _LLM:
            def chat_json(self, prompt, *, model_role="chat", system=None,
                          on_delta=None):
                captured["on_delta"] = on_delta
                return {"reply": "好的", "revealed": [], "mood": "neutral"}

        node = make_customer_reply_node(_LLM())
        result = node(base_state())
        assert captured["on_delta"] is None              # writer 不可用 → 不传
        assert result["customer_reply"] == "好的"
