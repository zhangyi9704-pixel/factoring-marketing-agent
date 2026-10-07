"""六类场景：公开、提问、递进、敏感、pending/未知隔离、申报硬校验。"""
import copy

import pytest

from app.engine.fact_cards import unlock_fact_cards, validate_revealed


def card(cid="INF-01", *, dim="付款人", level="on_ask", progressive=False,
         triggers=None, fact="付款方是市交通集团。", hint="问到付款方才披露"):
    return dict(id=cid, dim=dim, level=level, progressive=progressive,
                triggers=triggers if triggers is not None else ["付款方", "业主"],
                fact=fact, hint=hint)


def test_disclosed_first_turn_and_no_automatic_private_release():
    state = {"case": {"fact_cards": [card(level="disclosed"), card("INF-02")]}}
    assert unlock_fact_cards(state)["release_whitelist"] == ["INF-01"]
    state.update(unlock_fact_cards(state), turn=10)
    assert unlock_fact_cards(state)["release_whitelist"] == []


def test_on_ask_current_keywords_only():
    state = {"case": {"fact_cards": [card()]}, "sales_input": "你好",
             "conversation": [{"role": "sales", "content": "付款方是谁"}]}
    assert unlock_fact_cards(state)["release_whitelist"] == []
    state["sales_input"] = "业主是谁"
    assert unlock_fact_cards(state)["release_whitelist"] == ["INF-01"]


def test_progressive_dimension_once_per_turn_and_persistent_count():
    state = {"case": {"fact_cards": [card(progressive=True), card("INF-02", progressive=True)]},
             "sales_input": "付款方业主是谁"}
    original = copy.deepcopy(state)
    first = unlock_fact_cards(state)
    assert state == original
    assert first["dim_asked_count"] == {"付款人": 1}
    assert first["release_whitelist"] == []
    state.update(first, sales_input="你好", conversation=[])
    assert unlock_fact_cards(state)["dim_asked_count"] == {"付款人": 1}
    state.update(unlock_fact_cards(state), sales_input="业主配合吗")
    assert unlock_fact_cards(state)["release_whitelist"] == ["INF-01", "INF-02"]


@pytest.mark.parametrize("hint", ["提出费率异议", "回避财政问题", "反问其他费用"])
def test_sensitive_directive_and_one_time_firing(hint):
    state = {"case": {"fact_cards": [card(level="sensitive", hint=hint)]},
             "sales_input": "付款方是谁"}
    first = unlock_fact_cards(state)
    assert first["sensitive_fired"] == ["INF-01"]
    assert hint in first["sensitive_directives"][0]
    state.update(first, pending_directives=["INF-01"])
    assert unlock_fact_cards(state)["sensitive_fired"] == ["INF-01"]
    assert unlock_fact_cards(state)["sensitive_directives"] == first["sensitive_directives"]


def test_pending_union_release_removal_and_unknown_never_allowed():
    state = {"case": {"fact_cards": [card(), card("INF-02", dim="账期", triggers=["账期"])],
                      "unknown_info": ["需核实的事"]},
             "sales_input": "账期多久", "unlocked_ids": ["INF-01"],
             "pending_directives": ["INF-01", "INF-01", "需核实的事", "INF-99"]}
    assert unlock_fact_cards(state)["release_whitelist"] == ["INF-02", "INF-01"]
    state["released_info"] = ["INF-01"]
    assert unlock_fact_cards(state)["release_whitelist"] == ["INF-02"]
    state["pending_directives"] = ["INF-02"]
    state["sales_input"] = "你好"
    assert unlock_fact_cards(state)["release_whitelist"] == []


def test_revealed_hard_subset_and_redacted_warning(caplog):
    valid, count = validate_revealed(["INF-01", "private secret", 42, {"id": "INF-02"}, "INF-01"], ["INF-01"])
    assert valid == ["INF-01"] and count == 3
    assert "revealed_outside_whitelist" in caplog.text
    assert "private secret" not in caplog.text
    assert validate_revealed("INF-01", ["INF-01"]) == ([], 1)


def test_sensitive_progressive_requires_second_question():
    state = {"case": {"fact_cards": [card(level="sensitive", progressive=True)]},
             "sales_input": "业主是谁"}
    first = unlock_fact_cards(state)
    assert first["sensitive_fired"] == [] and first["sensitive_directives"] == []
    state.update(first)
    second = unlock_fact_cards(state)
    assert second["sensitive_fired"] == ["INF-01"]
    assert second["sensitive_directives"]
