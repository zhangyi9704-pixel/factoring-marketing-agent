"""2.4：真实三案例字段、来源分级、公开视图及发送前保护。"""
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from contextlib import contextmanager

from app.config import Settings
from app.schemas.case import CaseState, to_public_case, to_full_case, assert_no_internal_leak
from app.services.case_service import CaseService, CaseServiceError
from app.api.v1.case import get_case_service
from app.main import app
from app.engine.fact_cards import unlock_fact_cards
from app.agents.roleplay.case_loader import make_init_customer_state_node
from app.agents.roleplay.customer_reply import make_customer_reply_node, DEGRADED_REPLY
from app.agents.coach.briefing import make_briefing_node
from app.engine.leak_guard import SAFE_REPLY
from app.prompts.roleplay_prompt import build_reply_prompt

ROOT = Path(__file__).resolve().parents[1]
LEVELS = {"INFRA_001": (4, 9, 3), "PHARMA_001": (4, 8, 4), "ECOM_001": (3, 9, 4)}
REQUIRED = {"case_id", "industry", "company", "client_role", "basic_information", "persona",
            "transaction_chain", "payer_info", "receivable", "pain_points", "known_risks",
            "unknown_info", "hidden_risks", "fact_cards", "opening_line", "version", "source", "status"}


def cases():
    return CaseService().load_yaml_cases()


@pytest.mark.parametrize("case_id", list(LEVELS))
def test_case_complete_source_counts_hooks_and_legacy(case_id):
    case = next(c for c in cases() if c.case_id == case_id)
    data = case.model_dump()
    assert REQUIRED <= data.keys()
    assert all(data[key] for key in REQUIRED)
    assert len(case.fact_cards) == 16
    assert tuple(sum(c.level == level for c in case.fact_cards) for level in ("disclosed", "on_ask", "sensitive")) == LEVELS[case_id]
    assert all({"id", "dim", "fact", "level", "hint", "triggers", "progressive"} == c.model_dump().keys() for c in case.fact_cards)
    hook, phrase = {"INFRA_001": ("INF-16", "单据"), "PHARMA_001": ("PHA-15", "年化5%"), "ECOM_001": ("ECO-15", "500万")}[case_id]
    card = next(c for c in case.fact_cards if c.id == hook)
    assert card.level == "sensitive" and phrase in card.fact
    unlocked = unlock_fact_cards({"case": data, "sales_input": card.triggers[0]})
    assert hook in unlocked["release_whitelist"]
    assert hook in unlocked["sensitive_fired"]


@pytest.mark.parametrize("case_id", list(LEVELS))
def test_public_is_disclosed_only_and_nested_whitelisted(case_id):
    case = next(c for c in cases() if c.case_id == case_id)
    case.basic_information["企业性质"] = {"unknown_info": "SENTINEL_PRIVATE", "allowed": "SENTINEL_PRIVATE"}
    case.transaction_chain.description = "SENTINEL_CHAIN_PRIVATE"
    case.receivable.details["private"] = "SENTINEL_REC_PRIVATE"
    public = to_public_case(case).model_dump()
    assert_no_internal_leak(public)
    blob = json.dumps(public, ensure_ascii=False)
    assert "SENTINEL" not in blob
    assert public["known_facts"] == [c.fact for c in case.fact_cards if c.level == "disclosed"]
    assert public["receivable"] == {} and public["known_risks"] == []
    for card in case.fact_cards:
        if card.level != "disclosed":
            assert card.fact not in blob and card.hint not in blob
    for value in case.unknown_info + case.hidden_risks:
        assert value not in blob
    assert to_full_case(case)["fact_cards"] == case.model_dump()["fact_cards"]


@pytest.mark.parametrize("corrupt", ["count", "duplicate", "enum", "trigger", "source", "industry"])
def test_fact_case_validation_rejects_bad_payload(corrupt):
    data = cases()[0].model_dump()
    if corrupt == "count":
        data["fact_cards"].pop()
    elif corrupt == "duplicate":
        data["fact_cards"][1]["id"] = data["fact_cards"][0]["id"]
    elif corrupt == "enum":
        data["fact_cards"][0]["level"] = "internal"
    elif corrupt == "trigger":
        data["fact_cards"][-1]["triggers"] = []
    elif corrupt == "source":
        data["source"] = ""
    else:
        data["industry"] = "基建"
    with pytest.raises(ValidationError):
        CaseState.model_validate(data)


def test_legacy_nested_fields_are_also_whitelisted():
    case = CaseState(case_id="X", industry="基建", company="C", client_role="R",
                     basic_information={"企业性质": "民营", "secret_alias": {"arbitrary": "SENTINEL"},
                                        "在手项目": {"unknown_info": "SENTINEL"}},
                     receivable={"amount": 1, "details": {"secret_alias": "SENTINEL"}})
    assert "SENTINEL" not in json.dumps(to_public_case(case).model_dump())


def test_case_mysql_payload_roundtrip_public_api_whitelist(tmp_path):
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE t_case (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id VARCHAR(32) UNIQUE, industry VARCHAR(16), company VARCHAR(128), client_role VARCHAR(64), case_json JSON, enabled BOOLEAN, created_at DATETIME DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME DEFAULT CURRENT_TIMESTAMP)"))
    @contextmanager
    def sf():
        with Session(engine) as session:
            yield session
            session.commit()
    service = CaseService(session_factory=sf)
    assert service.sync_to_mysql()["inserted"] == 3
    assert service.sync_to_mysql()["updated"] == 3
    stored = service.list_cases()
    assert {c.case_id for c in stored} == set(LEVELS)
    assert all(len(c.fact_cards) == 16 for c in stored)
    # Use preloaded case models for API thread to avoid SQLite thread connection constraints.
    class ApiService:
        def list_cases(self, industry=None):
            return [c for c in stored if not industry or c.industry == industry]
        def get_public_case(self, case_id):
            return next((to_public_case(c) for c in stored if c.case_id == case_id), None)
    app.dependency_overrides[get_case_service] = lambda: ApiService()
    try:
        client = TestClient(app)
        response = client.get("/api/v1/cases")
        assert response.status_code == 200
        assert len(response.json()) == 3
        for item in response.json():
            assert_no_internal_leak(item)
            assert item["receivable"] == {}
        for case in stored:
            assert client.get("/api/v1/cases/" + case.case_id).json() == to_public_case(case).model_dump()
    finally:
        app.dependency_overrides.pop(get_case_service, None)


def test_private_case_not_in_prompts_and_deterministic_opening():
    case = cases()[0]
    prompt = build_reply_prompt(case=case.model_dump(), conversation_window=[], sales_input="你好", whitelist=[], released_info=[])
    assert all(c.fact not in prompt for c in case.fact_cards if c.level != "disclosed")
    assert all(value not in prompt for value in case.unknown_info)
    class NeverLLM:
        def chat(self, *args, **kwargs):
            raise AssertionError("modern opening must not call LLM")
    state = make_init_customer_state_node(NeverLLM())({"case": case.model_dump()})
    assert state["conversation"][0]["content"] == case.opening_line


def test_private_model_output_never_sent_or_logged(monkeypatch, caplog):
    import app.agents.roleplay.customer_reply as reply_module
    case = next(c for c in cases() if c.case_id == "INFRA_001")
    private = next(c.fact for c in case.fact_cards if c.id == "INF-09")
    events = []
    monkeypatch.setattr(reply_module, "_stream_writer", lambda: events.append)
    class LeakyLLM:
        def chat_json(self, prompt, **kwargs):
            kwargs["on_delta"](json.dumps({"reply": private}, ensure_ascii=False))
            return {"reply": private, "revealed": [], "mood": "neutral"}
    result = make_customer_reply_node(LeakyLLM())({"case": case.model_dump(), "sales_input": "你好"})
    assert result["customer_reply"] == SAFE_REPLY
    assert private not in json.dumps(events, ensure_ascii=False)
    assert private not in caplog.text
    assert result["revealed_this_turn"] == []


def test_briefing_blocks_private_generated_facts(caplog):
    case = next(c for c in cases() if c.case_id == "INFRA_001")
    private = next(c.fact for c in case.fact_cards if c.id == "INF-09")
    class LeakyLLM:
        def chat_json(self, *args, **kwargs):
            return {"visit_goals": [private]}
    class Missing:
        def retrieve(self, *args, **kwargs):
            class Result:
                missing, items, message = True, [], "待核实"
            return Result()
    result = make_briefing_node(LeakyLLM(), settings=Settings(_env_file=None), retriever=Missing())(
        {"case": case.model_dump(), "public_case": to_public_case(case).model_dump()})
    assert private not in json.dumps(result["briefing"], ensure_ascii=False)
    assert private not in caplog.text


def test_validation_errors_do_not_echo_internal_values(tmp_path):
    (tmp_path / "bad.yaml").write_text("case_id: TEST\ncompany: PRIVATE_SENTINEL\n", encoding="utf-8")
    with pytest.raises(CaseServiceError) as caught:
        CaseService(case_dir=tmp_path).load_yaml_cases()
    assert "PRIVATE_SENTINEL" not in str(caught.value)
