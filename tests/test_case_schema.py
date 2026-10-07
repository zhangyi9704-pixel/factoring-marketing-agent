"""CaseState / 双视角过滤 / 防泄漏校验单元测试（TDD §10.2 / §10.3）。"""
import pytest

from app.schemas.case import (
    CaseLeakError,
    CaseState,
    assert_no_internal_leak,
    to_full_case,
    to_public_case,
)


def make_case() -> CaseState:
    return CaseState(
        case_id="INFRA_001", industry="基建", company="华建建设有限公司",
        client_role="施工企业财务总监",
        basic_information={"企业性质": "民营施工总承包", "在手项目": "市政高架桥"},
        transaction_chain={"payer": "市交通集团", "contractor": "华建建设有限公司",
                           "supplier": None},
        receivable={"amount": 50000000, "aging_days": 180},
        pain_points=["回款周期长，供应商催款压力大"],
        known_risks=["政府项目审批链条长"],
        unknown_info=["是否完成竣工验收", "是否完成对账"],
        hidden_risks=["未计量部分约30%，应收不成立"],
        behavior_rules=["turn==2 -> 透露付款方", "asked_about_对账 -> 视提问质量决定"],
    )


class TestCaseState:

    def test_minimal_defaults(self):
        case = CaseState(case_id="X_001", industry="电商", company="C", client_role="R")
        assert case.unknown_info == [] and case.behavior_rules == []
        assert case.basic_information == {} and case.pain_points == []

    def test_missing_required_field(self):
        with pytest.raises(Exception):
            CaseState(case_id="X", industry="基建", company="C")  # 缺 client_role


class TestPublicCaseView:

    def test_internal_fields_never_leak(self):
        public = to_public_case(make_case()).model_dump()
        assert "unknown_info" not in public
        assert "hidden_risks" not in public
        assert_no_internal_leak(public)                 # 二次校验通过

    def test_known_facts_aggregation(self):
        public = to_public_case(make_case())
        # 客观事实带"键: 值"标注，让数字/信息含义一目了然
        assert "企业性质: 民营施工总承包" in public.known_facts
        assert "应收账款: 5000万元" in public.known_facts
        assert "账期: 180 天" in public.known_facts
        assert "付款方: 市交通集团" in public.known_facts
        assert "回款周期长，供应商催款压力大" in public.known_facts
        assert "政府项目审批链条长" in public.known_facts  # known_risks
        assert len(public.known_facts) == 7

    def test_transaction_chain_excludes_none(self):
        public = to_public_case(make_case())
        assert public.transaction_chain == {
            "payer": "市交通集团", "contractor": "华建建设有限公司"}
        assert public.receivable == {"amount": 50000000, "aging_days": 180}


class TestFullCaseView:

    def test_internal_marked(self):
        full = to_full_case(make_case())
        assert full["unknown_info"] == [
            {"content": "是否完成竣工验收", "visibility": "internal"},
            {"content": "是否完成对账", "visibility": "internal"},
        ]
        assert full["hidden_risks"][0]["visibility"] == "internal"
        # agent 视角信息完整（§10.3 allow all）
        assert full["behavior_rules"] == ["turn==2 -> 透露付款方",
                                          "asked_about_对账 -> 视提问质量决定"]


class TestLeakCheck:

    def test_nested_leak_detected(self):
        with pytest.raises(CaseLeakError, match="unknown_info"):
            assert_no_internal_leak({"a": {"b": [{"unknown_info": ["x"]}]}})

    def test_visibility_key_detected(self):
        with pytest.raises(CaseLeakError, match="visibility"):
            assert_no_internal_leak({"msg": {"visibility": "internal"}})

    def test_plain_data_passes(self):
        assert_no_internal_leak({"company": "X", "facts": ["a", 1],
                                 "nested": {"payer": "Y"}})

    def test_internal_words_in_values_do_not_trigger(self):
        # 仅查键名：值文本提到 unknown_info 不误报
        assert_no_internal_leak({"note": "unknown_info 字段不对外"})
