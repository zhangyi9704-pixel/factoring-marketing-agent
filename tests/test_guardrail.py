"""Guardrail 违禁话术检测单元测试（TDD §14，纯规则无 LLM）。"""
import pytest

from app.config import Settings
from app.guardrail.detector import GuardrailService


def make_service() -> GuardrailService:
    return GuardrailService(Settings(_env_file=None))


VIOLATIONS = [
    ("我们保证这笔5000万的额度肯定能批下来", "保证融资额度"),
    ("放心，审批肯定能通过", "保证审批通过"),
    ("我保证一周内放款", "保证放款时间"),
    ("我们承诺利率固定在年化6%", "承诺固定利率"),
    ("这个利率绝对最低，保证给你最优惠", "承诺固定利率"),
    ("三天之内肯定到账", "保证放款时间"),
]

COMPLIANT = [
    "具体额度要根据审批结果来确定",
    "审批通过与否要看最终审核结果",
    "放款时间一般在一个星期左右，具体以流程为准",
    "利率会根据审批情况在合同里约定",
    "我们的产品是应收账款转让融资",
]


class TestDetect:

    @pytest.mark.parametrize("text,rule", VIOLATIONS)
    def test_violation_detected(self, text, rule):
        hits = make_service().detect(text)
        assert any(h.rule == rule for h in hits), f"应命中 {rule}: {text}"

    @pytest.mark.parametrize("text", COMPLIANT)
    def test_compliant_passes(self, text):
        assert make_service().detect(text) == []

    def test_multiple_rules_same_utterance(self):
        hits = make_service().detect("我保证额度批下来，利率也固定不变")
        rules = {h.rule for h in hits}
        assert "保证融资额度" in rules and "承诺固定利率" in rules

    def test_hit_carries_snippet_and_action(self):
        hit = make_service().detect("我保证一周内放款")[0]
        assert "放款" in hit.snippet
        assert hit.action == "log"

    def test_empty_text_passes(self):
        assert make_service().detect("") == []


class TestEscalate:

    def test_threshold_3(self):
        service = make_service()
        assert not service.should_escalate(2)
        assert service.should_escalate(3)
        assert service.should_escalate(5)

    def test_real_config_forbidden_rules(self):
        # config.yaml 四条规则全部有语义化正则
        service = make_service()
        assert set(service.forbidden) == {
            "保证融资额度", "保证审批通过", "保证放款时间", "承诺固定利率",
        }
