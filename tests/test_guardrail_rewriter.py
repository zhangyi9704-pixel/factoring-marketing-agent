"""Guardrail 系统侧改写单元测试（TDD §14 action=rewrite）。"""
from app.config import Settings
from app.guardrail.detector import GuardrailService
from app.guardrail.rewriter import GuardrailRewriter


def make_guardrail() -> GuardrailService:
    return GuardrailService(Settings(_env_file=None))


class FakeLLM:
    def __init__(self, rewrite="具体额度以审批结果为准。", raises=False):
        self.rewrite = rewrite
        self.raises = raises
        self.calls: list[str] = []

    def chat(self, prompt, *, model_role="chat", system=None):
        self.calls.append(prompt)
        if self.raises:
            raise RuntimeError("LLM 挂了")
        return self.rewrite


class TestRewrite:

    def test_clean_text_unchanged_no_llm_call(self):
        llm = FakeLLM()
        result = GuardrailRewriter(make_guardrail(), llm).rewrite("建议完善付款主体调查")
        assert result.rewritten == "建议完善付款主体调查"
        assert result.violations == [] and llm.calls == []

    def test_violation_rewritten_and_rechecked(self):
        llm = FakeLLM()
        result = GuardrailRewriter(make_guardrail(), llm).rewrite("我们保证额度一周内放款")
        assert result.violations                        # 首次命中（待改写）
        assert "以审批结果为准" in result.rewritten
        assert result.remaining == []                   # 改写后复检干净

    def test_violation_recorded_via_violations(self):
        result = GuardrailRewriter(make_guardrail(), FakeLLM()).rewrite("保证马上放款")
        rules = {h.rule for h in result.violations}
        assert "保证放款时间" in rules

    def test_no_llm_keeps_original(self):
        result = GuardrailRewriter(make_guardrail(), None).rewrite("我们保证额度肯定批")
        assert result.rewritten == "我们保证额度肯定批"
        assert result.violations

    def test_llm_error_keeps_original(self):
        result = GuardrailRewriter(make_guardrail(), FakeLLM(raises=True)).rewrite("保证马上放款")
        assert result.rewritten == "保证马上放款"
        assert result.violations
