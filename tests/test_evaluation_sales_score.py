"""销售能力评分单元测试（FakeLLM + 证据门控，TDD §13.3）。"""
from app.config import Settings
from app.evaluation.sales_score import SalesScorer


def make_settings() -> Settings:
    return Settings(_env_file=None)


RUBRIC_WEIGHTS = {
    "requirement_discovery": 25, "business_understanding": 25,
    "risk_awareness": 25, "compliance": 25,
}


class FakeLLM:
    def __init__(self, dims=None, raises=None):
        self.dims = dims or {}
        self.raises = raises

    def chat_json(self, prompt, *, model_role="reasoning", system=None):
        if self.raises:
            raise self.raises
        return {"dimensions": self.dims}


def evidence(except_keys=()):
    keys = ["requirement_discovery", "business_understanding", "risk_awareness", "compliance"]
    return {k: [{"turn": 1, "quote_sales": "x", "quote_customer": "y"}] for k in keys
            if k not in except_keys}


class TestSalesScorer:

    def test_total_and_dimensions(self):
        llm = FakeLLM(dims={
            "requirement_discovery": {"score": 80, "rationale": "主动挖需求"},
            "business_understanding": {"score": 70, "rationale": "主线清晰"},
            "risk_awareness": {"score": 60, "rationale": "问了对账"},
            "compliance": {"score": 90, "rationale": "未违规"},
        })
        result = SalesScorer(llm, make_settings()).score([{"role": "sales", "turn": 1, "content": "x"}], {}, evidence())
        assert result["total"] == 75.0        # (80+70+60+90)/4
        assert result["dimension_scores"]["compliance"]["score"] == 90

    def test_evidence_missing_zeroes(self):
        llm = FakeLLM(dims={
            "requirement_discovery": {"score": 80}, "business_understanding": {"score": 70},
            "risk_awareness": {"score": 60}, "compliance": {"score": 90},
        })
        result = SalesScorer(llm, make_settings()).score(
            [{"role": "sales", "turn": 1, "content": "x"}], {}, evidence(except_keys=("risk_awareness",)))
        # (80 + 70 + 0 + 90) / 4 = 60
        assert result["total"] == 60.0
        assert result["dimension_scores"]["risk_awareness"]["evidence_missing"] is True

    def test_llm_failure_degrades(self):
        result = SalesScorer(FakeLLM(raises=RuntimeError("x")), make_settings()).score([{"role": "sales", "turn": 1, "content": "x"}], {}, evidence())
        assert result["total"] == 0.0
        assert result["errors"]
