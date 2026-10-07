"""机会诊断 + 加权门控 + 等级映射单元测试（FakeLLM，TDD §13.2）。"""
import pytest

from app.config import Settings
from app.evaluation.common import build_dim_specs, map_level, weighted_gated_score
from app.evaluation.opportunity import OpportunityDiagnoser


def make_settings() -> Settings:
    return Settings(_env_file=None)


# 机会诊断 5 维（config 权重）
OPP_WEIGHTS = {
    "transaction_authenticity": 25, "receivable_quality": 25,
    "payer_confirmation": 20, "financing_need": 15, "risk_identification": 15,
}


class FakeLLM:
    """chat_json 返回固定维度分；Raises() 则抛异常。"""

    def __init__(self, dims=None, raises=None):
        self.dims = dims or {}
        self.raises = raises

    def chat_json(self, prompt, *, model_role="reasoning", system=None):
        if self.raises:
            raise self.raises
        return {"dimensions": self.dims}


def all_evidence(except_keys=()):
    """为全部评分维度构造非空证据（可排除若干维度模拟证据缺失）。"""
    keys = ["transaction_authenticity", "receivable_quality", "payer_confirmation",
            "financing_need", "risk_identification", "requirement_discovery",
            "business_understanding", "risk_awareness", "compliance"]
    return {k: [{"turn": 1, "quote_sales": "x", "quote_customer": "y"}] for k in keys
            if k not in except_keys}


class TestWeightedGatedScore:

    def test_weighted_sum(self):
        scores = {"transaction_authenticity": {"score": 90}, "receivable_quality": {"score": 80},
                  "payer_confirmation": {"score": 70}, "financing_need": {"score": 60},
                  "risk_identification": {"score": 50}}
        total, dims = weighted_gated_score(scores, OPP_WEIGHTS, all_evidence())
        assert total == 73.0
        assert not dims["risk_identification"]["evidence_missing"]

    def test_evidence_missing_zeroes_dimension(self):
        scores = {"transaction_authenticity": {"score": 90}, "receivable_quality": {"score": 80},
                  "payer_confirmation": {"score": 70}, "financing_need": {"score": 60},
                  "risk_identification": {"score": 50}}
        total, dims = weighted_gated_score(
            scores, OPP_WEIGHTS, all_evidence(except_keys=("risk_identification",)))
        assert total == 65.5                 # 缺证据维度贡献记 0
        assert dims["risk_identification"]["gated_score"] == 0.0
        assert dims["risk_identification"]["evidence_missing"] is True

    def test_llm_empty_scores_all_zero(self):
        total, dims = weighted_gated_score({}, OPP_WEIGHTS, all_evidence())
        assert total == 0.0
        assert all(d.get("gated_score") == 0.0 for d in dims.values())


class TestLevelMapping:

    def test_thresholds(self):
        assert map_level(82, high=80, observe=60) == "高潜机会"
        assert map_level(80, high=80, observe=60) == "高潜机会"
        assert map_level(79, high=80, observe=60) == "观察"
        assert map_level(60, high=80, observe=60) == "观察"
        assert map_level(59, high=80, observe=60) == "暂缓"


class TestOpportunityDiagnoser:

    def test_analyze_produces_score_level_reasons(self):
        llm = FakeLLM(dims={
            "transaction_authenticity": {"score": 90, "rationale": "有中标合同线索"},
            "receivable_quality": {"score": 80, "rationale": "应收成立"},
            "payer_confirmation": {"score": 70, "rationale": "付款方明确"},
            "financing_need": {"score": 60, "rationale": "缺口存在"},
            "risk_identification": {"score": 50, "rationale": "有逾期隐患"},
        })
        diagnoser = OpportunityDiagnoser(llm, make_settings())
        result = diagnoser.analyze([{"role": "sales", "turn": 1, "content": "x"}], {}, all_evidence())
        assert result["score"] == 73.0
        assert result["level"] == "观察"
        assert any("已验证销售原话" in r for r in result["reasons"])

    def test_evidence_missing_flagged_and_zeroed(self):
        llm = FakeLLM(dims={
            "transaction_authenticity": {"score": 90}, "receivable_quality": {"score": 80},
            "payer_confirmation": {"score": 70}, "financing_need": {"score": 60},
            "risk_identification": {"score": 50},
        })
        diagnoser = OpportunityDiagnoser(llm, make_settings())
        result = diagnoser.analyze([{"role": "sales", "turn": 1, "content": "x"}], {}, all_evidence(except_keys=("risk_identification",)))
        assert result["dimension_scores"]["risk_identification"]["evidence_missing"] is True
        assert any("证据缺失" in r for r in result["reasons"])

    def test_llm_failure_degrades(self):
        diagnoser = OpportunityDiagnoser(FakeLLM(raises=RuntimeError("LLM 挂了")), make_settings())
        result = diagnoser.analyze([{"role": "sales", "turn": 1, "content": "x"}], {}, all_evidence())
        assert result["score"] == 0.0
        assert result["errors"]
