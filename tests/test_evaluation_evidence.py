"""证据抽取单元测试（纯规则，无 LLM，TDD §13.4）。"""
from app.config import Settings
from app.evaluation.evidence import EvidenceExtractor, DIM_LABELS


def make_settings() -> Settings:
    return Settings(_env_file=None)


def make_evidence():
    return {
        "conversation": [
            {"role": "customer", "turn": 0, "content": "我们最近资金压力确实比较急。"},
            {"role": "sales", "turn": 1, "content": "王总，项目最终付款方是谁？回款来源确定吗？"},
            {"role": "customer", "turn": 1, "content": "是市交通集团，账期比较长。"},
            {"role": "sales", "turn": 2, "content": "那应收对账和验收完成了吗？"},
            {"role": "customer", "turn": 2, "content": "对账还没完全做完，验收也还差一层。"},
        ]
    }


class TestTopicEvidence:

    def test_extracts_best_turn_and_quotes(self):
        skills = make_settings()
        conv = make_evidence()["conversation"]
        result = EvidenceExtractor(skills).extract(conv)
        # payer: 付款方 + 回款来源 → turn 1；receivable: 应收/对账/验收 → turn 2
        payer = result["payer_confirmation"][0]
        assert payer["turn"] == 1
        assert "付款方" in payer["quote_sales"]
        receivable = result["receivable_quality"][0]
        assert receivable["turn"] == 2
        assert "验收" in receivable["quote_sales"]
        assert receivable["quote_customer"]

    def test_no_match_returns_empty(self):
        # 基建案例对话未涉风险/逾期/质押 → risk_identification 无证据
        conv = make_evidence()["conversation"]
        result = EvidenceExtractor(make_settings()).extract(conv)
        assert result["risk_identification"] == []
        assert "风险识别" in DIM_LABELS.values()

    def test_dimension_label_present(self):
        conv = make_evidence()["conversation"]
        result = EvidenceExtractor(make_settings()).extract(conv)
        assert result["payer_confirmation"][0]["dimension"] == "payer_confirmation"
        assert "付款来源" in result["payer_confirmation"][0]["comment"]


class TestComplianceEvidence:

    def test_no_violation_gives_positive_sample(self):
        conv = make_evidence()["conversation"]
        result = EvidenceExtractor(make_settings()).extract(conv, guardrail_hits=[])
        item = result["compliance"][0]
        assert "全程未出现违禁承诺" in item["comment"]
        assert item["turn"] == 2          # 最近销售轮

    def test_violation_cites_offending_turn(self):
        conv = make_evidence()["conversation"]
        conv[-2]["content"] = "保证一周内放款"
        hits = [{"turn": 2, "rule": "保证放款时间", "snippet": "保证一周内放款"}]
        result = EvidenceExtractor(make_settings()).extract(conv, guardrail_hits=hits)
        item = result["compliance"][0]
        assert item["turn"] == 2
        assert item["quote_sales"] == "保证一周内放款"
        assert "保证放款时间" in item["comment"]
