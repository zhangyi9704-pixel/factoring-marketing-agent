"""遗漏问题检测单元测试（行业 checklist 规则比对，TDD §13.5）。"""
from app.config import Settings
from app.evaluation.missing_question import MissingQuestionAnalyzer
from app.schemas.case import CaseState


def make_settings() -> Settings:
    return Settings(_env_file=None)


def make_case(industry="基建") -> CaseState:
    return CaseState(
        case_id="C_001", industry=industry, company="样本公司", client_role="财务总监",
        transaction_chain={"payer": "某集团"}, receivable={"amount": 1, "aging_days": 90},
        unknown_info=[], hidden_risks=[], behavior_rules=[],
    )


def conversation(*lines) -> list[dict]:
    return [{"role": "sales", "turn": i + 1, "content": line}
            for i, line in enumerate(lines)]


class TestMissingQuestionAnalyzer:

    def test_asked_payer_not_missing(self):
        # 问过付款方 → 付款主体 不遗漏；未问验收/质押/账期/融资需求
        conv = conversation("项目最终还是靠谁付钱啊？")
        result = MissingQuestionAnalyzer(make_settings()).analyze(conv, make_case().model_dump())
        items = {m["item"] for m in result}
        assert "付款主体" not in items
        assert {"验收状态", "质押担保", "账期与回款", "融资需求"} <= items

    def test_all_asked_no_missing(self):
        conv = conversation(
            "付款方是谁？", "验收完成了吗？", "账期和回款怎么样？", "有没有质押？", "资金缺口多大？",
        )
        result = MissingQuestionAnalyzer(make_settings()).analyze(conv, make_case().model_dump())
        assert result == []

    def test_unknown_industry_degrades_empty(self):
        case = make_case("其他").model_dump()
        result = MissingQuestionAnalyzer(make_settings()).analyze([], case)
        assert result == []
