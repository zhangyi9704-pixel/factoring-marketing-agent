"""复盘报告渲染单元测试（纯函数，TDD §20 Phase 4）。"""
from app.evaluation.report import render_markdown


def make_report(**overrides):
    report = {
        "session_id": "sess_r", "generated_at": "2026-09-06T12:00:00",
        "opportunity": {"score": 82, "level": "高潜机会", "reasons": ["付款主体明确", "合同真实"]},
        "sales_score": {"total": 74, "dimension_scores": {
            "需求挖掘": {"score": 80, "evidence_missing": False},
            "合规意识": {"score": 90, "evidence_missing": False},
        }},
        "missing_questions": [{"item": "验收状态", "impact": "未验收应收不成立", "suggestion": "索取验收资料"}],
        "next_actions": [{"action": "进入业务尽调", "priority": "high", "note": "尽快"}],
        "evidence": [{"dimension": "付款来源", "turn": 1, "score": 70.0,
                      "quote_sales": "付款方是谁？", "quote_customer": "交通集团", "comment": "明确"}],
        "total_score": 82.0,
    }
    report.update(overrides)
    return report


class TestRenderMarkdown:

    def test_all_sections_present(self):
        md = render_markdown(make_report())
        for section in ("# 复盘报告", "## 机会诊断", "## 销售能力评分",
                        "## 证据链", "## 遗漏问题", "## 下一步行动"):
            assert section in md
        assert "82" in md and "高潜机会" in md
        assert "付款主体明确" in md
        assert "付款方是谁？" in md

    def test_empty_sections_degrade_gracefully(self):
        md = render_markdown(make_report(
            opportunity={"score": None, "level": "", "reasons": []},
            missing_questions=[], next_actions=[], evidence=[],
        ))
        assert "（无证据项" in md
        assert "（无遗漏" in md
        assert "（暂无行动建议）" in md

    def test_evidence_missing_marked(self):
        md = render_markdown(make_report(sales_score={
            "total": 50, "dimension_scores": {"风险意识": {"score": 0, "evidence_missing": True}},
        }))
        assert "证据缺失" in md
