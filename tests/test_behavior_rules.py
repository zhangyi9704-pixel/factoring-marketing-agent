"""behavior_rules DSL 解析与执行单元测试（TDD §10.4 state_judge 规则引擎）。"""
from app.agents.roleplay.behavior_rules import (
    evaluate_rules,
    parse_rules,
    sales_history_text,
)

INFRA_RULES = [
    "turn==2 -> 透露付款方为市交通集团",
    "turn>=4 -> 透露验收尚未完成（若销售已询问验收类问题）",
    "asked_about_对账 -> 视提问质量决定是否透露对账状态",
    "asked_about_质押 -> 仅在销售主动提供融资方案后透露已质押部分应收",
]


class TestParseRules:

    def test_all_infra_rules_parse(self):
        parsed = parse_rules(INFRA_RULES)
        assert len(parsed) == 4
        kinds = {r.kind for r in parsed}
        assert kinds == {"turn", "asked"}

    def test_turn_rule_fields(self):
        rule = parse_rules(["turn==2 -> 透露付款方"])[0]
        assert rule.kind == "turn" and rule.op == "==" and rule.turn == 2
        assert rule.action == "透露付款方"

    def test_turn_ge_rule(self):
        rule = parse_rules(["turn>=4 -> 透露验收"])[0]
        assert rule.op == ">=" and rule.turn == 4

    def test_asked_rule_keyword(self):
        rule = parse_rules(["asked_about_对账 -> 视提问质量决定"])[0]
        assert rule.kind == "asked" and rule.keyword == "对账"

    def test_unparsable_rule_skipped(self):
        parsed = parse_rules(["这条不是合法规则", "turn==2 -> 合法"])
        assert len(parsed) == 1 and parsed[0].action == "合法"

    def test_empty_and_none(self):
        assert parse_rules([]) == []
        assert parse_rules(None) == []          # type: ignore[arg-type]


class TestEvaluateRules:

    def test_turn_exact_turn_match(self):
        rules = parse_rules(["turn==2 -> 透露付款方"])
        assert evaluate_rules(rules, turn=2, sales_text="你好") == ["透露付款方"]
        assert evaluate_rules(rules, turn=1, sales_text="你好") == []
        assert evaluate_rules(rules, turn=3, sales_text="你好") == []   # 严格等于

    def test_turn_ge_match(self):
        rules = parse_rules(["turn>=4 -> 透露验收"])
        assert evaluate_rules(rules, turn=4, sales_text="x") == ["透露验收"]
        assert evaluate_rules(rules, turn=9, sales_text="x") == ["透露验收"]
        assert evaluate_rules(rules, turn=3, sales_text="x") == []

    def test_asked_about_current_turn(self):
        rules = parse_rules(["asked_about_对账 -> 视提问质量决定是否透露对账状态"])
        hit = ["视提问质量决定是否透露对账状态"]
        assert evaluate_rules(rules, turn=1, sales_text="咱们工程计量和对账做了吗") == hit
        assert evaluate_rules(rules, turn=1, sales_text="你们资金紧张吗") == []

    def test_asked_about_cumulative_history(self):
        rules = parse_rules(["asked_about_质押 -> 透露已质押"])
        assert evaluate_rules(
            rules, turn=3, sales_text="换个话题", cumulative_text="第1轮提到过质押的事",
        ) == ["透露已质押"]

    def test_duplicate_actions_deduped(self):
        rules = parse_rules([
            "turn==1 -> 同一条指令", "asked_about_关键词 -> 同一条指令",
        ])
        assert evaluate_rules(rules, turn=1, sales_text="含关键词") == ["同一条指令"]

    def test_rule_order_preserved(self):
        rules = parse_rules(INFRA_RULES)
        actions = evaluate_rules(
            rules, turn=4, sales_text="想了解对账和质押情况",
        )
        # 严格语义：turn==2 只在第 2 轮命中；asked 规则按关键词命中
        assert "视提问质量决定是否透露对账状态" in actions
        assert "仅在销售主动提供融资方案后透露已质押部分应收" in actions
        assert "透露付款方为市交通集团" not in actions


class TestSalesHistoryText:

    def test_only_sales_messages(self):
        conversation = [
            {"role": "customer", "turn": 0, "content": "开场"},
            {"role": "sales", "turn": 1, "content": "问对账"},
            {"role": "customer", "turn": 1, "content": "客户答"},
            {"role": "sales", "turn": 2, "content": "问质押"},
        ]
        assert sales_history_text(conversation) == "问对账\n问质押"

    def test_empty(self):
        assert sales_history_text([]) == ""
