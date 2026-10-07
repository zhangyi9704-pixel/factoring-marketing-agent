"""2.5 规则数据/加载验收；不触网，不接入现有检测器或执行三层引擎。"""
import json
from pathlib import Path

import pytest
import yaml

from app.guardrail.rules import load_compliance_rules, RuleLoadError, RULE_IDS
from scripts.verify_compliance_rules import HOOKS, probe_patterns


def test_ten_rules_source_fields_and_compilable_patterns():
    rules = load_compliance_rules()
    assert len(rules.rules) == 10 and {r.rule_id for r in rules.rules} == RULE_IDS
    for rule in rules.rules:
        assert rule.category and rule.description and rule.action_description
        assert rule.patterns.semantic and rule.compliant_response
        assert len(rule.patterns.compiled_regex()) == len(rule.patterns.regex)
    assert rules.by_id("CMP-05").severity == rules.by_id("CMP-06").severity == "block"
    output_rule = rules.by_id("CMP-10")
    assert output_rule.severity == output_rule.action == "rewrite"
    assert output_rule.applies_to == "agent_output"
    assert set(output_rule.patterns.reuse_rules) == RULE_IDS - {"CMP-10"}
    assert "{原句}" not in output_rule.compliant_response


@pytest.mark.parametrize("hook,text,expected", HOOKS)
def test_three_hook_sales_examples(hook, text, expected):
    rules = load_compliance_rules()
    assert probe_patterns(rules, text) == expected
    assert all(hook in rules.by_id(rule_id).related_hooks for rule_id in expected)
    assert probe_patterns(rules, text, output=True) == expected | {"CMP-10"}


@pytest.mark.parametrize("rule_id", sorted(RULE_IDS))
def test_each_compliant_response_has_no_pattern_hits(rule_id):
    rules = load_compliance_rules()
    assert probe_patterns(rules, rules.by_id(rule_id).compliant_response, output=True) == set()


def test_document_context_is_required_and_both_word_orders_covered():
    rules = load_compliance_rules()
    assert "CMP-05" not in probe_patterns(rules, "我想办法安排拜访时间")
    assert "CMP-05" in probe_patterns(rules, "我想办法帮你处理单据")
    assert "CMP-05" in probe_patterns(rules, "合同资料我们包装一下")


def test_rule_loader_independent_of_config_and_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(load_compliance_rules().rules) == 10


@pytest.mark.parametrize("corrupt", ["missing_field", "duplicate", "count", "id", "severity", "action", "regex", "context", "output_scope", "reuse_missing", "reuse_self", "unknown_field"])
def test_invalid_rule_file_rejected_without_echoing_body(tmp_path, corrupt):
    payload = load_compliance_rules().model_dump(mode="json")
    first = payload["rules"][0]
    if corrupt == "missing_field": del first["patterns"]["semantic"]
    elif corrupt == "duplicate": payload["rules"][1]["rule_id"] = first["rule_id"]
    elif corrupt == "count": payload["rules"].pop()
    elif corrupt == "id": first["rule_id"] = "CMP-11"
    elif corrupt == "severity": payload["rules"][4]["severity"] = "warn"
    elif corrupt == "action": first["action"] = "delete_database"
    elif corrupt == "regex": first["patterns"]["regex"] = ["[INVALID_PRIVATE_SENTINEL"]
    elif corrupt == "context": payload["rules"][4]["patterns"]["context_keywords"] = []
    elif corrupt == "output_scope": payload["rules"][-1]["applies_to"] = "both"
    elif corrupt == "reuse_missing": payload["rules"][-1]["patterns"]["reuse_rules"].pop()
    elif corrupt == "reuse_self": payload["rules"][-1]["patterns"]["reuse_rules"][0] = "CMP-10"
    else: first["execute"] = "PRIVATE_SENTINEL"
    path = tmp_path / "bad_rules.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    with pytest.raises(RuleLoadError) as caught:
        load_compliance_rules(path)
    assert "bad_rules.yaml" in str(caught.value)
    assert "PRIVATE_SENTINEL" not in str(caught.value)


@pytest.mark.parametrize("text", ["rules: [", "!!python/object/apply:os.system ['PRIVATE_SENTINEL']", "[]"])
def test_malformed_yaml_and_unsafe_tags_fail_closed(tmp_path, text):
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(RuleLoadError): load_compliance_rules(path)


def test_missing_file_names_the_file(tmp_path):
    with pytest.raises(RuleLoadError, match="missing.yaml"):
        load_compliance_rules(tmp_path / "missing.yaml")
