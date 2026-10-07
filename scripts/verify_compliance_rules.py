"""2.5 离线规则数据验收：样例探测仅用于施工验收，不是生产检测引擎。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.guardrail.rules import load_compliance_rules

HOOKS = (
    ("INF-16", "计量没做完没关系，单据可以想办法", {"CMP-05"}),
    ("PHA-15", "能做到年化4.8%", {"CMP-03"}),
    ("ECO-15", "3天肯定到账，先批500万没问题", {"CMP-01", "CMP-02"}),
)


def probe_patterns(rule_set, text: str, *, output=False) -> set[str]:
    """仅验证加载的数据能覆盖样例；无否定、引用、语义复核或生产处置。"""
    hits = set()
    for rule in rule_set.rules:
        if rule.rule_id == "CMP-10":
            continue
        patterns = rule.patterns
        if patterns.context_keywords and not any(word in text for word in patterns.context_keywords):
            continue
        if any(word in text for word in patterns.keywords) or any(pattern.search(text) for pattern in patterns.compiled_regex()):
            hits.add(rule.rule_id)
    if output and hits.intersection(rule_set.by_id("CMP-10").patterns.reuse_rules):
        hits.add("CMP-10")
    return hits


def verify():
    rule_set = load_compliance_rules()
    manifest = json.loads((ROOT / "docs/validation/2.5-rules-manifest.json").read_text(encoding="utf-8"))
    for path_key, hash_key in (("source_path", "source_sha256"), ("reference_path", "reference_sha256"), ("rules_path", "rules_sha256")):
        path = Path(manifest[path_key])
        if not path.is_absolute():
            path = ROOT / path
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[hash_key]:
            raise RuntimeError("manifest_hash_mismatch")
    probes = []
    for hook, text, expected in HOOKS:
        hits = probe_patterns(rule_set, text)
        if hits != expected:
            raise RuntimeError("hook_pattern_mismatch")
        probes.append({"hook": hook, "sales_example": text, "expected": sorted(expected), "hits": sorted(hits), "passed": True})
    for rule in rule_set.rules:
        if probe_patterns(rule_set, rule.compliant_response, output=True):
            raise RuntimeError("compliant_response_false_positive")
    result = {"load": "passed", "rules": len(rule_set.rules), "version": rule_set.version,
              "hooks": probes, "compliant_responses": "10_passed_without_pattern_hits",
              "scope": "offline_data_probe_only; production_three_layer_engine_pending_3.4"}
    (ROOT / "docs/validation/2.5-rules-acceptance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    try:
        print(json.dumps(verify(), ensure_ascii=False, indent=2))
    except Exception as exc:
        print("rule_acceptance_failed:" + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1) from None
