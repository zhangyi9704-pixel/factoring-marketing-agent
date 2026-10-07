"""2.4 案例验收；只输出统计，--live 回读 MySQL 并探测当前代码的 HTTP 路由。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.schemas.case import assert_no_internal_leak, to_public_case
from app.services.case_service import CaseService
from app.agents.roleplay.behavior_rules import parse_rules


def require(condition, code):
    if not condition:
        raise RuntimeError(code)  # 不在错误中回显内部数据


def verify(live=False):
    manifest = json.loads((ROOT / "docs/validation/2.4-case-manifest.json").read_text(encoding="utf-8"))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest(Path(manifest["source_path"])) == manifest["source_sha256"], "source_changed")
    service = CaseService()
    cases = service.load_yaml_cases()
    require(len(cases) == 3, "case_count")
    stats = []
    for entry in manifest["cases"]:
        require(digest(ROOT / entry["path"]) == entry["file_sha256"], "case_file_changed")
        require(digest(ROOT / entry["legacy_backup"]) == entry["legacy_sha256"], "legacy_backup_changed")
        case = next(c for c in cases if c.case_id == entry["case_id"])
        require(dict(Counter(c.level for c in case.fact_cards)) == entry["levels"], "level_count")
        require(len(parse_rules(case.behavior_rules)) == len(case.behavior_rules), "legacy_dsl")
        public = to_public_case(case).model_dump()
        assert_no_internal_leak(public)
        blob = json.dumps(public, ensure_ascii=False)
        for card in case.fact_cards:
            require(card.level == "disclosed" or card.fact not in blob, "private_card_in_public")
        for secret in case.unknown_info + case.hidden_risks:
            require(secret not in blob, "internal_fact_in_public")
        stats.append({"case_id": case.case_id, "version": case.version, "cards": len(case.fact_cards),
                      "levels": entry["levels"], "public": "passed", "legacy_dsl": "passed"})
    result = {"offline": "passed", "cases": stats, "total_cards": 48}
    if live:
        stored = service.list_cases()
        require({c.case_id for c in stored} == {c.case_id for c in cases}, "mysql_case_ids")
        for case in cases:
            loaded = service.get_case(case.case_id)
            require(loaded is not None and loaded.model_dump() == case.model_dump(), "mysql_payload_mismatch")
        # 不进入 lifespan、不调用业务图；使用真实 CaseService/MySQL，验证 HTTP 对外投影。
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        response = client.get("/api/v1/cases")
        require(response.status_code == 200, "api_list_status")
        expected = [to_public_case(c).model_dump() for c in sorted(cases, key=lambda c: c.case_id)]
        require(response.json() == expected, "api_list_public_mismatch")
        assert_no_internal_leak(response.json())
        for case in cases:
            response = client.get("/api/v1/cases/" + case.case_id)
            require(response.status_code == 200 and response.json() == to_public_case(case).model_dump(), "api_detail_public_mismatch")
            response = client.get("/api/v1/cases", params={"industry": case.industry})
            require(response.status_code == 200 and response.json() == [to_public_case(case).model_dump()], "api_industry_filter")
        result.update({"mysql": "passed", "mysql_cases": len(stored), "api_list_detail_industry": "passed", "api_transport": "TestClient_current_app_real_mysql"})
    (ROOT / "docs/validation/2.4-case-acceptance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.live), ensure_ascii=False, indent=2))
    except Exception as exc:
        print("case_acceptance_failed:" + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1) from None
