r"""Collect only; persist node IDs/markers/parameterization without service calls.

PowerShell at repository root:
  .\.venv\Scripts\python.exe scripts\collect_test_inventory.py
"""
import ast
import argparse
import json
import sys
from importlib.metadata import distributions
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class Inventory:
    def __init__(self, report):
        self.report = report

    def pytest_collection_finish(self, session):
        items = session.items
        nodes = [{"nodeid": i.nodeid, "integration": bool(i.get_closest_marker("integration")),
                  "parameterized": hasattr(i, "callspec")} for i in items]
        functions = sum(
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
            for p in (ROOT / "tests").glob("test_*.py")
            for n in ast.walk(ast.parse(p.read_text(encoding="utf-8-sig")))
        )
        summary = {
            "python": sys.version.split()[0], "platform": sys.platform,
            "source_static_baseline": 381, "target_static_functions": functions,
            "new_functions": functions - 381, "collected": len(items),
            "integration": sum(n["integration"] for n in nodes),
            "non_integration": sum(not n["integration"] for n in nodes),
            "parameterized_cases": sum(n["parameterized"] for n in nodes),
            "parameterized_functions": len({i.nodeid.split("[")[0] for i in items if hasattr(i, "callspec")}),
            "parameterization_extra_cases": len(items) - functions,
            "nodes": nodes,
            "packages": dict(sorted((d.metadata["Name"], d.version) for d in distributions())),
        }
        self.report.parent.mkdir(parents=True, exist_ok=True)
        self.report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("AUDIT: " + json.dumps({k: v for k, v in summary.items() if k not in {"nodes", "packages"}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/validation/1.2-test-inventory.json")
    args = parser.parse_args()
    raise SystemExit(pytest.main(["--collect-only", "-q"], plugins=[Inventory(args.report)]))
