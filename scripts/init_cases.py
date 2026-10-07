"""Mock Case 库初始化 CLI（cases/*.yaml → MySQL t_case，幂等）。

用法：
    python scripts/init_cases.py            # 校验 + 同步全部 YAML 案例
    python scripts/init_cases.py --list     # 仅校验并列出案例（不写库）
"""
from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.case_service import CaseService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Mock Case 库初始化（YAML → MySQL t_case）")
    parser.add_argument("--list", action="store_true", help="仅校验并列出案例，不写库")
    args = parser.parse_args()

    service = CaseService()
    print("=" * 60)
    print("Mock Case 库（TDD §10.2）")
    print("=" * 60)
    cases = service.load_yaml_cases()
    for case in cases:
        levels = Counter(card.level for card in case.fact_cards)
        print(f"  {case.case_id:<12} {case.industry:<4} {case.company} "
              f"（unknown={len(case.unknown_info)} hidden={len(case.hidden_risks)} "
              f"rules={len(case.behavior_rules)} cards={len(case.fact_cards)} "
              f"levels={dict(levels)} version={case.version}）")

    if args.list:
        print(f"结论: 校验通过 {len(cases)} 个案例（--list 未写库）")
        return 0

    result = service.sync_to_mysql(cases)
    print("-" * 60)
    print(f"同步 MySQL t_case: total={result['total']} "
          f"inserted={result['inserted']} updated={result['updated']}")
    stored = service.list_cases()
    print(f"回读: {len(stored)} 个启用案例 → "
          f"{', '.join(c.case_id for c in stored)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
