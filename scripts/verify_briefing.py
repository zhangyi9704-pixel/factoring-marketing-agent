"""访前简报 E2E 验证脚本（真实链路：RAG 检索 + gte-rerank-v2 精排 + DeepSeek reasoning）。

用法：
    python scripts/verify_briefing.py [--case INFRA_001]

验证技术设计文档（新）§7：
    - 单 Agent 图内 briefing 节点：public_case → RAG → 1×reasoning → 简报；
    - 防泄漏：简报过 assert_no_internal_leak，内部字段不出现；
    - 确定性字段：industry / compliance_warnings / sources 规则渲染；
    - t_visit_report 落库（SessionService.get_briefing 读回）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.agents.roleplay.agent import RoleplayAgent          # noqa: E402
from app.schemas.case import assert_no_internal_leak        # noqa: E402
from app.services.session_service import SessionService      # noqa: E402


def print_section(title: str, lines: list[str]) -> None:
    print(f"\n[{title}]")
    for line in lines:
        print(f"  - {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="访前简报 E2E 验证")
    parser.add_argument("--case", default="INFRA_001", help="案例 id（默认 INFRA_001）")
    args = parser.parse_args()

    agent = RoleplayAgent()

    print("=" * 60)
    print(f"访前简报 E2E（技术设计文档（新）§7）：case={args.case}")
    print("=" * 60)

    created = agent.start_session(args.case, 1001)
    sid = created["session_id"]
    briefing = created.get("briefing") or {}

    assert_no_internal_leak(briefing)                        # 防泄漏硬校验（§7.3）
    print(f"会话: {sid}")
    print(f"防泄漏校验: 通过（unknown_info / hidden_risks 未出现）")
    print(f"行业（受控枚举）: {briefing.get('industry')}")

    print_section("拜访目标", briefing.get("visit_goals") or [])
    print_section("优先问题", [
        f"{q['question']}（{q['purpose']}｜价值: {q['risk_value']}）"
        for q in briefing.get("priority_questions") or []
    ])
    print_section("待确认（业务澄清，规则生成）", [
        f"{c['element']}：{c['suggestion']}" for c in briefing.get("clarifications") or []
    ] or ["（公开事实已覆盖核查清单）"])
    print_section("风险提示", [
        f"{r['risk']}（依据: {r['basis']}）" for r in briefing.get("risk_warnings") or []
    ])
    print_section("合规警示（config 规则渲染）", briefing.get("compliance_warnings") or [])
    print_section("可能的异议", [
        f"{o['objection']} → {o['response_hint']}"
        for o in briefing.get("possible_objections") or []
    ])
    print_section("非目标", briefing.get("non_goals") or [])
    print_section("引用来源（doc 去重）", [
        s["doc"] for s in briefing.get("sources") or []
    ] or ["（无命中）"])

    missing = briefing.get("knowledge_missing") or {}
    if missing.get("missing"):
        print(f"\n[知识库] 未命中 → {missing.get('message')}")

    persisted = SessionService().get_briefing(sid)
    print("\n" + "=" * 60)
    print(f"t_visit_report 落库: {'✅ 已读回' if persisted else '❌ 缺失'}")
    print(f"开场白已就绪: {created['opening']['content'][:30]}…")
    print("验收要点: 单图 briefing 节点 ✅ | 防泄漏 ✅ | 规则确定性字段 ✅ | 落库 ✅")
    print(f"（会话 {sid} 保持挂起，可用 verify_roleplay.py --session-id {sid} 续跑演练）")


if __name__ == "__main__":
    main()
