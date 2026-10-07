"""评估 E2E 验证脚本（真实链路：DeepSeek reasoning + MySQL 落库 + 规则模块）。

用法：
    python scripts/verify_evaluation.py [--case INFRA_001] [--turns 3] [--session-id s]

流程：跑一段真实 roleplay（或续跑既有会话）→ finish → EvaluationService.get()
惰性计算并落库，打印机会诊断 / 四维能力评分 / 证据链 / 遗漏问题 / 下一步行动。
输出为纯中文文本（Windows GBK 控制台打不出 emoji）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.agents.roleplay.agent import RoleplayAgent          # noqa: E402
from app.evaluation.service import EvaluationService         # noqa: E402

SCRIPTED_TURNS = [
    "王总您好，听说咱们项目回款压力比较大，主要卡在哪个环节？",
    "咱们和业主的工程计量与对账完成了吗？这关系到应收的确认。",
    "项目最终付款方是谁？合同签了吗？",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 E2E 验证")
    parser.add_argument("--case", default="INFRA_001", help="案例 id（默认 INFRA_001）")
    parser.add_argument("--turns", type=int, default=3, help="演练轮数（默认 3）")
    parser.add_argument("--session-id", default=None, help="指定 session_id（复用既有会话）")
    args = parser.parse_args()

    agent = RoleplayAgent()
    if args.session_id:
        sid = args.session_id
        status = agent.get_status(sid)
        print(f"复用会话 {sid}: turn={status['turn']} waiting={status['waiting_input']}")
        if status["waiting_input"]:
            print("注意：会话仍处于等待输入状态，请先在其上完成演练或用 verify_roleplay.py。")
        if status["finished"] is False and status["waiting_input"]:
            return
    else:
        created = agent.start_session(args.case, 1001)
        sid = created["session_id"]
        print(f"会话: {sid}")
        print(f"客户: {(created['public_case'] or {}).get('company')}")

    if args.session_id is None:
        for i in range(args.turns):
            text = SCRIPTED_TURNS[i % len(SCRIPTED_TURNS)]
            result = agent.send_message(sid, text)
            print(f"[销售·第{i + 1}轮] {text}")
            print(f"[客户·{result['turn']}] {result['reply']}")
            if result["finished"]:
                break
        agent.finish_session(sid)
    else:
        agent.finish_session(sid)

    print("\n" + "=" * 60)
    print("评估管线（TDD §13）：惰性计算 + 落库")
    print("=" * 60)
    t0 = time.time()
    report = EvaluationService().get(sid)
    opp = report.get("opportunity", {})
    sales = report.get("sales_score", {})
    print(f"机会诊断: 得分 {opp.get('score')} / {opp.get('level')}")
    for reason in opp.get("reasons", []):
        print(f"  - {reason}")
    print(f"能力评分: 总分 {sales.get('total')}")
    for key, res in (sales.get("dimension_scores") or {}).items():
        print(f"  - {key}: {res.get('score')}（证据{'缺失' if res.get('evidence_missing') else '完整'}）")
    print(f"证据链: {len(report.get('evidence', []))} 条")
    for ev in report.get("evidence", []):
        print(f"  · {ev['dimension']}: 第{ev['turn']}轮 销售「{ev['quote_sales']}」"
              f"客户「{ev['quote_customer']}」")
    print(f"遗漏问题: {len(report.get('missing_questions', []))} 项")
    for m in report.get("missing_questions", []):
        print(f"  - {m['item']}：{m['impact']}")
    print(f"下一步行动: {len(report.get('next_actions', []))} 条")
    for a in report.get("next_actions", []):
        print(f"  - [{a.get('priority')}] {a.get('action')}")
    print(f"\n评估完成，耗时 {time.time() - t0:.1f}s（含 2 次 reasoning 调用）")


if __name__ == "__main__":
    main()
