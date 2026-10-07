"""Phase 4 E2E 验证脚本（复盘报告 Markdown + 主管成绩画像）。

用法：
    python scripts/verify_phase4.py [--case INFRA_001] [--turns 2] [--session-id s]

流程：跑一段 roleplay（或复用既有会话）→ finish → EvaluationService.get()
惰性计算 → 打印 Markdown 复盘报告；再调 UserScoreService.scores() 打印主管
画像汇总。输出为纯中文文本（Windows GBK 控制台打不出 emoji）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.agents.roleplay.agent import RoleplayAgent          # noqa: E402
from app.evaluation.service import EvaluationService         # noqa: E402
from app.services.user_score_service import UserScoreService  # noqa: E402

SCRIPTED_TURNS = [
    "王总您好，咱们项目最终付款方是谁？合同签了吗？",
    "应收对账和验收完成了吗？有质押吗？",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 E2E 验证")
    parser.add_argument("--case", default="INFRA_001", help="案例 id（默认 INFRA_001）")
    parser.add_argument("--turns", type=int, default=2, help="演练轮数（默认 2）")
    parser.add_argument("--session-id", default=None, help="复用既有会话 id")
    args = parser.parse_args()

    agent = RoleplayAgent()
    if args.session_id:
        sid = args.session_id
        agent.finish_session(sid)
        print(f"复用并结束会话 {sid}")
    else:
        created = agent.start_session(args.case, 1001)
        sid = created["session_id"]
        print(f"会话: {sid}（user=1001）")
        for i in range(args.turns):
            text = SCRIPTED_TURNS[i % len(SCRIPTED_TURNS)]
            agent.send_message(sid, text)
            print(f"  [第{i + 1}轮] {text}")
        agent.finish_session(sid)

    report = EvaluationService().get(sid)
    print("\n" + "=" * 60)
    print("复盘报告（Markdown）")
    print("=" * 60)
    print(report.get("markdown", ""))

    print("\n" + "=" * 60)
    print("主管成绩画像（user=1001）")
    print("=" * 60)
    score = UserScoreService().scores(1001)
    print(f"  会话总数={score['total_sessions']} 已评估={score['evaluated_sessions']}")
    print(f"  机会均分={score['avg_opportunity']} 能力均分={score['avg_sales_score']}")
    print(f"  等级分布={score['level_distribution']}")
    print(f"  维度均值={score['dimension_averages']}")


if __name__ == "__main__":
    main()
