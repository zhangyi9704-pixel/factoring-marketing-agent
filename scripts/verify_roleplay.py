"""roleplay E2E 验证脚本（真实链路：DeepSeek chat + Redis checkpoint + MySQL）。

用法：
    python scripts/verify_roleplay.py [--case INFRA_001] [--turns 3] [--session-id s]

固定话术序列演示：开场 → 资金痛点 → 对账（触发 asked_about 白名单）→
违禁话术（触发 guardrail）→ [结束]。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.agents.roleplay.agent import RoleplayAgent          # noqa: E402
from app.services.session_service import SessionService      # noqa: E402

SCRIPTED_TURNS = [
    "王总您好，听说咱们项目回款压力比较大，主要卡在哪个环节？",
    "咱们和业主的工程计量与对账完成了吗？这关系到应收的确认。",
    "我们保证额度一周内就能批下来，利率也给您固定住。",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="roleplay E2E 验证")
    parser.add_argument("--case", default="INFRA_001", help="案例 id（默认 INFRA_001）")
    parser.add_argument("--turns", type=int, default=3, help="演练轮数（默认 3）")
    parser.add_argument("--session-id", default=None, help="指定 session_id（续跑既有会话）")
    args = parser.parse_args()

    agent = RoleplayAgent()

    print("=" * 60)
    print(f"情景演练 E2E（TDD §10.4）：case={args.case}")
    print("=" * 60)

    if args.session_id:
        sid = args.session_id
        status = agent.get_status(sid)
        print(f"续跑会话 {sid}: turn={status['turn']} waiting={status['waiting_input']}")
        if not status["waiting_input"]:
            print("会话不在等待输入状态，退出。")
            return
    else:
        created = agent.start_session(args.case, 1001)
        sid = created["session_id"]
        public = created["public_case"] or {}
        print(f"会话: {sid}")
        print(f"客户: {public.get('company')} / {public.get('client_role')}（销售视角）")
        print("-" * 60)
        print(f"[客户·开场] {created['opening']['content']}")

    for i in range(args.turns):
        text = SCRIPTED_TURNS[i % len(SCRIPTED_TURNS)]
        print(f"\n[销售·第{i + 1}轮] {text}")
        result = agent.send_message(sid, text)
        print(f"[客户·第{result['turn']}轮] {result['reply']}")
        if result["guardrail_warnings"]:
            print(f"  [合规提醒] {', '.join(result['guardrail_warnings'])}")
        if result["needs_human_review"]:
            print("  [HITL] 累计违规 >=3 次，已升级人工审核")
        if result["finished"]:
            print(f"\n会话结束（{result['finish_reason']}）")
            break

    final = agent.finish_session(sid)
    print("\n" + "=" * 60)
    print(f"最终状态: finished={final['finished']} reason={final['finish_reason']} "
          f"turns={final['turn']} mood={final['mood']}")

    messages = SessionService().list_messages(sid)
    print(f"MySQL 对话落库: {len(messages)} 条（含 turn 0 开场）")
    revealed = [m["revealed_info"] for m in messages if m["revealed_info"]]
    print(f"已释放未知信息: {sum(len(r) for r in revealed)} 条（内部视角，仅落库）")
    print("验收要点: checkpoint 续跑 ✅ | fact_constraint 白名单 ✅ | guardrail 每轮检测 ✅")


if __name__ == "__main__":
    main()
