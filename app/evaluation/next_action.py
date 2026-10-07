"""下一步行动建议（TDD §13.5，按机会等级分支输出，纯规则）。

- 高潜机会：机会明确 → 进入业务尽调、收集财务资料、推进融资方案；
- 观察：机会尚未闭合 → 补齐遗漏关键问题、索取验收/对账证据、复核付款主体；
- 暂缓：信息不足 → 先补充合同/验收证明/付款计划，暂不推进。
"""
from __future__ import annotations

from typing import Any

HIGH_LEVEL = "高潜机会"
OBSERVE_LEVEL = "观察"


class NextActionAdvisor:
    """按机会等级与遗漏项数输出行动清单（纯规则，无 LLM）。"""

    def advise(self, opportunity: dict, missing_count: int = 0) -> list[dict]:
        level = (opportunity or {}).get("level", "")
        if level == HIGH_LEVEL:
            return [
                {"action": "进入业务尽调，收集财务/审计/合同/验收资料",
                 "priority": "high", "note": "机会明确，尽快推动"},
                {"action": "量化融资需求与用款时点，初步输出保理方案",
                 "priority": "high", "note": "对齐其资金痛点"},
            ]
        if level == OBSERVE_LEVEL:
            return [
                {"action": "补齐遗漏关键问题", "priority": "high",
                 "note": f"当前遗漏 {missing_count} 项，需先补全"},
                {"action": "索取验收/对账/回款证据，复核付款主体",
                 "priority": "medium", "note": "提升机会确定性"},
            ]
        return [
            {"action": "信息不足，先补充合同/验收证明/付款计划",
             "priority": "high", "note": "关键信息缺失，暂缓推进"},
            {"action": "二次沟通确认真实应收与资金缺口",
             "priority": "medium", "note": "避免无效营销"},
        ]
