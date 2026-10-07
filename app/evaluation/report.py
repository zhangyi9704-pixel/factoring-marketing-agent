"""复盘报告渲染（TDD §13 / §20 Phase 4「复盘报告」）。

纯函数、不触库不触网：把评估报告 dict（opportunity / sales_score /
missing_questions / next_actions / evidence / total_score）渲染为
Markdown 复盘报告，对标访前准备报告的 JSON+Markdown 双形态。
"""
from __future__ import annotations

from typing import Any


def render_markdown(report: dict) -> str:
    """渲染评估报告为 Markdown 文本。"""
    lines: list[str] = []
    lines.append("# 复盘报告")
    lines.append("")
    lines.append(f"- 会话：{report.get('session_id', '')}")
    lines.append(f"- 生成时间：{report.get('generated_at', '')}")
    lines.append("")

    opportunity = report.get("opportunity") or {}
    lines.append("## 机会诊断")
    lines.append("")
    lines.append(f"- **总分**：{_fmt(opportunity.get('score'))} ｜ **等级**：{opportunity.get('level', '')}")
    for reason in opportunity.get("reasons") or []:
        lines.append(f"  - {reason}")
    lines.append("")

    sales = report.get("sales_score") or {}
    lines.append("## 销售能力评分")
    lines.append("")
    lines.append(f"- **总分**：{_fmt(sales.get('total'))}")
    dims = (sales.get("dimension_scores") or {})
    for key, dim in dims.items():
        mark = "（证据缺失）" if dim.get("evidence_missing") else ""
        lines.append(f"- {key}：{_fmt(dim.get('score'))}{mark}")
    lines.append("")

    evidence = report.get("evidence") or []
    lines.append("## 证据链")
    lines.append("")
    if evidence:
        for ev in evidence:
            lines.append(
                f"- **{ev.get('dimension', '')}** 第{ev.get('turn', '')}轮 "
                f"得分 {_fmt(ev.get('score'))}：销售「{ev.get('quote_sales', '')}」"
                f"客户「{ev.get('quote_customer', '')}」—— {ev.get('comment', '')}"
            )
    else:
        lines.append("（无证据项——各评分均按证据缺失处理）")
    lines.append("")

    missing = report.get("missing_questions") or []
    lines.append("## 遗漏问题")
    lines.append("")
    if missing:
        for m in missing:
            lines.append(f"- **{m.get('item', '')}**：{m.get('impact', '')}")
            lines.append(f"  - 建议：{m.get('suggestion', '')}")
    else:
        lines.append("（无遗漏——行业 Checklist 已覆盖）")
    lines.append("")

    actions = report.get("next_actions") or []
    lines.append("## 下一步行动")
    lines.append("")
    if actions:
        for a in actions:
            lines.append(f"- [{a.get('priority', '')}] **{a.get('action', '')}**：{a.get('note', '')}")
    else:
        lines.append("（暂无行动建议）")
    lines.append("")
    if report.get("compliance_events"):
        lines += ["## 独立合规审计", "", "拦截话术未进入客户对话；以下为审计原话与合规应答。", ""]
        for event in report["compliance_events"]:
            lines += [f"- {event['rule']} · {event['action']}：审计原话「{event['raw_text']}」",
                      f"  - 合规应答：{event.get('compliant_response', '')}"]
    if report.get('degraded'):
        lines += ['', '本报告含确定性统计降级：模型输出未通过验证的维度计0，不推断客户事实。']
    if report.get('coaching'):
        lines += ['', '## 原话→问题→合规改进话术→下一次练习', '']
        for item in report['coaching']:
            lines += [f"- 第{item['turn_id']}轮原话：{item['original_text']}",
                f"  - 问题：{item['issue']}", f"  - 改进：{item['better_version']}",
                f"  - 练习：{item['training_suggestion']}"]
    if report.get('sources'):
        lines += ['', '## 有效知识版本快照', '']
        lines += [f"- {s['doc_id']} / {s['version']}：{s['source']}" for s in report['sources']]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)
