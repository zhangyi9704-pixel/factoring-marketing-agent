"""访前简报提示词（技术设计文档（新）§7：单 Agent 简报节点，1 次 reasoning 调用）。

由旧版六节点访前准备 Agent 的四个 LLM 提示词（parser/classifier/risk/
report）收敛为一个；确定性字段（case_id/行业/公司/合规警示/来源）不交给
模型，由 briefing.py 规则后处理覆盖（继承"确定性字段不交给模型"原则）。

防泄漏硬约束（§7.3）：提示词只注入 public_case（销售视角），绝不注入
full_case——unknown_info/hidden_risks 不得经简报文本泄漏给销售。
"""
from __future__ import annotations

import json

BRIEFING_SYSTEM_PROMPT = """你是保理公司的资深销售教练，为销售人员生成一次客户拜访的"访前简报"。
铁律：
1. 只依据给定的【客户公开事实】与【知识上下文】作答；其中没有的信息一律不编造，不得虚构数字、进度或承诺；
2. 待核实信息只能出现在"优先问题""待澄清问题"与"风险提示"中，表述保持"待确认"口径；
3. 合规优先：任何字段不得出现承诺额度、审批结果、放款时间或固定利率的表述；
4. 仅输出一个合法 JSON 对象，不包含任何解释文字或 markdown 代码围栏。"""


def _format_public_facts(public_case: dict) -> str:
    """销售视角公开事实（briefing 唯一的案例信息来源）。"""
    chain = {k: v for k, v in (public_case.get("transaction_chain") or {}).items() if v}
    receivable = {
        k: v for k, v in (public_case.get("receivable") or {}).items() if v is not None
    }
    return json.dumps({
        "案例": public_case.get("case_id"),
        "企业": public_case.get("company"),
        "客户角色": public_case.get("client_role"),
        "行业": public_case.get("industry"),
        "基本情况": public_case.get("basic_information") or {},
        "交易链": chain,
        "应收账款": receivable,
        "经营痛点": public_case.get("pain_points") or [],
        "已知风险": public_case.get("known_risks") or [],
    }, ensure_ascii=False, indent=2)


def _format_knowledge(knowledge_context: list[dict]) -> str:
    """知识上下文（含来源与优先级，低优先级标注"仅供参考"）。"""
    if not knowledge_context:
        return "（知识库未命中——本次简报仅基于案例事实，相关结论需人工确认）"
    lines = []
    for item in knowledge_context:
        note = "（仅供参考）" if item.get("reference_only") else ""
        lines.append(
            f"- 【{item.get('doc_name')}｜来源ID:{item.get('doc_id') or item.get('doc_name')}｜版本:{item.get('version') or '未标注'}｜{item.get('source')}｜优先级{item.get('priority')}】{note}\n"
            f"  {item.get('content', '')[:500]}"
        )
    return "\n".join(lines)


def _format_clarifications(clarifications: list[dict]) -> str:
    if not clarifications:
        return "（公开事实已覆盖行业核查清单要素）"
    return "\n".join(
        f"- {c['element']}：{c['question']}（影响：{c['impact']}）"
        for c in clarifications
    )


def build_briefing_prompt(
    *,
    public_case: dict,
    knowledge_context: list[dict],
    clarifications: list[dict],
) -> str:
    """briefing 用户提示词（1 次 reasoning 调用产出全部 LLM 字段，§7.1）。"""
    industry = public_case.get("industry") or "其他"
    return f"""请基于以下信息生成访前简报。

【客户公开事实】（唯一案例信息来源，未列出的信息一律视为待确认）
{_format_public_facts(public_case)}

【知识上下文】（行业准入规则 / 产品要求 / 合规红线，含来源文档名）
{_format_knowledge(knowledge_context)}

【待澄清要素】（规则比对行业核查清单得出，围绕它们设计优先问题）
{_format_clarifications(clarifications)}

行业：{industry}

输出 JSON：
{{
  "visit_goals": ["拜访目标（3-5 条，每条一句话，围绕 确认交易真实性→应收形成→付款来源→融资需求 主线）"],
  "priority_questions": [
    {{
      "question": "要向客户提出的问题",
      "purpose": "提问目的",
      "risk_value": "该问题对风险判断的价值",
      "order_rationale": "该问题排在当前位置的依据（一句话）"
    }}
  ],
  "possible_objections": [
    {{
      "objection": "客户可能提出的异议",
      "response_hint": "应对思路提示（不得含任何承诺性表述）"
    }}
  ],
  "non_goals": ["本次沟通的非目标（2-4 条，如：不涉及具体额度与利率、不承诺审批结果）"],
  "risk_warnings": [
    {{
      "risk": "风险描述（一句话，待确认信息只能出现在这里）",
      "basis": "判断依据（引用公开事实或知识上下文的具体内容与来源）",
      "source_id": "公开事实使用case；知识使用上文给出的来源ID",
      "evidence": "该来源中连续、完整的原文片段；不得改写或虚构"
    }}
  ]
}}

要求：
1. priority_questions 每条必须完整包含 question / purpose / risk_value / order_rationale 四段，围绕待澄清要素与风险生成，5-8 条，按优先级排序并说明排序依据；
2. possible_objections 结合行业与客户角色生成 2-4 条，response_hint 只给沟通思路；
3. risk_warnings 每条必须注明source_id和连续原文evidence；未证实的客户风险保持待核实口径，最多 6 条，按严重程度排序；
4. 结合行业知识体现行业差异（基建/医药/电商的核实重点不同），不要只替换行业名称；
5. 未知信息禁止编造。"""
