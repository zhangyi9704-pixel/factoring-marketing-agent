"""情景演练提示词（TDD §10.4 customer_reply + fact_constraint 防幻觉）。

防幻觉铁律（§10.4）：系统提示词注入完整 Case State，明确"仅可使用以下
事实作答，未列出的信息一律回答不清楚"。信息释放白名单来自
事实卡引擎的 state_judge 判定。
"""
from __future__ import annotations

import json
from app.schemas.case import public_prompt_case
from app.engine.fact_cards import allowed_card_texts

CUSTOMER_SYSTEM_PROMPT = """你是保理业务营销演练中的"客户扮演引擎"，负责扮演指定企业的客户角色。铁律：

1. 【事实约束】你只能使用用户提供给你的【客户事实】作答；事实中没有的信息，一律回答"这个我不清楚"或"这个需要核实"，绝对禁止编造数字、进度或承诺。
2. 【释放纪律】信息释放严格遵守【本轮可释放信息】白名单：白名单外的未知信息一律不透露；白名单内的条目若附有条件（如"若销售已询问""仅在销售主动提供融资方案后"），解锁条件已由规则引擎判定，你只按卡片提示自然表达。
3. 【真实感】你是真实的企业客户：有痛点但不会主动掏底牌，对过度承诺保持警惕，回答口语化、贴合角色身份。
4. 【输出格式】仅输出一个合法 JSON 对象，不包含任何其他文字或 markdown 围栏。"""

OPENING_SYSTEM_PROMPT = """你是保理业务营销演练中的"客户扮演引擎"。请以指定企业客户角色的身份说一段开场白（2-3 句）：
- 自然点出企业当前的资金痛点，但不要释放任何未知信息（验收/对账/质押等一律不提）；
- 口语化、贴合角色身份，结尾给销售留下接话的空间；
- 直接输出开场白文本，不要 JSON、不要任何解释。"""


def _format_case_facts(case: dict) -> str:
    """按公开白名单注入，未释放事实卡留在受控内部State。"""
    case = public_prompt_case(case)
    basic = case.get("basic_information") or {}
    chain = {k: v for k, v in (case.get("transaction_chain") or {}).items() if v}
    receivable = {k: v for k, v in (case.get("receivable") or {}).items() if v is not None}
    return json.dumps({
        "企业": case.get("company"),
        "你的角色": case.get("client_role"),
        "行业": case.get("industry"),
        "基本情况": basic,
        "交易链": chain,
        "应收账款": receivable,
        "公知事实": case.get("known_facts") or [],
        "经营痛点": case.get("pain_points") or [],
        "已知风险（客户自知）": case.get("known_risks") or [],
    }, ensure_ascii=False, indent=2)


def build_opening_prompt(case: dict) -> str:
    """开场白生成（turn 0，客户先开口）。"""
    return (
        f"【客户事实】\n{_format_case_facts(case)}\n\n"
        "请以该角色身份说开场白。"
    )


def build_reply_prompt(
    *,
    case: dict,
    conversation_window: list[dict],
    sales_input: str,
    whitelist: list[str],
    released_info: list[str],
) -> str:
    """customer_reply 用户提示词。

    Args:
        conversation_window: 记忆窗口内的最近对话（config agents.roleplay.memory.window）。
        whitelist: 本轮可释放信息（事实卡 ID）。
        released_info: 已释放过的信息（避免当新信息重复抛出）。
    """
    transcript = "\n".join(
        f"第{m.get('turn')}轮 [{'销售' if m.get('role') == 'sales' else '客户'}] {m.get('content', '')}"
        for m in conversation_window
    ) or "（尚无对话）"
    whitelist_items = allowed_card_texts(case, whitelist) if case.get("fact_cards") else whitelist
    released_items = allowed_card_texts(case, released_info) if case.get("fact_cards") else released_info
    whitelist_text = "\n".join(f"- {d}" for d in whitelist_items) or "（无——本轮不得释放任何未知信息）"
    released_text = "\n".join(f"- {d}" for d in released_items) or "（无）"
    return f"""【客户事实】
{_format_case_facts(case)}

【对话记录（最近{len(conversation_window)}条）】
{transcript}

【已释放过的信息】（自然带过即可，不要当作新信息重复抛出）
{released_text}

【本轮可释放信息】（白名单，含释放条件；未列入的未知信息一律不透露）
{whitelist_text}

【销售本轮发言】
{sales_input}

请以客户角色回复本轮销售。输出 JSON：
{{
  "reply": "你的回复原文（口语化）",
  "revealed": ["本轮你实际释放的白名单卡片 ID（如 INF-05），未释放则为空数组"],
  "mood": "neutral / positive / impatient / angry"
}}"""
