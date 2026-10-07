"""Case State 数据模型与双视角信息隔离（TDD §10.2 / §10.3）。

视角矩阵（§10.3）：
    | 视角  | allow                          | deny                       |
    | sales | basic_information、known_facts | unknown_info、hidden_risk |
    | agent | all（完整 Case State）          | —                          |

- CaseState：完整 Case（agent 侧 / MySQL case_json 存储结构）；
- PublicCase：销售视角视图（API / visit_report 使用），绝不包含
  unknown_info 与 hidden_risks；
- to_public_case / to_full_case：两份视图的生成入口；
- assert_no_internal_leak：API 层二次校验（§19.4，防 State 泄漏）。

behavior_rules 为 TDD §10.2 原文 DSL（如 "turn==2 -> 透露付款方"），
仅兼容历史存储；运行时由事实卡引擎判定，不再解析该字段。
"""
from __future__ import annotations

from typing import Any, Optional, Literal
import json
import re

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

SUPPORTED_INDUSTRIES = ("基建", "医药", "电商", "其他")

# 销售视角禁止出现的键（含内部标记），防泄漏二次校验用
INTERNAL_KEYS = ("unknown_info", "hidden_risk", "hidden_risks", "visibility",
                 "persona", "fact_cards", "payer_info", "behavior_rules", "hint", "triggers", "progressive")

BASIC_PUBLIC_KEYS = {"企业性质", "在手项目", "成立时间", "年营收规模", "公开画像", "业务模式"}
CHAIN_PUBLIC_KEYS = {"payer", "contractor", "supplier", "summary"}


def _public_scalars(data, allowed):
    return {k: v for k, v in (data or {}).items() if k in allowed and isinstance(v, (str, int, float, bool))}


class FactCard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, pattern=r"^(INF|PHA|ECO)-\d{2}$")
    dim: str = Field(min_length=1)
    fact: str = Field(min_length=1)
    level: Literal["disclosed", "on_ask", "sensitive"]
    hint: str = Field(min_length=1)
    triggers: list[str]
    progressive: bool

    @model_validator(mode="after")
    def validate_trigger(self):
        if any(not value.strip() for value in self.triggers):
            raise ValueError("empty fact trigger")
        if self.level != "disclosed" and not self.triggers:
            raise ValueError("non-disclosed card requires triggers")
        if self.level == "disclosed" and self.progressive:
            raise ValueError("disclosed card cannot be progressive")
        return self


class CaseLeakError(RuntimeError):
    """销售视角数据泄漏了内部信息（unknown_info / hidden_risks / 内部标记）。"""


class TransactionChain(BaseModel):
    description: Optional[str] = None
    payer: Optional[str] = Field(default=None, description="最终付款方")
    contractor: Optional[str] = Field(default=None, description="合同/订单对手方")
    supplier: Optional[str] = Field(default=None, description="供应商")


class Receivable(BaseModel):
    details: dict[str, str] = Field(default_factory=dict)
    amount: Optional[float] = Field(default=None, description="应收账款金额（元）")
    aging_days: Optional[int] = Field(default=None, description="账期（天）")


class CaseState(BaseModel):
    """完整 Case State（agent 视角，MySQL t_case.case_json 结构）。"""

    case_id: str = Field(..., description="案例编号，如 INFRA_001")
    industry: str = Field(..., description="基建/医药/电商/其他")
    company: str = Field(..., description="客户企业名称")
    persona: dict[str, str] = Field(default_factory=dict)
    payer_info: dict[str, str] = Field(default_factory=dict)
    fact_cards: list[FactCard] = Field(default_factory=list)
    opening_line: str = ""
    version: str = ""
    source: str = ""
    status: Literal["active", "archived"] = "active"
    client_role: str = Field(..., description="模拟客户角色，如 施工企业财务总监")
    basic_information: dict[str, Any] = Field(
        default_factory=dict, description="销售可见的客观事实（键值对）")
    transaction_chain: TransactionChain = Field(default_factory=TransactionChain)
    receivable: Receivable = Field(default_factory=Receivable)
    pain_points: list[str] = Field(default_factory=list, description="客户痛点")
    known_risks: list[str] = Field(default_factory=list, description="已知风险（销售可见）")
    unknown_info: list[str] = Field(
        default_factory=list, description="未知信息（销售不可见，未满足释放条件不得透露）")
    hidden_risks: list[str] = Field(
        default_factory=list, description="隐藏风险（销售不可见，评分用）")
    behavior_rules: list[str] = Field(
        default_factory=list, description="信息释放节奏 DSL，如 turn==2 -> 透露付款方")

    @model_validator(mode="after")
    def validate_fact_case(self):
        if not self.fact_cards:
            if self.version:
                raise ValueError("versioned case requires fact cards")
            return self  # 旧测试/存储结构的过渡兼容
        prefix = {"INFRA_001": "INF", "PHARMA_001": "PHA", "ECOM_001": "ECO"}.get(self.case_id)
        if not prefix or self.industry != {"INF": "基建", "PHA": "医药", "ECO": "电商"}[prefix]:
            raise ValueError("case id/industry mismatch")
        if {c.id for c in self.fact_cards} != {f"{prefix}-{i:02}" for i in range(1, 17)} or len(self.fact_cards) != 16:
            raise ValueError("case requires 16 unique matching fact ids")
        if not self.version or not self.source or not self.opening_line or not self.persona or not self.payer_info:
            raise ValueError("versioned case metadata missing")
        return self


class PublicCase(BaseModel):
    """销售视角 Case（API / visit_report 使用，双视角过滤后）。"""

    case_id: str
    industry: str
    company: str
    client_role: str
    basic_information: dict[str, Any]
    transaction_chain: dict[str, Any]
    receivable: dict[str, Any]
    pain_points: list[str]
    known_risks: list[str]
    known_facts: list[str] = Field(
        default_factory=list, description="客观事实汇总（basic_information 值 + 痛点 + 已知风险）")

    @field_validator("basic_information", "transaction_chain", "receivable", mode="before")
    @classmethod
    def nested_whitelist(cls, value, info):
        allowed = {"basic_information": BASIC_PUBLIC_KEYS, "transaction_chain": CHAIN_PUBLIC_KEYS,
                   "receivable": {"amount", "aging_days"}}[info.field_name]
        return _public_scalars(value, allowed)


def _fmt_amount(amount: float) -> str:
    """金额（元）→ 可读文本：≥1 万显示万元，否则显示元（50000000 → 5000万元）。"""
    if amount is None:
        return ""
    if amount >= 10000:
        wan = amount / 10000
        wan = int(wan) if float(wan).is_integer() else round(wan, 1)
        return f"{wan}万元"
    return f"{int(amount)}元"


def to_public_case(case: CaseState) -> PublicCase:
    """生成销售视角视图（§10.3：deny unknown_info / hidden_risks）。

    客观事实以"键: 值"标注：basic_information、应收金额、账期、付款方都带明确
    标签，避免出现"约2.8亿元"这类无标注数字让人不明其意（需求·案例简报注明含义）。
    """
    if case.fact_cards:
        disclosed = [c for c in case.fact_cards if c.level == "disclosed"]
        basic = {"公开画像": c.fact for c in disclosed if c.dim == "主体"}
        chain = {"contractor": case.company}
        trade = [c.fact for c in disclosed if c.dim == "基础交易"]
        if trade:
            chain["summary"] = "；".join(trade)
        return PublicCase(case_id=case.case_id, industry=case.industry, company=case.company,
                          client_role=case.client_role, basic_information=basic, transaction_chain=chain,
                          receivable={}, pain_points=[c.fact for c in disclosed if c.dim in {"融资用途", "痛点延伸"}],
                          known_risks=[], known_facts=[c.fact for c in disclosed])
    basic = _public_scalars(case.basic_information, BASIC_PUBLIC_KEYS)
    known_facts = [f"{k}: {v}" for k, v in basic.items()]
    rec = case.receivable
    if rec.amount is not None:
        known_facts.append(f"应收账款: {_fmt_amount(rec.amount)}")
    if rec.aging_days is not None:
        known_facts.append(f"账期: {rec.aging_days} 天")
    if case.transaction_chain.payer:
        known_facts.append(f"付款方: {case.transaction_chain.payer}")
    known_facts += case.pain_points + case.known_risks
    return PublicCase(
        case_id=case.case_id,
        industry=case.industry,
        company=case.company,
        client_role=case.client_role,
        basic_information=basic,
        transaction_chain=_public_scalars(case.transaction_chain.model_dump(exclude_none=True), CHAIN_PUBLIC_KEYS),
        receivable=_public_scalars(case.receivable.model_dump(exclude_none=True), {"amount", "aging_days"}),
        pain_points=list(case.pain_points),
        known_risks=list(case.known_risks),
        known_facts=known_facts,
    )


def to_full_case(case: CaseState) -> dict[str, Any]:
    """生成 agent 视角数据（完整 Case State + 内部字段标记 visibility）。

    §10.3：任何写回 State 的未知信息均带 visibility: internal 标记，
    API 层据此过滤（assert_no_internal_leak 同时校验键名）。
    """
    data = case.model_dump()
    data["unknown_info"] = [
        {"content": item, "visibility": "internal"} if isinstance(item, str) else item
        for item in data.get("unknown_info", [])
    ]
    data["hidden_risks"] = [
        {"content": item, "visibility": "internal"} if isinstance(item, str) else item
        for item in data.get("hidden_risks", [])
    ]
    return data


def assert_no_internal_leak(data: Any) -> None:
    """API 层二次校验（§19.4）：销售视角数据不得含内部键。

    仅检查容器键名（不递归文本内容），命中 INTERNAL_KEYS 抛 CaseLeakError。
    """
    if isinstance(data, dict):
        for key, value in data.items():
            if key in INTERNAL_KEYS:
                raise CaseLeakError(f"销售视角数据包含内部字段: {key!r}")
            assert_no_internal_leak(value)
    elif isinstance(data, (list, tuple)):
        for item in data:
            assert_no_internal_leak(item)


def public_prompt_case(data: dict) -> dict:
    """仅向客户模型注入公开事实，获准卡片另走白名单。"""
    if not data.get("fact_cards"):
        return data
    return to_public_case(CaseState.model_validate(data)).model_dump()


def assert_case_text_safe(case: dict, output: Any, allowed: list[str] | None = None) -> None:
    """发送前防内部文本泄漏；不做事实卡解锁或追问计数。"""
    if not case.get("fact_cards"):
        return
    model = CaseState.model_validate(case)
    # Import here: leak_guard imports these schemas when building the public view.
    # Use the same currency normalization for briefing and customer replies.
    from app.engine.leak_guard import fingerprints, normalize
    visible = normalize(json.dumps(to_public_case(model).model_dump(), ensure_ascii=False) + " ".join(allowed or []))
    text = normalize(json.dumps(output, ensure_ascii=False))
    secrets = [c.fact for c in model.fact_cards if c.level != "disclosed"] + model.unknown_info + model.hidden_risks + list(model.persona.values())
    approved = fingerprints(visible)
    observed = fingerprints(text)
    for secret in secrets:
        secret = normalize(secret)
        if secret not in visible and secret.casefold() in text.casefold():
            raise CaseLeakError("private_case_text")
        hidden = fingerprints(secret)
        for kind in ("number", "code", "name"):
            denied = hidden[kind] - approved[kind]
            hit = (any(normalize(v).casefold() in text.casefold() for v in denied)
                   if kind == "name" else bool(denied & observed[kind]))
            if hit:
                raise CaseLeakError("private_case_fingerprint")
