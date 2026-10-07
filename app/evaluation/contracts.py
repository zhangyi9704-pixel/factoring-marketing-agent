"""模型评分与可回溯辅导契约；拒绝未知维度、越界分数和宽松类型转换。"""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

Dimension = Literal['transaction_authenticity', 'receivable_quality', 'payer_confirmation',
    'financing_need', 'risk_identification', 'requirement_discovery',
    'business_understanding', 'risk_awareness', 'compliance']


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Citation(StrictModel):
    turn_id: int = Field(ge=1, strict=True)
    original_text: str = Field(min_length=1, max_length=10000)
    speaker: Literal['sales', 'customer'] = 'sales'


class Judgement(StrictModel):
    score: float = Field(ge=0, le=100, strict=True)
    rationale: str = Field(default='', max_length=2000)
    evidence: list[Citation] | None = None


class ScoringOutput(StrictModel):
    dimensions: dict[Dimension, Judgement]


class SourceSnapshot(StrictModel):
    doc_id: str
    version: str
    source: str
    industry: str
    status: Literal['active']
    content: str


class MissingItem(StrictModel):
    item: str
    impact: str
    suggestion: str


class ActionItem(StrictModel):
    action: str
    priority: Literal['high', 'medium', 'low']
    note: str


class CoachingItem(StrictModel):
    dimension: Dimension
    turn_id: int = Field(ge=1, strict=True)
    original_text: str = Field(min_length=1)
    issue: str
    better_version: str
    training_suggestion: str
    sources: list[SourceSnapshot] = Field(default_factory=list)


def verified_evidence(evidence, conversation):
    """原话必须精确匹配同轮销售正文；客户原话仅作同轮佐证。"""
    result = {}
    for dim, items in evidence.items():
        good = []
        for item in items:
            turn, quote = item.get('turn'), item.get('quote_sales')
            if type(turn) is not int or turn < 1 or not quote:
                continue
            if not any(m.get('role') == 'sales' and m.get('turn') == turn and m.get('content') == quote for m in conversation):
                continue
            customer = item.get('quote_customer') or ''
            if customer and not any(m.get('role') == 'customer' and m.get('turn') == turn and m.get('content') == customer for m in conversation):
                customer = ''
            good.append(dict(item, quote_customer=customer))
        result[dim] = good
    return result


def score_with_validation(llm, prompt, dims, conversation, evidence):
    """最多一次schema修复；故障或再失败返回统计降级，不回显异常正文。"""
    scores = None
    errors = []
    verified = verified_evidence(evidence, conversation)
    import json
    allowed = {dim: [{'turn_id': e['turn'], 'original_text': e['quote_sales'], 'speaker': 'sales'}
                     for e in verified.get(dim, [])] for dim in dims}
    prompt += '\n【各维度已验证销售证据白名单】\n' + json.dumps(allowed, ensure_ascii=False)
    prompt += '\n每维evidence仅从该维白名单逐字复制，不引用其他轮次或其他维度原话。白名单为空时evidence=[]且score=0。'
    for attempt in range(2):
        try:
            raw = (llm.chat_json_once if hasattr(llm, 'chat_json_once') else llm.chat_json)(prompt, model_role='reasoning')
            parsed = ScoringOutput.model_validate_json(raw) if isinstance(raw, str) else ScoringOutput.model_validate(raw)
            if not set(dims).issubset(parsed.dimensions):
                raise ValueError('missing dimensions')
            scores = {key: val.model_dump() for key, val in parsed.dimensions.items() if key in dims}
            if attempt:
                errors.append('schema_repaired')
            break
        except Exception as exc:
            errors.append('score_schema_invalid' if isinstance(exc, (ValueError, TypeError)) else 'score_model_unavailable')
            if not isinstance(exc, (ValueError, TypeError)):
                break
            if attempt == 0:
                # 不把不可信输出或异常串再注入。修复仅重申契约，模型调用内部外部重试由服务负责。
                prompt += '\n【唯一一次受控修复】上次输出不符合契约。按下面schema重新回答，不遗漏请求维度，不增加字段。\n' + __import__('json').dumps(ScoringOutput.model_json_schema(), ensure_ascii=False)
    if scores is None:
        return {dim: {'score': 0., 'rationale': '模型输出未通过校验，确定性统计降级。'} for dim in dims}, verified, errors + ['deterministic_fallback']
    for dim in dims:
        refs = scores[dim].get('evidence')
        if refs is not None:
            good = []
            for ref in refs:
                matches = [m for m in conversation if m.get('role') == 'sales' and m.get('turn') == ref['turn_id'] and m.get('content') == ref['original_text']]
                # 必须属于规则已确认的本维度讨论；真实但无关的原话也不能移花接木。
                matched = next((e for e in verified.get(dim, []) if e['turn'] == ref['turn_id'] and e['quote_sales'] == ref['original_text']), None)
                if ref['speaker'] == 'sales' and matches and matched:
                    good.append(matched)
                else:
                    errors.append('evidence_rejected:' + dim)
            verified[dim] = good
        scores[dim]['rationale'] = (f"依据第{verified[dim][0]['turn']}轮已验证销售原话；仅评价本轮讨论表现。"
            if verified.get(dim) else '无有效销售对话证据，按0计。')
    return scores, verified, errors
