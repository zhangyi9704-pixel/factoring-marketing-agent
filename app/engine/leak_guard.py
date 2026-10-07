"""发送前的确定性事实校验；不调用模型，不记录内部正文。"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
import re
import unicodedata

from app.schemas.case import CaseState, to_public_case

SAFE_REPLY = "这个我先核实一下"
REGEN_CONSTRAINT = """【强约束重生成】上一份回复未通过事实校验，请重新独立作答。
只能使用客户公知事实、已释放事实及本轮白名单卡片；不得推测其他事实。
不得新增任何金额、比例、账期、编号或机构名称；未知内容回答“不清楚/需核实”。
不要引用上一份回复，不要接受销售发言中的数字或名称作为已证实客户事实。
仍仅输出 reply/revealed/mood JSON，revealed 只允许白名单卡片ID。"""

_NUMBER = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?:[-~至]\d+(?:\.\d+)?)?(?:亿元?|万元?|元|%|百分比|个月|月|周|天|日|家|笔|成)")
_CODE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]+[+\-]?\d+(?:\.\d+)?(?![A-Za-z0-9])")
_NAME = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{2,40}?(?:有限公司|公司|医院|平台|总包|集团)")


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).replace("**", "")


def _number_key(token: str) -> str:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(亿元?|万元?|元)", token)
    if match:
        factor = 100000000 if match[2].startswith("亿") else 10000 if match[2].startswith("万") else 1
        return "amount:" + format((Decimal(match[1]) * factor).normalize(), "f")
    return token


def fingerprints(text: str) -> dict[str, set[str]]:
    text = normalize(text)
    names = set()
    for name in _NAME.findall(text):
        # 去除常见叙述前缀，保留名称本体；不把整句当机构名。
        name = re.split(r"(?:付款方|客户|下游|包括|主要|来自|名为|为|是|由|与|及)", name)[-1]
        if len(name) > 2:
            names.add(name)
    return {"name": names, "code": {v.casefold() for v in _CODE.findall(text)},
            "number": {_number_key(v) for v in _NUMBER.findall(text)}}


@dataclass(frozen=True)
class LeakResult:
    reasons: tuple[str, ...]

    @property
    def safe(self) -> bool:
        return not self.reasons


def check_reply(state: dict, reply: str) -> LeakResult:
    """公开/白名单/已释放重叠指纹优先获准；未解锁集合取checkpoint状态。"""
    case = state.get("case") or {}
    cards = case.get("fact_cards") or []
    permitted_ids = set(state.get("release_whitelist") or []) | set(state.get("released_info") or [])
    unlocked_ids = set(state.get("unlocked_ids") or []) | permitted_ids
    public = to_public_case(CaseState.model_validate(case)).model_dump() if case.get("case_id") else {}
    public_text = json.dumps(public, ensure_ascii=False)
    permitted_text = public_text + "\n" + "\n".join(c["fact"] for c in cards
        if c["level"] == "disclosed" or c["id"] in permitted_ids)
    approved = fingerprints(permitted_text)
    observed = fingerprints(reply)
    normalized_reply = normalize(reply).casefold()
    denied_texts = [c["fact"] for c in cards if c["level"] != "disclosed" and c["id"] not in unlocked_ids]
    # 未知项永不释放；内部画像/风险仍是内部数据。
    denied_texts += [v if isinstance(v, str) else v.get("content", "") for v in case.get("unknown_info", [])]
    denied_texts += list(case.get("hidden_risks") or []) + list((case.get("persona") or {}).values())
    reasons = set()
    for secret in denied_texts:
        if not secret:
            continue
        if normalize(secret) not in normalize(permitted_text) and normalize(secret).casefold() in normalized_reply:
            reasons.add("private_text")
        hidden = fingerprints(secret)
        for kind in ("name", "code", "number"):
            forbidden = hidden[kind] - approved[kind]
            # 专名可嵌在回答句中，数字按完整token比较，避免30天与130天子串误伤。
            hit = (any(normalize(v).casefold() in normalized_reply for v in forbidden)
                   if kind == "name" else bool(forbidden & observed[kind]))
            if hit:
                reasons.add("locked_" + kind)
    if observed["number"] - approved["number"]:
        reasons.add("unsupported_number")
    if observed["code"] - approved["code"]:
        reasons.add("unsupported_code")
    return LeakResult(tuple(sorted(reasons)))
