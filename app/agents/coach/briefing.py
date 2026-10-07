"""访前简报节点（技术设计文档（新）§7：原访前准备 Agent 的降级形态）。

单节点替代旧六节点流水线（parser/classifier/clarification/retrieval/
risk/report 约 680 行）：行业取案例受控枚举（零 LLM），业务澄清走行业
核查清单规则比对，1 次 reasoning 调用产出全部 LLM 字段。

防泄漏硬约束（§7.3）：LLM 输入只有 public_case（销售视角）+ RAG 检索
结果，绝不注入 full_case；组装结果过 assert_no_internal_leak 二次校验。

容错（继承节点容错模式）：检索失败/LLM 失败均降级为规则拼装简报，
errors 留痕不中断会话创建。
"""
from __future__ import annotations

import logging
import copy
import json
from time import perf_counter
from app.services.session_service import SessionServiceError
from typing import Any, Optional

from app.config import Settings
from app.prompts.briefing_prompt import BRIEFING_SYSTEM_PROMPT, build_briefing_prompt
from app.schemas.case import assert_no_internal_leak, assert_case_text_safe, CaseLeakError
from app.services.external_calls import request_deadline, check_deadline

logger = logging.getLogger(__name__)

# 拜访目标主线（降级模板，围绕"交易→应收→付款→融资"）
DEFAULT_GOALS = [
    "确认交易背景真实性",
    "确认应收账款形成条件",
    "确认付款来源与付款主体",
    "判断融资需求与用途",
]

DEFAULT_NON_GOALS = [
    "本次沟通不涉及具体融资额度与利率",
    "不承诺审批结果与放款时间",
    "以信息核实为主，不强推产品方案",
]


class _LazyRetriever:
    """延迟构造 HybridRetriever（首次 retrieve 才连 Milvus/Embedding）。

    单测注入替身时可整体绕开；生产由 agent 门面懒加载传入。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._inner: Any = None

    def retrieve(self, query: str, *, industry=None):
        if self._inner is None:
            from app.rag.retriever import HybridRetriever
            self._inner = HybridRetriever(self._settings)
        return self._inner.retrieve(query, industry=industry)


def make_briefing_node(
    llm,
    *,
    settings: Settings,
    retriever: Any = None,
    session_service: Any = None,
    guardrail: Any = None,
):
    """briefing：public_case → RAG 检索 → 1×reasoning → 简报（落 t_visit_report）。"""

    def _briefing_node(state: dict) -> dict:
        if state.get("briefing") or state.get("finished"):
            return {}                                   # checkpoint 恢复幂等
        public_case = state.get("public_case") or {}
        node_started = perf_counter()
        timings = {}
        errors = list(state.get("errors", []))
        degradation_reasons = []

        # 1) RAG 检索（外挂知识库，§9）
        retriever_ = retriever or _LazyRetriever(settings)
        knowledge_context: list[dict] = []
        knowledge_missing = False
        knowledge_message = ""
        knowledge_mode, knowledge_reason, escalation_id = "missing", None, None
        phase_started = perf_counter()
        try:
            check_deadline()
            result = retriever_.retrieve(_build_query(public_case), industry=public_case.get("industry") or None)
            knowledge_mode = getattr(result, "mode", "missing" if result.missing else "rag")
            knowledge_reason = getattr(result, "reason", None)
            escalation_id = getattr(result, "escalation_id", None)
            if knowledge_mode == "conflict":
                knowledge_missing = True
                knowledge_message = result.message
            if result.missing or not result.items:
                knowledge_missing = True
                knowledge_message = result.message or settings.knowledge.missing.message
            else:
                knowledge_context = [item.to_dict() for item in result.items]
        except Exception as exc:                        # 基础设施故障留痕
            error_text = type(exc).__name__ if (state.get("case") or {}).get("fact_cards") else str(exc)
            logger.warning("briefing 检索失败，降级为无知识上下文: %s", error_text)
            errors.append(f"briefing retrieval: {error_text}")
            knowledge_missing = True
            knowledge_message = settings.knowledge.missing.message

        timings['retrieval_s'] = perf_counter() - phase_started

        if knowledge_mode in {"missing", "conflict"} and session_service and hasattr(session_service, "register_escalation"):
            from app.rag.revision import revision
            kind = "out_of_scope" if knowledge_reason == "out_of_scope" else knowledge_mode
            versions = [{"doc_id":i.get("doc_id"), "version":i.get("version")} for i in knowledge_context]
            versions.append({"knowledge_revision":revision()})
            escalation_id = session_service.register_escalation(kind, _build_query(public_case),
                user_id=state["user_id"], session_id=state["session_id"],
                context={"turn":0,"items":knowledge_context,"reason":knowledge_reason}, versions=versions)

        # 2) 业务澄清（核心任务 1，纯规则：行业核查清单比对公开事实）
        clarifications = rule_clarifications(public_case, settings)

        # 3) LLM 生成（1 次 reasoning，§7.1）
        raw: dict = {}
        output_scan = None
        phase_started = perf_counter()
        try:
            check_deadline()
            raw = llm.chat_json(
                build_briefing_prompt(
                    public_case=public_case,
                    knowledge_context=knowledge_context if knowledge_mode == "rag" else [],
                    clarifications=clarifications,
                ),
                model_role=settings.agents.briefing.llm.model,
                system=BRIEFING_SYSTEM_PROMPT,
            )
        except Exception as exc:
            error_text = type(exc).__name__ if (state.get("case") or {}).get("fact_cards") else str(exc)
            logger.warning("briefing LLM 失败，降级为规则拼装: %s", error_text)
            errors.append(f"briefing llm: {error_text}")
            degradation_reasons.append("deadline_exceeded" if getattr(exc, "category", None) == "deadline_exceeded"
                                       else "generation_unavailable")
        timings['llm_s'] = perf_counter() - phase_started
        if not isinstance(raw, dict):                    # 畸形输出（非 JSON 字典）
            if raw:
                errors.append(f"briefing llm: 输出非对象，降级为规则拼装")
            raw = {}

        # Check private fingerprints before grounding: changing currency units must
        # never make an undisclosed fact appear to be an ordinary missing citation.
        try:
            assert_case_text_safe(state.get("case") or {}, raw)
        except CaseLeakError:
            raw = {}
            errors.append("briefing: private_case_blocked")
            degradation_reasons.append("private_case_blocked")
        raw, rejected = ground_briefing_raw(public_case, raw,
            knowledge_context if knowledge_mode == "rag" else [])
        if rejected:
            errors.append("briefing: grounding_rejected")
            degradation_reasons.append("grounding_rejected")

        # 模型建议在落库/发送前统一扫描；确定性规则文案不交模型改写。
        phase_started = perf_counter()
        if guardrail is not None and hasattr(guardrail, "protect_tree"):
            output_scan = guardrail.protect_tree(raw, session_id=state["session_id"], session_service=session_service)
            if output_scan.l2_degraded:
                raw = {}
                errors.append("briefing: compliance_unavailable")
                degradation_reasons.append("compliance_unavailable")
        timings['guardrail_s'] = perf_counter() - phase_started
        # 4) 组装（确定性字段覆盖 + 防泄漏校验，§7.3）
        phase_started = perf_counter()
        briefing = assemble_briefing(
            public_case=public_case,
            raw=raw,
            knowledge_context=knowledge_context,
            clarifications=clarifications,
            knowledge_missing=knowledge_missing,
            knowledge_message=knowledge_message,
            settings=settings,
        )
        try:
            assert_case_text_safe(state.get("case") or {}, briefing)
        except CaseLeakError as exc:
            errors.append("briefing: private_case_blocked")
            reason = str(exc) if str(exc) in {"private_case_text", "private_case_fingerprint"} else "unknown"
            errors.append("briefing_private_reason: " + reason)
            logger.warning("briefing: private_case_blocked")
            briefing = assemble_briefing(public_case=public_case, raw={}, knowledge_context=knowledge_context,
                clarifications=clarifications, knowledge_missing=knowledge_missing,
                knowledge_message=knowledge_message, settings=settings)
            degradation_reasons.append("private_case_blocked")
        assert_no_internal_leak(briefing)
        briefing.update(knowledge_mode=knowledge_mode, knowledge_reason=knowledge_reason, escalation_id=escalation_id)
        if knowledge_mode == "conflict":
            briefing["knowledge_conflicts"] = knowledge_context
        needs_review = bool(state.get("needs_human_review")) or bool(degradation_reasons)
        if output_scan:
            needs_review = needs_review or any(h.severity == "block" for h in output_scan.hits) or len(output_scan.hits) >= 3
            briefing["compliance_status"] = dict(l2_degraded=output_scan.l2_degraded,
                l1_elapsed_ms=output_scan.l1_elapsed_ms, l2_elapsed_ms=output_scan.l2_elapsed_ms)

        review_id = state.get("compliance_escalation_id")
        if degradation_reasons:
            degradation_reasons = list(dict.fromkeys(degradation_reasons))
            briefing["degraded"] = True
            briefing["degradation_reasons"] = degradation_reasons
            briefing["review_message"] = "部分模型建议未通过完整验证，已使用规则或待核实提示，请人工确认。"
            briefing["compliance_warnings"].append(briefing["review_message"])
            if session_service and hasattr(session_service, "mark_human_review"):
                session_service.mark_human_review(state["session_id"])
            if session_service and hasattr(session_service, "register_escalation") and "user_id" in state:
                review_id = session_service.register_escalation("compliance", "访前简报验证未完成，请人工确认。",
                    user_id=state["user_id"], session_id=state["session_id"], severity="warn",
                    context={"turn": 0, "phase": "briefing_validation", "reasons": degradation_reasons})
            briefing["review_pending_id"] = review_id

        timings['assembly_s'] = perf_counter() - phase_started

        # 5) 落库（t_visit_report 复用，幂等 upsert）
        phase_started = perf_counter()
        if session_service:
            try:
                session_service.save_briefing(state["session_id"], briefing)
            except Exception as exc:
                raise SessionServiceError("briefing_write_failed") from None

        timings['persist_s'] = perf_counter() - phase_started
        timings['total_s'] = perf_counter() - node_started
        logger.info('briefing phase timings: %s', timings)

        return {
            "briefing": briefing,
            "knowledge_context": knowledge_context,
            "knowledge_missing": knowledge_missing,
            "knowledge_mode": knowledge_mode, "knowledge_reason": knowledge_reason, "escalation_id": escalation_id,
            "needs_human_review": needs_review,
            "compliance_escalation_id": review_id,
            "errors": errors,
            "briefing_timings": timings,
        }

    def briefing_node(state: dict) -> dict:
        # The budget is cooperative: external clients receive the remaining
        # timeout, while already submitted provider computation cannot be recalled.
        with request_deadline(settings.agents.briefing.timeout_seconds):
            return _briefing_node(state)

    return briefing_node


UNVERIFIED_GUIDANCE = "请核实该信息的原始依据与数据，本次不形成确定结论。"
UNVERIFIED_RISK = {"risk": "待核实交易、应收及回款风险", "basis": "依据不足，请人工确认原始资料。"}


def ground_briefing_raw(public_case: dict, raw: dict, knowledge_context: list[dict]):
    """Gate measurable claims and risk citations; this is not entailment proof.

    Numbers/codes must appear in data actually supplied to the model. A risk is
    always a question to verify, with a source-bound continuous quote; unsupported
    conclusions are replaced, never echoed as facts or appended to disclaimers.
    """
    from app.engine.leak_guard import fingerprints, normalize
    raw = copy.deepcopy(raw)
    public_text = json.dumps(public_case, ensure_ascii=False) + "\n" + _facts_text(public_case)
    sources = {"case": public_text}
    for item in knowledge_context:
        identifier = item.get("doc_id") or item.get("doc_name")
        if identifier:
            sources.setdefault(str(identifier), "")
            sources[str(identifier)] += "\n" + str(item.get("content") or "")[:500]
    approved = fingerprints("\n".join(sources.values()) + "\n" + " ".join(sources))
    rejected = False

    def clean(value):
        nonlocal rejected
        if isinstance(value, str):
            observed = fingerprints(value)
            if any(observed[k] - approved[k] for k in ("number", "code")):
                rejected = True
                return UNVERIFIED_GUIDANCE
            return value
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        return value

    raw = clean(raw)
    risks = []
    raw_risks = raw.get("risk_warnings")
    if raw_risks is not None and not isinstance(raw_risks, list):
        rejected = True
        raw_risks = []
    for risk in raw_risks or []:
        source = sources.get(str(risk.get("source_id") or ""), "") if isinstance(risk, dict) else ""
        quote = str(risk.get("evidence") or "").strip() if isinstance(risk, dict) else ""
        if not source or len(normalize(quote)) < 4 or normalize(quote) not in normalize(source):
            rejected = True
            risks.append(dict(UNVERIFIED_RISK))
            continue
        text = str(risk.get("risk") or "").strip()
        if not text:
            rejected = True
            risks.append(dict(UNVERIFIED_RISK))
            continue
        risks.append({"risk": text if text.startswith("待核实") else "待核实：" + text,
                      "basis": f"{risk['source_id']}：{quote}"})
    if "risk_warnings" in raw:
        raw["risk_warnings"] = risks
    return raw, rejected


# -----------------------------------------------------
# 规则步骤（简单规则不交给 LLM，附录 A.1）
# -----------------------------------------------------

def _build_query(public_case: dict) -> str:
    """检索 query：案例结构化字段拼装（§9：替代旧版自由文本输入）。"""
    parts: list[str] = [public_case.get("industry", ""), public_case.get("company", "")]
    parts += [str(p) for p in public_case.get("pain_points") or []]
    receivable = public_case.get("receivable") or {}
    if receivable.get("amount") is not None:
        amount = float(receivable["amount"])
        if amount.is_integer():                         # 50000000.0 → 50000000
            amount = int(amount)
        parts.append(f"应收账款 {amount} 元")
    if receivable.get("aging_days") is not None:
        parts.append(f"账期 {receivable['aging_days']} 天")
    chain = public_case.get("transaction_chain") or {}
    labels = {"payer": "付款方", "contractor": "合同对手方", "supplier": "供应商"}
    for key, label in labels.items():
        if chain.get(key):
            parts.append(f"{label} {chain[key]}")
    return " ".join(p for p in parts if p).strip() or (public_case.get("industry") or "保理")


def _facts_text(public_case: dict) -> str:
    """公开事实文本（核查清单关键词命中判定用）。"""
    parts = [str(v) for v in public_case.get("known_facts") or []]
    parts += [f"{k} {v}" for k, v in (public_case.get("basic_information") or {}).items()]
    for key in ("transaction_chain", "receivable"):
        parts += [f"{k} {v}" for k, v in (public_case.get(key) or {}).items() if v]
    return " ".join(parts)


def rule_clarifications(public_case: dict, settings: Settings) -> list[dict]:
    """业务澄清（核心任务 1）：行业核查清单项未被公开事实覆盖 → 待确认问题。

    复用 evaluation.modules.missing_question.industry_checklist（config 三行业），
    纯规则零 LLM（继承旧 business_clarification 节点语义）。
    """
    industry = public_case.get("industry", "")
    checklist = (
        settings.evaluation.modules.missing_question.industry_checklist.get(industry) or []
    )
    facts = _facts_text(public_case)
    clarifications: list[dict] = []
    for entry in checklist:
        if any(kw in facts for kw in entry.keywords):
            continue                                  # 公开事实已覆盖
        clarifications.append({
            "element": entry.item,
            "question": entry.suggestion,
            "impact": entry.impact,
            "suggestion": entry.suggestion,
        })
    return clarifications


# -----------------------------------------------------
# 组装（确定性字段不交给模型）
# -----------------------------------------------------

def _normalize_str_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:limit]


def _normalize_questions(
    raw_questions: Any, clarifications: list[dict], limit: int,
) -> list[dict]:
    """priority_questions 归一化（四段完整）；LLM 无输出时由澄清清单降级生成。"""
    questions: list[dict] = []
    if isinstance(raw_questions, list):
        for item in raw_questions:
            if isinstance(item, dict) and str(item.get("question") or "").strip():
                questions.append({
                    "question": str(item["question"]).strip(),
                    "purpose": str(item.get("purpose") or "待确认").strip(),
                    "risk_value": str(item.get("risk_value") or "待确认").strip(),
                    "order_rationale": str(
                        item.get("order_rationale") or "按对风险判断的优先级排序"
                    ).strip(),
                })
    if not questions:                                 # 降级：澄清问题转优先问题
        for c in clarifications[:limit]:
            questions.append({
                "question": f"{c['element']}：{c['question']}",
                "purpose": f"确认{c['element']}",
                "risk_value": c["impact"],
                "order_rationale": "行业核查清单要求（LLM 降级，规则生成）",
            })
    return questions[:limit]


def _normalize_objections(raw_objections: Any) -> list[dict]:
    if not isinstance(raw_objections, list):
        return []
    result: list[dict] = []
    for item in raw_objections:
        if isinstance(item, dict) and str(item.get("objection") or "").strip():
            result.append({
                "objection": str(item["objection"]).strip(),
                "response_hint": str(item.get("response_hint") or "待确认").strip(),
            })
        elif isinstance(item, str) and item.strip():
            result.append({"objection": item.strip(), "response_hint": "待确认"})
    return result


def _normalize_risks(raw_risks: Any, public_case: dict) -> list[dict]:
    """risk_warnings 归一化（每条含依据）；LLM 无输出时取案例已知风险。"""
    risks: list[dict] = []
    if isinstance(raw_risks, list):
        for item in raw_risks:
            if isinstance(item, dict) and str(item.get("risk") or "").strip():
                risks.append({
                    "risk": str(item["risk"]).strip(),
                    "basis": str(item.get("basis") or "待确认").strip(),
                })
            elif isinstance(item, str) and item.strip():
                risks.append({"risk": item.strip(), "basis": "待确认"})
    if not risks:                                      # 降级：案例已知风险
        risks = [
            {"risk": r, "basis": "案例已知风险"} for r in public_case.get("known_risks") or []
        ]
    return risks[:6]


def _sources(knowledge_context: list[dict]) -> list[dict]:
    """引用溯源：按 doc 去重（确定性字段，规则渲染）。"""
    seen = {}
    for item in knowledge_context:
        doc = item.get("doc_name")
        key = (item.get("doc_id") or doc, item.get("version"))
        if doc and key not in seen:
            entry = {"doc": doc, "priority": int(item.get("priority") or 0)}
            entry.update({k: item[k] for k in ("doc_id", "category", "industry", "source", "version", "effective_date", "status", "topic") if k in item})
            seen[key] = entry
    return list(seen.values())


def assemble_briefing(
    *,
    public_case: dict,
    raw: dict,
    knowledge_context: list[dict],
    clarifications: list[dict],
    knowledge_missing: bool,
    knowledge_message: str,
    settings: Settings,
) -> dict:
    """组装简报（§7.2 Schema）：LLM 字段 + 确定性字段覆盖。"""
    briefing_cfg = settings.agents.briefing
    industry = public_case.get("industry", "其他")
    return {
        "case_id": public_case.get("case_id", ""),
        "industry": industry,                          # 受控枚举，覆盖 LLM 输出
        "customer_profile": {
            "company": public_case.get("company", ""),
            "industry": industry,
            "role_in_chain": public_case.get("client_role") or "待确认",
        },
        "visit_goals": (
            _normalize_str_list(raw.get("visit_goals"), briefing_cfg.max_goals)
            or list(DEFAULT_GOALS)
        ),
        "priority_questions": _normalize_questions(
            raw.get("priority_questions"), clarifications, briefing_cfg.max_questions,
        ),
        "possible_objections": _normalize_objections(raw.get("possible_objections")),
        "non_goals": _normalize_str_list(raw.get("non_goals"), 4) or list(DEFAULT_NON_GOALS),
        "clarifications": clarifications,
        "risk_warnings": _normalize_risks(raw.get("risk_warnings"), public_case),
        "compliance_warnings": [                        # config 规则渲染，不交给 LLM
            f"禁止{rule}" for rule in settings.guardrail.rules.forbidden
        ],
        "sources": _sources(knowledge_context),
        "knowledge_missing": {
            "missing": knowledge_missing,
            "message": knowledge_message if knowledge_missing else "",
        },
    }


__all__ = [
    "make_briefing_node",
    "rule_clarifications",
    "assemble_briefing",
    "DEFAULT_GOALS",
    "DEFAULT_NON_GOALS",
]
