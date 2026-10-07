"""评估编排与持久化（TDD §13，非独立 Agent）。

EvaluationService 是 `GET /sessions/{id}/evaluation` 的惰性计算入口：
读 t_conversation_message（对话）→ 案例行业（不含内部事实）→ 依次执行
evidence → opportunity / sales_score / missing_question / next_action →
幂等写入 t_evaluation + t_evaluation_evidence，并返回可读报告。

幂等：MySQL领取锁覆盖模型调用及事务提交，session_id唯一约束；返回不可变报告快照。
依赖可注入（llm / case_service / session_service / session_factory），
单测用替身，集成测试走真实 DeepSeek + MySQL。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, ContextManager, Optional

from app.config import Settings, get_settings
from app.db.mysql import get_mysql_database
from app.evaluation.common import DIM_LABELS
from app.evaluation.evidence import EvidenceExtractor
from app.evaluation.missing_question import MissingQuestionAnalyzer
from app.evaluation.next_action import NextActionAdvisor
from app.evaluation.opportunity import OpportunityDiagnoser
from app.evaluation.report import render_markdown
from app.evaluation.sales_score import SalesScorer
from app.guardrail.detector import GuardrailService
from app.guardrail.rewriter import GuardrailRewriter
from app.models import Evaluation, EvaluationEvidence
from app.services.session_service import SessionServiceError
import copy
from app.evaluation.lease import DatabaseLease, local_lease
from app.evaluation.contracts import verified_evidence, MissingItem, ActionItem, CoachingItem, SourceSnapshot
from app.evaluation.knowledge import active_sources, coaching

logger = logging.getLogger(__name__)

# 流式评估阶段（key → 中文标签）：section 键与报告字段同名
EVAL_STAGES = (
    ("evidence", "抽取评分证据"),
    ("opportunity", "机会诊断评分"),
    ("sales", "销售能力评分"),
    ("missing", "遗漏问题核对"),
    ("next_action", "生成下一步行动"),
    ("persist", "评估落库与报告生成"),
)


class EvaluationError(RuntimeError):
    """评估错误（API 层翻译 404 等）。"""


class EvaluationService:
    """评估编排器（机会诊断 + 能力评分 + 证据 + 遗漏 + 行动 + 落库）。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        llm: Any = None,
        case_service: Any = None,
        session_service: Any = None,
        guardrail_service: Any = None,
        session_factory: Optional[Callable[[], ContextManager]] = None,
        lease_provider=None,
        knowledge_provider=None,
    ) -> None:
        self.settings = settings or get_settings()
        self._llm = llm
        self._case_service = case_service
        self._session_service = session_service
        self._guardrail_service = guardrail_service
        self.session_factory = session_factory or get_mysql_database().session_scope
        self.lease_provider = lease_provider or (local_lease if session_factory is not None else DatabaseLease(get_mysql_database().engine))
        self.knowledge_provider = knowledge_provider or ((lambda industry: []) if session_factory is not None else (lambda industry: active_sources(self.session_factory, industry)))
        self.job_namespace = id(session_factory) if session_factory is not None else 'mysql:' + self.settings.database.mysql.database

    # -------------------------------------------------
    # 懒加载依赖（真实组件首次使用才构造，便于测试注入替身）
    # -------------------------------------------------

    @property
    def llm(self):
        if self._llm is None:
            from app.services.llm_service import LLMService
            self._llm = LLMService(self.settings)
        return self._llm

    @property
    def case_service(self):
        if self._case_service is None:
            from app.services.case_service import CaseService
            self._case_service = CaseService(self.settings)
        return self._case_service

    @property
    def session_service(self):
        if self._session_service is None:
            from app.services.session_service import SessionService
            self._session_service = SessionService(self.settings)
        return self._session_service

    @property
    def guardrail_service(self):
        if self._guardrail_service is None:
            from app.compliance.engine import ComplianceEngine
            self._guardrail_service = ComplianceEngine(self.settings, llm=self.llm)
        return self._guardrail_service

    # -------------------------------------------------
    # 入口
    # -------------------------------------------------

    def get(self, session_id: str) -> dict:
        """GET 惰性入口：已评估读库返回，否则计算并落库。"""
        existing = self._load_existing(session_id)
        if existing is not None:
            return existing
        return self.run(session_id)

    def run(self, session_id: str) -> dict:
        for event in self.run_stream(session_id):
            if event.get('type') == 'done':
                return event['report']
        raise EvaluationError('evaluation_incomplete')

    def run_stream(self, session_id: str):
        existing = self._load_existing(session_id)
        if existing is not None:
            yield {'type': 'done', 'report': existing}
            return
        from app.evaluation.jobs import stream_job
        yield from stream_job((self.job_namespace, session_id), lambda: self._run_stream_owned(session_id))

    def _run_stream_owned(self, session_id: str):
        existing = self._load_existing(session_id)
        if existing is not None:
            yield {'type': 'done', 'report': existing}
            return
        with self.lease_provider(session_id):
            # 锁取得后再次读库；并发赢家已提交则零模型短路。
            existing = self._load_existing(session_id)
            if existing is not None:
                yield {'type': 'done', 'report': existing}
                return
            session = self.session_service.get_session(session_id)
            if session is None:
                raise EvaluationError('会话不存在: ' + session_id)
            if session.get('status') not in (None, 'finished'):
                raise EvaluationError('演练尚未结束')
            case = self.case_service.get_case(session['case_id'])
            if case is None:
                raise EvaluationError('案例不存在: ' + session['case_id'])
            # 只保留行业；不把完整案例、事实卡或hidden_risks送入评估。
            safe_case = {'industry': case.get('industry', '') if isinstance(case, dict) else getattr(case, 'industry', '')}
            conversation = self.session_service.list_messages(session_id)
            audits = self.session_service.list_compliance(session_id) if hasattr(self.session_service, 'list_compliance') else []
            safe_case['_compliance_events'] = audits
            safe_case['_sources'] = [SourceSnapshot.model_validate(s).model_dump() for s in self.knowledge_provider(safe_case['industry'])]
            hits = _collect_guardrail_hits(conversation)
            yield from self._run_stages(session_id, conversation, safe_case, hits)

    # -------------------------------------------------
    # 编排 + 落库
    # -------------------------------------------------

    def _run_stages(
        self,
        session_id: str,
        conversation: list[dict],
        case_full: dict,
        guardrail_hits: list[dict],
    ):
        """分阶段执行评估（生成器，run()/run_stream() 共用）。

        每阶段先 yield stage(running)，完成后 yield 该阶段产物
        （section 事件，键与报告字段同名）或 stage(done)。
        """
        labels = dict(EVAL_STAGES)

        def stage(key: str, status: str) -> dict:
            return {"type": "stage", "stage": key,
                    "label": labels[key], "status": status}

        yield stage("evidence", "running")
        evidence = EvidenceExtractor(self.settings).extract(
            conversation, guardrail_hits=guardrail_hits, case=case_full,
        )
        evidence = verified_evidence(evidence, conversation)
        yield stage("evidence", "done")

        yield stage("opportunity", "running")
        opportunity = OpportunityDiagnoser(self.llm, self.settings).analyze(
            conversation, case_full, evidence,
        )


        yield stage("sales", "running")
        sales = SalesScorer(self.llm, self.settings).score(
            conversation, case_full, evidence,
        )


        yield stage("missing", "running")
        missing = [MissingItem.model_validate(m).model_dump() for m in MissingQuestionAnalyzer(self.settings).analyze(conversation, case_full)]


        yield stage("next_action", "running")
        next_actions = NextActionAdvisor().advise(
            opportunity, missing_count=len(missing),
        )


        next_actions = [ActionItem.model_validate(a).model_dump() for a in next_actions]
        # 模型引用被剔除后，最终证据集合以门控后的真实引用为准。
        dimensions = {**opportunity['dimension_scores'], **sales['dimension_scores']}
        evidence = {dim: [e for e in items if any(r['turn_id'] == e['turn'] and r['original_text'] == e['quote_sales']
            for r in dimensions.get(dim, {}).get('evidence_refs', []))] for dim, items in evidence.items()}
        training = coaching(evidence, case_full.get('_sources', []))
        yield stage("persist", "running")
        errors = list(opportunity.get("errors") or []) + list(sales.get("errors") or [])
        total_score = opportunity.get("score", 0)
        # 系统输出侧合规改写（TDD §14 action=rewrite）：建议/行动文本命中违禁则改写并留痕
        self._apply_output_guardrail(session_id, opportunity, missing, next_actions, sales)
        if hasattr(self.guardrail_service, 'protect_tree') and self.guardrail_service.enabled:
            self.guardrail_service.protect_tree(training, session_id=session_id, session_service=self.session_service)
        # 合规改写后再次检查最终结构，保留固定优先级及证据字段。
        missing = [MissingItem.model_validate(m).model_dump() for m in missing]
        next_actions = [ActionItem.model_validate(a).model_dump() for a in next_actions]
        training = [CoachingItem.model_validate(c).model_dump() for c in training]
        report = self._compose(session_id, opportunity, sales, missing, next_actions, total_score, evidence, None)
        report.update(schema_version='2.0', sources=case_full.get('_sources', []), coaching=training,
            degraded=any(e == 'deterministic_fallback' or e.startswith('evidence_rejected:') for e in errors), markdown='')
        report['markdown'] = render_markdown(report)
        self._persist(session_id, opportunity, sales, missing, next_actions, total_score, evidence, errors, snapshot=report)
        # 完整报告与证据同事务提交后才发业务段；断开SSE不能留下半份报告。
        for key, data in (("opportunity", opportunity), ("sales_score", sales),
                          ("missing_questions", missing), ("next_actions", next_actions)):
            yield {"type": "section", "key": key, "data": data}
        logger.info("评估完成: session=%s total=%s(%s)", session_id, total_score, opportunity.get("level"))
        yield {"type": "done", "report": report}

    def _persist(
        self,
        session_id: str,
        opportunity: dict,
        sales: dict,
        missing: list[dict],
        next_actions: list[dict],
        total_score: float,
        evidence: dict[str, list],
        errors: list[str],
        snapshot=None,
    ) -> int:
        """写入 t_evaluation + t_evaluation_evidence，返回 evaluation id。"""
        with self.session_factory() as session:
            row = Evaluation(
                session_id=session_id,
                opportunity=opportunity,
                sales_score=sales,
                missing={"items": missing, "errors": errors},
                next_action={"actions": next_actions, "errors": errors},
                total_score=total_score,
                report_snapshot=copy.deepcopy(snapshot),
            )
            session.add(row)
            session.flush()                             # 取 evaluation id
            ev_id = int(row.id)
            for ev in self._evidence_rows(evidence, opportunity, sales):
                session.add(EvaluationEvidence(
                    evaluation_id=ev_id,
                    dimension=ev["dimension"],
                    score=ev["score"],
                    turn=ev["turn"],
                    quote_sales=ev["quote_sales"],
                    quote_customer=ev["quote_customer"],
                    comment=ev["comment"],
                ))
            return ev_id

    # -------------------------------------------------
    # 系统输出侧合规改写（TDD §14 action=rewrite）
    # -------------------------------------------------

    def _apply_output_guardrail(
        self,
        session_id: str,
        opportunity: dict,
        missing: list[dict],
        next_actions: list[dict],
        sales: dict | None = None,
    ) -> None:
        """对报告/建议文本做违禁检测 + 命中 LLM 改写 + 落 rewrite 留痕。

        仅在命中时产生 LLM 调用（多数报告不命中为零成本）；在持久化之前调用，
        保证落库的是已改写文本。
        """
        guardrail = self.guardrail_service
        if not getattr(guardrail, "enabled", False):
            return
        if hasattr(guardrail, "protect_tree"):
            result = guardrail.protect_tree([opportunity, missing, next_actions, sales or {}],
                session_id=session_id, session_service=self.session_service)
            opportunity["compliance_status"] = dict(l2_degraded=result.l2_degraded,
                l1_elapsed_ms=result.l1_elapsed_ms, l2_elapsed_ms=result.l2_elapsed_ms)
            if any(h.severity == "block" for h in result.hits) and hasattr(self.session_service, "mark_human_review"):
                self.session_service.mark_human_review(session_id)
            return
        rewriter = GuardrailRewriter(guardrail, self.llm)
        for action in next_actions:
            for key in ("action", "note"):
                if isinstance(action.get(key), str) and action[key]:
                    action[key] = self._rewrite_and_log(session_id, action[key], rewriter)
        for item in missing:
            if isinstance(item.get("suggestion"), str) and item["suggestion"]:
                item["suggestion"] = self._rewrite_and_log(session_id, item["suggestion"], rewriter)
        reasons = opportunity.get("reasons") or []
        opportunity["reasons"] = [
            self._rewrite_and_log(session_id, r, rewriter) if isinstance(r, str) else r
            for r in reasons
        ]

    def _rewrite_and_log(self, session_id: str, text: str, rewriter: GuardrailRewriter) -> str:
        """改写一段文本；命中（含改写后仍残留的）记录到 t_compliance_log（action=rewrite）。"""
        result = rewriter.rewrite(text)
        if result.violations:
            for hit in result.violations:
                try:
                    self.session_service.record_compliance(
                        session_id, 0, hit.rule, hit.snippet, action="rewrite",
                    )
                except Exception as exc:
                    raise SessionServiceError("evaluation_compliance_write_failed") from None
        return result.rewritten

    def _evidence_rows(
        self,
        evidence: dict[str, list],
        opportunity: dict,
        sales: dict,
    ) -> list[dict]:
        """合并机会/能力评分各维度的 gated_score 与证据项，生成证据行。"""
        gated: dict[str, float] = {}
        for key, res in (opportunity.get("dimension_scores") or {}).items():
            gated[key] = res.get("gated_score", 0)
        for key, res in (sales.get("dimension_scores") or {}).items():
            gated[key] = res.get("gated_score", 0)
        rows: list[dict] = []
        for dimension, items in evidence.items():
            if not items:
                continue
            # 已迁移为DECIMAL(5,2)，100分原值保存，不损失满分。
            score = gated.get(dimension, 0)
            for item in items:
                rows.append({
                    "dimension": DIM_LABELS.get(dimension, dimension),
                    "score": score,
                    "turn": item.get("turn"),
                    "quote_sales": item.get("quote_sales", ""),
                    "quote_customer": item.get("quote_customer", ""),
                    "comment": item.get("comment", ""),
                })
        return rows

    # -------------------------------------------------
    # 读库重建报告（幂等路径）
    # -------------------------------------------------

    def _load_existing(self, session_id: str) -> Optional[dict]:
        with self.session_factory() as session:
            row = session.query(Evaluation).filter(
                Evaluation.session_id == session_id,
            ).one_or_none()
            if row is None:
                return None
            if isinstance(getattr(row, 'report_snapshot', None), dict):
                return copy.deepcopy(row.report_snapshot)
            ev_id = row.id
            ev_rows = session.query(EvaluationEvidence).filter(
                EvaluationEvidence.evaluation_id == ev_id,
            ).all()
        report = {
            "session_id": session_id,
            "opportunity": row.opportunity or {},
            "sales_score": row.sales_score or {},
            "missing_questions": (row.missing or {}).get("items", []),
            "next_actions": (row.next_action or {}).get("actions", []),
            "evidence": _read_evidence(ev_rows),
            "total_score": float(row.total_score) if row.total_score is not None else 0.0,
            "generated_at": _iso(row.created_at),
        }
        self._attach_compliance(report, session_id, allow_current_rules=False)
        report["schema_version"] = "legacy"
        report["snapshot_unavailable"] = True
        report["markdown"] = render_markdown(report)
        return report

    def _compose(self, session_id, opportunity, sales, missing, next_actions,
                 total_score, evidence, ev_id) -> dict:
        rows = self._evidence_rows(evidence, opportunity, sales)
        report = {
            "session_id": session_id,
            "opportunity": opportunity,
            "sales_score": sales,
            "missing_questions": missing,
            "next_actions": next_actions,
            "evidence": rows,
            "total_score": total_score,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._attach_compliance(report, session_id)
        report["markdown"] = render_markdown(report)
        return report

    def _attach_compliance(self, report, session_id, *, allow_current_rules=True):
        if hasattr(self.session_service, "list_compliance"):
            from app.guardrail.rules import load_compliance_rules
            rules = {r.rule_id: r for r in load_compliance_rules().rules} if allow_current_rules else {}
            events = self.session_service.list_compliance(session_id)
            for event in events:
                rule = rules.get(event["rule"])
                event["compliant_response"] = (event.get("details") or {}).get("compliant_response") or (rule.compliant_response if rule else "历史审计未保存应答快照，请核实当时规则。")
            report["compliance_events"] = events
            report["needs_human_review"] = any(e["action"] == "block" or (e.get("details") or {}).get("severity") == "block" for e in events) or len(events) >= 3



# -------------------------------------------------
# 工具
# -------------------------------------------------

def _collect_guardrail_hits(conversation: list[dict]) -> list[dict]:
    """从 sales 行的 guardrail_hit 汇总违禁命中（evidence 用）。"""
    hits: list[dict] = []
    for m in conversation or []:
        if m.get("role") != "sales":
            continue
        turn = m.get("turn", 0)
        for h in m.get("guardrail_hit") or []:
            hits.append({
                "turn": turn, "rule": h.get("rule"), "snippet": h.get("snippet"), "severity": h.get("severity", "warn"),
            })
    return hits


def _read_evidence(rows) -> list[dict]:
    return [
        {
            "dimension": r.dimension, "score": float(r.score) if r.score is not None else 0.0,
            "turn": r.turn, "quote_sales": r.quote_sales or "",
            "quote_customer": r.quote_customer or "", "comment": r.comment or "",
        }
        for r in rows
    ]


def _iso(value) -> str:
    return value.isoformat(timespec="seconds") if value else datetime.now().isoformat(timespec="seconds")
