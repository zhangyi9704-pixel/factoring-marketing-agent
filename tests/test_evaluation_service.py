"""EvaluationService 编排与持久化单元测试（全替身，不触库不触网）。"""
import contextlib

import pytest

from app.config import Settings
from app.evaluation.service import EvaluationError, EvaluationService
from app.models import Evaluation, EvaluationEvidence
from tests.test_case_schema import make_case

SID = "sess_eval_1"

CONVERSATION = [
    {"role": "customer", "turn": 0, "content": "我们最近资金压力大，需要融资解决缺口。"},
    {"role": "sales", "turn": 1, "content": "王总，项目付款方是谁？合同签了吗？需要融资多少资金解决缺口？"},
    {"role": "customer", "turn": 1, "content": "是市交通集团，账期180天。"},
    {"role": "sales", "turn": 2, "content": "应收对账和验收完成了吗？有质押吗？"},
    {"role": "customer", "turn": 2, "content": "对账还差一层，验收没完成。"},
]

EVAL_DIMS = {
    "transaction_authenticity": {"score": 90, "rationale": "有合同"},
    "receivable_quality": {"score": 80, "rationale": "应收成立"},
    "payer_confirmation": {"score": 70, "rationale": "付款方明确"},
    "financing_need": {"score": 60, "rationale": "有缺口"},
    "risk_identification": {"score": 50, "rationale": "有逾期"},
    "requirement_discovery": {"score": 80, "rationale": "挖需求"},
    "business_understanding": {"score": 70, "rationale": "主线清晰"},
    "risk_awareness": {"score": 60, "rationale": "问了对账"},
    "compliance": {"score": 90, "rationale": "未违规"},
}


class FakeLLM:
    def __init__(self, dims=None, raises=None):
        self.dims = dims or EVAL_DIMS
        self.raises = raises
        self.calls = []

    def chat_json(self, prompt, *, model_role="reasoning", system=None):
        self.calls.append((model_role, prompt))
        if self.raises:
            raise self.raises
        return {"dimensions": self.dims}


class FakeEvalCaseService:
    def get_case(self, case_id):
        return make_case()


class FakeEvalSessionService:
    def __init__(self, session_row=None, messages=None):
        self.session_row = session_row
        self.messages = messages or CONVERSATION

    def get_session(self, session_id):
        row = self.session_row
        if row is None:
            return None
        return row if row.get("session_id") == session_id else None

    def list_messages(self, session_id):
        return self.messages


class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *conditions):
        rows = self._rows
        for cond in conditions:
            column = getattr(cond, "left", None)
            right = getattr(cond, "right", None)
            value = getattr(right, "value", right)
            if column is not None:
                rows = [r for r in rows if getattr(r, column.name, None) == value]
        return _Query(rows)

    def one_or_none(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class FakeEvalSessionFactory:
    def __init__(self):
        self.evals = []
        self.evidences = []
        self._next = 1

    def __call__(self):
        factory = self

        class _Session:
            def __init__(self):
                self.added = []

            def add(self, obj):
                if isinstance(obj, Evaluation):
                    obj.id = factory._next
                    factory._next += 1
                    factory.evals.append(obj)
                elif isinstance(obj, EvaluationEvidence):
                    obj.id = factory._next
                    factory._next += 1
                    factory.evidences.append(obj)
                else:
                    self.added.append(obj)

            def flush(self):
                pass

            def query(self, model):
                return _Query(factory.evals if model is Evaluation else factory.evidences)

            def close(self):
                pass

        return contextlib.closing(_Session())


_UNSET = object()


def make_service(llm=None, *, session_row=_UNSET, messages=None, factory=None):
    settings = Settings(_env_file=None)
    factory = factory or FakeEvalSessionFactory()
    if session_row is _UNSET:
        session_row = {"session_id": SID, "case_id": "INFRA_001"}
    service = EvaluationService(
        settings,
        llm=llm or FakeLLM(),
        case_service=FakeEvalCaseService(),
        session_service=FakeEvalSessionService(session_row, messages),
        session_factory=factory,
    )
    return service, factory


class TestRun:

    def test_computes_and_persists(self):
        service, factory = make_service()
        report = service.run(SID)
        assert report["session_id"] == SID
        assert report["total_score"] == 73.0
        assert report["opportunity"]["level"] == "观察"
        assert report["sales_score"]["total"] == 75.0
        assert isinstance(report["missing_questions"], list)  # checklist 比对结果（本对话全问到故为空）
        assert report["next_actions"]
        assert report["markdown"]                             # 复盘报告 Markdown
        assert "复盘报告" in report["markdown"]
        assert report["evidence"]                          # 证据链非空
        assert len(factory.evals) == 1
        assert len(factory.evidences) >= 1
        assert factory.evals[0].session_id == SID

    def test_idempotent_no_duplicate(self):
        service, factory = make_service()
        first = service.run(SID)
        second = service.run(SID)
        assert first["total_score"] == second["total_score"]
        assert len(factory.evals) == 1                     # 不重复落库
        assert len(factory.evidences) == len(factory.evidences)

    def test_get_lazy_equivalent_to_run(self):
        service, factory = make_service()
        report = service.get(SID)
        assert report["total_score"] == 73.0
        assert len(factory.evals) == 1

    def test_missing_session_raises(self):
        service, _ = make_service(session_row=None)
        with pytest.raises(EvaluationError, match="会话不存在"):
            service.run(SID)

    def test_llm_failure_degrades_but_persists(self):
        service, factory = make_service(FakeLLM(raises=RuntimeError("LLM 挂了")))
        report = service.run(SID)
        assert report["total_score"] == 0.0
        assert report["opportunity"]["level"] == "暂缓"
        # 降级留痕落库：missing/next_action 的 errors 非空
        assert factory.evals[0].missing["errors"]
        assert factory.evals[0].next_action["errors"]

    def test_evidence_score_stored_at_full_score(self):
        # t_evaluation_evidence.score 为 DECIMAL(4,2)，满分的 gated 100 需钳制到 99.99
        dims = dict(EVAL_DIMS)
        dims["compliance"] = {"score": 100, "rationale": "满分"}
        service, factory = make_service(FakeLLM(dims=dims))
        report = service.run(SID)
        comp = [e for e in report["evidence"] if e["dimension"] == "合规意识"]
        assert comp and comp[0]["score"] == 100

    def test_output_guardrail_rewrites_and_logs_action_rewrite(self):
        # 报告/建议文本命中违禁 → LLM 改写为合规表述 + t_compliance_log(action=rewrite)
        from app.guardrail.detector import GuardrailService

        recorder = RewriteRecorder()
        service, _ = make_service(RewriteLLM())
        service._session_service = recorder
        service._guardrail_service = GuardrailService(Settings(_env_file=None))

        next_actions = [{"action": "我们保证额度肯定批", "priority": "high", "note": "尽快"}]
        opportunity = {"score": 82, "level": "高潜机会", "reasons": ["保证一周内放款"]}
        missing = [{"item": "验收", "impact": "n", "suggestion": "保证审批通过"}]

        service._apply_output_guardrail(SID, opportunity, missing, next_actions)

        assert "以审批结果为准" in next_actions[0]["action"]
        assert "以审批结果为准" in opportunity["reasons"][0]
        assert "以审批结果为准" in missing[0]["suggestion"]
        assert recorder.rows and all(r[2] == "rewrite" for r in recorder.rows)


class RewriteRecorder:
    """记录 record_compliance(action=...) 的替身（评估改写留痕验证用）。"""

    def __init__(self):
        self.rows: list[tuple] = []

    def record_compliance(self, session_id, turn, rule, raw_text, *, action="log"):
        self.rows.append((rule, raw_text, action))


class RewriteLLM:
    """带 chat 面的替身：改写为合规表述。"""

    def chat(self, prompt, *, model_role="chat", system=None):
        return "具体额度以审批结果为准。"

    def chat_json(self, prompt, *, model_role="reasoning", system=None):
        return {"dimensions": {}}


class TestRunStream:

    def test_stage_and_section_sequence(self):
        service, factory = make_service()
        events = list(service.run_stream(SID))
        sections = [e["key"] for e in events if e["type"] == "section"]
        assert sections == ["opportunity", "sales_score",
                            "missing_questions", "next_actions"]
        assert events[0]["type"] == "stage" and events[0]["status"] == "running"
        assert events[-1]["type"] == "done"
        report = events[-1]["report"]
        assert report["total_score"] == 73.0
        # section 与终帧报告同源
        opp = next(e for e in events if e.get("key") == "opportunity")
        assert opp["data"] == report["opportunity"]
        assert len(factory.evals) == 1                       # 流式路径同样落库

    def test_stage_running_precedes_each_section(self):
        service, _ = make_service()
        events = list(service.run_stream(SID))
        order = [(e["type"], e.get("stage") or e.get("key")) for e in events
                 if e["type"] in ("stage", "section")]
        # evidence 先 running 后 done；其余阶段 running 后跟 section
        assert order[0] == ("stage", "evidence")
        assert order[1] == ("stage", "evidence")             # 第二个是 done 态
        assert events[1]["status"] == "done"
        for key, stage in [("opportunity", "opportunity"), ("sales_score", "sales"),
                           ("missing_questions", "missing"),
                           ("next_actions", "next_action")]:
            assert ("stage", stage) in order
            assert order.index(("stage", stage)) < order.index(("section", key))

    def test_idempotent_short_circuits_to_done(self):
        service, factory = make_service()
        list(service.run_stream(SID))                        # 首次流式评估
        second = list(service.run_stream(SID))
        assert [e["type"] for e in second] == ["done"]       # 已评估 → 直接终帧
        assert second[0]["report"]["total_score"] == 73.0
        assert len(factory.evals) == 1

    def test_run_and_run_stream_equivalent(self):
        service_a, _ = make_service()
        service_b, _ = make_service()
        via_run = service_a.run(SID)
        via_stream = [e for e in service_b.run_stream(SID)
                      if e["type"] == "done"][0]["report"]
        assert via_run["total_score"] == via_stream["total_score"]
        assert via_run["markdown"] == via_stream["markdown"]

    def test_missing_session_raises(self):
        service, _ = make_service(session_row=None)
        with pytest.raises(EvaluationError, match="会话不存在"):
            list(service.run_stream(SID))
