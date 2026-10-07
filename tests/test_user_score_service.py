"""用户成绩画像聚合单元测试（FakeSession 注入，不触库）。"""
import contextlib
from datetime import datetime
from types import SimpleNamespace

from app.models import Evaluation, EvaluationEvidence
from app.services.user_score_service import UserScoreService


def _cond_value(cond):
    """从 SQLAlchemy 条件中取绑定值（等值 → 标量；in_ → 集合）。"""
    node = getattr(cond, "right", None)
    for _ in range(3):
        if node is None:
            return None
        if isinstance(node, (list, tuple, set)):
            return node
        value = getattr(node, "value", None)
        if isinstance(value, (list, tuple, set)):
            return value
        if value is not None:
            return value
        node = getattr(node, "element", None)
    return None


class FakeUserSessionService:
    """只读会话列表 + 会话工厂（评估按 session_id 回查 / 证据按 evaluation_id 回查）。"""

    def __init__(self, sessions, evaluations, evidence_rows=None):
        self.sessions = sessions
        self.evaluations = evaluations      # session_id -> Evaluation row
        self.evidence_rows = evidence_rows or []

    def list_sessions_by_user(self, user_id):
        return [s for s in self.sessions if s["user_id"] == user_id]

    @property
    def session_factory(self):
        evaluations = self.evaluations
        evidence_rows = [SimpleNamespace(**r) for r in self.evidence_rows]

        class _EvalQuery:
            def __init__(self, rows):
                self._rows = rows
            def filter(self, *conditions):
                sid = None
                for cond in conditions:
                    sid = _cond_value(cond)
                row = evaluations.get(sid)
                return _EvalQuery([row] if row else [])
            def one_or_none(self):
                return self._rows[0] if self._rows else None
            def all(self):
                return list(self._rows)

        class _EvidenceQuery:
            def __init__(self, rows):
                self._rows = rows
            def filter(self, *conditions):
                ids = None
                for cond in conditions:
                    value = _cond_value(cond)
                    if isinstance(value, (list, tuple, set)):
                        ids = set(value)
                if ids is None:
                    return _EvidenceQuery([])
                return _EvidenceQuery([r for r in evidence_rows
                                       if r.evaluation_id in ids])
            def order_by(self, *args, **kwargs):
                return self
            def all(self):
                return list(self._rows)

        class _S:
            def query(self, model):
                if model is EvaluationEvidence:
                    return _EvidenceQuery(evidence_rows)
                return _EvalQuery([])
            def close(self):
                pass

        return lambda: contextlib.closing(_S())


def make_eval(session_id, opp, sales, created=None, id=None, missing=None,
              next_action=None):
    return Evaluation(
        id=id, session_id=session_id, opportunity=opp, sales_score=sales,
        missing=missing or {"items": [], "errors": []},
        next_action=next_action or {"actions": [], "errors": []},
        created_at=created or datetime(2026, 9, 6, 12, 0, 0),
    )


DEF_SESSIONS = [
    {"session_id": "s1", "user_id": 7, "case_id": "INFRA_001", "status": "finished", "turn_count": 3},
    {"session_id": "s2", "user_id": 7, "case_id": "PHARMA_001", "status": "finished", "turn_count": 2},
    {"session_id": "s3", "user_id": 7, "case_id": "ECOM_001", "status": "active", "turn_count": 1},
]


class TestAggregate:

    def test_mixed_sessions_aggregate(self):
        evals = {
            "s1": make_eval("s1",
                            {"score": 82, "level": "高潜机会",
                             "dimension_scores": {"payer_confirmation": {"gated_score": 70}}},
                            {"total": 74, "dimension_scores": {"compliance": {"gated_score": 90}}}),
            "s2": make_eval("s2",
                            {"score": 60, "level": "观察", "dimension_scores": {}},
                            {"total": 66, "dimension_scores": {}}),
        }
        svc = UserScoreService(session_service=FakeUserSessionService(DEF_SESSIONS, evals))
        report = svc.scores(7)
        assert report["total_sessions"] == 3
        assert report["evaluated_sessions"] == 2        # active s3 不计入
        assert report["avg_opportunity"] == 71.0        # (82+60)/2
        assert report["avg_sales_score"] == 70.0        # (74+66)/2
        assert report["level_distribution"] == {"高潜机会": 1, "观察": 1, "暂缓": 0}
        assert report["dimension_averages"] == {"payer_confirmation": 70.0, "compliance": 90.0}
        assert len(report["latest_sessions"]) == 2

    def test_no_evaluations_empty_profile(self):
        svc = UserScoreService(session_service=FakeUserSessionService(DEF_SESSIONS, {}))
        report = svc.scores(7)
        assert report["evaluated_sessions"] == 0
        assert report["avg_opportunity"] == 0.0
        assert report["avg_sales_score"] == 0.0
        assert report["level_distribution"] == {"高潜机会": 0, "观察": 0, "暂缓": 0}
        assert report["dimension_averages"] == {}

    def test_unfinished_session_ignored(self):
        # s1 finished 有评估；s3 active 无评估
        evals = {"s1": make_eval("s1", {"score": 90, "level": "高潜机会", "dimension_scores": {}},
                                 {"total": 80, "dimension_scores": {}})}
        svc = UserScoreService(session_service=FakeUserSessionService(DEF_SESSIONS, evals))
        report = svc.scores(7)
        assert report["evaluated_sessions"] == 1
        assert report["avg_opportunity"] == 90.0


class TestCoaching:
    """五段式辅导板块（一机会诊断 / 二复盘辅导 / 三证据 / 四遗漏 / 五改善）。"""

    def _make_service(self):
        evals = {
            "s1": make_eval(
                "s1",
                {"score": 82, "level": "高潜机会", "dimension_scores": {
                    "payer_confirmation": {"gated_score": 70, "evidence_missing": False},
                    "financing_need": {"gated_score": 40, "evidence_missing": True}}},
                {"total": 74, "dimension_scores": {
                    "compliance": {"gated_score": 90, "evidence_missing": False}}},
                id=11,
                missing={"items": [
                    {"item": "验收状态", "impact": "未验收部分应收不成立",
                     "suggestion": "索取竣工验收资料"},
                ], "errors": []},
                next_action={"actions": [
                    {"action": "补问验收状态", "priority": "高", "note": "下次拜访优先"},
                ], "errors": []},
            ),
            "s2": make_eval(
                "s2",
                {"score": 60, "level": "观察", "dimension_scores": {
                    "payer_confirmation": {"gated_score": 50}}},
                {"total": 66, "dimension_scores": {}},
                created=datetime(2026, 9, 5, 10, 0, 0), id=22,
                missing={"items": [
                    {"item": "验收状态", "impact": "未验收部分应收不成立",
                     "suggestion": "索取竣工验收资料"},
                ], "errors": []},
            ),
        }
        evidence = [
            {"evaluation_id": 11, "dimension": "payer_confirmation", "turn": 1,
             "quote_sales": "项目最终付款方是谁", "quote_customer": "市交通集团",
             "comment": "已确认付款来源", "score": 70.0},
            {"evaluation_id": 11, "dimension": "payer_confirmation", "turn": 2,
             "quote_sales": "账期多久", "quote_customer": "180 天",
             "comment": "账期确认", "score": 68.0},
        ]
        return UserScoreService(session_service=FakeUserSessionService(
            DEF_SESSIONS, evals, evidence,
        ))

    def test_opportunity_and_review_sections(self):
        c = self._make_service().scores(7)["coaching"]
        # 一、机会诊断：均分 + 等级分布 + 逐维中文均值
        assert c["opportunity"]["avg_score"] == 71.0            # (82+60)/2
        assert c["opportunity"]["level_distribution"] == {"高潜机会": 1, "观察": 1, "暂缓": 0}
        opp_dims = {d["key"]: d for d in c["opportunity"]["dimension_averages"]}
        assert opp_dims["payer_confirmation"]["avg"] == 60.0    # (70+50)/2
        assert opp_dims["payer_confirmation"]["label"] == "付款来源"
        # 二、复盘辅导：能力均值 + 逐维
        assert c["review"]["avg_total"] == 70.0                 # (74+66)/2
        assert c["review"]["dimension_averages"][0]["label"] == "合规意识"

    def test_evidence_section(self):
        c = self._make_service().scores(7)["coaching"]
        by_dim = c["evidence"]["by_dimension"]
        assert by_dim == [{"key": "payer_confirmation", "label": "付款来源", "quotes": 2}]
        latest = c["evidence"]["latest"]
        assert latest["session_id"] == "s1"                      # 最近一次（09-06）
        assert latest["items"][0]["quote_sales"] == "项目最终付款方是谁"
        assert latest["items"][0]["dimensions"][0]["label"] == "付款来源"

    def test_latest_evidence_items_sorted_by_turn(self):
        # 同一评估下，多项证据按轮次乱序给出 → 应按问答顺序（turn 升序）重新编排
        evals = {"s1": make_eval(
            "s1",
            {"score": 82, "level": "高潜机会",
             "dimension_scores": {"payer_confirmation": {"gated_score": 70, "evidence_missing": False}}},
            {"total": 74, "dimension_scores": {"compliance": {"gated_score": 90, "evidence_missing": False}}},
            id=11,
        )}
        evidence = [
            {"evaluation_id": 11, "dimension": "payer_confirmation", "turn": 2,
             "quote_sales": "账期多久", "quote_customer": "180 天", "comment": "", "score": 68.0},
            {"evaluation_id": 11, "dimension": "payer_confirmation", "turn": 0,
             "quote_sales": "", "quote_customer": "款子压了快半年", "comment": "", "score": 0.0},
            {"evaluation_id": 11, "dimension": "payer_confirmation", "turn": 1,
             "quote_sales": "项目最终付款方是谁", "quote_customer": "市交通集团", "comment": "", "score": 70.0},
        ]
        svc = UserScoreService(
            session_service=FakeUserSessionService(DEF_SESSIONS, evals, evidence),
        )
        latest = svc.scores(7)["coaching"]["evidence"]["latest"]
        assert latest["session_id"] == "s1"
        assert [it["turn"] for it in latest["items"]] == [0, 1, 2]   # 按问答顺序
        assert latest["items"][0]["quote_sales"] == ""               # turn 0：客户开场
        assert latest["items"][1]["quote_sales"] == "项目最终付款方是谁"

    def test_evidence_group_merges_same_turn_dimensions(self):
        # 同一轮的多条维度证据 → 合并为一个 turn 组：原话一次，维度列表多条
        evals = {"s1": make_eval(
            "s1",
            {"score": 82, "level": "高潜机会",
             "dimension_scores": {"payer_confirmation": {"gated_score": 70, "evidence_missing": False}}},
            {"total": 74, "dimension_scores": {"compliance": {"gated_score": 90, "evidence_missing": False}}},
            id=11,
        )}
        evidence = [
            {"evaluation_id": 11, "dimension": "transaction_authenticity", "turn": 1,
             "quote_sales": "付款方是谁", "quote_customer": "市交通集团", "comment": "", "score": 0.0},
            {"evaluation_id": 11, "dimension": "receivable_quality", "turn": 1,
             "quote_sales": "付款方是谁", "quote_customer": "市交通集团", "comment": "", "score": 20.0},
            {"evaluation_id": 11, "dimension": "payer_confirmation", "turn": 0,
             "quote_sales": "", "quote_customer": "款子压了半年", "comment": "", "score": 0.0},
        ]
        svc = UserScoreService(
            session_service=FakeUserSessionService(DEF_SESSIONS, evals, evidence),
        )
        items = svc.scores(7)["coaching"]["evidence"]["latest"]["items"]
        assert len(items) == 2                          # turn0 一组 + turn1 一组
        assert items[0]["turn"] == 0
        assert items[1]["turn"] == 1
        assert items[1]["quote_sales"] == "付款方是谁"     # 同轮原话只显示一次
        assert len(items[1]["dimensions"]) == 2          # 同轮合并两个维度
        assert [d["label"] for d in items[1]["dimensions"]] == ["交易真实性", "应收质量"]
        assert items[0]["dimensions"][0]["label"] == "付款来源"

    def test_missing_and_improvement_sections(self):
        c = self._make_service().scores(7)["coaching"]
        # 四、遗漏问题：跨会话计数
        assert c["missing"][0]["item"] == "验收状态"
        assert c["missing"][0]["count"] == 2
        # 五、改善空间：最低维 financing_need(40) 排首；证据缺口 = financing_need
        weakest = c["improvement"]["weakest_dimensions"]
        assert weakest[0]["key"] == "financing_need" and weakest[0]["avg"] == 40.0
        gaps = c["improvement"]["evidence_gaps"]
        assert gaps[0]["key"] == "financing_need" and gaps[0]["sessions"] == 1
        assert c["improvement"]["next_actions"][0]["action"] == "补问验收状态"

    def test_no_evaluations_empty_coaching(self):
        svc = UserScoreService(session_service=FakeUserSessionService(DEF_SESSIONS, {}))
        c = svc.scores(7)["coaching"]
        assert c["opportunity"]["avg_score"] == 0.0
        assert c["opportunity"]["dimension_averages"] == []
        assert c["review"]["dimension_averages"] == []
        assert c["evidence"]["by_dimension"] == []
        assert c["evidence"]["latest"]["items"] == []
        assert c["missing"] == []
        assert c["improvement"]["weakest_dimensions"] == []
        assert c["improvement"]["evidence_gaps"] == []
        assert c["improvement"]["next_actions"] == []

    def test_latest_sessions_public_shape(self):
        report = self._make_service().scores(7)
        entry = report["latest_sessions"][0]
        assert set(entry.keys()) == {
            "session_id", "case_id", "opportunity_score", "level",
            "sales_score", "generated_at",
        }                                                       # 聚合原料不出 API


class TestScoresStream:

    def _fixture(self):
        return TestCoaching()._make_service()               # 复用五段式数据

    def test_event_sequence(self):
        svc = self._fixture()
        events = list(svc.scores_stream(7))
        types = [e["type"] for e in events]
        assert types[0] == "profile" and types[-1] == "done"
        keys = [e["key"] for e in events if e["type"] == "section"]
        assert keys == ["opportunity", "review", "evidence",
                        "missing", "improvement"]           # 一 → 五 逐段产出
        profile = events[0]["data"]
        assert profile["evaluated_sessions"] == 2
        assert "coaching" not in profile                    # 画像帧不含辅导段

    def test_stream_assembles_into_scores(self):
        svc = self._fixture()
        events = list(svc.scores_stream(7))
        streamed = dict(next(e["data"] for e in events if e["type"] == "profile"))
        streamed["coaching"] = {e["key"]: e["data"] for e in events
                                if e["type"] == "section"}
        assert streamed == svc.scores(7)                    # 两条路径同构

    def test_empty_user_still_full_sequence(self):
        svc = UserScoreService(session_service=FakeUserSessionService(DEF_SESSIONS, {}))
        events = list(svc.scores_stream(7))
        keys = [e["key"] for e in events if e["type"] == "section"]
        assert keys == ["opportunity", "review", "evidence", "missing", "improvement"]
        assert events[0]["data"]["evaluated_sessions"] == 0
