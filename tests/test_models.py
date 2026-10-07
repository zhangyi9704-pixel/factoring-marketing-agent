"""ORM 模型与业务表测试。

元数据测试（TestMetadata）不触网；
CRUD 集成测试（TestCrudRoundtrip）需要可达的 MySQL（本地 phpStudy），
使用带 __test_ 前缀的标记数据，finally 中清理。
"""
import pytest
from sqlalchemy import delete, select

from app.models import (
    Base,
    Case,
    CoachSession,
    ComplianceLog,
    ConversationMessage,
    Evaluation,
    EvaluationEvidence,
    KnowledgeDoc,
    User,
    VisitReport,
)

EXPECTED_TABLES = {
    "t_user", "t_case", "t_session", "t_conversation_message",
    "t_visit_report", "t_evaluation", "t_evaluation_evidence",
    "t_compliance_log", "t_knowledge_doc", "t_agent_error",
    "t_message_request",
    "t_knowledge_pending",
    "t_knowledge_draft", "t_business_audit", "t_knowledge_ingestion_operation", "t_session_create_request",
}

TEST_USER = "__test_coach_user__"
TEST_CASE = "__TEST_001__"
TEST_SESSION = "__test_sess_001__"


class TestMetadata:
    """纯元数据校验，不触网。"""

    def test_business_and_request_tables_registered(self):
        assert set(Base.metadata.tables) == EXPECTED_TABLES

    def test_key_columns(self):
        cols = Base.metadata.tables["t_conversation_message"].columns
        for name in ("session_id", "turn", "role", "content", "revealed_info", "guardrail_hit"):
            assert name in cols
        eval_cols = Base.metadata.tables["t_evaluation"].columns
        for name in ("opportunity", "sales_score", "missing", "next_action", "total_score"):
            assert name in eval_cols

    def test_unique_constraints(self):
        tables = Base.metadata.tables
        assert tables["t_user"].columns["username"].unique
        assert tables["t_case"].columns["case_id"].unique
        assert tables["t_session"].columns["session_id"].unique
        assert tables["t_evaluation"].columns["session_id"].unique
        assert tables["t_visit_report"].columns["session_id"].unique

    def test_indexes(self):
        tables = Base.metadata.tables
        session_indexes = {ix.name for ix in tables["t_session"].indexes}
        assert {"idx_user", "idx_case"} <= session_indexes
        msg_indexes = {ix.name for ix in tables["t_conversation_message"].indexes}
        assert "idx_session" in msg_indexes


class TestCrudRoundtrip:
    """建表 + 增删查回环（集成测试）。"""

    @pytest.mark.integration
    def test_create_all_and_crud(self):
        from app.db.mysql import get_mysql_database

        from uuid import uuid4
        run_id = uuid4().hex[:16]
        TEST_USER = "__test_user_" + run_id
        TEST_CASE = "__TEST_" + run_id
        TEST_SESSION = "__test_sess_" + run_id
        db = get_mysql_database()

        # 幂等建表（连续两次）
        Base.metadata.create_all(db.engine)
        Base.metadata.create_all(db.engine)

        try:
            # 写入：覆盖核心业务链路 用户→案例→会话→对话→评估→证据
            with db.session_scope() as s:
                user = User(username=TEST_USER, password_hash="hash", role="sales", team="华东组")
                case = Case(
                    case_id=TEST_CASE, industry="基建", company="测试建设有限公司",
                    client_role="施工企业财务总监",
                    case_json={"transaction_chain": {"payer": "某交通集团"},
                               "unknown_info": ["是否完成验收"], "behavior_rules": []},
                )
                s.add(user)
                s.add(case)
                s.flush()
                assert user.id is not None and case.id is not None

                sess = CoachSession(session_id=TEST_SESSION, user_id=user.id, case_id=TEST_CASE)
                s.add(sess)
                s.add(ConversationMessage(
                    session_id=TEST_SESSION, turn=1, role="sales", content="项目最终付款方是谁？",
                ))
                s.add(ConversationMessage(
                    session_id=TEST_SESSION, turn=2, role="customer", content="某交通集团。",
                    revealed_info={"payer": True},
                ))
                evaluation = Evaluation(
                    session_id=TEST_SESSION,
                    opportunity={"score": 82, "level": "高潜机会"},
                    sales_score={"risk_awareness": 8},
                    total_score=82.0,
                )
                s.add(evaluation)
                s.flush()
                s.add(EvaluationEvidence(
                    evaluation_id=evaluation.id, dimension="付款主体确认",
                    score=8.0, turn=2, quote_sales="项目最终付款方是谁？",
                    quote_customer="某交通集团。", comment="完成关键付款来源确认",
                ))
                s.add(ComplianceLog(session_id=TEST_SESSION, turn=1, rule="保证融资额度",
                                    action="rewrite"))
                s.add(VisitReport(session_id=TEST_SESSION,
                                  report_json={"visit_goals": ["确认付款来源"]}))

            # 读回验证：默认值、JSON 回读、关联查询
            with db.session_scope() as s:
                got_sess = s.execute(
                    select(CoachSession).where(CoachSession.session_id == TEST_SESSION)
                ).scalar_one()
                assert got_sess.status == "active"      # 默认值
                assert got_sess.turn_count == 0
                assert got_sess.phase is None

                got_case = s.execute(select(Case).where(Case.case_id == TEST_CASE)).scalar_one()
                assert got_case.enabled is True
                assert got_case.case_json["unknown_info"] == ["是否完成验收"]  # JSON 回读

                evaluation = s.execute(
                    select(Evaluation).where(Evaluation.session_id == TEST_SESSION)
                ).scalar_one()
                assert float(evaluation.total_score) == 82.0
                assert evaluation.opportunity["level"] == "高潜机会"

                evidences = s.execute(
                    select(EvaluationEvidence).where(EvaluationEvidence.evaluation_id == evaluation.id)
                ).scalars().all()
                assert len(evidences) == 1
                assert evidences[0].turn == 2

                logs = s.execute(
                    select(ComplianceLog).where(ComplianceLog.session_id == TEST_SESSION)
                ).scalar_one()
                assert logs.review_status == "pending"

                turns = s.execute(
                    select(ConversationMessage.turn)
                    .where(ConversationMessage.session_id == TEST_SESSION)
                    .order_by(ConversationMessage.turn)
                ).scalars().all()
                assert turns == [1, 2]
        finally:
            # 清理测试数据（按依赖逆序）
            with db.session_scope() as s:
                ev_ids = select(Evaluation.id).where(Evaluation.session_id == TEST_SESSION)
                s.execute(delete(EvaluationEvidence).where(EvaluationEvidence.evaluation_id.in_(ev_ids)))
                s.execute(delete(Evaluation).where(Evaluation.session_id == TEST_SESSION))
                s.execute(delete(ConversationMessage).where(ConversationMessage.session_id == TEST_SESSION))
                s.execute(delete(ComplianceLog).where(ComplianceLog.session_id == TEST_SESSION))
                s.execute(delete(VisitReport).where(VisitReport.session_id == TEST_SESSION))
                s.execute(delete(CoachSession).where(CoachSession.session_id == TEST_SESSION))
                s.execute(delete(Case).where(Case.case_id == TEST_CASE))
                s.execute(delete(User).where(User.username == TEST_USER))
