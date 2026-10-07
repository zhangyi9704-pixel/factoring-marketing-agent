"""评估真实链路集成测试（DeepSeek reasoning + MySQL 落库 + checklist 规则）。

验收 TDD §13 / §20 Phase 3「结束评估衔接」：一次真实对话 → finish → 惰性
计算评估 → t_evaluation + t_evaluation_evidence 落库。依赖 .env 真实凭证；
只用 2 次 reasoning 调用（机会诊断 + 能力评分），不含 roleplay chat 循环。
"""
import pytest
from sqlalchemy import delete, select

from app.models import CoachSession, ConversationMessage, Evaluation, EvaluationEvidence
from app.services.case_service import CaseService
from app.services.session_service import SessionService




@pytest.fixture
def ready_session():
    """建会话 + 录几轮对话 + 置 finished（MysQL），结束后清理。"""
    from uuid import uuid4
    sid = "__eval_itest_" + uuid4().hex
    CaseService().sync_to_mysql()                      # 幂等，确保案例在库
    ss = SessionService()
    ss.create_session(sid, 1001, "INFRA_001")
    ss.record_message(sid, 1, "sales", "王总，项目最终付款方是谁？合同签了吗？")
    ss.record_message(sid, 1, "customer", "是市交通集团，账期180天。",
                      revealed_info=["透露付款方"])
    ss.record_message(sid, 2, "sales", "应收对账和验收完成了吗？有质押吗？")
    ss.record_message(sid, 2, "customer", "对账还差一层，验收没完成。")
    ss.touch_session(sid, 2, finished=True)
    yield sid
    _cleanup(sid)


@pytest.mark.integration
class TestEvaluationIntegration:

    def test_real_evaluation_pipeline(self, ready_session):
        from app.evaluation.service import EvaluationService
        from app.db.mysql import get_mysql_database

        service = EvaluationService()
        report = service.get(ready_session)

        assert report["session_id"] == ready_session
        assert report["opportunity"]["score"] >= 0
        assert report["opportunity"]["level"] in ("高潜机会", "观察", "暂缓")
        assert report["sales_score"]["total"] >= 0
        assert isinstance(report["evidence"], list)
        assert report["missing_questions"] is not None
        assert report.get("markdown") and "复盘报告" in report["markdown"]   # §20 Phase 4 复盘报告

        db = get_mysql_database()
        with db.session_scope() as s:
            row = s.execute(
                select(Evaluation).where(Evaluation.session_id == ready_session)
            ).scalar_one()
            assert row.opportunity["score"] >= 0
            assert row.sales_score["total"] >= 0
            evs = s.execute(
                select(EvaluationEvidence).where(EvaluationEvidence.evaluation_id == row.id)
            ).scalars().all()
            assert len(evs) >= 1

    def test_evaluation_idempotent(self, ready_session):
        from app.evaluation.service import EvaluationService
        from app.db.mysql import get_mysql_database

        service = EvaluationService()
        first = service.get(ready_session)
        second = service.get(ready_session)
        assert first["total_score"] == second["total_score"]

        db = get_mysql_database()
        with db.session_scope() as s:
            rows = s.execute(
                select(Evaluation).where(Evaluation.session_id == ready_session)
            ).scalars().all()
            assert len(rows) == 1                     # 不重复落库


def _cleanup(sid: str) -> None:
    from app.db.mysql import get_mysql_database

    db = get_mysql_database()
    with db.session_scope() as s:
        ev_ids = select(Evaluation.id).where(Evaluation.session_id == sid)
        s.execute(delete(EvaluationEvidence).where(EvaluationEvidence.evaluation_id.in_(ev_ids)))
        s.execute(delete(Evaluation).where(Evaluation.session_id == sid))
        s.execute(delete(ConversationMessage).where(ConversationMessage.session_id == sid))
        s.execute(delete(CoachSession).where(CoachSession.session_id == sid))
