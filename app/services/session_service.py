"""演练会话服务（TDD §12.1 t_session / t_conversation_message / t_compliance_log）。

职责：
    - 会话行生命周期（创建 / 轮次推进 / 结束）；
    - 访前简报落库/读回（t_visit_report，briefing 节点 §7）；
    - 逐轮对话落库（logging.save.conversation，§10.4）；
    - 合规命中留痕（t_compliance_log，§14）。

session_factory 可注入（单测 FakeSession），默认 MySQL 事务作用域。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, ContextManager, Optional

from app.config import Settings, get_settings
from app.db.mysql import get_mysql_database
from app.models import CoachSession, ComplianceLog, ConversationMessage, VisitReport, MessageRequestRecord, BusinessAudit, SessionCreateRequestRecord

logger = logging.getLogger(__name__)


class SessionServiceError(RuntimeError):
    """会话服务错误。"""


class SessionService:
    """roleplay 会话与对话记录持久化。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        session_factory: Optional[Callable[[], ContextManager]] = None,
        cache=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.session_factory = session_factory or get_mysql_database().session_scope
        self._pending_engine = get_mysql_database().engine if session_factory is None else None
        self._cache = cache
        self._use_cache = self.settings.database.redis.cache_enabled and (cache is not None or session_factory is None)
        self.cache_status = "unverified" if self._use_cache else "disabled"

    def _cached(self, key, authoritative):
        if not self._use_cache:
            return authoritative
        try:
            if self._cache is None:
                from app.db.redis import get_redis_client
                self._cache = get_redis_client()
            cached = self._cache.get_json(key)
            # MySQL gates identity and freshness before even reading Redis.
            if cached != authoritative:
                self._cache.set_json(key, authoritative, self.settings.database.redis.cache_ttl)
            self.cache_status = "up"
            return cached if cached == authoritative else authoritative
        except Exception:
            self.cache_status = "degraded"
            logger.warning("session cache unavailable; mysql source used")
            return authoritative

    # -------------------------------------------------
    # 会话行（t_session）
    # -------------------------------------------------

    def create_session(
        self, session_id: str, user_id: int, case_id: str, *, phase: str = "roleplay",
    ) -> None:
        """创建会话行（session_id 唯一；重复创建视为幂等成功）。"""
        with self.session_factory() as session:
            row = session.query(CoachSession).filter(
                CoachSession.session_id == session_id,
            ).one_or_none()
            if row is None:
                session.add(CoachSession(
                    session_id=session_id, user_id=user_id,
                    case_id=case_id, phase=phase, status="active", turn_count=0,
                ))
                logger.info("会话创建: %s (user=%s case=%s)", session_id, user_id, case_id)

    def touch_session(
        self, session_id: str, turn_count: int, *,
        finished: bool = False, phase: Optional[str] = None,
    ) -> None:
        """推进会话轮次；finished 时置 status=finished / finished_at / phase=evaluate。"""
        with self.session_factory() as session:
            row = session.query(CoachSession).filter(
                CoachSession.session_id == session_id,
            ).one_or_none()
            if row is None:
                logger.warning("会话行不存在，跳过更新: %s", session_id)
                return
            row.turn_count = turn_count
            if finished:
                row.status = "finished"
                row.finished_at = datetime.now()
                row.phase = phase or "evaluate"
            elif phase:
                row.phase = phase

    def get_session(self, session_id: str, *, user_id: Optional[int] = None) -> Optional[dict]:
        """读会话行（API 状态查询）。"""
        with self.session_factory() as session:
            query = session.query(CoachSession).filter(CoachSession.session_id == session_id)
            if user_id is not None:
                query = query.filter(CoachSession.user_id == user_id)
            row = query.one_or_none()
            if row is None:
                if user_id is not None and session.query(CoachSession).filter(
                        CoachSession.session_id == session_id).one_or_none() is not None:
                    raise PermissionError("session_owner_mismatch")
                return None
            if user_id is not None and row.user_id != user_id:
                raise PermissionError("session_owner_mismatch")
            payload = {
                "session_id": row.session_id, "user_id": row.user_id,
                "case_id": row.case_id, "phase": row.phase, "status": row.status,
                "turn_count": row.turn_count,
                "needs_human_review": bool(row.needs_human_review),
            }

        return self._cached(f"factoring:session:{payload['user_id']}:{session_id}", payload)

    def list_sessions_by_user(self, user_id: int) -> list[dict]:
        """按用户列会话（主管看板 / 成绩画像用，§20 Phase 4）。"""
        with self.session_factory() as session:
            rows = session.query(CoachSession).filter(
                CoachSession.user_id == user_id,
            ).order_by(CoachSession.created_at, CoachSession.id).all()
            return [
                {
                    "session_id": row.session_id, "user_id": row.user_id,
                    "case_id": row.case_id, "phase": row.phase, "status": row.status,
                    "turn_count": row.turn_count,
                }
                for row in rows
            ]

    # -------------------------------------------------
    # 访前简报（t_visit_report，briefing 节点 upsert 复用）
    # -------------------------------------------------

    def save_briefing(self, session_id: str, briefing: dict) -> None:
        """简报落库（session_id 唯一，重复生成幂等覆盖）。"""
        with self.session_factory() as session:
            row = session.query(VisitReport).filter(
                VisitReport.session_id == session_id,
            ).one_or_none()
            if row is None:
                session.add(VisitReport(
                    session_id=session_id, report_json=briefing,
                ))
            else:
                row.report_json = briefing

    def get_briefing(self, session_id: str, *, user_id: Optional[int] = None) -> Optional[dict]:
        """读回简报（会话详情 / 前端回显用）。"""
        with self.session_factory() as session:
            row = session.query(VisitReport).filter(
                VisitReport.session_id == session_id,
            ).one_or_none()
            payload = row.report_json if row else None
        if payload is None:
            return None
        if not self._use_cache and user_id is None:
            return payload
        owner = self.get_session(session_id, user_id=user_id)
        if owner is None:
            return None
        if not self._use_cache:
            return payload
        return self._cached(f"factoring:briefing:{owner['user_id']}:{session_id}", payload)

    # -------------------------------------------------
    # 对话记录（t_conversation_message，逐轮双方向）
    # -------------------------------------------------

    def record_message(
        self,
        session_id: str,
        turn: int,
        role: str,
        content: str,
        *,
        revealed_info: Optional[list] = None,
        guardrail_hit: Optional[list] = None,
    ) -> None:
        """写入单条对话消息（内部字段只入 MySQL，绝不回传销售侧 API）。"""
        with self.session_factory() as session:
            session.add(ConversationMessage(
                session_id=session_id, turn=turn, role=role, content=content or "",
                revealed_info=revealed_info, guardrail_hit=guardrail_hit,
            ))

    def list_messages(self, session_id: str) -> list[dict]:
        """按轮次读回对话（评估 / 回放用）。"""
        with self.session_factory() as session:
            rows = session.query(ConversationMessage).filter(
                ConversationMessage.session_id == session_id,
            ).order_by(ConversationMessage.turn, ConversationMessage.id).all()
            return [
                {
                    "turn": row.turn, "role": row.role, "content": row.content,
                    "revealed_info": row.revealed_info, "guardrail_hit": row.guardrail_hit,
                }
                for row in rows
            ]

    def persist_turn(self, state, turn, *, finished, blocked):
        """销售/客户消息与会话推进同事务；失败不会只保存半轮。"""
        with self.session_factory() as db:
            row = db.query(CoachSession).filter(CoachSession.session_id == state['session_id']).one()
            if not blocked and (not finished or state.get('finish_reason') == 'max_turns'):
                db.add(ConversationMessage(session_id=state['session_id'], turn=turn, role='sales',
                    content=state.get('sales_input', ''), guardrail_hit=state.get('guardrail_hits') or None))
                db.add(ConversationMessage(session_id=state['session_id'], turn=turn, role='customer',
                    content=state.get('customer_reply', ''), revealed_info=state.get('revealed_this_turn') or None))
            row.turn_count = turn
            if finished:
                row.status, row.phase, row.finished_at = 'finished', 'evaluate', datetime.now()
            rid = state.get('active_request_id')
            if rid:
                from app.schemas.session import TurnResponse
                from app.services.message_requests import turn_result
                request = db.query(MessageRequestRecord).filter_by(session_id=state['session_id'], request_id=rid).one()
                if (request.status != 'running' or request.payload_hash != state.get('active_payload_hash')
                        or request.expected_turn != state.get('active_expected_turn')):
                    raise SessionServiceError('request_binding_mismatch')
                # Still running until the final checkpoint is confirmed. This
                # snapshot commits with both messages and the session counter.
                request.result = TurnResponse(**turn_result(dict(state, turn=turn, finished=finished)), request_id=rid).model_dump()

    def abort_session(self, session_id, actor_id, reason):
        """Terminate uncertain work without replaying/deleting messages or checkpoints."""
        with self.session_factory() as db:
            row = db.query(CoachSession).filter_by(session_id=session_id).one()
            if row.status == 'aborted':
                return
            row.status, row.phase, row.finished_at = 'aborted', 'report', datetime.now()
            row.needs_human_review = 1
            creation = db.query(SessionCreateRequestRecord).filter_by(session_id=session_id).one_or_none()
            if creation is not None and creation.status == 'initializing':
                creation.status, creation.error_code = 'aborted', 'operator_aborted'
            for request in db.query(MessageRequestRecord).filter(MessageRequestRecord.session_id == session_id,
                    MessageRequestRecord.status.in_(['running', 'failed'])).all():
                request.status, request.error_code = 'aborted', 'operator_aborted'
            db.add(BusinessAudit(actor_id=actor_id, queue='session', action='session_aborted',
                details=dict(session_id=session_id, reason=reason)))

    # -------------------------------------------------
    # 合规留痕（t_compliance_log，§14）
    # -------------------------------------------------

    def record_compliance(
        self, session_id: str, turn: int, rule: str, raw_text: str, *, action: str = "log", details=None,
    ) -> str | None:
        """记录一次违禁命中（review_status=pending，累计 ≥3 升级 HITL 队列）。"""
        pending_owner = None
        with self.session_factory() as session:
            session.add(ComplianceLog(
                session_id=session_id, turn=turn, rule=rule,
                raw_text=raw_text, action=action, review_status="pending", details=details,
            ))
            session.flush()
            events = session.query(ComplianceLog).filter(ComplianceLog.session_id == session_id).all()
            events = [e for e in events if e.action != "human_review"]
            warns = sum((e.details or {}).get("severity", "warn") == "warn" and e.action != "block" for e in events)
            if any(e.action == "block" or (e.details or {}).get("severity") == "block" for e in events) or warns >= 3:
                row = session.query(CoachSession).filter(CoachSession.session_id == session_id).first()
                if row:
                    row.needs_human_review = 1
                    pending_owner = row.user_id
            logger.warning("guardrail 命中: session=%s turn=%s rule=%s", session_id, turn, rule)
        if pending_owner is not None:
            return self.register_escalation("compliance", raw_text or "合规累计升级", user_id=pending_owner,
                session_id=session_id, context={"turn":turn,"rule":rule,"details":details or {},"warn_count":warns},
                versions=[{"rule":rule,"version":(details or {}).get("rule_version", (details or {}).get("version"))}],
                severity="block" if action == "block" or (details or {}).get("severity") == "block" else "warn")

    def mark_human_review(self, session_id):
        with self.session_factory() as session:
            row = session.query(CoachSession).filter(CoachSession.session_id == session_id).first()
            if row:
                row.needs_human_review = 1

    def register_escalation(self, kind, question, *, user_id, session_id, context=None, versions=None, severity=None):
        from app.services.pending_service import PendingService
        return PendingService(self.session_factory, engine=self._pending_engine).register(kind, question, user_id=user_id,
            session_id=session_id, context=context, versions=versions, severity=severity)["external_id"]

    def latest_compliance_pending(self, session_id):
        with self.session_factory() as db:
            row = db.query(ComplianceLog).filter(ComplianceLog.session_id == session_id,
                ComplianceLog.action == "human_review").order_by(ComplianceLog.created_at.desc()).first()
            return row.external_id if row else None

    def list_compliance(self, session_id):
        with self.session_factory() as session:
            rows = session.query(ComplianceLog).filter(ComplianceLog.session_id == session_id).order_by(ComplianceLog.id).all()
            rows = [r for r in rows if r.action != "human_review"]
            return [dict(turn=r.turn, rule=r.rule, raw_text=r.raw_text, action=r.action,
                         details=r.details or {}, review_status=r.review_status) for r in rows]
