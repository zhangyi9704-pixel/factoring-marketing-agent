"""Durable HITL queues. SQL conditional updates arbitrate claim/close races."""
from contextlib import contextmanager
from datetime import datetime, timedelta
import hashlib
import json
import threading
import unicodedata
import uuid

from sqlalchemy import select, text, update
from app.db.mysql import get_mysql_database
from app.models import CoachSession, ComplianceLog, KnowledgePending, User, KnowledgeDraft, BusinessAudit
from app.config import get_settings

KINDS = {"missing", "conflict", "out_of_scope", "compliance"}
STATUSES = {"pending", "processing", "resolved", "dismissed"}
_LOCKS = [threading.RLock() for _ in range(64)]


class PendingConflict(RuntimeError):
    pass


def normalize_question(question):
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


class PendingService:
    def __init__(self, session_factory=None, engine=None, clock=None, feedback_service=None):
        if session_factory is None:
            database = get_mysql_database()
            session_factory, engine = database.session_scope, database.engine
        self.session_factory, self.engine = session_factory, engine
        self.clock = clock or datetime.utcnow
        self.feedback_service = feedback_service

    @contextmanager
    def registration_lock(self, key):
        # MySQL named lock held on a separate connection until the write commits.
        if self.engine is not None and self.engine.dialect.name == "mysql":
            with self.engine.connect() as conn:
                name = "pending:" + key[:48]
                if conn.execute(text("SELECT GET_LOCK(:name, 2)"), {"name":name}).scalar() != 1:
                    raise PendingConflict("registration_busy")
                try:
                    yield
                finally:
                    conn.execute(text("SELECT RELEASE_LOCK(:name)"), {"name":name})
        else:
            with _LOCKS[int(key[:8], 16) % len(_LOCKS)]:
                yield

    def register(self, kind, question, *, user_id, session_id=None, context=None, versions=None, severity=None):
        if kind not in KINDS or not question.strip() or not user_id or int(user_id) <= 0:
            raise ValueError("invalid_pending_registration")
        context = dict(context or {})
        # Stable version set; a new version must produce a new registration key.
        version_strings = sorted({json.dumps(v, sort_keys=True, ensure_ascii=False) for v in (versions or [])})
        key = hashlib.sha256(json.dumps([user_id, session_id, kind, normalize_question(question), version_strings],
                                     ensure_ascii=False).encode()).hexdigest()
        now = self.clock()
        model = ComplianceLog if kind == "compliance" else KnowledgePending
        with self.registration_lock(key):
            with self.session_factory() as db:
                if session_id:
                    owner = db.query(CoachSession).filter(CoachSession.session_id == session_id,
                                                         CoachSession.user_id == user_id).one_or_none()
                    if owner is None:
                        raise PermissionError("session_owner_forbidden")
                row = db.query(model).filter(model.dedup_key == key,
                        model.created_at > now - timedelta(hours=24)).order_by(model.created_at.desc()).first()
                if row is not None:
                    return self.serialize(row, kind)
                context.update(user_id=user_id, session_id=session_id, versions=list(versions or []))
                identifier = "esc_" + uuid.uuid4().hex
                if kind == "compliance":
                    row = ComplianceLog(user_id=user_id, session_id=session_id, turn=context.get("turn",0),
                        rule="human_review", raw_text=question, details=dict(context, severity=severity),
                        action="human_review", status="pending", dedup_key=key, external_id=identifier,
                        review_status="pending", created_at=now)
                else:
                    row = KnowledgePending(type=kind, question=question, context=context, severity=severity,
                        status="pending", user_id=user_id, session_id=session_id, dedup_key=key,
                        external_id=identifier, created_at=now)
                db.add(row)
                db.flush()
                return self.serialize(row, kind)

    @staticmethod
    def serialize(row, kind=None):
        compliance = isinstance(row, ComplianceLog)
        result = dict(id=row.id, external_id=row.external_id, type="compliance" if compliance else row.type,
            question=row.raw_text if compliance else row.question,
            context=(row.details if compliance else row.context) or {},
            severity=(row.details or {}).get("severity") if compliance else row.severity,
            status=row.status, resolution=row.resolution, assignee=row.assignee,
            created_at=row.created_at, resolved_at=row.resolved_at,
            user_id=row.user_id, session_id=row.session_id)
        if compliance:
            context = row.details or {}
            details = context.get("details") or {}
            response = details.get("compliant_response") or context.get("compliant_response")
            source = "登记时应答快照" if response else "应答未记录"
            if not response:
                # Existing queues may lack an answer snapshot. Label the current
                # rule reference explicitly; never invent a historical answer.
                from app.guardrail.rules import load_compliance_rules
                rules = load_compliance_rules()
                rule = next((r for r in rules.rules if r.rule_id == context.get("rule")), None)
                if rule:
                    response = rule.compliant_response
                    source = f"当前规则参考 · {rules.version}（原记录未保存应答快照）"
            result.update(compliant_response=response or "", compliant_response_source=source)
        return result

    @staticmethod
    def scope(db, actor):
        if actor.role == "admin":
            return None
        if actor.role != "coach":
            raise PermissionError("role_forbidden")
        names = get_settings().auth.coach_sales.get(actor.username, [])
        ids = db.scalars(select(User.id).where(User.username.in_(names), User.role == "sales",
                                             User.enabled.is_(True))).all()
        return [actor.id, *ids]

    def list(self, queue, actor, *, kind=None, status=None, since=None, until=None, offset=0, limit=100):
        model = ComplianceLog if queue == "compliance" else KnowledgePending
        with self.session_factory() as db:
            query = db.query(model).filter(model.status.is_not(None))
            scope = self.scope(db, actor)
            if scope is not None:
                query = query.filter(model.user_id.in_(scope))
            if kind:
                if queue == "compliance" and kind != "compliance":
                    return []
                if queue != "compliance":
                    query = query.filter(model.type == kind)
            if status:
                query = query.filter(model.status == status)
            if since:
                query = query.filter(model.created_at >= since)
            if until:
                query = query.filter(model.created_at <= until)
            return [self.serialize(row) for row in query.order_by(model.created_at.desc(), model.id.desc()).offset(offset).limit(limit)]

    def get(self, queue, item_id, actor):
        model = ComplianceLog if queue == "compliance" else KnowledgePending
        with self.session_factory() as db:
            row = db.get(model, item_id)
            scope = self.scope(db, actor)
            if row is None or row.status is None:
                return None
            if scope is not None and row.user_id not in scope:
                raise PermissionError("sales_scope_forbidden")
            result = self.serialize(row)
            if queue == "knowledge":
                draft = db.scalars(select(KnowledgeDraft).where(KnowledgeDraft.pending_id == item_id)).first()
                if draft:
                    result["knowledge_draft"] = dict(id=draft.id, status=draft.status)
            return result

    def transition(self, queue, item_id, actor, status, resolution=None, *, feedback=None, feedback_service=None):
        # Permission lookup precedes conditional write, but scope is also in SQL.
        existing = self.get(queue, item_id, actor)
        if existing is None:
            return None
        if status not in {"processing", "resolved", "dismissed"}:
            raise ValueError("invalid_transition")
        if status != "processing" and (not resolution or not resolution.strip()):
            raise ValueError("resolution_required")
        if feedback and (queue != "knowledge" or status != "resolved"):
            raise ValueError("feedback_requires_resolved_knowledge")
        model = ComplianceLog if queue == "compliance" else KnowledgePending
        with self.session_factory() as db:
            stmt = update(model).where(model.id == item_id)
            scope = self.scope(db, actor)
            if scope is not None:
                stmt = stmt.where(model.user_id.in_(scope))
            if status == "processing":
                stmt = stmt.where(model.status == "pending", model.assignee.is_(None)).values(
                    status="processing", assignee=actor.id)
            else:
                stmt = stmt.where(model.status == "processing", model.assignee == actor.id).values(
                    status=status, resolution=resolution.strip(), resolved_at=self.clock())
            changed = db.execute(stmt).rowcount
            if changed == 1:
                db.add(BusinessAudit(actor_id=actor.id, pending_id=item_id, queue=queue,
                    action="pending_claimed" if status == "processing" else "pending_closed",
                    details=dict(before=existing["status"], after=status, resolution=resolution)))
                if feedback:
                    from app.services.knowledge_feedback import KnowledgeFeedback
                    publisher = feedback_service or self.feedback_service or KnowledgeFeedback(self.session_factory, self.engine)
                    publisher.create_in_transaction(db, actor, db.get(model, item_id), feedback)
        if changed != 1:
            raise PendingConflict("pending_state_or_assignee_conflict")
        return self.get(queue, item_id, actor)

    def reassign(self, queue, item_id, actor, *, expected_assignee, assignee_id=None, reason):
        """Explicit admin takeover with optimistic identity and reviewer scope."""
        if actor.role != "admin":
            raise PermissionError("role_forbidden")
        if not reason or not reason.strip():
            raise ValueError("reassignment_reason_required")
        model = ComplianceLog if queue == "compliance" else KnowledgePending
        with self.session_factory() as db:
            row = db.get(model, item_id)
            if row is None or row.status is None:
                return None
            if assignee_id is not None:
                target = db.get(User, assignee_id)
                if target is None or not target.enabled or target.role not in {"coach", "admin"}:
                    raise ValueError("invalid_reviewer")
                from app.services.auth_service import Principal
                scope = self.scope(db, Principal(target.id, target.username, target.role))
                if scope is not None and row.user_id not in scope:
                    raise PermissionError("sales_scope_forbidden")
            changed = db.execute(update(model).where(model.id == item_id, model.status == "processing",
                model.assignee == expected_assignee).values(
                    status="pending" if assignee_id is None else "processing", assignee=assignee_id)).rowcount
            if changed != 1:
                raise PendingConflict("pending_state_or_assignee_conflict")
            db.add(BusinessAudit(actor_id=actor.id, pending_id=item_id, queue=queue,
                action="pending_released" if assignee_id is None else "pending_reassigned",
                details=dict(previous_assignee=expected_assignee, assignee=assignee_id, reason=reason.strip())))
        return self.get(queue, item_id, actor)
