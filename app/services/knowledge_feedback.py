"""Human-authored drafts and verified publication. No model execution or path input."""
from contextlib import contextmanager, ExitStack
from datetime import datetime
from pathlib import Path
import hashlib
import json
import threading
import uuid
import yaml
from sqlalchemy import select, text, update
from app.config import PROJECT_ROOT, get_settings
from app.db.mysql import get_mysql_database
from app.models import KnowledgeDraft, KnowledgeDoc, KnowledgePending, BusinessAudit, AgentErrorLog, KnowledgeIngestionOperation
from app.services.domain_errors import FeedbackConflict, FeedbackValidation, FeedbackUnavailable
from app.services.pending_service import PendingService
from app.rag.loader import parse_metadata
from app.rag.revision import invalidate
from app.rag.integrity import document_hash, vector_hash

_LOCKS = [threading.RLock() for _ in range(64)]


def audit(db, actor, action, *, pending_id=None, queue="knowledge", draft_id=None, **details):
    db.add(BusinessAudit(actor_id=actor.id,action=action,pending_id=pending_id,queue=queue,draft_id=draft_id,details=details))


def invalidate_safely():
    # Database catalog digest is also checked on every cache read. An unavailable
    # local revision file must never turn an already committed publish into 500.
    try:
        invalidate()
    except OSError:
        pass


class KnowledgeFeedback:
    def __init__(self, session_factory=None, engine=None, ingestor=None, root=None):
        if session_factory is None:
            database = get_mysql_database()
            session_factory, engine = database.session_scope,database.engine
        self.factory,self.engine,self._ingestor = session_factory,engine,ingestor
        self.root = Path(root or PROJECT_ROOT / "data/knowledge_feedback").resolve()

    @property
    def ingestor(self):
        if self._ingestor is None:
            from app.rag.ingest import KnowledgeIngestor
            self._ingestor = KnowledgeIngestor(session_factory=self.factory)
        return self._ingestor

    @contextmanager
    def locks(self, identities):
        with ExitStack() as stack:
            for identity in sorted(set(identities)):
                digest = hashlib.sha256(identity.encode()).hexdigest()
                if self.engine is not None and self.engine.dialect.name == "mysql":
                    conn = stack.enter_context(self.engine.connect())
                    name = "kb:" + digest[:60]
                    if conn.execute(text("SELECT GET_LOCK(:name, 2)"),{"name":name}).scalar() != 1:
                        raise FeedbackConflict("knowledge_busy")
                    def release(connection=conn, lock=name):
                        try:
                            connection.execute(text("SELECT RELEASE_LOCK(:name)"),{"name":lock})
                        except Exception:
                            connection.invalidate()
                    stack.callback(release)
                else:
                    stack.enter_context(_LOCKS[int(digest[:8],16)%len(_LOCKS)])
            yield

    def path(self, draft_id):
        if len(draft_id)!=32 or any(c not in "0123456789abcdef" for c in draft_id):
            raise FeedbackValidation("invalid_draft_id")
        path = (self.root / draft_id / "entry.md").resolve()
        if not path.is_relative_to(self.root):
            raise FeedbackValidation("unsafe_draft_path")
        return path

    @staticmethod
    def render(metadata,content,active=False):
        return "---\n" + yaml.safe_dump(dict(metadata,status="active" if active else "archived"),allow_unicode=True,sort_keys=False) + "---\n\n" + content.strip() + "\n"

    @staticmethod
    def digest(metadata,content):
        return hashlib.sha256(KnowledgeFeedback.render(metadata,content).encode()).hexdigest()

    def verify_file(self, draft_id, metadata, content):
        # MySQL JSON canonicalizes key order. Compare parsed facts and body,
        # rather than YAML serialization order; whitespace-only YAML changes
        # cannot change the metadata/body eventually sent to ingestion.
        try:
            body, actual = parse_metadata(self.path(draft_id).read_text(encoding="utf-8"), "entry.md")
        except (OSError, RuntimeError):
            raise FeedbackConflict("draft_file_changed") from None
        if actual != dict(metadata, status="archived") or body != content.strip():
            raise FeedbackConflict("draft_file_changed")

    @staticmethod
    def data(row):
        return dict(id=row.id,pending_id=row.pending_id,status=row.status,creator_id=row.creator_id,confirmed_by=row.confirmed_by,
                    metadata=row.metadata_json,content=row.content,adopted=row.adopted,supersedes=row.supersedes,
                    content_hash=row.content_hash,failure=row.failure,created_at=row.created_at,
                    confirmed_at=row.confirmed_at,published_at=row.published_at)

    def create_in_transaction(self,db,actor,pending,payload):
        if pending.type == "out_of_scope":
            raise FeedbackValidation("out_of_scope_cannot_publish_as_fact")
        draft_id = uuid.uuid4().hex
        adopted = payload.get("adopted")
        items = (pending.context or {}).get("items",[])
        refs = {(i.get("doc_id"),i.get("version")):i for i in items if i.get("doc_id") and i.get("version")}
        if pending.type == "conflict":
            if len(refs) < 2:
                raise FeedbackValidation("incomplete_conflict_requires_full_manual_verification")
            if payload["action"] != "adopt" or not adopted or (adopted["doc_id"],adopted["version"]) not in refs:
                raise FeedbackValidation("conflict_requires_recorded_adopted_version")
            selected = refs[(adopted["doc_id"],adopted["version"])]
            if any(i.get("topic") != selected.get("topic") for i in refs.values()):
                raise FeedbackValidation("multiple_conflict_topics_require_separate_resolution")
        elif adopted:
            raise FeedbackValidation("adopted_only_for_conflict")
        target = (adopted or {}).get("doc_id") or payload.get("doc_id") or "KB-FB-"+draft_id[:12]
        if adopted and payload.get("doc_id") and payload["doc_id"] != target:
            raise FeedbackValidation("adopted_target_mismatch")
        if payload.get("doc_id") and not adopted:
            target_rows = db.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id==target,
                                      KnowledgeDoc.status.in_(["active","conflicted"]))).all()
            if not target_rows:
                raise FeedbackValidation("target_document_not_effective")
        old = db.scalars(select(KnowledgeDoc).where(KnowledgeDoc.status.in_(["active","conflicted"]))).all()
        supersedes = [dict(doc_id=r.doc_id,version=r.version,status=r.status) for r in old
                      if r.doc_id==target or (r.doc_id,r.version) in refs and pending.type=="conflict"]
        if pending.type=="conflict" and set(refs) != {(r["doc_id"],r["version"]) for r in supersedes if (r["doc_id"],r["version"]) in refs}:
            raise FeedbackConflict("conflict_versions_changed")
        metadata = {k:payload[k] for k in ("source","category","industry","topic","effective_date")}
        metadata.update(doc_id=target,version="v-fb-"+draft_id[:16])
        content = payload["content"].strip()
        parse_metadata(self.render(metadata,content),"entry.md")
        if not metadata["source"].strip():
            raise FeedbackValidation("source_required")
        # Files contain archived metadata until a separate human confirmation.
        path = self.path(draft_id)
        path.parent.mkdir(parents=True,exist_ok=False)
        path.write_text(self.render(metadata,content),encoding="utf-8")
        row = KnowledgeDraft(id=draft_id,pending_id=pending.id,creator_id=actor.id,metadata_json=metadata,
            content=content,adopted=adopted,supersedes=supersedes,content_hash=self.digest(metadata,content),status="draft")
        db.add(row)
        audit(db,actor,"draft_saved",pending_id=pending.id,draft_id=draft_id,content_hash=row.content_hash,
              metadata=metadata,adopted=adopted,supersedes=supersedes)
        return self.data(row)

    def get(self,draft_id,actor):
        self.path(draft_id)
        with self.factory() as db:
            row = db.get(KnowledgeDraft,draft_id)
            if row is None:
                return None
            pending = db.get(KnowledgePending,row.pending_id)
            scope = PendingService.scope(db,actor)
            if scope is not None and pending.user_id not in scope:
                raise PermissionError("sales_scope_forbidden")
            return self.data(row)

    def confirm(self,draft_id,actor,content_hash):
        if actor.role!="admin":
            raise PermissionError("role_forbidden")
        with self.locks(["draft:"+draft_id]):
            with self.factory() as db:
                row = db.get(KnowledgeDraft,draft_id)
                if row is None:
                    return None
                if content_hash != row.content_hash:
                    raise FeedbackConflict("draft_content_changed")
                if row.status=="confirmed":
                    return self.data(row)
                if row.status!="draft":
                    raise FeedbackConflict("invalid_draft_state")
                self.verify_file(draft_id, row.metadata_json, row.content)
                row.status="confirmed"; row.confirmed_by=actor.id; row.confirmed_at=datetime.utcnow()
                audit(db,actor,"draft_confirmed",pending_id=row.pending_id,draft_id=draft_id,content_hash=content_hash,
                      source=row.metadata_json["source"])
                return self.data(row)

    def history(self, draft_id, actor):
        draft = self.get(draft_id, actor)  # Same stored-owner scope as draft read.
        if draft is None:
            return None
        with self.factory() as db:
            records = db.scalars(select(BusinessAudit).where(BusinessAudit.pending_id == draft["pending_id"],
                BusinessAudit.queue == "knowledge").order_by(BusinessAudit.id)).all()
            return [dict(id=r.id, actor_id=r.actor_id, action=r.action, details=r.details,
                         created_at=r.created_at, draft_id=r.draft_id) for r in records]

    def publish(self,draft_id,actor):
        if actor.role!="admin":
            raise PermissionError("role_forbidden")
        initial = self.get(draft_id,actor)
        if initial is None:
            return None
        identities = [initial["metadata"]["doc_id"], *[r["doc_id"] for r in initial["supersedes"]], "draft:"+draft_id]
        with self.locks(identities):
            draft = self.get(draft_id,actor)
            if draft["status"]=="published":
                return draft  # Exactly the committed operation; no model/vector writes.
            if draft["status"]!="confirmed":
                raise FeedbackConflict("draft_must_be_confirmed_or_reconciled")
            self._check_targets(draft)
            self.verify_file(draft_id, draft["metadata"], draft["content"])
            store = self.ingestor.store
            snapshots = {identity: store.snapshot_doc(identity)
                         for identity in set(identities)-{"draft:"+draft_id}}
            with self.factory() as db:
                row=db.get(KnowledgeDraft,draft_id); row.status="publishing"
                catalog = db.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id.in_(snapshots))).all()
                # Commit a durable pre-write snapshot with the publishing state.
                # Recovery does not need an in-memory snapshot from a dead worker.
                row.publication_snapshot = dict(documents=snapshots, catalog=[dict(doc_id=r.doc_id,
                    version=r.version, status=r.status, content_hash=r.content_hash) for r in catalog
                    if (r.doc_id, r.version) != (draft["metadata"]["doc_id"], draft["metadata"]["version"])])
                audit(db,actor,"publish_started",pending_id=row.pending_id,draft_id=draft_id,content_hash=row.content_hash)
            try:
                path=self.path(draft_id)
                path.write_text(self.render(draft["metadata"],draft["content"],True),encoding="utf-8")
                cfg=get_settings().knowledge.sources[draft["metadata"]["category"]]
                result=self.ingestor.ingest_file(path,source=draft["metadata"]["category"],priority=cfg.priority,
                                                 staged=True,already_locked=True)
                if result.status!="indexed":
                    raise FeedbackUnavailable("staged_ingestion_failed")
                doc,version=draft["metadata"]["doc_id"],draft["metadata"]["version"]
                new=[r for r in store.snapshot_doc(doc) if r["version"]==version]
                if len(new)!=result.chunk_cnt or not new:
                    raise FeedbackUnavailable("staged_verification_failed")
                changed=[dict(r,status="active") for r in new]
                for old in draft["supersedes"]:
                    rows=[r for r in snapshots[old["doc_id"]] if r["version"]==old["version"]]
                    if not rows:
                        raise FeedbackUnavailable("old_version_snapshot_missing")
                    changed.extend(dict(r,status="archived") for r in rows)
                # Batch per version: VectorStore requires same-status parent/child.
                if store.upsert(changed)!=len(changed):
                    raise FeedbackUnavailable("activation_count_mismatch")
                for identity in snapshots:
                    actual={r["id"]:r for r in store.snapshot_doc(identity)}
                    for expected in [r for r in changed if r["doc_id"]==identity]:
                        if expected["id"] not in actual or any(actual[expected["id"]].get(k)!=expected.get(k)
                            for k in ("content","status","doc_id","version","source","parent_id")):
                            raise FeedbackUnavailable("activation_verification_failed")
                with self.factory() as db:
                    self._check_targets(draft,db)
                    row=db.get(KnowledgeDraft,draft_id)
                    current=db.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id==doc,KnowledgeDoc.version==version)).one()
                    current.status="active"
                    for old in draft["supersedes"]:
                        db.execute(update(KnowledgeDoc).where(KnowledgeDoc.doc_id==old["doc_id"],
                            KnowledgeDoc.version==old["version"]).values(status="archived"))
                    row.status="published"; row.published_at=datetime.utcnow(); row.failure=None
                    audit(db,actor,"ingestion_verified",pending_id=row.pending_id,draft_id=draft_id,chunk_cnt=len(new))
                    audit(db,actor,"knowledge_published",pending_id=row.pending_id,draft_id=draft_id,metadata=draft["metadata"],
                          adopted=draft["adopted"],content_hash=draft["content_hash"],confirmed_by=row.confirmed_by)
                    audit(db,actor,"versions_archived",pending_id=row.pending_id,draft_id=draft_id,versions=draft["supersedes"])
            except Exception as exc:
                # A commit acknowledgement may be lost. Inspect persisted state;
                # never blindly undo a publication that actually committed.
                try:
                    saved=self.get(draft_id,actor)
                except Exception:
                    raise FeedbackUnavailable("publication_outcome_unknown") from None
                if saved["status"]=="published":
                    invalidate_safely()
                    return saved
                compensation_failed=False
                for identity, rows in snapshots.items():
                    try:
                        self.ingestor.store.restore_doc(identity,draft["metadata"]["version"],rows)
                    except Exception:
                        compensation_failed=True
                try:
                    self.path(draft_id).write_text(self.render(draft["metadata"],draft["content"]),encoding="utf-8")
                except OSError:
                    compensation_failed=True
                with self.factory() as db:
                    row=db.get(KnowledgeDraft,draft_id)
                    row.status="reconcile_required" if compensation_failed else "failed"
                    row.failure=type(exc).__name__
                    error_code = str(exc) if getattr(exc, "domain_error", False) else "publication_exception"
                    audit(db,actor,"publish_failed",pending_id=row.pending_id,draft_id=draft_id,
                          error_type=type(exc).__name__,error_code=error_code,compensation_failed=compensation_failed)
                    db.add(AgentErrorLog(level="ERROR",logger_name="knowledge_feedback",user_id=actor.id,
                        message="knowledge_publication_failed:"+error_code,exc_type=type(exc).__name__,
                        path="/api/v1/knowledge/drafts/"+draft_id+"/publish",method="POST"))
                raise FeedbackUnavailable("knowledge_publication_failed") from None
            finally:
                invalidate_safely()  # Catalog digest also invalidates across a crash here.
            return self.get(draft_id,actor)

    def recover(self, draft_id, actor):
        """Explicit admin reconciliation; never re-embeds or silently republishes."""
        if actor.role != "admin":
            raise PermissionError("role_forbidden")
        initial = self.get(draft_id, actor)
        if initial is None:
            return None
        identities = [initial["metadata"]["doc_id"], *[r["doc_id"] for r in initial["supersedes"]], "draft:"+draft_id]
        with self.locks(identities):
            draft = self.get(draft_id, actor)
            if self.digest(draft["metadata"], draft["content"]) != draft["content_hash"]:
                raise FeedbackConflict("draft_content_changed")
            if draft["status"] not in {"failed", "reconcile_required", "publishing", "published", "confirmed"}:
                raise FeedbackConflict("draft_not_recoverable")
            if draft["status"] == "confirmed":
                self.verify_file(draft_id, draft["metadata"], draft["content"])
                return draft
            doc, version = draft["metadata"]["doc_id"], draft["metadata"]["version"]
            store = self.ingestor.store
            with self.factory() as db:
                row = db.get(KnowledgeDraft, draft_id)
                snapshot = row.publication_snapshot
                current = db.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id == doc,
                                                               KnowledgeDoc.version == version)).one_or_none()
                if row.status == "published":
                    expected_hash = document_hash(dict(draft["metadata"], status="staged"), draft["content"])
                    operation = db.get(KnowledgeIngestionOperation, current.ingestion_operation_id) if current and current.ingestion_operation_id else None
                    actual = [r for r in store.snapshot_doc(doc) if r["version"] == version]
                    if (current is None or current.status != "active" or current.content_hash != expected_hash
                        or current.content_snapshot != draft["content"] or operation is None
                        or operation.input_hash != expected_hash or operation.doc_id != doc or operation.version != version
                        or any(getattr(current, key) != value for key, value in draft["metadata"].items())
                        or len(actual) != operation.chunk_cnt
                        or vector_hash(actual, status="staged") != operation.vector_hash):
                        raise FeedbackUnavailable("committed_publication_requires_manual_repair")
                    for old in draft["supersedes"]:
                        prior = db.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id == old["doc_id"],
                            KnowledgeDoc.version == old["version"])).one_or_none()
                        if prior is None or prior.status != "archived":
                            raise FeedbackUnavailable("committed_publication_requires_manual_repair")
                    audit(db, actor, "publication_reconciled", pending_id=row.pending_id, draft_id=draft_id,
                          outcome="already_published", content_hash=draft["content_hash"])
                    return draft
                self._check_targets(draft, db)
                if not snapshot or "documents" not in snapshot or "catalog" not in snapshot:
                    raise FeedbackUnavailable("recovery_snapshot_unavailable")
                # A later same-version ingestion must not be overwritten by an
                # old recovery operation, even if its active status is unchanged.
                for previous in snapshot["catalog"]:
                    saved = db.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id == previous["doc_id"],
                        KnowledgeDoc.version == previous["version"])).one_or_none()
                    if (saved is None or saved.content_hash != previous["content_hash"]
                        or saved.status != previous["status"]):
                        raise FeedbackConflict("recovery_catalog_changed")
                if current is not None and current.status not in {"staged", "archived"}:
                    raise FeedbackConflict("recovery_target_effective")
                if current is not None and current.content_hash != document_hash(
                        dict(draft["metadata"], status="staged"), draft["content"]):
                    raise FeedbackConflict("recovery_target_content_changed")
            try:
                for identity, rows in snapshot["documents"].items():
                    store.restore_doc(identity, version, rows)
                    actual = store.snapshot_doc(identity)
                    # Includes content, metadata, parent links and dense/sparse
                    # vectors; restore count alone cannot establish reconciliation.
                    expected_ids = {r["id"] for r in rows}
                    restored = [r for r in actual if r["id"] in expected_ids]
                    if vector_hash(restored) != vector_hash(rows) or any(
                            r["version"] == version and r["id"] not in expected_ids for r in actual):
                        raise FeedbackUnavailable("recovery_vector_verification_failed")
                self.path(draft_id).write_text(self.render(draft["metadata"], draft["content"]), encoding="utf-8")
                self.verify_file(draft_id, draft["metadata"], draft["content"])
                with self.factory() as db:
                    self._check_targets(draft, db)
                    row = db.get(KnowledgeDraft, draft_id)
                    target = db.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id == doc,
                                                                 KnowledgeDoc.version == version)).one_or_none()
                    if target is not None:
                        target.status = "archived"
                    row.status = "confirmed"; row.failure = None
                    audit(db, actor, "publication_reconciled", pending_id=row.pending_id, draft_id=draft_id,
                          outcome="restored_for_explicit_retry", content_hash=draft["content_hash"])
            except Exception:
                with self.factory() as db:
                    row = db.get(KnowledgeDraft, draft_id)
                    # A lost recovery acknowledgement cannot undo a recovery that
                    # committed; a later request safely rechecks the file above.
                    if row.status != "confirmed":
                        row.status = "reconcile_required"
                        audit(db, actor, "recovery_failed", pending_id=row.pending_id, draft_id=draft_id,
                              error_code="recovery_verification_failed")
                raise FeedbackUnavailable("recovery_verification_failed") from None
            finally:
                invalidate_safely()
            return self.get(draft_id, actor)

    def _check_targets(self,draft,db=None):
        if db is None:
            with self.factory() as session:
                return self._check_targets(draft,session)
        doc=draft["metadata"]["doc_id"]
        expected={(r["doc_id"],r["version"]):r["status"] for r in draft["supersedes"]}
        current={ (r.doc_id,r.version):r.status for r in db.scalars(select(KnowledgeDoc).where(
            KnowledgeDoc.doc_id.in_({doc,*[a for a,_ in expected]}),KnowledgeDoc.status.in_(["active","conflicted"]))) }
        if current!=expected:
            raise FeedbackConflict("effective_versions_changed")
