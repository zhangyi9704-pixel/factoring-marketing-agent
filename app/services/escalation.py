"""Unbound retrieval markers; caller binds identity before durable registration."""
import logging
import uuid
from typing import Literal

logger = logging.getLogger(__name__)


def register_escalation(kind: Literal["missing", "conflict", "out_of_scope", "compliance"]) -> str:
    if kind not in {"missing", "conflict", "out_of_scope", "compliance"}:
        raise ValueError("unknown escalation kind")
    identifier = "esc_" + uuid.uuid4().hex
    logger.debug("escalation_unbound id=%s kind=%s", identifier, kind)
    return identifier


def register_retrieval(result, *, user_id, session_id=None, turn=0, service=None):
    if result.mode not in {"missing", "conflict"}:
        return result
    from app.services.pending_service import PendingService
    from app.rag.revision import revision
    kind = "out_of_scope" if result.reason == "out_of_scope" else result.mode
    items = [item.to_dict() for item in result.items]
    versions = [{"doc_id":i.get("doc_id"), "version":i.get("version")} for i in items]
    versions.append({"knowledge_revision":revision()})
    row = (service or PendingService()).register(kind, result.query, user_id=user_id, session_id=session_id,
            context={"turn":turn,"reason":result.reason,"items":items}, versions=versions)
    result.escalation_id = row["external_id"]
    return result
