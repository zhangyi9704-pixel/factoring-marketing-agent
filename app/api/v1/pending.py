"""Knowledge/compliance pending routes; owner is always read from stored records."""
from datetime import datetime, timezone
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from app.api.deps import require_reviewer, require_admin
from app.services.pending_service import PendingConflict, PendingService
from app.schemas.knowledge_feedback import FeedbackInput
from app.services.domain_errors import FeedbackConflict, FeedbackValidation

router = APIRouter(tags=["pending"], dependencies=[Depends(require_reviewer)])


def get_pending_service():
    return PendingService()


class PendingPatch(BaseModel):
    model_config = {"extra": "forbid"}
    status: Literal["processing", "resolved", "dismissed"]
    resolution: str | None = Field(default=None, max_length=10000)
    feedback: FeedbackInput | None = None


class PendingReassignment(BaseModel):
    model_config = {"extra": "forbid"}
    expected_assignee: int = Field(gt=0)
    assignee_id: int | None = Field(default=None, gt=0)
    reason: str = Field(min_length=1, max_length=1000)


def utc_naive(value):
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value and value.tzinfo else value


def queue_routes(queue):
    base = "/" + queue + "/pending"

    @router.get(base)
    def list_pending(type: Literal["missing","conflict","out_of_scope","compliance"] | None = None,
                     status: Literal["pending","processing","resolved","dismissed"] | None = None,
                     since: datetime | None = None, until: datetime | None = None,
                     offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=200),
                     actor=Depends(require_reviewer), service=Depends(get_pending_service)):
        since, until = utc_naive(since), utc_naive(until)
        if since and until and since > until:
            raise HTTPException(422, "invalid_time_range")
        return service.list(queue, actor, kind=type, status=status, since=since, until=until, offset=offset, limit=limit)

    @router.get(base + "/{item_id}")
    def get_pending(item_id: int, actor=Depends(require_reviewer), service=Depends(get_pending_service)):
        try:
            result = service.get(queue, item_id, actor)
        except PermissionError:
            raise HTTPException(403, "sales_scope_forbidden") from None
        if result is None:
            raise HTTPException(404, "pending_not_found")
        return result

    @router.patch(base + "/{item_id}")
    def patch_pending(item_id: int, payload: PendingPatch, actor=Depends(require_reviewer), service=Depends(get_pending_service)):
        try:
            result = service.transition(queue, item_id, actor, payload.status, payload.resolution,
                feedback=payload.feedback.model_dump(mode="json", exclude_none=True) if payload.feedback else None)
        except PermissionError:
            raise HTTPException(403, "sales_scope_forbidden") from None
        except (PendingConflict, FeedbackConflict):
            raise HTTPException(409, "pending_state_or_assignee_conflict") from None
        except (ValueError, FeedbackValidation) as exc:
            raise HTTPException(422, str(exc)) from None
        if result is None:
            raise HTTPException(404, "pending_not_found")
        return result

    @router.post(base + "/{item_id}/reassign")
    def reassign_pending(item_id: int, payload: PendingReassignment, actor=Depends(require_admin),
                         service=Depends(get_pending_service)):
        try:
            result = service.reassign(queue, item_id, actor, expected_assignee=payload.expected_assignee,
                                      assignee_id=payload.assignee_id, reason=payload.reason)
        except PermissionError:
            raise HTTPException(403, "sales_scope_forbidden") from None
        except PendingConflict:
            raise HTTPException(409, "pending_state_or_assignee_conflict") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        if result is None:
            raise HTTPException(404, "pending_not_found")
        return result


queue_routes("knowledge")
queue_routes("compliance")
