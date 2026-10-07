"""演练会话接口（TDD §15：POST /sessions、messages、finish、resume）。"""
from app.db.mysql import MySQLError
from app.services.session_service import SessionServiceError
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.agents.roleplay.agent import RoleplayAgent, RoleplayAgentError
from app.api.deps import get_current_user, require_owned_session, get_ownership_service
from app.services.auth_service import Principal
from app.api.sse import SSE_HEADERS, sse_event, RequestStreamingResponse
from app.schemas.session import (
    CreateSessionRequest,
    MessageRequest,
    SessionCreatedResponse,
    SessionStatusResponse,
    TurnResponse,
    VisibleDelta,
    SessionReconcileRequest,
    SessionRecoveryResponse,
)

router = APIRouter(prefix="/sessions", tags=["session"])


def get_roleplay_agent() -> RoleplayAgent:
    """RoleplayAgent 构建依赖（测试用 dependency_overrides 覆盖）。"""
    return RoleplayAgent()


@router.post("", response_model=SessionCreatedResponse, status_code=201,
             summary="创建演练会话",
             description="选择案例开始演练；图内 briefing 节点产出访前简报，"
                         "随销售视角 public_case 与客户开场白一并返回（§7/§12）。")
def create_session(
    payload: CreateSessionRequest,
    user: Principal = Depends(get_current_user),
    agent: RoleplayAgent = Depends(get_roleplay_agent),
) -> SessionCreatedResponse:
    try:
        result = agent.start_session(payload.case_id, user.id, request_id=payload.request_id)
    except RoleplayAgentError as exc:
        message = str(exc)
        status = 404 if "案例不存在" in message else 409
        raise HTTPException(status_code=status, detail=message) from exc
    return SessionCreatedResponse(**result)


def require_recovery_scope(session_id: str, user=Depends(get_current_user), service=Depends(get_ownership_service)):
    row = service.get_session(session_id)
    if row is None:
        raise HTTPException(404, 'session_not_found')
    if user.role != 'admin' and row['user_id'] != user.id:
        raise HTTPException(403, 'session_owner_forbidden')
    return user


@router.post('/{session_id}/reconcile', response_model=SessionRecoveryResponse,
             summary='核对中断请求或明确终止会话（本人/管理员）')
def reconcile_session(session_id: str, payload: SessionReconcileRequest,
                      user=Depends(require_recovery_scope), agent=Depends(get_roleplay_agent)):
    try:
        return agent.reconcile_session(session_id, payload.request_id,
            actor_id=user.id, action=payload.action, reason=payload.reason)
    except RoleplayAgentError as exc:
        raise HTTPException(409, str(exc)) from None


@router.post("/{session_id}/messages", response_model=TurnResponse, dependencies=[Depends(require_owned_session)],
             summary="发送销售话术",
             description="注入一轮销售发言，返回客户回复（fact_constraint 防幻觉）。")
def send_message(
    session_id: str,
    payload: MessageRequest,
    agent: RoleplayAgent = Depends(get_roleplay_agent),
) -> TurnResponse:
    try:
        result = agent.send_message(session_id, payload.content, request_id=payload.request_id, expected_turn=payload.expected_turn)
    except RoleplayAgentError as exc:
        message = str(exc)
        status = 404 if "不存在" in message else 409
        raise HTTPException(status_code=status, detail=message) from exc
    return TurnResponse(**result)


@router.post("/{session_id}/messages/stream", dependencies=[Depends(require_owned_session)],
             summary="发送销售话术（SSE 流式回复）",
             description="客户回复逐段推送：delta 事件携带文本增量（打字机），"
                         "done 事件返回完整轮结果（turn/reply/guardrail_warnings/"
                         "finished/needs_human_review）；业务错误走 error 事件。")
def send_message_stream(
    session_id: str,
    payload: MessageRequest,
    agent: RoleplayAgent = Depends(get_roleplay_agent),
) -> StreamingResponse:
    def gen():
        try:
            for event in agent.send_message_stream(session_id, payload.content, request_id=payload.request_id, expected_turn=payload.expected_turn):
                if event["type"] == "delta":
                    yield sse_event("delta", VisibleDelta(text=event["text"]).model_dump())
                else:
                    yield sse_event("done", TurnResponse(**{
                        k: v for k, v in event.items() if k != "type"
                    }).model_dump())
        except (MySQLError, SessionServiceError):
            yield sse_event("error", {"detail": "mysql_write_failed"})
        except RoleplayAgentError as exc:
            yield sse_event("error", {"detail": str(exc)})
        except Exception:
            yield sse_event('error', {'detail': 'checkpoint_or_request_unavailable'})

    return RequestStreamingResponse(
        gen(), media_type="text/event-stream", headers=SSE_HEADERS,
    )


@router.post("/{session_id}/finish", response_model=SessionStatusResponse, dependencies=[Depends(require_owned_session)],
             summary="结束演练",
             description="注入 [结束] 收敛会话（幂等）；Phase 4 将在此触发 Evaluation。")
def finish_session(
    session_id: str,
    agent: RoleplayAgent = Depends(get_roleplay_agent),
) -> SessionStatusResponse:
    try:
        return SessionStatusResponse(**agent.finish_session(session_id))
    except RoleplayAgentError as exc:
        raise HTTPException(status_code=404 if '不存在' in str(exc) else 409, detail=str(exc)) from exc


@router.post("/{session_id}/resume", response_model=SessionStatusResponse, dependencies=[Depends(require_owned_session)],
             summary="从 Checkpoint 中断恢复",
             description="返回会话 checkpoint 快照；waiting_input=true 时下一条 "
                         "messages 请求即从挂起点续跑（TDD §8.3）；仅返回白名单状态及送达消息。")
def resume_session(
    session_id: str,
    agent: RoleplayAgent = Depends(get_roleplay_agent),
) -> SessionStatusResponse:
    try:
        return SessionStatusResponse(**agent.get_status(session_id))
    except RoleplayAgentError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
