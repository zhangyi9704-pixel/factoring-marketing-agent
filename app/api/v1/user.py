"""主管视角用户成绩接口（TDD §20 Phase 4：GET /users/{id}/scores）。

流式版 /scores/stream：profile 帧（聚合画像）先行，五个 section 帧
（一机会诊断/二复盘辅导/三评分证据/四遗漏问题/五改善空间）逐段推送。
"""
from app.db.mysql import MySQLError
from app.services.session_service import SessionServiceError
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.sse import SSE_HEADERS, sse_event, RequestStreamingResponse
from app.schemas.user import UserScoreReport
from app.services.user_score_service import UserScoreService, UserScoreServiceError

router = APIRouter(prefix="/users", tags=["user"])


def get_user_score_service() -> UserScoreService:
    """UserScoreService 构建依赖（测试用 dependency_overrides 覆盖）。"""
    return UserScoreService()


@router.get("/{user_id}/scores", response_model=UserScoreReport,
            summary="销售成绩画像（主管视角）",
            description="跨会话聚合该用户的评估数据：机会/能力均分、机会等级分布、"
                        "各维度均值、最近评估时间线；仅统计已落库的评估。")
def user_scores(
    user_id: int,
    service: UserScoreService = Depends(get_user_score_service),
) -> UserScoreReport:
    try:
        return UserScoreReport(**service.scores(user_id))
    except UserScoreServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{user_id}/scores/stream",
            summary="销售成绩画像（SSE 流式）",
            description="profile 帧先到先渲染（画像卡片/等级分布/维度均值），"
                        "随后 opportunity/review/evidence/missing/improvement "
                        "五个 section 帧逐段填充辅导报告。")
def user_scores_stream(
    user_id: int,
    service: UserScoreService = Depends(get_user_score_service),
) -> StreamingResponse:
    def gen():
        try:
            for event in service.scores_stream(user_id):
                if event["type"] == "profile":
                    yield sse_event("profile", event["data"])
                elif event["type"] == "section":
                    yield sse_event("section", {
                        "key": event["key"], "data": event["data"],
                    })
                else:
                    yield sse_event("done", {})
        except (MySQLError, SessionServiceError):
            yield sse_event("error", {"detail": "mysql_write_failed"})
        except UserScoreServiceError as exc:
            yield sse_event("error", {"detail": str(exc)})

    return RequestStreamingResponse(
        gen(), media_type="text/event-stream", headers=SSE_HEADERS,
    )
