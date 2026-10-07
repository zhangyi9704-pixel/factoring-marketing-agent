"""复盘评估接口（TDD §15：GET /sessions/{id}/evaluation）。

惰性计算：首次 GET 时对指定会话执行完整评估并幂等落库（TDD §13），
返回机会诊断 / 能力评分 / 证据链 / 遗漏问题 / 下一步行动。
流式版 /evaluation/stream：stage 进度 + section 阶段产物逐段推送。
"""
from app.db.mysql import MySQLError
from app.services.session_service import SessionServiceError
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.sse import SSE_HEADERS, sse_event, RequestStreamingResponse
from app.evaluation.service import EvaluationError, EvaluationService
from app.schemas.evaluation import EvaluationReport

router = APIRouter(tags=["evaluation"])


def get_evaluation_service() -> EvaluationService:
    """EvaluationService 构建依赖（测试用 dependency_overrides 覆盖）。"""
    return EvaluationService()


@router.get("/sessions/{session_id}/evaluation", response_model=EvaluationReport,
            summary="查询复盘评估报告",
            description="惰性计算：首次调用执行评估并幂等落库；"
                        "返回机会诊断/能力评分/证据/遗漏问题/下一步行动。")
def get_evaluation(
    session_id: str,
    service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationReport:
    try:
        return EvaluationReport(**service.get(session_id))
    except EvaluationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/sessions/{session_id}/evaluation/stream",
            summary="复盘评估（SSE 流式）",
            description="逐阶段推送：stage 事件（stage/label/status）标注进度，"
                        "section 事件携带该阶段产物（opportunity/sales_score/"
                        "missing_questions/next_actions，可提前渲染卡片），"
                        "done 事件携带完整报告；已评估会话直接 done。")
def get_evaluation_stream(
    session_id: str,
    service: EvaluationService = Depends(get_evaluation_service),
) -> StreamingResponse:
    def gen():
        try:
            for event in service.run_stream(session_id):
                kind = event.get("type")
                if kind == "stage":
                    yield sse_event("stage", {
                        k: v for k, v in event.items() if k != "type"
                    })
                elif kind == "section":
                    yield sse_event("section", {
                        k: v for k, v in event.items() if k != "type"
                    })
                else:
                    yield sse_event("done", {"report": event["report"]})
        except (MySQLError, SessionServiceError):
            yield sse_event("error", {"detail": "mysql_write_failed"})
        except EvaluationError as exc:
            yield sse_event("error", {"detail": str(exc)})

    return RequestStreamingResponse(
        gen(), media_type="text/event-stream", headers=SSE_HEADERS,
    )
