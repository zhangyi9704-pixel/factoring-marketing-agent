"""SSE（text/event-stream）帧组装小工具。

事件协议（三条流式端点共用）：
    delta   {"text": ...}                 客户回复文本增量（messages/stream）
    stage   {"stage","label","status"}    评估阶段进度（evaluation/stream）
    section {"key","data"}                阶段产物 / 看板辅导段（提前渲染）
    profile {...}                         看板聚合画像（scores/stream 首帧）
    done    {...}                         终帧：完整轮结果 / 报告
    error   {"detail": "..."}             业务错误（流已开 200，错误走事件）
"""
from __future__ import annotations

import json
from typing import Any

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",          # nginx 代理不缓冲，保证逐帧到达
}


def sse_event(event: str, data: Any) -> str:
    """组装一帧 SSE（event + data 两行 + 空行结束）。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


from starlette.responses import StreamingResponse


class RequestStreamingResponse(StreamingResponse):
    """Always watch disconnects; sync iterator work remains in AnyIO's thread pool."""
    async def __call__(self, scope, receive, send):
        import anyio
        import threading
        from app.services.external_calls import cancel_event
        event = threading.Event()
        token = cancel_event.set(event)
        try:
            async with anyio.create_task_group() as group:
                async def listen():
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            event.set()
                            group.cancel_scope.cancel()
                            return
                group.start_soon(listen)
                try:
                    await self.stream_response(send)
                finally:
                    event.set()
                    group.cancel_scope.cancel()
        finally:
            event.set()
            cancel_event.reset(token)
        if self.background is not None:
            await self.background()
