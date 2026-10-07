"""FastAPI 应用入口。

单 Agent 架构（技术设计文档（新））：
    POST /api/v1/sessions —— 创建会话，图内 briefing 节点产出访前简报
    （POST /api/v1/sessions 系列，Redis checkpoint 续跑）；
    GET /api/v1/sessions/{id}/evaluation —— 复盘评估（惰性计算落库）；
    GET /api/v1/users/{id}/scores + POST /api/v1/knowledge/ingest ——
    主管画像 + 知识库注入；
    前端：静态 SPA（frontend/，/ 与 /static 托管）。
"""
import logging
from app.services.session_service import SessionServiceError
from app.db.mysql import MySQLError
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler

from app.api.v1 import router as api_router
from app.config import PROJECT_ROOT, get_settings
from app.logging_config import clear_audit_context, set_audit_context, setup_logging

logger = logging.getLogger(__name__)

settings = get_settings()

# 请求审计上下文：从路径解析 session_id / user_id（关联 t_agent_error）
_SESSION_PATH = re.compile(r"/api/v1/sessions/([^/]+)")
_USER_PATH = re.compile(r"/api/v1/users/(\d+)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动：配置运行期日志（滚动文件全级别 + stderr WARNING+ + 集中错误审计）。"""
    from app.services.auth_service import jwt_secret
    jwt_secret()  # Fail closed before serving when the secret is absent/too short.
    from app.db.runtime_schema import verify_runtime_schema
    from app.db.mysql import get_mysql_database
    verify_runtime_schema(get_mysql_database().engine)
    setup_logging()
    logger.info("Agent 运行期日志已配置（file=%s）", settings.logging.file.path)
    yield


app = FastAPI(
    title=settings.application.name,
    version=settings.application.version,
    description="保理营销教练 Agent（单 Agent 架构：访前简报 + 情景演练 + 复盘评估 + "
                "主管画像 + 知识库注入；前端 SPA 已附）",
    lifespan=lifespan,
)

# 本地演示：跨域放开，前端可独立调试（同源托管时实际不触发）
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


def audit_kwargs_for(path: str, method: str) -> dict:
    """从请求路径解析审计上下文（session_id / user_id / path / method）。

    供中间件与单测复用；session_id 取 /sessions/{id}，user_id 取 /users/{id}。
    """
    kwargs = {"path": path, "method": method}
    session_m = _SESSION_PATH.search(path)
    user_m = _USER_PATH.search(path)
    if session_m:
        kwargs["session_id"] = session_m.group(1)
    if user_m:
        kwargs["user_id"] = int(user_m.group(1))
    return kwargs


@app.exception_handler(SessionServiceError)
async def session_write_error(request, exc):
    return JSONResponse(status_code=503, content={"detail": "mysql_write_failed"})


@app.exception_handler(MySQLError)
async def database_error(request, exc):
    return JSONResponse(status_code=503, content={"detail": "mysql_unavailable"})


from app.services.pending_service import PendingConflict


@app.exception_handler(PendingConflict)
async def pending_registration_busy(request, exc):
    # Claim/close conflicts are translated to 409 by their route. A registration
    # lock timeout is temporary unavailability, never a successful fake marker.
    return JSONResponse(status_code=503, content={"detail": "pending_registration_busy"})


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    if request.url.path == settings.server.api_prefix + "/auth/login":
        # Default validation responses echo input, including plaintext passwords.
        return JSONResponse(status_code=422, content={"detail": "invalid_login_payload"})
    return await request_validation_exception_handler(request, exc)


@app.middleware("http")
async def audit_context_middleware(request: Request, call_next):
    """填充请求级审计上下文（AUDIT_CONTEXT），供 DatabaseErrorHandler 关联会话/用户/路径。

    捕获未处理异常并 logger.exception 上报（既落滚动文件，也经集中错误表审计），
    然后继续抛出由 Starlette 返回 500。
    """
    set_audit_context(**audit_kwargs_for(request.url.path, request.method))
    try:
        return await call_next(request)
    except Exception:
        logger.exception("请求未处理异常: %s %s", request.method, request.url.path)
        raise
    finally:
        clear_audit_context()


# 路由挂载（TDD §15）
app.include_router(api_router, prefix=settings.server.api_prefix)

# 前端静态托管（frontend/ 存在才挂载，避免无前端环境报错）
_frontend_dir = PROJECT_ROOT / "frontend"
if _frontend_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(str(_frontend_dir / "index.html"))
