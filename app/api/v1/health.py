"""健康检查接口 `GET /api/v1/health`（Milvus/Redis/MySQL/checkpoint 连通性及模型配置状态矩阵）。

复用 app/services/health.py::HealthService（组件懒加载、可注入替身），
客户端维度：状态码 200 恒返回；业务"是否全部健康"由响应体 status 表达
（healthy / degraded），异常组件在 components.<name>.detail 给出原因。
"""
from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
import time

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api.deps import bearer, get_auth_service, get_current_user
from app.services.health import HealthService

router = APIRouter(tags=["health"])


def authorize_cloud_probe(
    llm: bool = Query(False),
    credentials: HTTPAuthorizationCredentials = Depends(bearer),
    auth_service=Depends(get_auth_service),
):
    if not llm:
        return None
    user = get_current_user(credentials, auth_service)
    if user.role != "admin":
        raise HTTPException(403, "role_forbidden")
    return user


class CloudProbeGate:
    """A process-wide cooldown and single in-flight probe bound cloud cost."""
    def __init__(self, interval=30, clock=time.monotonic):
        self.interval = interval
        self.clock = clock
        self.lock = Lock()
        self.busy = False
        self.next_at = 0

    @contextmanager
    def enter(self):
        with self.lock:
            now = self.clock()
            if self.busy or now < self.next_at:
                retry_after = max(1, int(self.next_at - now + 0.999))
                raise HTTPException(429, "cloud_probe_rate_limited", headers={"Retry-After": str(retry_after)})
            self.busy = True
            self.next_at = now + self.interval
        try:
            yield
        finally:
            with self.lock:
                self.busy = False


cloud_probe_gate = CloudProbeGate()


def get_health_service() -> HealthService:
    """HealthService 构建依赖（测试用 dependency_overrides 覆盖）。"""
    return HealthService()


@router.get(
    "/health",
    response_model=dict,
    summary="健康检查",
    description="逐组件探测 Milvus/Redis/MySQL/checkpoint 连通性及模型配置状态，返回矩阵。"
                "模型默认 configured/unverified；llm=1 仅管理员可探测 LLM（会请求云 API，每进程30秒一次）。",
)
def health_check(
    llm: bool = Query(False, description="是否执行 LLM 连通探测（默认 0；1 调用云 API）"),
    service: HealthService = Depends(get_health_service),
    cloud_user=Depends(authorize_cloud_probe),
) -> dict:
    if llm:
        with cloud_probe_gate.enter():
            return service.check(include_llm=True)
    return service.check(include_llm=llm)


__all__ = ["router", "get_health_service"]
