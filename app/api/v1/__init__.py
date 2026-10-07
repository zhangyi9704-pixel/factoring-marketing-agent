"""API v1 路由聚合（TDD §15）。

各业务域路由分模块维护，由本模块聚合为统一 router，
main.py 以 settings.server.api_prefix（/api/v1）挂载。
"""
from fastapi import APIRouter, Depends
from app.api.deps import get_current_user, require_admin, require_owned_session, require_score_scope

from app.api.v1 import auth, case, evaluation, health, knowledge, session, user, pending

router = APIRouter()
router.include_router(health.router)
router.include_router(auth.router)
router.include_router(pending.router)
router.include_router(session.router, dependencies=[Depends(get_current_user)])
router.include_router(evaluation.router, dependencies=[Depends(require_owned_session)])
router.include_router(case.router, dependencies=[Depends(get_current_user)])
router.include_router(user.router, dependencies=[Depends(require_score_scope)])
router.include_router(knowledge.router, dependencies=[Depends(get_current_user)])
