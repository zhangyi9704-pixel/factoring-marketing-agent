"""案例管理接口（TDD §15：GET /cases、GET /cases/{case_id}，销售视角过滤）。"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.schemas.case import PublicCase, assert_no_internal_leak, to_public_case
from app.services.case_service import CaseService

router = APIRouter(prefix="/cases", tags=["cases"])


def get_case_service() -> CaseService:
    """CaseService 构建依赖（测试用 dependency_overrides 覆盖）。"""
    return CaseService()


@router.get("", response_model=list[PublicCase],
            summary="案例列表（销售视角）",
            description="列出启用案例，可按行业过滤；绝不包含 unknown_info / hidden_risks。")
def list_cases(
    industry: Optional[str] = Query(default=None, description="行业过滤：基建/医药/电商"),
    service: CaseService = Depends(get_case_service),
) -> list[PublicCase]:
    cases = [to_public_case(c) for c in service.list_cases(industry)]
    for case in cases:
        assert_no_internal_leak(case.model_dump())    # API 层二次校验（§19.4）
    return cases


@router.get("/{case_id}", response_model=PublicCase,
            summary="案例详情（销售视角）",
            description="返回 public_case（双视角过滤后），绝不包含 unknown_info / hidden_risks。")
def get_case_detail(
    case_id: str,
    service: CaseService = Depends(get_case_service),
) -> PublicCase:
    public = service.get_public_case(case_id)
    if public is None:
        raise HTTPException(status_code=404, detail=f"案例不存在: {case_id}")
    assert_no_internal_leak(public.model_dump())      # API 层二次校验（§19.4）
    return public
