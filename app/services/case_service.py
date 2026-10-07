"""Case 库服务（TDD §10.2 / §15）。

职责：
    - YAML Mock 案例加载与校验（cases/*.yaml → CaseState）；
    - 同步到 MySQL t_case（按 case_id 幂等 upsert）；
    - 查询：list_cases / get_case（agent 全量视角）、get_public_case
      （销售视角，双视角过滤见 app/schemas/case.py）。

session_factory 可注入（单测 FakeSession），默认 MySQL 事务作用域。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, ContextManager, Optional

import yaml

from app.config import PROJECT_ROOT, Settings, get_settings
from app.db.mysql import get_mysql_database
from app.models import Case
from app.schemas.case import CaseState, PublicCase, to_public_case

logger = logging.getLogger(__name__)

SUPPORTED_CASE_SUFFIXES = (".yaml", ".yml")


class CaseServiceError(RuntimeError):
    """Case 加载或同步错误。"""


class CaseService:
    """Mock Case 库（YAML 种子 → MySQL t_case）。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        session_factory: Optional[Callable[[], ContextManager]] = None,
        case_dir: Optional[Path] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.session_factory = session_factory or get_mysql_database().session_scope
        self.case_dir = Path(case_dir) if case_dir else PROJECT_ROOT / "cases"

    # -------------------------------------------------
    # YAML 加载与校验
    # -------------------------------------------------

    def load_yaml_cases(self) -> list[CaseState]:
        """加载 case_dir 下全部 YAML 案例并做 Schema 校验。"""
        if not self.case_dir.is_dir():
            raise CaseServiceError(f"案例目录不存在: {self.case_dir}")
        cases: list[CaseState] = []
        for path in sorted(self.case_dir.iterdir()):
            if path.suffix.lower() not in SUPPORTED_CASE_SUFFIXES:
                continue
            cases.append(self._load_case_file(path))
        if not cases:
            raise CaseServiceError(f"案例目录为空: {self.case_dir}")
        return cases

    def _load_case_file(self, path: Path) -> CaseState:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return CaseState(**data)
        except Exception as exc:
            raise CaseServiceError(f"案例文件解析失败 ({path.name}): {type(exc).__name__}") from None

    # -------------------------------------------------
    # 同步（YAML → MySQL t_case，按 case_id 幂等 upsert）
    # -------------------------------------------------

    def sync_to_mysql(self, cases: Optional[list[CaseState]] = None) -> dict:
        """同步案例到 t_case；重复执行安全（存在则更新，不存在则插入）。

        Returns:
            {"total": n, "inserted": x, "updated": y, "case_ids": [...]}
        """
        if cases is None:
            cases = self.load_yaml_cases()
        inserted = updated = 0
        with self.session_factory() as session:
            for case in cases:
                row = session.query(Case).filter(
                    Case.case_id == case.case_id,
                ).one_or_none()
                if row is None:
                    session.add(Case(
                        case_id=case.case_id,
                        industry=case.industry,
                        company=case.company,
                        client_role=case.client_role,
                        case_json=case.model_dump(),
                        enabled=True,
                    ))
                    inserted += 1
                else:
                    row.industry = case.industry
                    row.company = case.company
                    row.client_role = case.client_role
                    row.case_json = case.model_dump()
                    row.enabled = True
                    updated += 1
        result = {
            "total": len(cases), "inserted": inserted, "updated": updated,
            "case_ids": [c.case_id for c in cases],
        }
        logger.info("Case 同步完成: %s", result)
        return result

    # -------------------------------------------------
    # 查询（storage: mysql，TDD §10.2）
    # -------------------------------------------------

    def list_cases(self, industry: Optional[str] = None) -> list[CaseState]:
        """列出启用案例（可按行业过滤）。"""
        with self.session_factory() as session:
            query = session.query(Case).filter(Case.enabled.is_(True))
            if industry:
                query = query.filter(Case.industry == industry)
            rows = query.order_by(Case.case_id).all()
            return [self._decode_row(row) for row in rows]

    def get_case(self, case_id: str) -> Optional[CaseState]:
        """取单个案例（agent 全量视角）；不存在返回 None。"""
        with self.session_factory() as session:
            row = session.query(Case).filter(
                Case.case_id == case_id, Case.enabled.is_(True),
            ).one_or_none()
            return self._decode_row(row) if row else None

    def get_public_case(self, case_id: str) -> Optional[PublicCase]:
        """取单个案例的销售视角视图（双视角过滤，§10.3）。"""
        case = self.get_case(case_id)
        return to_public_case(case) if case else None

    @staticmethod
    def _decode_row(row) -> CaseState:
        try:
            return CaseState.model_validate(row.case_json)
        except Exception as exc:
            raise CaseServiceError("stored_case_invalid:" + type(exc).__name__) from None
