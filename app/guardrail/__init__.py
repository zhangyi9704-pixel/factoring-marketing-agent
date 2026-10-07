"""合规护栏子系统（TDD §14）：违禁话术检测 + 留痕 + 升级 HITL。"""
from app.guardrail.detector import GuardrailService, GuardrailHit

__all__ = ["GuardrailService", "GuardrailHit"]
