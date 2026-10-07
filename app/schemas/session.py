"""演练会话 API Schema（TDD §15 sessions 四端点）。"""
from typing import Optional, Literal

from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """POST /sessions 请求体。"""

    case_id: str = Field(min_length=1, max_length=32, description="案例 id，如 INFRA_001")
    request_id: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$', description="创建请求幂等ID；重试必须保持原ID")
    user_id: Optional[int] = Field(default=None, description="兼容旧客户端；忽略此字段，身份取JWT sub")


class VisibleMessage(BaseModel):
    role: Literal['sales', 'customer']
    turn: int = Field(ge=0)
    content: str


class VisibleDelta(BaseModel):
    text: str


class MessageRequest(BaseModel):
    """POST /sessions/{id}/messages 请求体。"""

    content: str = Field(min_length=1, max_length=4000, description="销售话术")
    request_id: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')
    expected_turn: int = Field(ge=0, strict=True)


class SessionCreatedResponse(BaseModel):
    """POST /sessions 响应（public_case 双视角过滤，§7.3/§10.3）。"""

    session_id: str
    case_id: str
    user_id: int
    public_case: Optional[dict] = None     # PublicCase.model_dump()（销售视角）
    briefing: Optional[dict] = None        # 访前简报（§7.2 Schema，同口径防泄漏）
    opening: Optional[VisibleMessage] = None
    turn: int = 0
    knowledge_mode: Literal["rag", "conflict", "missing"] = "missing"


class GuardrailEvent(BaseModel):
    """前端警示专用白名单，无违规原话或案例状态。"""
    rule: str
    severity: Literal["warn", "block"]
    action: Literal["log", "block"]
    compliant_response: str = ""


class TurnResponse(BaseModel):
    """POST /sessions/{id}/messages 与 finish 的轮次响应。

    注意：revealed_info / unknown_info / hidden_risks 一律不出现在响应。
    """

    turn: int
    reply: str = ""
    finished: bool = False
    finish_reason: str = ""
    guardrail_warnings: list[str] = Field(default_factory=list,
                                          description="本轮销售话术命中的违禁规则名")
    l2_degraded: bool = False
    escalation_id: Optional[str] = None
    input_blocked: bool = False
    guardrail_events: list[GuardrailEvent] = Field(default_factory=list, description="规则、级别、动作及标准合规应答；无原始话术")
    needs_human_review: bool = False
    request_id: Optional[str] = None


class SessionStatusResponse(BaseModel):
    """GET/resume/finish 的会话状态响应（checkpoint 快照）。"""

    session_id: str
    case_id: str = ""
    turn: int = 0
    finished: bool = False
    finish_reason: str = ""
    waiting_input: bool = False            # interrupt 挂起中（可继续发消息）
    needs_human_review: bool = False
    recovery_degraded: bool = False
    interrupted: bool = False
    recovery_request_id: Optional[str] = None
    messages: list[VisibleMessage] = Field(default_factory=list)


class SessionReconcileRequest(BaseModel):
    action: Literal['reconcile', 'abort'] = 'reconcile'
    request_id: str = Field(min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')
    reason: str = Field(min_length=1, max_length=1000)


class SessionRecoveryResponse(BaseModel):
    action: Literal['reconciled', 'aborted']
    status: SessionStatusResponse
    result: Optional[TurnResponse] = None
