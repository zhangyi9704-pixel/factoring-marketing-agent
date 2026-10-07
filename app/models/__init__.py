"""SQLAlchemy ORM 模型（10张业务表及消息幂等账本）。

表清单：
    t_user                  用户（销售/主管/管理员）
    t_case                  案例库（Case State）
    t_session               演练会话
    t_conversation_message  对话记录（逐轮落库）
    t_visit_report          访前准备报告
    t_evaluation            复盘评估结果
    t_evaluation_evidence   评分证据（关联原始对话）
    t_compliance_log        合规 / HITL 审核日志
    t_knowledge_doc         知识库文档元数据
    t_agent_error           运行期错误/异常审计（审计增强：集中落库）

说明：
    - JSON 字段统一使用 sqlalchemy.JSON（MySQL 原生 JSON 类型）
    - t_case.updated_at 的 ON UPDATE 在 ORM 侧由 onupdate 触发（Python 端），
      与 scripts/init_mysql.sql 的 ON UPDATE CURRENT_TIMESTAMP 等效
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    DECIMAL,
    BigInteger,
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    func,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """ORM 基类。"""


class User(Base):
    """用户。"""

    __tablename__ = "t_user"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    salt: Mapped[Optional[str]] = mapped_column(String(64))
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False, comment="sales / coach / admin")
    team: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Case(Base):
    """案例库（Case State 完整结构存 case_json）。"""

    __tablename__ = "t_case"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(String(32), nullable=False, unique=True, comment="如 INFRA_001")
    industry: Mapped[str] = mapped_column(String(16), nullable=False, comment="基建/医药/电商")
    company: Mapped[Optional[str]] = mapped_column(String(128))
    client_role: Mapped[Optional[str]] = mapped_column(String(64))
    case_json: Mapped[dict] = mapped_column(JSON, nullable=False, comment="完整 Case State")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )


class CoachSession(Base):
    """演练会话（session_id 即 LangGraph thread_id）。"""

    __tablename__ = "t_session"
    __table_args__ = (
        Index("idx_user", "user_id"),
        Index("idx_case", "case_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, comment="= LangGraph thread_id")
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    case_id: Mapped[str] = mapped_column(String(32), nullable=False)
    phase: Mapped[Optional[str]] = mapped_column(String(16), comment="prepare/roleplay/evaluate/report")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active", comment="active/finished/interrupted")
    turn_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    needs_human_review: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class ConversationMessage(Base):
    """对话记录（逐轮落库，logging.save.conversation）。"""

    __tablename__ = "t_conversation_message"
    __table_args__ = (Index("idx_session", "session_id", "turn"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    turn: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, comment="sales / customer")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    revealed_info: Mapped[Optional[dict]] = mapped_column(JSON, comment="本轮释放的 unknown_info（内部标记）")
    guardrail_hit: Mapped[Optional[dict]] = mapped_column(JSON, comment="本轮触发的合规规则")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class MessageRequestRecord(Base):
    """轮次领取及结果账本；唯一键使重连不能重跑演练图。"""
    __tablename__ = "t_message_request"
    __table_args__ = (UniqueConstraint("session_id", "request_id", name="uq_message_request"),)
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    expected_turn: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    result: Mapped[Optional[dict]] = mapped_column(JSON)
    error_code: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class SessionCreateRequestRecord(Base):
    """Durable creation request identity; retry never creates another session."""
    __tablename__ = "t_session_create_request"
    __table_args__ = (UniqueConstraint("user_id", "request_id", name="uq_session_create_request"),)
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    case_id: Mapped[str] = mapped_column(String(32), nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    result: Mapped[Optional[dict]] = mapped_column(JSON)
    error_code: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class VisitReport(Base):
    """访前简报（单 Agent 架构 briefing 节点写入，复用原表）。"""

    __tablename__ = "t_visit_report"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    report_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Evaluation(Base):
    """复盘评估结果。"""

    __tablename__ = "t_evaluation"
    __table_args__ = (Index("idx_user_score", "session_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    opportunity: Mapped[Optional[dict]] = mapped_column(JSON, comment="机会诊断（分数/等级/理由）")
    sales_score: Mapped[Optional[dict]] = mapped_column(JSON, comment="四维能力评分")
    missing: Mapped[Optional[dict]] = mapped_column(JSON, comment="遗漏问题")
    next_action: Mapped[Optional[dict]] = mapped_column(JSON, comment="下一步行动")
    total_score: Mapped[Optional[float]] = mapped_column(DECIMAL(5, 2))
    report_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, comment="已验证报告及来源版本的不可变快照")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class EvaluationEvidence(Base):
    """评分证据（所有评分必须关联原始对话，PRD 强制）。"""

    __tablename__ = "t_evaluation_evidence"
    __table_args__ = (Index("idx_eval", "evaluation_id"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    evaluation_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    dimension: Mapped[Optional[str]] = mapped_column(String(64), comment="评分维度")
    score: Mapped[Optional[float]] = mapped_column(DECIMAL(5, 2))
    turn: Mapped[Optional[int]] = mapped_column(Integer, comment="证据轮次")
    quote_sales: Mapped[Optional[str]] = mapped_column(Text, comment="销售原话")
    quote_customer: Mapped[Optional[str]] = mapped_column(Text, comment="客户原话")
    comment: Mapped[Optional[str]] = mapped_column(String(512), comment="评价")
    # 注：TDD DDL 中本表无 created_at，保持一致不加


class ComplianceLog(Base):
    """合规 / HITL 审核日志（TDD §14）。"""

    __tablename__ = "t_compliance_log"

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(64))
    turn: Mapped[Optional[int]] = mapped_column(Integer)
    rule: Mapped[Optional[str]] = mapped_column(String(64), comment="触发规则")
    raw_text: Mapped[Optional[str]] = mapped_column(Text)
    details: Mapped[Optional[dict]] = mapped_column(JSON, comment="判定层/耗时/降级/规则版本")
    action: Mapped[Optional[str]] = mapped_column(String(16), comment="rewrite / block / human_review")
    review_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", comment="pending/approved/rejected"
    )
    reviewer: Mapped[Optional[str]] = mapped_column(String(64))
    status: Mapped[Optional[str]] = mapped_column(String(16))
    resolution: Mapped[Optional[str]] = mapped_column(Text)
    assignee: Mapped[Optional[int]] = mapped_column(BigInteger)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    user_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    dedup_key: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    external_id: Mapped[Optional[str]] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class KnowledgePending(Base):
    __tablename__ = "t_knowledge_pending"
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    type: Mapped[str] = mapped_column(String(16), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[dict] = mapped_column(JSON, nullable=False)
    severity: Mapped[Optional[str]] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    resolution: Mapped[Optional[str]] = mapped_column(Text)
    assignee: Mapped[Optional[int]] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(64))
    dedup_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)


class KnowledgeDraft(Base):
    __tablename__ = "t_knowledge_draft"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    pending_id: Mapped[int] = mapped_column(BigInteger, unique=True, nullable=False)
    creator_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    confirmed_by: Mapped[Optional[int]] = mapped_column(BigInteger)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    adopted: Mapped[Optional[dict]] = mapped_column(JSON)
    supersedes: Mapped[list] = mapped_column(JSON, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    publication_snapshot: Mapped[Optional[dict]] = mapped_column(JSON, comment="Pre-write vector snapshots for explicit recovery")
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="draft")
    failure: Mapped[Optional[str]] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    confirmed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


class BusinessAudit(Base):
    __tablename__ = "t_business_audit"
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer,"sqlite"), primary_key=True, autoincrement=True)
    pending_id: Mapped[Optional[int]] = mapped_column(BigInteger, index=True)
    queue: Mapped[str] = mapped_column(String(16), default="knowledge")
    draft_id: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    actor_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    details: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class KnowledgeDoc(Base):
    """知识库文档元数据。"""

    __tablename__ = "t_knowledge_doc"
    __table_args__ = (UniqueConstraint("doc_id", "version", name="uq_knowledge_doc_version"),)

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer,"sqlite"), primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(512), nullable=False, comment="知识来源文献")
    doc_id: Mapped[Optional[str]] = mapped_column(String(64))
    version: Mapped[Optional[str]] = mapped_column(String(64))
    effective_date: Mapped[Optional[str]] = mapped_column(String(10))
    industry: Mapped[Optional[str]] = mapped_column(String(16))
    category: Mapped[Optional[str]] = mapped_column(String(16))
    topic: Mapped[Optional[str]] = mapped_column(String(256))
    content_snapshot: Mapped[Optional[str]] = mapped_column(Text, comment="Committed document body, independent of source files")
    content_hash: Mapped[Optional[str]] = mapped_column(String(64))
    chunk_manifest: Mapped[Optional[dict]] = mapped_column(JSON, comment="Committed scalar hashes prevent same-version partial visibility")
    ingestion_operation_id: Mapped[Optional[str]] = mapped_column(String(32))
    doc_name: Mapped[str] = mapped_column(String(256), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_cnt: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="indexed")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class KnowledgeIngestionOperation(Base):
    """Committed with the catalog; identifies the exact same-version overwrite."""
    __tablename__ = "t_knowledge_ingestion_operation"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    vector_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    chunk_cnt: Mapped[int] = mapped_column(Integer, nullable=False)
    committed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AgentErrorLog(Base):
    """运行期错误/异常审计（集中落库，审计增强）。

    DatabaseErrorHandler 仅对 ERROR / CRITICAL / 携带堆栈的记录写入，
    并通过 AUDIT_CONTEXT 关联请求路径、会话与用户。
    """

    __tablename__ = "t_agent_error"
    __table_args__ = (
        Index("idx_error_session", "session_id"),
        Index("idx_error_level", "level"),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer,"sqlite"), primary_key=True, autoincrement=True)
    level: Mapped[str] = mapped_column(String(16), nullable=False, comment="ERROR/CRITICAL")
    logger_name: Mapped[Optional[str]] = mapped_column(String(64), comment="来源 logger")
    message: Mapped[Optional[str]] = mapped_column(Text, comment="日志消息")
    exc_type: Mapped[Optional[str]] = mapped_column(String(128), comment="异常类型")
    traceback: Mapped[Optional[str]] = mapped_column(Text, comment="完整堆栈")
    session_id: Mapped[Optional[str]] = mapped_column(String(64), comment="关联会话")
    user_id: Mapped[Optional[int]] = mapped_column(BigInteger, comment="关联用户")
    path: Mapped[Optional[str]] = mapped_column(String(255), comment="请求路径")
    method: Mapped[Optional[str]] = mapped_column(String(8), comment="HTTP 方法")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
