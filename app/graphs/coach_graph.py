"""单 Agent 教练图 coach_graph（技术设计文档（新）§6.1）。

    START → load_case → briefing（访前简报，§7）→ init_customer_state
       ↓
    ┌─▶ sales_input（interrupt 挂起，等 Command(resume) 注入销售发言）
    │     ↓ [结束]→ 标记 finished（后续节点跳过，直达出口）
    │  state_judge        # 前置block门禁 + 事实卡 → 释放白名单
    │     ↓
    │  customer_reply     # chat 模型 + fact_constraint + 记忆窗口
    │     ↓
    │  guardrail          # 销售话术违禁检测（每轮）+ 合规留痕
    │     ↓
    │  update_state       # 双向落库、轮次 +1、pending 跨轮保留、max_turns 出口
    │     ↓
    └─ continue? ──否──▶ END（END 后 GET /sessions/{id}/evaluation 惰性复盘）

单图单 Agent：无父子图、无子图 interrupt 上抛（旧 P1 难点消除）；
briefing 为普通节点，销售发言的 Command(resume) 语义与 checkpoint
续跑机制不变。旧 roleplay_graph / coach_workflow / visit_prep_graph 随
双 Agent 架构退役。
"""
from __future__ import annotations

import logging

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agents.coach.briefing import make_briefing_node
from app.agents.roleplay.case_loader import (
    make_init_customer_state_node,
    make_load_case_node,
)
from app.agents.roleplay.customer_reply import make_customer_reply_node
from app.agents.roleplay.guardrail_node import make_guardrail_node, make_update_state_node
from app.agents.roleplay.state_judge import make_state_judge_node, sales_input
from app.config import Settings
from app.graphs.state import CoachState
from app.guardrail.detector import GuardrailService
from app.services.case_service import CaseService
from app.services.llm_service import LLMService
from app.services.session_service import SessionService

logger = logging.getLogger(__name__)

NODE_ORDER = (
    "load_case", "briefing", "init_customer_state", "sales_input",
    "state_judge", "customer_reply", "guardrail", "update_state",
)


def build_coach_graph(
    llm: LLMService,
    *,
    case_service: CaseService,
    guardrail: GuardrailService,
    settings: Settings,
    session_service: SessionService | None = None,
    retriever=None,
    checkpointer: BaseCheckpointSaver | None = None,
):
    """单 Agent 图（briefing + roleplay 六节点循环 + interrupt HITL）。"""
    roleplay = settings.agents.roleplay

    builder = StateGraph(CoachState)
    builder.add_node("load_case", make_load_case_node(case_service))
    builder.add_node("briefing", make_briefing_node(
        llm, settings=settings, retriever=retriever, session_service=session_service, guardrail=guardrail,
    ))
    builder.add_node("init_customer_state", make_init_customer_state_node(llm, session_service,
        guardrail=guardrail, timeout_seconds=settings.agents.briefing.timeout_seconds))
    builder.add_node("sales_input", sales_input)
    builder.add_node("state_judge", make_state_judge_node(guardrail))
    builder.add_node("customer_reply", make_customer_reply_node(
        llm, window=roleplay.memory.window, model_role=roleplay.llm.model,
        guardrail=guardrail, session_service=session_service,
    ))
    builder.add_node("guardrail", make_guardrail_node(guardrail, session_service))
    builder.add_node("update_state", make_update_state_node(
        max_turns=roleplay.max_turns, session_service=session_service,
    ))

    builder.add_edge(START, "load_case")
    builder.add_conditional_edges("load_case", _route_finished,
        {"wait": "briefing", "end": END})  # Invalid cases never generate/persist an opening.
    builder.add_edge("briefing", "init_customer_state")
    builder.add_conditional_edges(                        # load 失败不进 interrupt
        "init_customer_state", _route_finished,
        {"wait": "sales_input", "end": END},
    )
    builder.add_edge("sales_input", "state_judge")
    builder.add_edge("state_judge", "customer_reply")
    builder.add_edge("customer_reply", "guardrail")
    builder.add_edge("guardrail", "update_state")
    builder.add_conditional_edges(
        "update_state", _route_continue, {"continue": "sales_input", "end": END},
    )
    return builder.compile(checkpointer=checkpointer)


def _route_continue(state: CoachState) -> str:
    return "end" if state.get("finished") else "continue"


def _route_finished(state: CoachState) -> str:
    return "end" if state.get("finished") else "wait"


def checkpoint_ttl_minutes(ttl_seconds: int) -> float:
    """RedisSaver 0.5.2 accepts float minutes: preserve even sub-minute TTLs."""
    if ttl_seconds <= 0:
        raise ValueError("Checkpoint TTL must be positive seconds")
    return ttl_seconds / 60.0


def build_redis_checkpointer(
    settings: Settings, *, checkpoint_prefix="checkpoint", checkpoint_write_prefix="checkpoint_write",
) -> BaseCheckpointSaver:
    """RedisSaver（TTL 86400s，thread_id 粒度恢复）。

    后端须为 Redis Stack（RedisJSON + RediSearch 模块）；地址取
    langgraph.checkpoint.host/port（缺省回落 database.redis）。
    模块缺失时 setup() 失败提前暴露（连接错配不应静默降级）。
    """
    import redis as redis_lib
    from langgraph.checkpoint.redis import RedisSaver

    checkpoint_cfg = settings.langgraph.checkpoint
    if checkpoint_cfg.backend == 'memory':
        from langgraph.checkpoint.memory import InMemorySaver
        logger.warning('Explicit development memory checkpoint: restart recovery unavailable')
        return InMemorySaver()
    if not checkpoint_cfg.enabled or checkpoint_cfg.backend != 'redis':
        raise RuntimeError('persistent_checkpoint_required; use explicit backend=memory only for development')
    host = checkpoint_cfg.host or settings.database.redis.host
    port = checkpoint_cfg.port or settings.database.redis.port
    client = redis_lib.Redis(
        host=host, port=port, password=checkpoint_cfg.password,
        decode_responses=True, socket_connect_timeout=5, socket_timeout=30,
    )
    saver = RedisSaver(
        redis_client=client,
        checkpoint_prefix=checkpoint_prefix,
        checkpoint_write_prefix=checkpoint_write_prefix,
        ttl={                                    # default_ttl 单位为分钟
            "default_ttl": checkpoint_ttl_minutes(checkpoint_cfg.ttl),
            "refresh_on_read": True,
        },
    )
    saver.setup()                                 # 建索引；模块缺失在此抛错
    logger.info("RedisSaver 就绪: %s:%s (ttl=%ss)", host, port, checkpoint_cfg.ttl)
    return saver


__all__ = [
    "NODE_ORDER",
    "build_coach_graph",
    "build_redis_checkpointer",
    "Command",
]
