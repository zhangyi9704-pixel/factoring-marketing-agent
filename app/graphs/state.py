"""LangGraph 统一 State 定义（技术设计文档（新）§6.2）。

单 Agent 架构：一张 coach_graph、一个 thread（session）、一条 checkpoint
时间线。旧 MarketingCoachState / VisitPrepState 随双 Agent 架构退役
（访前准备降级为 briefing 节点，§0 结论 3/4）。
"""
from typing import TypedDict


class RoleplayState(TypedDict, total=False):
    """情景演练循环的中继数据（继承旧 §10.4，字段不变）。

    一轮 = sales_input（interrupt 等待销售发言）→ state_judge →
    customer_reply → guardrail → update_state。
    """

    session_id: str             # = LangGraph thread_id
    user_id: int                # 销售用户 id
    case_id: str                # 如 INFRA_001
    case: dict                  # 完整 Case State（agent 视角，§10.3 all）
    initialized: bool           # init_customer_state 已执行（防重复初始化）
    turn: int                   # 已完成轮次（第 N 轮销售发言前为 N-1）
    sales_input: str            # 本轮销售输入（Command(resume=...) 注入）
    active_request_id: str      # 对账证明：最后执行请求的持久账本身份
    active_payload_hash: str
    active_expected_turn: int
    conversation: list          # [{"role": "sales|customer", "turn": int, "content": str}]
    mood: str                   # 客户当前情绪（customer_reply 输出）
    release_whitelist: list     # 本轮允许释放的事实卡 ID
    dim_asked_count: dict       # 每个维度命中的销售轮次
    unlocked_ids: list         # 累计已解锁卡片 ID
    sensitive_fired: list      # 已触发敏感卡 ID
    sensitive_directives: list # 本轮敏感表达指令（内部）
    pending_directives: list    # 已触发未释放的 directive（跨轮保留直至释放）
    released_info: list         # 已释放事实卡 ID（累计，写 t_conversation_message）
    customer_reply: str         # 本轮客户回复文本
    revealed_this_turn: list    # 本轮实际释放的 directive（须 ⊆ release_whitelist）
    guardrail_hits: list        # 本轮销售话术命中 [{"rule": …, "snippet": …}]
    guardrail_checked: bool     # 本轮前置门禁已执行，审计节点不重复检测
    input_blocked: bool         # block：不送达模型、不入对话/计数
    audit_input: str            # 仅供独立审计，update后清空，不注入客户提示词
    blocked_turns: int          # 阻断次数，不增加已送达的对话轮次
    guardrail_hit_count: int    # 同会话累计命中（≥3 升级 human_review，§14）
    compliance_status: dict
    compliance_escalation_id: str
    knowledge_mode: str
    knowledge_reason: str
    escalation_id: str
    needs_human_review: bool    # HITL 队列标记
    finished: bool              # 循环出口（§8.1：[结束] / max_turns）
    finish_reason: str          # sales_end / max_turns
    errors: list                # 节点失败留痕（容错不中断循环）


class CoachState(RoleplayState, total=False):
    """单 Agent 图 State（技术设计文档（新）§6.2）。

    CoachState = RoleplayState 全部字段 + 访前简报字段；
    briefing 节点在 load_case 之后、init_customer_state 之前执行（§6.1）。
    """

    public_case: dict           # 销售视角 Case（briefing 输入专用，防泄漏边界 §7.3）
    briefing: dict              # 访前简报结构化输出（§7.2 Schema）
    knowledge_context: list     # RAG 检索结果（RetrievalItem.to_dict()，含来源优先级）
    knowledge_missing: bool     # 检索空/全低于 min_score → 简报附 human_review 提示
