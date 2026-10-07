"""情景演练 Agent 包（TDD §10.4，Phase 2 实现）。

customer_simulator（模型路由 chat / deepseek-v4-flash）：
    - behavior_rules.py    DSL 解析与执行（state_judge 规则引擎）
    - state_judge.py       sales_input（interrupt HITL）+ 释放白名单判定
    - customer_reply.py    fact_constraint 防幻觉客户回复
    - case_loader.py       load_case / init_customer_state（开场白）
    - guardrail_node.py    每轮合规检测 + update_state 状态推进
    - agent.py             RoleplayAgent 门面（会话生命周期）
"""
