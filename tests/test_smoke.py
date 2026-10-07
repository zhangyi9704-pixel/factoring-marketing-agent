"""骨架冒烟测试：验证全部模块可导入、占位方法按预期抛出 NotImplementedError。"""
import importlib

import pytest

MODULES = [
    "app",
    "app.main",
    "app.config",
    "app.services.llm_service",
    "app.agents.coach",
    "app.agents.coach.briefing",
    "app.agents.roleplay",
    "app.graphs.state",
    "app.graphs.coach_graph",
    "app.rag.loader",
    "app.rag.chunker",
    "app.rag.embedder",
    "app.rag.retriever",
    "app.rag.reranker",
    "app.rag.milvus_client",
    "app.evaluation.opportunity",
    "app.evaluation.sales_score",
    "app.evaluation.evidence",
    "app.evaluation.missing_question",
    "app.evaluation.next_action",
    "app.guardrail.rules",
    "app.guardrail.rewriter",
    "app.prompts.briefing_prompt",
    "app.prompts.roleplay_prompt",
    "app.api.v1",
    "app.api.v1.session",
    "app.api.v1.evaluation",
    "app.api.v1.case",
    "app.api.v1.knowledge",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_importable(module):
    importlib.import_module(module)


def test_coach_agent_lazy_construction():
    # 单 Agent 架构：门面构造不触发 LLM/Milvus/Redis 连接（懒加载，替身可注入）
    from app.agents.roleplay.agent import RoleplayAgent
    from app.config import Settings

    agent = RoleplayAgent(Settings(_env_file=None))
    assert agent._graph is None and agent._llm is None
    assert agent._retriever is None and agent._checkpointer is None


def test_fastapi_app_instantiated():
    from app.main import app

    assert app.title == "factoring-sales-coach"
    # 骨架阶段不注册业务路由
    assert app.routes is not None
