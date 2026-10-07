"""访前简报节点单元测试（FakeLLM / FakeRetriever / FakeSession 注入）。

覆盖技术设计文档（新）§7：
    - 组装：LLM 字段归一化 + 确定性字段（industry/compliance_warnings/sources）覆盖；
    - 防泄漏（§7.3）：LLM 输出仅基于 public_case，内部字段绝不出现在简报；
    - 业务澄清（核心任务 1）：行业核查清单规则比对；
    - 容错：检索失败 / LLM 失败降级为规则拼装，errors 留痕不中断。
"""
import pytest

from app.agents.coach.briefing import (
    DEFAULT_GOALS,
    DEFAULT_NON_GOALS,
    assemble_briefing,
    make_briefing_node,
    rule_clarifications,
)
from app.config import Settings
from app.rag.retriever import RetrievalItem
from app.schemas.case import assert_no_internal_leak, to_public_case
from tests.test_case_schema import make_case
from tests.test_roleplay_nodes import (
    FakeLLM,
    FakeRetriever,
    FakeSessionService,
    FAKE_BRIEFING_LLM_OUTPUT,
)


def make_public_case() -> dict:
    return to_public_case(make_case()).model_dump()


def retrieval_item(chunk_id="c1", doc="product.md", source="product", priority=2):
    return RetrievalItem(
        chunk_id=chunk_id, parent_id="p1", chunk_type="child",
        source=source, doc_name=doc, priority=priority,
        content=f"{chunk_id} 内容", score=0.9, parent_content="P1 全文",
    )


def base_state(**overrides) -> dict:
    state = {
        "session_id": "sess_brief", "case_id": "INFRA_001",
        "public_case": make_public_case(), "errors": [],
    }
    state.update(overrides)
    return state


def make_node(llm=None, *, retriever=None, session_service=None, settings=None):
    settings = settings or Settings(_env_file=None)
    return make_briefing_node(
        llm or FakeLLM(), settings=settings,
        retriever=retriever if retriever is not None else FakeRetriever(
            items=[retrieval_item("c1", "product.md"),
                   retrieval_item("c2", "product.md")],
        ),
        session_service=session_service or FakeSessionService(),
    )


# =====================================================
# 正常路径
# =====================================================

class TestBriefingNode:

    def test_assembles_llm_fields_and_persists(self):
        recorder = FakeSessionService()
        node = make_node(session_service=recorder)
        result = node(base_state())

        briefing = result["briefing"]
        assert briefing["visit_goals"] == FAKE_BRIEFING_LLM_OUTPUT["visit_goals"]
        assert briefing["priority_questions"][0]["question"] == "项目业主是哪家？"
        assert briefing["possible_objections"][0]["objection"] == "我们不需要融资"
        # Absence of a disclosure is not evidence that a pledge exists.
        assert briefing["risk_warnings"][0]["basis"] == "依据不足，请人工确认原始资料。"
        assert briefing["degraded"] and result["needs_human_review"]
        assert result["knowledge_missing"] is False
        assert recorder.briefings["sess_brief"] == briefing        # t_visit_report 落库

    def test_deterministic_fields_rule_rendered(self):
        settings = Settings(_env_file=None)
        node = make_node(settings=settings)
        briefing = node(base_state())["briefing"]

        # 行业受控枚举（案例字段，非 LLM 输出）
        assert briefing["industry"] == "基建"
        assert briefing["customer_profile"]["company"] == "华建建设有限公司"
        # 合规警示由 config guardrail.rules.forbidden 渲染
        assert briefing["compliance_warnings"][:4] == [
            f"禁止{r}" for r in settings.guardrail.rules.forbidden
        ]
        # 引用溯源按 doc 去重（两条命中同一 product.md → 一条）
        assert briefing["sources"][0]["doc"] == "product.md"
        assert briefing["sources"][0]["priority"] == 2
        assert briefing["sources"][0]["source"] == "product"

    def test_llm_extra_keys_dropped_and_leak_checked(self):
        # LLM 越权输出内部字段 → 组装白名单键直接丢弃，简报过防泄漏校验
        poisoned = dict(FAKE_BRIEFING_LLM_OUTPUT)
        poisoned["unknown_info"] = ["是否完成对账"]
        poisoned["hidden_risks"] = ["未计量30%"]
        node = make_node(FakeLLM(briefing=poisoned))
        briefing = node(base_state())["briefing"]

        assert "unknown_info" not in briefing and "hidden_risks" not in briefing
        assert_no_internal_leak(briefing)                          # 不抛即通过

    def test_idempotent_on_rerun(self):
        node = make_node()
        assert node(base_state(briefing={"case_id": "INFRA_001"})) == {}
        assert node(base_state(finished=True, finish_reason="error")) == {}

    def test_query_built_from_case_fields(self):
        retriever = FakeRetriever(items=[])
        node = make_node(retriever=retriever)
        node(base_state())
        query = retriever.queries[0]
        assert "基建" in query and "华建建设有限公司" in query
        assert "付款方 市交通集团" in query and "应收账款 50000000 元" in query


# =====================================================
# 业务澄清（核心任务 1，纯规则）
# =====================================================

class TestRuleClarifications:

    def test_uncovered_checklist_items_become_clarifications(self):
        clarifications = rule_clarifications(make_public_case(), Settings(_env_file=None))
        items = [c["element"] for c in clarifications]
        # fixture 公开事实已标注"付款方: 市交通集团"（键:值）→ 付款主体视为已覆盖，不再待确认
        assert "付款主体" not in items
        # 验收状态 / 质押担保 未被公开事实覆盖 → 进入待确认清单
        assert "验收状态" in items and "质押担保" in items
        # "回款周期长" 命中 账期与回款 → 不再出现在澄清清单
        assert "账期与回款" not in items
        entry = next(c for c in clarifications if c["element"] == "验收状态")
        assert entry["impact"] == "未验收部分应收不成立"

    def test_unknown_industry_empty(self):
        public = make_public_case()
        public["industry"] = "未知行业"
        assert rule_clarifications(public, Settings(_env_file=None)) == []


# =====================================================
# 容错降级
# =====================================================

class TestBriefingDegradation:

    def test_retrieval_failure_degrades_with_missing_flag(self):
        retriever = FakeRetriever(error=RuntimeError("Milvus 连接失败"))
        node = make_node(retriever=retriever)
        result = node(base_state())

        assert result["knowledge_missing"] is True
        assert result["knowledge_context"] == []
        assert any("Milvus 连接失败" in e for e in result["errors"])
        assert result["briefing"]["knowledge_missing"]["missing"] is True

    def test_no_hits_treated_as_missing(self):
        node = make_node(retriever=FakeRetriever(items=[], missing=True))
        result = node(base_state())
        assert result["knowledge_missing"] is True
        assert result["briefing"]["sources"] == []

    def test_llm_failure_falls_back_to_rules(self):
        class Boom:
            def chat_json(self, *a, **kw):
                raise RuntimeError("LLM 超时")

        node = make_node(Boom())
        result = node(base_state())
        briefing = result["briefing"]

        assert briefing["visit_goals"] == list(DEFAULT_GOALS)
        assert briefing["non_goals"] == list(DEFAULT_NON_GOALS)
        # 澄清清单降级为优先问题（标注规则生成）
        assert briefing["priority_questions"][0]["order_rationale"] == \
            "行业核查清单要求（LLM 降级，规则生成）"
        # 已知风险降级为风险提示
        assert {"risk": "政府项目审批链条长", "basis": "案例已知风险"} in \
            briefing["risk_warnings"]
        assert any("LLM 超时" in e for e in result["errors"])

    def test_malformed_llm_output_tolerated(self):
        node = make_node(FakeLLM(briefing="不是字典"))
        briefing = node(base_state())["briefing"]
        assert briefing["visit_goals"] == list(DEFAULT_GOALS)      # 规则兜底

    def test_persist_failure_is_explicit(self):
        class BadSession:
            def save_briefing(self, *a, **kw):
                raise RuntimeError("MySQL 不可用")

        node = make_node(session_service=BadSession())
        import pytest
        from app.services.session_service import SessionServiceError
        with pytest.raises(SessionServiceError, match="briefing_write_failed"):
            node(base_state())


# =====================================================
# assemble_briefing 归一化细节
# =====================================================

class TestAssembleNormalization:

    def test_questions_limit_and_string_objections(self):
        settings = Settings(_env_file=None)
        raw = {
            "priority_questions": [
                {"question": f"Q{i}"} for i in range(8)
            ],                                              # 超 max_questions=5
            "possible_objections": ["我们要找银行",          # 字符串形态
                                    {"objection": "", "response_hint": "x"}],
        }
        briefing = assemble_briefing(
            public_case=make_public_case(), raw=raw,
            knowledge_context=[], clarifications=[],
            knowledge_missing=False, knowledge_message="", settings=settings,
        )
        assert len(briefing["priority_questions"]) == 5
        assert briefing["priority_questions"][0]["purpose"] == "待确认"   # 缺省补齐
        assert briefing["possible_objections"] == [
            {"objection": "我们要找银行", "response_hint": "待确认"},
        ]

    def test_sources_dedup_keeps_first_priority(self):
        ctx = [
            {"doc_name": "a.md", "priority": 3},
            {"doc_name": "a.md", "priority": 1},
            {"doc_name": "b.md", "priority": 2},
        ]
        settings = Settings(_env_file=None)
        briefing = assemble_briefing(
            public_case=make_public_case(), raw={},
            knowledge_context=ctx, clarifications=[],
            knowledge_missing=False, knowledge_message="", settings=settings,
        )
        assert briefing["sources"] == [
            {"doc": "a.md", "priority": 3}, {"doc": "b.md", "priority": 2},
        ]
