"""配置加载基础测试。

验证：
    1. config.yaml 是否正常加载（核心段落字段值）
    2. ${VAR} 占位符是否解析为环境变量
    3. APP_ 前缀 + __ 嵌套分隔符的环境变量是否覆盖成功

说明：
    统一使用 Settings(_env_file=None) 构造，隔离真实 .env，
    保证测试确定性；占位符与覆盖均通过 monkeypatch 注入环境变量。
"""
from app.config import CONFIG_FILE, Settings, load_raw_config


def make_settings() -> Settings:
    """构造不受真实 .env 干扰的 Settings 实例。"""
    return Settings(_env_file=None)


# =====================================================
# 1. config.yaml 正常加载
# =====================================================

class TestYamlLoading:

    def test_config_file_exists(self):
        assert CONFIG_FILE.exists()

    def test_application_section(self):
        s = make_settings()
        assert s.application.name == "factoring-sales-coach"
        assert s.application.version == "1.0.0"
        assert s.application.environment == "development"

    def test_server_section(self):
        s = make_settings()
        assert s.server.host == "127.0.0.1"
        assert s.server.port == 8000
        assert s.server.workers == 4
        assert s.server.api_prefix == "/api/v1"

    def test_llm_section(self):
        s = make_settings()
        assert s.llm.provider == "deepseek"
        assert s.llm.api.base_url == "https://api.deepseek.com"
        # 双模型路由
        assert s.llm.models.reasoning.name == "deepseek-v4-pro"
        assert s.llm.models.chat.name == "deepseek-v4-flash"
        assert s.llm.models.reasoning.thinking.enabled is True
        assert s.llm.models.chat.thinking.enabled is False
        assert s.llm.models.reasoning.reasoning_effort == "high"
        # 超时与重试
        assert s.llm.timeout.connect == 10
        assert s.llm.timeout.read == 120
        assert s.llm.retry.max_attempts == 3

    def test_langgraph_section(self):
        s = make_settings()
        assert s.langgraph.workflow_name == "coach_graph"
        assert s.langgraph.state.name == "CoachState"
        assert s.langgraph.checkpoint.backend == "redis"
        assert s.langgraph.checkpoint.ttl == 86400

    def test_agents_section(self):
        s = make_settings()
        # 单 Agent 架构：briefing（节点）+ roleplay（循环）
        assert s.agents.briefing.enabled is True
        assert s.agents.briefing.llm.model == "reasoning"
        assert s.agents.briefing.max_questions == 5
        assert s.agents.roleplay.llm.model == "chat"
        assert s.agents.roleplay.max_turns == 20

    def test_rag_section(self):
        s = make_settings()
        assert s.rag.embedding.model == "qwen3.7-text-embedding"
        assert s.rag.embedding.dimension == 1024
        assert s.rag.vector_database.type == "milvus"
        assert s.rag.vector_database.port == 19530
        assert s.rag.vector_database.collection == "factoring_knowledge_v2"
        assert s.rag.chunk.parent_size == 1200
        assert s.rag.chunk.child_size == 300
        assert s.rag.chunk.overlap == 50
        assert s.rag.retrieval.top_k == 10
        assert s.rag.retrieval.hybrid.fusion == "RRF"
        # 精排：阿里云线上 gte-rerank-v2（§9.1）
        assert s.rag.retrieval.rerank.provider == "dashscope"
        assert s.rag.retrieval.rerank.model == "qwen3.7-text-rerank"
        assert s.rag.retrieval.rerank.min_score == 0.867799
        assert s.rag.retrieval.fallback_min_score is None
        assert s.rag.retrieval.rerank.top_n == 5

    def test_database_section(self):
        s = make_settings()
        assert s.database.mysql.port == 3307
        assert s.database.mysql.database == "factoring_coach"
        # Redis 标准端口 6379
        assert s.database.redis.port == 6379

    def test_raw_config_contains_unmodeled_sections(self):
        raw = load_raw_config()
        # 未建模业务段可通过原始字典访问
        for section in ("agents", "case", "knowledge", "evaluation", "guardrail"):
            assert section in raw, f"缺少业务段: {section}"


# =====================================================
# 2. ${VAR} 占位符解析
# =====================================================

class TestEnvPlaceholder:

    def test_api_key_resolved_from_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-placeholder-123")
        s = make_settings()
        assert s.llm.api.api_key == "sk-test-placeholder-123"

    def test_api_key_none_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        s = make_settings()
        assert s.llm.api.api_key is None


# =====================================================
# 3. 环境变量覆盖
# =====================================================

class TestEnvOverride:

    def test_override_redis_port(self, monkeypatch):
        # 用不同于 yaml 的值（yaml 为 6379），验证环境变量优先级高于 config.yaml
        monkeypatch.setenv("APP_DATABASE__REDIS__PORT", "6400")
        s = make_settings()
        assert s.database.redis.port == 6400

    def test_override_nested_llm_fields(self, monkeypatch):
        monkeypatch.setenv("APP_LLM__API__BASE_URL", "https://api.example.com")
        monkeypatch.setenv("APP_LLM__MODELS__CHAT__TEMPERATURE", "0.7")
        s = make_settings()
        assert s.llm.api.base_url == "https://api.example.com"
        assert s.llm.models.chat.temperature == 0.7

    def test_override_server_port(self, monkeypatch):
        monkeypatch.setenv("APP_SERVER__PORT", "9000")
        s = make_settings()
        assert s.server.port == 9000

    def test_override_takes_priority_over_yaml(self, monkeypatch):
        monkeypatch.setenv("APP_RAG__VECTOR_DATABASE__HOST", "8.133.229.0")
        s = make_settings()
        # yaml 中为 localhost，环境变量覆盖为公网 IP
        assert s.rag.vector_database.host == "8.133.229.0"
