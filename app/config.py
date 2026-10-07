"""config.yaml 配置加载模块。

加载机制（优先级从高到低）：
    1. init 参数
    2. 环境变量（前缀 APP_ + 嵌套分隔符 __，如 APP_DATABASE__REDIS__PORT=6400）
    3. .env 文件（同样支持 APP_ 前缀覆盖）
    4. config.yaml（默认值来源，禁止在代码中硬编码业务配置）

环境变量占位符：
    config.yaml 中 "${VAR}" 形式的值会被解析为同名环境变量
    （如 llm.api.api_key: ${DEEPSEEK_API_KEY}）；
    环境变量不存在时解析为 None，由使用方在启动时校验。

说明：
    本阶段已强类型建模：application/server/llm/langgraph/rag/knowledge/
    agents.briefing+roleplay/guardrail/database/logging/evaluation；
    case/evaluation 与 agents.roleplay 等业务段后续阶段再建模，
    可通过 load_raw_config() 获取原始字典。
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# =====================================================
# 路径
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # app/config.py → 项目根
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
ENV_FILE = PROJECT_ROOT / ".env"

_ENV_PLACEHOLDER = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


# =====================================================
# yaml 数据源：支持 ${VAR} 占位符解析
# =====================================================

class YamlPlaceholderSource(YamlConfigSettingsSource):
    """读取 config.yaml，并将 ${VAR} 占位符解析为环境变量值。"""

    def __init__(self, settings_cls, *, placeholder_values=None):
        super().__init__(settings_cls)
        self.placeholder_values = placeholder_values or {}

    def _resolve(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {k: self._resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve(v) for v in value]
        if isinstance(value, str):
            matched = _ENV_PLACEHOLDER.match(value)
            if matched:
                name = matched.group(1)
                return os.environ.get(name, self.placeholder_values.get(name.lower()))
        return value

    def __call__(self) -> dict[str, Any]:
        return self._resolve(super().__call__())


# =====================================================
# 配置模型（与 config.yaml 结构对应）
# =====================================================

class ApplicationSettings(BaseModel):
    name: str
    version: str
    environment: str


class ServerSettings(BaseModel):
    host: str
    port: int
    workers: int
    api_prefix: str


class LLMApiSettings(BaseModel):
    base_url: str
    api_key: Optional[str] = None  # ${DEEPSEEK_API_KEY}，缺失时为 None


class ThinkingSettings(BaseModel):
    enabled: bool = False


class LLMModelSettings(BaseModel):
    name: str
    temperature: float
    max_tokens: int
    thinking: ThinkingSettings
    reasoning_effort: Optional[str] = None  # 仅 reasoning 模型有


class LLMModelsSettings(BaseModel):
    reasoning: LLMModelSettings
    chat: LLMModelSettings


class LLMTimeoutSettings(BaseModel):
    connect: int = Field(gt=0)
    read: int = Field(gt=0)


class LLMRetrySettings(BaseModel):
    enabled: bool = True
    max_attempts: int = Field(default=3, ge=1, le=5)  # Total attempts, not retries.
    backoff_initial: float = Field(default=0.25, ge=0, le=5)
    backoff_max: float = Field(default=2.0, ge=0, le=10)


class LLMSettings(BaseModel):
    provider: str
    api: LLMApiSettings
    models: LLMModelsSettings
    timeout: LLMTimeoutSettings
    retry: LLMRetrySettings


class CheckpointSettings(BaseModel):
    enabled: bool
    backend: str
    ttl: int = Field(gt=0)  # Application unit: seconds; SDK conversion happens at boundary.
    # Redis Stack 专用地址（缺省回落 database.redis）。
    # checkpoint 后端需要 RedisJSON/RediSearch 模块（redis-stack 镜像），
    # 与普通业务 Redis（session 缓存等）可分开部署。
    host: Optional[str] = None
    port: Optional[int] = None
    password: Optional[str] = None  # Never fall back to the business Redis password.


class StateNameSettings(BaseModel):
    name: str


class LangGraphSettings(BaseModel):
    enabled: bool
    workflow_name: str
    checkpoint: CheckpointSettings
    state: StateNameSettings


class EmbeddingSettings(BaseModel):
    model: str
    dimension: int
    # provider: local（进程内 FlagEmbedding）/ http（自建 embedding 服务，TDD §11.4 方案 A）
    #           / dashscope（阿里云托管 embedding，OpenAI 兼容协议）
    provider: str = "dashscope"
    url: Optional[str] = None        # http 模式的服务地址，如 http://8.133.229.0:8100
    timeout: float = 60.0            # 远端模式超时（秒）
    api_key: Optional[str] = None    # dashscope 模式：${ALIYUN_API_KEY}
    base_url: Optional[str] = None   # dashscope 模式：OpenAI 兼容端点


class VectorDatabaseSettings(BaseModel):
    type: str
    host: str
    port: int
    collection: str


class ChunkSettings(BaseModel):
    strategy: str
    parent_size: int
    child_size: int
    overlap: int


class RerankSettings(BaseModel):
    enabled: bool
    model: str
    top_n: int = 5                 # 精排后保留条数（TDD §11.3：top10 → top5）
    # provider: passthrough（占位透传）/ dashscope（阿里云 gte-rerank-v2，§9.1）
    provider: str = "passthrough"
    min_score: float = 0.1         # 相关性下限：全部低于该值 → 视为检索缺失走 human_review
    api_key: Optional[str] = None  # dashscope 模式：${ALIYUN_API_KEY}（与 embedding 复用）
    base_url: Optional[str] = None  # dashscope 端点（缺省官方地址）
    timeout: float = 10.0          # dashscope 请求超时（秒）


class HybridSettings(BaseModel):
    enabled: bool
    sparse: str
    fusion: str
    rrf_k: int = 60                # RRF 融合常数 k（TDD §11.3）


class RetrievalSettings(BaseModel):
    top_k: int
    recall_k: int = 20
    fallback_min_score: Optional[float] = None  # 单独校准的 RRF 分数；未校准则安全返回 missing
    cache_ttl: float = 30.0
    rerank: RerankSettings
    hybrid: HybridSettings


class RagSettings(BaseModel):
    enabled: bool
    embedding: EmbeddingSettings
    vector_database: VectorDatabaseSettings
    chunk: ChunkSettings
    retrieval: RetrievalSettings


class KnowledgeSourceSettings(BaseModel):
    path: str
    priority: int


class KnowledgeConflictSettings(BaseModel):
    strategy: str
    order: list[str]


class KnowledgeMissingSettings(BaseModel):
    action: str
    message: str


class KnowledgeSettings(BaseModel):
    sources: dict[str, KnowledgeSourceSettings]
    conflict: KnowledgeConflictSettings
    missing: KnowledgeMissingSettings


class AgentLLMRefSettings(BaseModel):
    model: str                       # 引用 llm.models 的角色名（reasoning / chat）


class BriefingSettings(BaseModel):
    """访前简报节点配置（单 Agent 架构，技术设计文档（新）§7）。"""
    enabled: bool
    name: str
    llm: AgentLLMRefSettings         # config.yaml: reasoning（deepseek-v4）
    max_questions: int = 5           # priority_questions 上限
    max_goals: int = 4               # visit_goals 上限
    timeout_seconds: float = Field(default=120.0, gt=0, le=600)


class RoleplayBehaviorSettings(BaseModel):
    fact_constraint: bool = True     # 仅可使用 Case 事实作答（防幻觉）
    hallucination: bool = False      # 允许幻觉（生产恒为 False）
    reveal_unknown: str = "dynamic"  # unknown_info 释放节奏（dynamic=behavior_rules 驱动）


class RoleplayMemorySettings(BaseModel):
    enabled: bool = True
    window: int = 20                 # 记忆窗口：最近 20 轮（TDD §10.4）


class RoleplaySettings(BaseModel):
    enabled: bool
    name: str
    llm: AgentLLMRefSettings         # config.yaml: chat（deepseek-v4-flash）
    type: str
    objective: str
    behavior: RoleplayBehaviorSettings = RoleplayBehaviorSettings()
    memory: RoleplayMemorySettings = RoleplayMemorySettings()
    max_turns: int = 20              # 循环出口：达到最大轮数（TDD §8.1，缺省 20）


class AgentsSettings(BaseModel):
    briefing: BriefingSettings
    roleplay: RoleplaySettings


class GuardrailRulesSettings(BaseModel):
    forbidden: list[str]
    action: str


class GuardrailSettings(BaseModel):
    enabled: bool
    rules: GuardrailRulesSettings


class MySQLSettings(BaseModel):
    host: str
    port: int
    database: str
    user: Optional[str] = None
    password: Optional[str] = None


class RedisSettings(BaseModel):
    cache_enabled: bool = True
    cache_ttl: int = Field(default=60, gt=0)
    host: str
    port: int
    password: Optional[str] = None  # 缺省无密码；可用 APP_DATABASE__REDIS__PASSWORD 覆盖


class DatabaseSettings(BaseModel):
    mysql: MySQLSettings
    redis: RedisSettings


class AuthSettings(BaseModel):
    """JWT secret and explicit demo coach-to-sales username grants."""
    jwt_secret: Optional[SecretStr] = None
    coach_sales: dict[str, list[str]] = Field(default_factory=dict)


class LoggingFileSettings(BaseModel):
    """运行期日志文件（滚动，全级别；修复 uvicorn 默认下 INFO 被丢弃的问题）。"""
    path: str = "logs/agent.log"
    max_bytes: int = 5 * 1024 * 1024     # 单文件 5MB 滚动
    backup_count: int = 3


class LoggingSaveSettings(BaseModel):
    conversation: bool
    agent_trace: bool
    evidence: bool
    workflow_state: bool
    audit_error: bool = True    # 集中错误/异常审计：写 t_agent_error（审计增强）


class LoggingSettings(BaseModel):
    level: str
    file: LoggingFileSettings = LoggingFileSettings()
    save: LoggingSaveSettings


# =====================================================
# Evaluation Pipeline（TDD §13）
# 机会诊断 / 能力评分 / 证据 / 遗漏问题 / 下一步行动 的权重、关键词与行业 checklist
# =====================================================

class EvaluationLevelSettings(BaseModel):
    high: float = 80            # 机会等级：≥high 高潜机会
    observe: float = 60         # ≥observe 观察；<observe 暂缓


class EvaluationDimSettings(BaseModel):
    weight: int = Field(ge=0, le=100, strict=True)


class EvaluationEvidenceSettings(BaseModel):
    required: bool = True
    source: str = "conversation"
    # 各评分维度关键词 → 从对话按 turn 抽原话（纯规则证据抽取 §13.4）
    topics: dict[str, list[str]] = Field(default_factory=dict)


class ChecklistItemSettings(BaseModel):
    item: str                   # 待核查要素（如 付款主体）
    keywords: list[str]         # 命中即视为已问
    impact: str                 # 遗漏的影响
    suggestion: str             # 建议


class EvaluationMissingQuestionSettings(BaseModel):
    enabled: bool = True
    source: str = "industry_checklist"
    # 行业 → Checklist 要素列表（基建/医药/电商）
    industry_checklist: dict[str, list[ChecklistItemSettings]] = Field(default_factory=dict)


class EvaluationNextActionSettings(BaseModel):
    enabled: bool = True


class EvaluationOpportunitySettings(BaseModel):
    enabled: bool = True
    dimensions: dict[str, EvaluationDimSettings] = Field(default_factory=dict)


class EvaluationSalesScoreSettings(BaseModel):
    rubric: dict[str, EvaluationDimSettings] = Field(default_factory=dict)


class EvaluationModulesSettings(BaseModel):
    opportunity_diagnosis: EvaluationOpportunitySettings = Field(default_factory=EvaluationOpportunitySettings)
    sales_score: EvaluationSalesScoreSettings = Field(default_factory=EvaluationSalesScoreSettings)
    evidence: EvaluationEvidenceSettings = Field(default_factory=EvaluationEvidenceSettings)
    missing_question: EvaluationMissingQuestionSettings = Field(default_factory=EvaluationMissingQuestionSettings)
    next_action: EvaluationNextActionSettings = Field(default_factory=EvaluationNextActionSettings)


class EvaluationSettings(BaseModel):
    enabled: bool = True
    warn_penalty: int = Field(default=5, ge=0, le=100)
    llm: AgentLLMRefSettings
    levels: EvaluationLevelSettings = Field(default_factory=EvaluationLevelSettings)
    modules: EvaluationModulesSettings = Field(default_factory=EvaluationModulesSettings)


# =====================================================
# Settings 主类
# =====================================================

class Settings(BaseSettings):
    """应用配置。默认值来自 config.yaml，可被环境变量覆盖。"""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_nested_delimiter="__",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        yaml_file=CONFIG_FILE,
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    application: ApplicationSettings
    server: ServerSettings
    llm: LLMSettings
    langgraph: LangGraphSettings
    rag: RagSettings
    knowledge: KnowledgeSettings
    agents: AgentsSettings
    evaluation: EvaluationSettings
    guardrail: GuardrailSettings
    database: DatabaseSettings
    auth: AuthSettings = Field(default_factory=AuthSettings)
    logging: LoggingSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # 优先级：init > 环境变量 > .env > config.yaml
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlPlaceholderSource(settings_cls, placeholder_values=dotenv_settings.env_vars),
            file_secret_settings,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取配置单例；文件值只在配置源内解析，不注入 os.environ。"""
    return Settings()


def load_raw_config() -> dict[str, Any]:
    """加载 config.yaml 原始字典（含未建模的业务段：agents/case/knowledge 等）。"""
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
