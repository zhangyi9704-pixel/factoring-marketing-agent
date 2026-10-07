-- =====================================================
-- 历史结构参考（TDD §12.1），不适用于当前项目初始化。
-- 当前初始化请使用 scripts/init_tables.py 及根目录 README.md 所列迁移。
-- 目标库：sales_coach（utf8mb4）
-- 幂等：CREATE TABLE IF NOT EXISTS，不删除已有数据
-- 以下 SQL 保留供历史结构对照，请勿在当前环境执行。
-- =====================================================

CREATE DATABASE IF NOT EXISTS sales_coach
  DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE sales_coach;

-- 用户
CREATE TABLE IF NOT EXISTS t_user (
  id            BIGINT PRIMARY KEY AUTO_INCREMENT,
  username      VARCHAR(64)  NOT NULL UNIQUE,
  password_hash VARCHAR(128) NOT NULL,
  role          VARCHAR(16)  NOT NULL COMMENT 'sales / manager / admin',
  team          VARCHAR(64),
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 案例库（Case State）
CREATE TABLE IF NOT EXISTS t_case (
  id          BIGINT PRIMARY KEY AUTO_INCREMENT,
  case_id     VARCHAR(32)  NOT NULL UNIQUE COMMENT '如 INFRA_001',
  industry    VARCHAR(16)  NOT NULL COMMENT '基建/医药/电商',
  company     VARCHAR(128),
  client_role VARCHAR(64),
  case_json   JSON NOT NULL COMMENT '完整 Case State（transaction_chain/receivable/behavior_rules…）',
  enabled     TINYINT(1) NOT NULL DEFAULT 1,
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 演练会话
CREATE TABLE IF NOT EXISTS t_session (
  id          BIGINT PRIMARY KEY AUTO_INCREMENT,
  session_id  VARCHAR(64) NOT NULL UNIQUE COMMENT '= LangGraph thread_id',
  user_id     BIGINT NOT NULL,
  case_id     VARCHAR(32) NOT NULL,
  phase       VARCHAR(16) COMMENT 'prepare/roleplay/evaluate/report',
  status      VARCHAR(16) NOT NULL DEFAULT 'active' COMMENT 'active/finished/interrupted',
  turn_count  INT NOT NULL DEFAULT 0,
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  finished_at DATETIME,
  INDEX idx_user (user_id),
  INDEX idx_case (case_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 对话记录（逐轮落库）
CREATE TABLE IF NOT EXISTS t_conversation_message (
  id            BIGINT PRIMARY KEY AUTO_INCREMENT,
  session_id    VARCHAR(64) NOT NULL,
  turn          INT NOT NULL,
  role          VARCHAR(16) NOT NULL COMMENT 'sales / customer',
  content       TEXT NOT NULL,
  revealed_info JSON COMMENT '本轮释放的 unknown_info（内部标记）',
  guardrail_hit JSON COMMENT '本轮触发的合规规则',
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_session (session_id, turn)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 访前报告
CREATE TABLE IF NOT EXISTS t_visit_report (
  id          BIGINT PRIMARY KEY AUTO_INCREMENT,
  session_id  VARCHAR(64) NOT NULL UNIQUE,
  report_json JSON NOT NULL,
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 复盘评估结果
CREATE TABLE IF NOT EXISTS t_evaluation (
  id           BIGINT PRIMARY KEY AUTO_INCREMENT,
  session_id   VARCHAR(64) NOT NULL UNIQUE,
  opportunity  JSON COMMENT '机会诊断（分数/等级/理由）',
  sales_score  JSON COMMENT '四维能力评分',
  missing      JSON COMMENT '遗漏问题',
  next_action  JSON COMMENT '下一步行动',
  total_score  DECIMAL(5,2),
  created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_user_score (session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 评分证据（关联原始对话）
CREATE TABLE IF NOT EXISTS t_evaluation_evidence (
  id            BIGINT PRIMARY KEY AUTO_INCREMENT,
  evaluation_id BIGINT NOT NULL,
  dimension     VARCHAR(64) COMMENT '评分维度',
  score         DECIMAL(4,2),
  turn          INT COMMENT '证据轮次',
  quote_sales   TEXT COMMENT '销售原话',
  quote_customer TEXT COMMENT '客户原话',
  comment       VARCHAR(512) COMMENT '评价',
  INDEX idx_eval (evaluation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 合规 / HITL 审核日志
CREATE TABLE IF NOT EXISTS t_compliance_log (
  id            BIGINT PRIMARY KEY AUTO_INCREMENT,
  session_id    VARCHAR(64),
  turn          INT,
  rule          VARCHAR(64) COMMENT '触发规则',
  raw_text      TEXT,
  action        VARCHAR(16) COMMENT 'rewrite / block / human_review',
  review_status VARCHAR(16) NOT NULL DEFAULT 'pending' COMMENT 'pending/approved/rejected',
  reviewer      VARCHAR(64),
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 知识库文档元数据
CREATE TABLE IF NOT EXISTS t_knowledge_doc (
  id         BIGINT PRIMARY KEY AUTO_INCREMENT,
  source     VARCHAR(512) NOT NULL COMMENT '知识来源文献',
  doc_id     VARCHAR(64),
  version    VARCHAR(64),
  effective_date VARCHAR(10),
  industry   VARCHAR(16),
  category   VARCHAR(16),
  topic      VARCHAR(256),
  content_snapshot TEXT,
  content_hash VARCHAR(64),
  chunk_manifest JSON,
  ingestion_operation_id VARCHAR(32),
  doc_name   VARCHAR(256) NOT NULL,
  priority   INT NOT NULL,
  chunk_cnt  INT,
  status     VARCHAR(16) NOT NULL DEFAULT 'indexed',
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uq_knowledge_doc_version (doc_id, version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 精确的知识入库提交记录，与目录版本在同一个事务提交
CREATE TABLE IF NOT EXISTS t_knowledge_ingestion_operation (
  id VARCHAR(32) PRIMARY KEY,
  doc_id VARCHAR(64) NOT NULL,
  version VARCHAR(64) NOT NULL,
  input_hash VARCHAR(64) NOT NULL,
  vector_hash VARCHAR(64) NOT NULL,
  metadata_json JSON NOT NULL,
  chunk_cnt INT NOT NULL,
  committed_at DATETIME NOT NULL,
  INDEX ix_t_knowledge_ingestion_operation_doc_id (doc_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 会话创建请求账本；业务会话身份在模型执行前固定
CREATE TABLE IF NOT EXISTS t_session_create_request (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  user_id BIGINT NOT NULL,
  request_id VARCHAR(64) NOT NULL,
  case_id VARCHAR(32) NOT NULL,
  session_id VARCHAR(64) NOT NULL UNIQUE,
  status VARCHAR(16) NOT NULL,
  result JSON,
  error_code VARCHAR(64),
  created_at DATETIME NOT NULL,
  updated_at DATETIME NOT NULL,
  UNIQUE KEY uq_session_create_request (user_id, request_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 运行期错误/异常审计（审计增强）
CREATE TABLE IF NOT EXISTS t_agent_error (
  id          BIGINT PRIMARY KEY AUTO_INCREMENT,
  level       VARCHAR(16) NOT NULL COMMENT 'ERROR/CRITICAL',
  logger_name VARCHAR(64) COMMENT '来源 logger',
  message     TEXT COMMENT '日志消息',
  exc_type    VARCHAR(128) COMMENT '异常类型',
  traceback   TEXT COMMENT '完整堆栈',
  session_id  VARCHAR(64) COMMENT '关联会话',
  user_id     BIGINT COMMENT '关联用户',
  path        VARCHAR(255) COMMENT '请求路径',
  method      VARCHAR(8) COMMENT 'HTTP 方法',
  created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_error_session (session_id),
  INDEX idx_error_level (level)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
