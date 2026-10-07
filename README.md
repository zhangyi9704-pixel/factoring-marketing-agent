# 保理营销教练 Agent

面向保理销售培训的专业 AI 教练系统，提供访前案例简报生成、客户情景演练、证据化复盘评估、主管审核看板、知识检索及人工审核等核心功能。案例与知识为模拟或脱敏教学资产，不代替融资审批。

## 项目概述

本系统采用 FastAPI 构建后端服务，使用 LangGraph `coach_graph` 实现单 Agent 架构，前端由 FastAPI 托管。系统集成 MySQL 存储业务数据、Redis 提供缓存、Redis Stack 保存演练 checkpoint、Milvus 提供知识向量检索能力，DeepSeek 和阿里云 API 提供大模型、嵌入及精排能力。

### 核心功能

- **案例管理**：销售视角的公开案例浏览，支持行业筛选，绝不泄露内部敏感信息
- **访前简报**：基于案例事实与知识检索自动生成准备材料，含客户背景、关键问题、异议处理建议
- **情景演练**：与模拟客户进行多轮对话，支持流式输出与断点恢复
- **合规审查**：实时检测销售话术中的违规风险，支持 L1 规则匹配与 L2 语义分析
- **复盘评估**：会话结束后自动生成评估报告，包括机会诊断、能力评分、遗漏问题与改进建议
- **知识管理**：RAG 知识库，支持文档入库、版本管理、冲突检测与人工审核发布
- **主管看板**：查看授权销售的会话历史、评估得分分布与待审核待办

## 技术架构

| 组件 | 用途 |
|------|------|
| FastAPI | Web 框架 |
| LangGraph | Agent 编排 |
| MySQL | 业务持久化 |
| Redis | 缓存与分布式锁 |
| Redis Stack | 演练状态 Checkpoint |
| Milvus | 向量检索 |
| DeepSeek | 主模型与推理 |
| 阿里云 | 嵌入与精排 |

## 快速开始

### 环境要求

- Windows + PowerShell 7 或 Windows PowerShell 5.1，Python 3.11.17
- Docker Desktop 已启动，使用 Linux 容器引擎
- DeepSeek、阿里云 API Key

### 安装步骤

```powershell
# 克隆项目
git clone https://gitee.com/ZanYiZhang/factoring-marketing-agent.git
cd factoring-marketing-agent

# 创建虚拟环境并安装依赖
uv python install 3.11.17
uv venv --python 3.11.17 .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.lock.txt

# 复制环境配置
Copy-Item -LiteralPath '.env.example' -Destination '.env'
```

编辑 `.env` 填写必要配置：

| 配置项 | 说明 |
|--------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `ALIYUN_API_KEY` | 阿里云嵌入与精排 API Key |
| `MYSQL_ROOT_PASSWORD` | MySQL 根口令 |
| `REDIS_PASSWORD` | Redis 口令 |
| `CHECKPOINT_REDIS_PASSWORD` | Redis Stack 口令 |
| `MINIO_ROOT_PASSWORD` | MinIO 口令 |
| `APP_AUTH__JWT_SECRET` | JWT 密钥，至少 32 字符 |

### 初始化

```powershell
# 启动中间件服务
docker compose --env-file .env -f deploy/docker/compose.yaml up -d --wait --wait-timeout 180

# 执行数据库迁移
.\.venv\Scripts\python.exe -X utf8 scripts\init_tables.py
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_knowledge_metadata.py
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_compliance.py
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_evaluation.py
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_auth.py
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_pending.py
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_message_requests.py
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_session_reliability.py
.\.venv\Scripts\python.exe -X utf8 scripts\create_milvus_collection.py

# 导入知识与案例
.\.venv\Scripts\python.exe -X utf8 scripts\ingest_knowledge.py --source all
.\.venv\Scripts\python.exe -X utf8 scripts\init_cases.py
.\.venv\Scripts\python.exe -B -X utf8 scripts\init_demo_users.py
```

### 启动服务

```powershell
& .\scripts\start_local.ps1
```

服务启动后访问：
- 系统首页：`http://127.0.0.1:8000`
- API 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/api/v1/health`

### 演示账号

| 账号 | 密码 | 角色 |
|------|------|------|
| `sales_demo` | 12345 | 销售：简报、演练、个人复盘 |
| `coach_demo` | 12345 | 主管：授权销售看板与审核待办 |
| `admin_demo` | 12345 | 管理员：全部权限及知识发布 |

## 项目结构

```
factoring-marketing-agent/
├── app/                      # 应用核心代码
│   ├── agents/               # Agent 实现
│   │   ├── coach/            # 简报生成节点
│   │   └── roleplay/         # 角色扮演 Agent
│   ├── api/                  # API 路由
│   ├── compliance/           # 合规引擎
│   ├── db/                   # 数据库连接与迁移
│   ├── evaluation/           # 复盘评估模块
│   ├── graphs/               # LangGraph 图定义
│   ├── guardrail/            # 风险检测服务
│   ├── models/               # SQLAlchemy 模型
│   ├── rag/                  # RAG 组件（检索、嵌入、精排）
│   ├── schemas/              # Pydantic 模式
│   └── services/             # 业务服务层
├── cases/                    # 案例定义（YAML）
├── data/                     # 知识库文档
├── deploy/                   # Docker 部署配置
├── frontend/                 # 前端页面
├── scripts/                  # 初始化与迁移脚本
├── tests/                    # 单元与集成测试
├── config.yaml               # 应用配置
└── requirements.lock.txt     # 锁定依赖
```

## 核心模块说明

### Agent 系统

- **简报生成** (`app/agents/coach/briefing.py`)：根据案例事实与检索到的知识生成访前简报，支持安全降级
- **角色扮演** (`app/agents/roleplay/agent.py`)：管理演练会话生命周期，支持 SSE 流式回复与 Checkpoint 恢复

### 合规审查

- **规则检测** (`app/guardrail/detector.py`)：L1 正则匹配 + L2 语义分析，双命中触发阻断
- **内容改写** (`app/guardrail/rewriter.py`)：LLM 生成合规替代表述

### RAG 知识检索

- **混合检索** (`app/rag/retriever.py`)：Dense + Sparse 向量融合，RRF 排序
- **知识入库** (`app/rag/ingest.py`)：文档加载 → 切片 → 嵌入 → 入库，支持版本管理

### 评估复盘

- **证据提取** (`app/evaluation/evidence.py`)：从对话中提取各维度证据
- **能力评分** (`app/evaluation/sales_score.py`)：LLM 评分 + 证据加权
- **机会诊断** (`app/evaluation/opportunity.py`)：评估销售机会把握能力

## 测试

```powershell
# 离线单元测试
.\.venv\Scripts\python.exe -B -m pytest -q -m "not integration" --basetemp local_artifacts/test-runs

# 集成测试（需要中间件运行）
.\.venv\Scripts\python.exe -B -m pytest -q -m "integration"
```

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| 端口 8000 被占用 | `.\scripts\start_local.ps1 -Port 8016` |
| JWT 验证失败 | 检查 `APP_AUTH__JWT_SECRET` 至少 32 字符 |
| 登录失败 | 确认账号已初始化，检查密码与环境变量 |
| 健康检查 LLM 失败 | 使用管理员 Bearer Token 调用 `?llm=1` |

## 许可证

请参阅项目根目录 LICENSE 文件。