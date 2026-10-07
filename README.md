# 保理营销教练 Agent

面向保理销售培训，提供访前案例简报、客户情景演练、证据化复盘、主管看板、知识检索和人工审核。案例与知识为模拟或脱敏教学资产，不代替融资审批。

项目使用 FastAPI、单 Agent / LangGraph `coach_graph`，前端由 FastAPI 托管。MySQL 保存业务数据，Redis 提供缓存，Redis Stack 保存演练 checkpoint，Milvus 提供知识检索；DeepSeek 和阿里云接口提供模型、嵌入及精排能力。

## 环境要求

- Windows + Windows PowerShell 5.1 或 PowerShell 7，Python 3.11.17，使用 uv 创建项目虚拟环境。
- Docker Desktop 已启动，使用 Linux 容器引擎 / WSL2；Docker Compose 支持 `up --wait`。
- 可用的 DeepSeek、阿里云 API Key，以及对应模型调用权限。安装、知识入库和演练需要网络，云接口调用会产生费用。
- 前端无需单独启动或安装 Node.js。

以下命令均在项目根目录执行，将路径替换为实际克隆位置：

```powershell
Set-Location -LiteralPath 'D:\learning\factoring-marketing-agent'
```

## 首次安装与配置

先安装 uv、Docker Desktop 并启动 Docker，再执行：

```powershell
docker version
docker compose version
uv --version
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    uv python install 3.11.17
    if ($LASTEXITCODE -ne 0) { throw 'Python 安装失败' }
    uv venv --python 3.11.17 .venv
    if ($LASTEXITCODE -ne 0) { throw '虚拟环境创建失败' }
}
uv pip install --python .\.venv\Scripts\python.exe -r requirements.lock.txt
if ($LASTEXITCODE -ne 0) { throw '依赖安装失败' }
uv pip check --python .\.venv\Scripts\python.exe
if ($LASTEXITCODE -ne 0) { throw '依赖检查失败' }
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
```

使用编辑器在本机填写 `.env`。模板中的 `${...}` 用于引用已填配置值，可保留。

| 配置项 | 填写内容 |
| --- | --- |
| `MYSQL_ROOT_PASSWORD`、`MYSQL_PASSWORD` | MySQL 根账号及应用账号口令 |
| `REDIS_PASSWORD`、`CHECKPOINT_REDIS_PASSWORD` | 两个 Redis 服务各自的口令 |
| `MINIO_ROOT_PASSWORD` | MinIO 口令，至少 8 字符 |
| `DEEPSEEK_API_KEY` | DeepSeek API Key |
| `ALIYUN_API_KEY` | 阿里云嵌入与精排 API Key |
| `APP_AUTH__JWT_SECRET` | 至少 32 字符的随机密钥，用于登录鉴权 |

本地部署保留模板中的数据库名称、账号名称、回环地址和端口。模型名称、接口地址及嵌入维度见 `.env.example` 和 `config.yaml`，确认与你的云账号权限一致；嵌入维度需与 Milvus 集合一致。

`.env` 不上传 Git。已有 `.env` 不覆盖；已有 Docker 数据卷的账号口令不会因编辑 `.env` 自动修改。

检查配置，每条命令成功后再继续：

```powershell
docker compose --env-file .env -f deploy/docker/compose.yaml config --quiet
& .\scripts\start_local.ps1 -CheckOnly
```

## 首次初始化

只在新环境或需要明确升级时执行。每条命令退出码为 0 后再继续，失败时先处理对应问题。

先启动六个中间件服务、建表及增量迁移：

```powershell
docker compose --env-file .env -f deploy/docker/compose.yaml up -d --wait --wait-timeout 180
docker compose --env-file .env -f deploy/docker/compose.yaml ps
& .\deploy\docker\verify.ps1 -Mode Health -WaitSeconds 180
.\.venv\Scripts\python.exe -X utf8 scripts\init_tables.py
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_knowledge_metadata.py
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_compliance.py
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_evaluation.py
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_auth.py
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_pending.py
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_message_requests.py
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_session_reliability.py
.\.venv\Scripts\python.exe -X utf8 scripts\create_milvus_collection.py
```

再导入知识、案例，创建演示账号：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\ingest_knowledge.py --source all
.\.venv\Scripts\python.exe -X utf8 scripts\init_cases.py
.\.venv\Scripts\python.exe -B -X utf8 scripts\init_demo_users.py
```

成功标准：六服务健康，表及迁移完成，向量集合已创建，正式知识 35 篇入库 `failed=0`，3 个案例同步成功，新库演示账号 `created=3`。知识入库会调用云嵌入；独立合规规则 YAML 无须入向量库。日常启动不重复初始化，不使用建表脚本的 `--drop` 或删除数据卷处理一般启动问题。`scripts/init_mysql.sql` 仅为历史结构参考，不等同于当前 ORM，不用于初始化。

已有环境升级本次可靠性修复时，先备份已有数据库，再显式执行以下命令；不需要重新入库知识或重建数据卷：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_knowledge_metadata.py
if ($LASTEXITCODE -ne 0) { throw '知识元数据迁移失败，停止升级' }
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_pending.py
if ($LASTEXITCODE -ne 0) { throw '审核结构迁移失败，停止升级' }
.\.venv\Scripts\python.exe -X utf8 scripts\migrate_session_reliability.py
if ($LASTEXITCODE -ne 0) { throw '会话创建账本迁移失败，停止升级' }
```

升级增加 `t_session_create_request`、`t_knowledge_ingestion_operation`，为 `t_knowledge_doc` 增加正文快照、正文哈希、切片清单与入库操作编号，为 `t_knowledge_draft` 增加发布前快照。迁移只补缺失结构，不删除历史记录；MySQL DDL 可能持有元数据锁，不能靠事务回滚全部 DDL。旧环境若尚未执行早期迁移，仍需先补齐上面的完整首次迁移列表。历史条目不会因此重新生成嵌入；已有发布失败但缺少发布前快照的草稿可能仍需人工核对。

当前检索使用 `config.yaml` 中的阈值。更换模型或知识后可另行进行阈值校准及人工复核，校准不属于基本启动步骤。

## 启动与登录

完成首次初始化后，日常只需运行：

```powershell
& .\scripts\start_local.ps1
```

脚本检查配置，启动并等待中间件健康，再以前台进程启动应用。应用启动只读检查运行表与字段，缺失时拒绝启动并提示显式迁移，不再自动执行审核 DDL。出现 `Application startup complete` 后访问：

- [系统首页](http://127.0.0.1:8000)
- [API 说明](http://127.0.0.1:8000/docs)
- [健康检查](http://127.0.0.1:8000/api/v1/health)，同时查看 `components` 状态。

健康检查默认不调用云模型，模型项显示 configured/unverified 不表示真实服务验证通过。`GET /api/v1/health?llm=1` 必须携带有效管理员 Bearer token，会调用 DeepSeek 并可能产生费用；每个应用进程最多每 30 秒发起一次且只允许一个云探测调用在途。超时后底层调用尚未结束时仍占用名额，后续响应的 LLM 项显示 CloudProbeBusy，不再发起云请求。多进程部署时限制按进程计算。嵌入与精排仍需单独实测。

| 账号 | 默认密码 | 角色 |
| --- | --- | --- |
| `sales_demo` | `12345` | 销售：简报、演练、个人复盘 |
| `coach_demo` | `12345` | 主管：授权销售的看板与审核待办，默认关联 `sales_demo` |
| `admin_demo` | `12345` | 管理员：全部审核范围及知识导入、确认、发布 |

默认密码用于本地演示。普通账号初始化保留已有密码；如需重置三个演示账号密码，执行：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts\init_demo_users.py --reset-passwords
```

进程环境变量 `APP_DEMO_SALES_PASSWORD`、`APP_DEMO_COACH_PASSWORD`、`APP_DEMO_ADMIN_PASSWORD` 可覆盖默认密码，命令参数优先级最高。重置保留账号身份、角色和启用状态。

使用流程：销售登录 → 选择案例 → 生成访前简报 → 与模拟客户演练 → 查看复盘；主管查看授权销售表现并处理待办；管理员确认并发布知识。单 Agent `coach_graph` 保持原流程。

创建会话接口 `POST /api/v1/sessions` 现在要求 `case_id` 和 `request_id`（建议 UUID）。前端自动保存创建编号；同一次创建重试必须复用编号，成功后开启新演练才换编号。重复请求回放持久结果，不再重复生成简报或创建会话；编号被不同案例复用会返回 409。无效案例在创建记录与调用模型前拒绝，初始化失败的记录保留为 failed，不当作可演练会话。

演练断网后先使用“重连核对本轮”，沿用原编号。存在 failed/running 请求时可点击“核对已保存结果”；只有图快照、SQL 对话及结果账本一致才恢复，不能确认时不会重跑图。本人或管理员还可通过 `POST /api/v1/sessions/{sid}/reconcile`，提交 `action`（`reconcile` / `abort`）、原 `request_id`（核对时必填）及非空 `reason`。无法恢复时可“终止本次演练（保留记录）”后开新会话；终止不伪造复盘或删除旧数据。

简报需要检索、模型生成和合规检查，未给出真实云耗时承诺。公开检索缓存按进程复用，知识修订与数据库目录哈希变化时失效；审核待办仍按用户和会话登记。`agents.briefing.timeout_seconds` 默认 120 秒，分别约束简报与开场阶段的协作式预算；各 HTTP 尝试和重试使用剩余预算。这不能强制停止已提交的云计算，也不保证整个创建请求严格在 120 秒内完成。预算耗尽或合规语义扫描不可用时采用安全降级并标记审核，不发送未经完整验证的模型建议。

`llm.models.*.thinking.enabled` 现通过 `extra_body.thinking` 传递，启用思考时也传递 `reasoning_effort`；思考模式下 temperature 不生效，详见 [DeepSeek 官方说明](https://api-docs.deepseek.com/guides/thinking_mode/)。模型名及账号权限仍需真实服务确认，离线参数测试不能代替云验证。

管理员可在审核台为处理中待办填写理由，将其回收为待领取或转派给启用且有该销售授权的主管/管理员。请求携带原处理人编号，状态发生变化会拒绝覆盖。知识草稿的 failed / reconcile_required / publishing 状态提供“核对并恢复”：核对数据库、正文哈希及发布前向量快照，必要时写回快照、草稿文件和审计记录；该操作不调用嵌入。恢复到 confirmed 后仍需另行点击发布。没有可信快照、内容发生变化或数据库无法确认时保持阻止，不自动重试发布。实际发布仍可能调用云嵌入并写入 MySQL / Milvus。

## 停止与常见问题

在启动应用的终端按 `Ctrl+C` 停止应用，随后可停止中间件：

```powershell
docker compose --env-file .env -f deploy/docker/compose.yaml stop
```

这会保留数据卷；再次运行启动脚本即可启动已有数据环境。

| 问题 | 处理方式 |
| --- | --- |
| 找不到 `.venv` 或 `.env` | 执行首次安装，填写本机配置 |
| Docker 无法连接或容器不健康 | 启动 Docker Desktop，确认 Linux 引擎，执行上面的 Compose `ps` 命令查看服务状态 |
| JWT 检查失败 | 确认 `APP_AUTH__JWT_SECRET` 至少 32 字符 |
| `runtime_schema_missing` | 查看列出的缺表/字段，按首次初始化或已有环境升级步骤显式迁移；启动不会自动建表 |
| 8000 端口被占用 | 使用 `& .\scripts\start_local.ps1 -Port 8016`，访问 `http://127.0.0.1:8016` |
| 登录失败 | 确认账号已初始化、密码未被环境变量覆盖、账号处于启用状态 |
| 没有案例或检索不到知识 | 检查首次导入结果、云 Key、模型权限及数据库 / 向量集合配置 |
| Windows PowerShell 报 `File "<string>", line 1` | 使用仓库当前启动脚本；前置检查已改为执行 `scripts.check_startup` 模块 |
| 健康检查 `llm=1` 返回 401 / 403 / 429 | 使用启用管理员身份；429 按 Retry-After 等待，默认健康检查不需要云探测 |

默认本地端口：应用 8000、MySQL 3307、Redis 6379、Redis Stack 6380、Milvus 19530 / 9091。Compose 项目名为 `factoring`，数据库为 `factoring_coach`，向量集合为 `factoring_knowledge_v2`。

## 仓库内容与验证

| 路径 | 作用 |
| --- | --- |
| `app/` | API、鉴权、演练图、检索、合规、评分及持久化 |
| `frontend/` | 由 FastAPI 托管的静态前端 |
| `cases/`、`data/` | 案例、正式知识、独立规则及评测数据 |
| `scripts/` | 初始化、迁移、启动及验证工具 |
| `deploy/` | Docker Compose 及配置检查 |
| `tests/` | 自动化测试 |
| `config.yaml`、`.env.example` | 配置默认值及本机环境变量模板 |
| `requirements.txt`、`requirements.lock.txt`、`.python-version` | 依赖清单、锁定依赖及 Python 版本 |

`docs/`、`reports/`、`archive/` 为本地文档、验收报告和归档，不上传 Gitee。`.env`、虚拟环境、日志、缓存、备份和本地知识反馈也由 `.gitignore` 排除。根目录 `README.md` 保留上述独立启动说明。

数据库中的账号、历史演练和已发布知识，以及 Docker 数据卷不随源码上传；新机器需要自行配置并初始化。部分历史验收、归档完整性和文档验证脚本依赖未上传的目录，不属于日常启动流程。

离线回归测试在项目根目录运行，不需要先启动应用：

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -B -m pytest -q -m "not integration" -o cache_dir=.pytest_cache --basetemp .pytest_tmp_readme --tb=short
```

测试会隔离本机凭据，并禁止单元测试真实网络访问。前端身份、会话、审核和 SSE 测试可选安装 Node.js 后运行 `node --test tests/auth_client.test.cjs tests/session_client.test.cjs tests/review_client.test.cjs tests/sse_client.test.cjs`；日常运行无需 Node.js。

克隆后应确认仓库包含上表中的运行资产；`.gitignore` 不等于文件已上传。离线回归涵盖部分权限、并发替身与失败补偿，真实 MySQL / Redis Stack / Milvus、云模型质量、费用、时延及完整业务验收仍需单独验证。严重合规规则的销售输入双命中策略保留现状，业务规则调整需另行确认。
