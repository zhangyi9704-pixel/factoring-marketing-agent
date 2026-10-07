# factoring 本地六服务编排

用途：在 Docker Desktop 的 Linux 引擎上准备中间件。不依赖 Python，不运行业务建表、知识入库或应用图。所有以下命令使用 PowerShell，工作目录为 `D:\learning\factoring-marketing-agent`。

## 文件作用

| 文件 | 作用 |
| --- | --- |
| `compose.yaml` | 固定六个服务的版本、健康检查、依赖、回环端口及命名卷 |
| 根目录 `.env.example` | 0.2 的八个中间件键及 0.4 补充的应用键，共 42 键；空中间件口令故意不能通过 Compose 验证 |
| 根目录 `.env` | 本机开发凭据；本次由系统密码学随机数生成器生成五个独立的 32 字节随机口令，以十六进制保存；不提交、不展示 |
| 根目录 `.gitignore` | 排除真实凭据、虚拟环境、日志和缓存 |
| `verify.ps1` | 认证/模块/AOF/端口/卷检查，以及独立测试数据写入和读回；不输出凭据 |

已有 `.env` 不覆盖。修改数据库初始化口令不能自动修改已存在卷里的账号；以后调整凭据需单独执行账号变更流程，不清卷重建。

## 固定版本与依据

| 服务 | 镜像标签 | 选择依据 |
| --- | --- | --- |
| Milvus | `milvusdb/milvus:v2.5.27` | 保持方案的 2.5.x；使用该版本官方发布附件的镜像组合 |
| etcd | `quay.io/coreos/etcd:v3.5.18` | 同一 Milvus 发布附件指定版本 |
| MinIO | `minio/minio:RELEASE.2024-05-28T17-19-04Z` | 同一 Milvus 发布附件指定版本 |
| MySQL | `mysql:8.0.45` | 保持方案的 8.0 系列，官方具体补丁版；默认 caching_sha2_password 认证 |
| Redis | `redis:7.2.16-alpine` | 保持方案的 7.2 系列，官方镜像具体补丁标签；实际拉取成功 |
| Redis Stack | `redis/redis-stack-server:7.4.0-v8` | 服务端镜像，不包含 RedisInsight 前端；官方发布包含 Redis 7.4.7、RediSearch 2.10.20、RedisJSON 2.8.9，本次已实测模块版本 |

来源（核对日期 2026-10-05）：

- [Milvus v2.5.27 官方 Compose 发布附件](https://github.com/milvus-io/milvus/releases/download/v2.5.27/milvus-standalone-docker-compose.yml)。仓库相同 tag 下的示例 YAML 仍写 v2.5.26，故以实际发布附件为依据，未照搬该滞后字段。
- [MySQL 8.0.45 官方发布说明](https://dev.mysql.com/doc/relnotes/mysql/8.0/en/news-8-0-45.html)。镜像成功拉取，运行版本另由容器查询。
- [Redis 官方镜像标签清单](https://github.com/docker-library/official-images/blob/master/library/redis)。网页清单会继续变化，本文件与 Compose 保留本次选定的具体标签。
- [Redis Stack 7.4.0-v8 官方发布说明](https://github.com/redis-stack/redis-stack/releases/tag/v7.4.0-v8)。
- [RedisSaver 官方安装要求](https://redis-developer.github.io/langgraph-redis/main/user_guide/installation.html)：需要 RedisJSON 和 RediSearch，Redis 8 以前可使用 Redis Stack。模块能力检查与 JSON/搜索实际操作在 0.2 验证；`langgraph-checkpoint-redis`、`redisvl`、`redis-py`、`langgraph-checkpoint` 的兼容版本组合在 M1 锁定并做 RedisSaver setup/写读，当前不把服务器探测写成客户端集成通过。
- [Milvus REST v2.5 Create](https://milvus.io/api-reference/restful/v2.5.x/v2/Collection%20(v2)/Create.md)、[Flush](https://milvus.io/api-reference/restful/v2.5.x/v2/Collection%20(v2)/Flush.md)、[Query](https://milvus.io/api-reference/restful/v2.5.x/v2/Vector%20(v2)/Query.md)：仅用于本步不依赖 Python 的隔离向量持久化探测。

适配说明：不使用固定 container_name 和教程网络名；MinIO 使用当前 MINIO_ROOT_USER/PASSWORD，并把相同值接入 Milvus 的 MINIO_ACCESS_KEY_ID/SECRET_ACCESS_KEY。MinIO 命名卷绑定镜像声明的 `/data`，避免运行容器额外挂载匿名卷。Milvus 保留官方的 `seccomp:unconfined`，不使用 privileged。未改变全局镜像源、Python 或 pip 设置；cryptography 随 M1 应用依赖保留。

## 启动与只读验证

```powershell
Set-Location -LiteralPath 'D:\learning\factoring-marketing-agent'
docker compose --env-file .env -f deploy/docker/compose.yaml config --quiet
docker compose --env-file .env -f deploy/docker/compose.yaml up -d
docker compose --env-file .env -f deploy/docker/compose.yaml ps
& .\deploy\docker\verify.ps1 -Mode Health -WaitSeconds 180
git check-ignore .env
```

成功输出：`config --quiet` 无输出且退出 0；`ps` 六行 healthy；脚本所有检查显示 PASS；`git check-ignore` 返回 `.env`。脚本只重复只读健康检查，不自动重试写入。

暴露端口仅为 `127.0.0.1:3307`、`6379`、`6380`、`19530`、`9091`；etcd/MinIO 没有宿主机发布端口。`docker ps` 中的 `2379-2380/tcp`、`9000/tcp` 是镜像声明的内部端口，没有 `主机地址->端口` 映射。Milvus 等待 etcd/MinIO healthy 后启动。

项目网络：`factoring_default`。命名卷：`factoring_etcd_data`、`factoring_minio_data`、`factoring_milvus_data`、`factoring_mysql_data`、`factoring_redis_data`、`factoring_redis_stack_data`。Docker Desktop 数据盘在 `D:\DockerDesktopWSL\disk\docker_data.vhdx`，命名卷存储由引擎管理，不是项目根目录中的 Windows 文件夹。

## 隔离的重启持久化验收

本次 RunId 为 `a0261005b001`。已写入的测试位置：MySQL 独立库 `factoring_envcheck_a0261005b001.persistence_probe`；etcd `/factoring/envcheck/a0261005b001`；两种 Redis 的 `envcheck:a0261005b001` 与 Stack 索引 `envcheck_idx_a0261005b001`；MinIO 桶 `factoring-envcheck-a0261005b001/probe.txt`；Milvus 集合 `envcheck_a0261005b001`。这些数据保留作复核，没有触及业务表或 `factoring_knowledge`。

只读复核已有测试数据（Milvus load 仅恢复查询加载状态）：

```powershell
Set-Location -LiteralPath 'D:\learning\factoring-marketing-agent'
& .\deploy\docker\verify.ps1 -Mode Verify -RunId a0261005b001 -WaitSeconds 180
```

以后如需再次实施完整重启测试，使用新 RunId，确认只有本项目容器会被重启。以下代码写入独立测试数据并重启六服务；不要作为日常健康检查反复执行：

```powershell
Set-Location -LiteralPath 'D:\learning\factoring-marketing-agent'
$probeRunId = [guid]::NewGuid().ToString('N').Substring(0, 12)
& .\deploy\docker\verify.ps1 -Mode Seed -RunId $probeRunId -WaitSeconds 180
if (-not $?) { throw 'Seed failed; diagnose the failing operation before proceeding.' }
docker compose --env-file .env -f deploy/docker/compose.yaml restart --timeout 30
if ($LASTEXITCODE -ne 0) { throw 'Restart failed.' }
& .\deploy\docker\verify.ps1 -Mode Verify -RunId $probeRunId -WaitSeconds 180
```

成功输出：六个服务重启后重新 healthy；MySQL、etcd、Redis、Stack JSON/搜索索引、MinIO 对象、Milvus 向量标记全部 PASS。部分写入失败时只修失败操作，不能对同一 RunId 重跑整个 Seed。测试空间不会自动删除；本脚本不执行 down -v、清卷或任何业务初始化脚本。
