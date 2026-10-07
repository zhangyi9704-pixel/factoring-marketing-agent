"""M0 dotenv inventory/merge using only the standard library; never print values.

Run from the project root with .venv/Scripts/python.exe. This validates file-level
mapping only; it does not import source applications, run graphs or call services.
"""
from __future__ import annotations

import argparse
import ast
import re
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"D:\BaiduNetdiskDownload\HM\保理营销教练Agent")
PAIR = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
SENSITIVE = re.compile(r"PASSWORD|SECRET|API_KEY|TOKEN", re.I)
MAPPINGS = {
    "APP_DATABASE__MYSQL__DATABASE": "MYSQL_DATABASE",
    "APP_DATABASE__MYSQL__USER": "MYSQL_USER",
    "APP_DATABASE__MYSQL__PASSWORD": "MYSQL_PASSWORD",
    "APP_DATABASE__REDIS__PASSWORD": "REDIS_PASSWORD",
    "APP_LANGGRAPH__CHECKPOINT__PASSWORD": "CHECKPOINT_REDIS_PASSWORD",
    "APP_LLM__API__API_KEY": "DEEPSEEK_API_KEY",
    "APP_RAG__EMBEDDING__API_KEY": "ALIYUN_API_KEY",
    "APP_RAG__RETRIEVAL__RERANK__API_KEY": "ALIYUN_API_KEY",
}


def read_env(path: Path) -> tuple[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig")
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = PAIR.match(line)
        if not match:
            raise ValueError("Unsupported dotenv syntax; values suppressed")
        key, value = match.groups()
        if key in values:
            raise ValueError(f"Duplicate key {key}; values suppressed")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        values[key] = value
    return text, values


def resolve(key: str, values: dict[str, str], seen: tuple[str, ...] = ()) -> str:
    if key in seen:
        raise ValueError(f"Cyclic reference at {key}; values suppressed")
    if key not in values:
        raise ValueError(f"Missing referenced key {key}")
    return REF.sub(lambda m: resolve(m[1], values, seen + (key,)), values[key])


def fields(path: Path) -> dict[str, set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    return {
        node.name: {item.target.id for item in node.body
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
        for node in tree.body if isinstance(node, ast.ClassDef)
    }


def consumer(key: str) -> str:
    if key.startswith("APP_"):
        field = key[4:].lower().replace("__", ".")
        if key == "APP_AUTH__JWT_SECRET":
            return "规划 auth.jwt_secret；旧 Settings 缺 auth，M1-1.3 补模型，M5 实现鉴权"
        if key == "APP_LANGGRAPH__CHECKPOINT__PASSWORD":
            return "规划 langgraph.checkpoint.password；旧字段/独立口令接线缺失，M1-1.3"
        if key.startswith("APP_DATABASE__MYSQL__"):
            target = "app/db/mysql.py::MySQLDatabase"
        elif key.startswith("APP_DATABASE__REDIS__"):
            target = "app/db/redis.py::RedisClient"
        elif key.startswith("APP_RAG__VECTOR_DATABASE__"):
            target = "app/rag/milvus_client.py::VectorStore"
        elif key.startswith("APP_RAG__EMBEDDING__"):
            target = "app/rag/embedder.py::EmbeddingService"
        elif key.startswith("APP_RAG__RETRIEVAL__RERANK__"):
            target = "app/rag/reranker.py::Reranker"
        elif key.startswith("APP_LLM__"):
            target = "app/services/llm_service.py::LLMService"
        elif key.startswith("APP_LANGGRAPH__"):
            target = "app/graphs/coach_graph.py"
        else:
            target = "app/main.py / app/config.py"
        return f"旧 Settings.{field} → {target}；新仓库运行消费待 M1"
    if key == "DEEPSEEK_API_KEY":
        return "旧 config.yaml llm.api.api_key 占位符；APP_LLM__API__API_KEY 同源引用"
    if key == "ALIYUN_API_KEY":
        return "旧 config.yaml embedding/rerank api_key 占位符；两 APP_API_KEY 同源引用"
    if key.startswith("MINIO_"):
        return "compose.yaml minio / milvus 环境注入"
    if key.startswith("MYSQL_"):
        return "compose.yaml mysql 环境注入；应用映射见下表（root 不给应用）"
    if key == "REDIS_PASSWORD":
        return "compose.yaml redis requirepass / 认证健康检查"
    return "compose.yaml redis-stack REDIS_ARGS / 认证健康检查"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true", help="Append missing template keys only")
    parser.add_argument("--report", action="store_true", help="Write the masked Markdown inventory")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise ValueError("Run from project root")
    template_text, template = read_env(ROOT / ".env.example")
    original_text, original = read_env(ROOT / ".env")
    if args.merge:
        lines = ["", "# M0-0.4 application configuration; existing values preserved."]
        for key, value in template.items():
            if key not in original:
                if key == "APP_AUTH__JWT_SECRET":
                    value = secrets.token_hex(32)
                lines.append(f"{key}={value}")
        if len(lines) > 2:
            with (ROOT / ".env").open("a", encoding="utf-8", newline="") as handle:
                if not original_text.endswith("\n"):
                    handle.write("\n")
                handle.write("\n".join(lines) + "\n")
    merged_text, current = read_env(ROOT / ".env")
    if any(current.get(key) != value for key, value in original.items()):
        raise ValueError("Existing values changed; values suppressed")
    if not merged_text.startswith(original_text):
        raise ValueError("Existing text changed; values suppressed")
    missing = set(template) - set(current)
    if missing:
        raise ValueError("Missing keys: " + ", ".join(sorted(missing)))
    for target, source in MAPPINGS.items():
        if resolve(target, current) != resolve(source, current):
            raise ValueError(f"Credential/config mismatch: {target} versus {source}; values suppressed")
    old_text, old = read_env(SOURCE / ".env")
    config_fields = fields(SOURCE / "app/config.py")
    if not {"user", "password"} <= config_fields["MySQLSettings"]:
        raise ValueError("Source MySQL field inventory changed")
    if "password" not in config_fields["RedisSettings"]:
        raise ValueError("Source Redis password field inventory changed")
    checkpoint_missing = "password" not in config_fields["CheckpointSettings"]
    auth_missing = "auth" not in config_fields["Settings"]
    yaml_values = "\n".join(line.split("#", 1)[0] for line in (SOURCE / "config.yaml").read_text(encoding="utf-8-sig").splitlines())
    placeholders = sorted(set(REF.findall(yaml_values)))
    print(f"PASS: {len(original)} existing values preserved; {len(current)} current keys; {len(MAPPINGS)} mappings equal")
    print(f"Source inventory: {len(old)} keys (names/status only); placeholders: {', '.join(placeholders)}")
    print(f"Source fields: MySQL user/password present; Redis password present; checkpoint password missing={checkpoint_missing}; auth missing={auth_missing}")
    pending = [key for key in ("DEEPSEEK_API_KEY", "ALIYUN_API_KEY") if not resolve(key, current)]
    print("Cloud status: " + (", ".join(f"{key}=UNFILLED" for key in pending) or "configured, not externally verified"))
    if not args.report:
        return
    report = ["# 环境变量对照表（0.4）", "", "更新日期：2026-10-05。仅记录键名、默认/目标值及填写状态，不记录真实凭据。", "",
              "消费位置依据只读旧基座的静态分析；当前仓库没有业务 app/config.py。文件级映射通过不等于配置加载或外部调用通过。", "",
              "APP_ 字段目标默认值以当前模板为准；旧 YAML 的 localhost/3306/sales_coach/local/BGE 默认值待 M1-1.3 同步。所有真实调用、Pydantic 加载和占位符消费仍待验证。", "",
              "## 当前键名、消费位置、默认值与填写状态", "",
              "| 键名 | 消费位置 / 状态 | 模板默认值 / 目标 | 本地填写状态 |", "| --- | --- | --- | --- |"]
    for key, default in template.items():
        filled = bool(resolve(key, current))
        if key in MAPPINGS:
            default_text = "引用 " + MAPPINGS[key]
        elif SENSITIVE.search(key) or key == "APP_DATABASE__MYSQL__USER":
            default_text = "空占位；真实值仅在 .env"
        else:
            default_text = default or "空占位"
        status = "已填写（脱敏）" if filled and SENSITIVE.search(key) else ("已填写" if filled else "未填写")
        if REF.search(current[key]):
            status += "，同源引用" if filled else "，引用源未填写"
        report.append(f"| `{key}` | {consumer(key)} | {default_text} | {status} |")
    report += ["", "## 编排与应用的同源映射", "", "| 编排 / 来源键 | 应用键 | 文件级一致性 |", "| --- | --- | --- |"]
    for target, source in MAPPINGS.items():
        report.append(f"| `{source}` | `{target}` | 相等；{'已填写' if resolve(source, current) else '源未填写'} |")
    report += ["", "MYSQL_ROOT_PASSWORD 仅给 MySQL 容器，不作为应用账号口令。MINIO_ROOT_USER/PASSWORD 给 MinIO，Milvus 使用相同值的 MINIO_ACCESS_KEY_ID/SECRET_ACCESS_KEY；应用通过 Milvus 客户端访问，不直接复制 MinIO root 凭据。", "",
               "引用形式为 ${SOURCE_KEY}；源键在文件内先于应用键。Compose dotenv 和旧 python-dotenv 加载器支持引用，但后者未在本仓库运行，M1 需验证。外部进程环境覆盖和未来口令轮换也需要重新核对；当前校验仅使用本地文件。", "",
               "## 旧 .env 键名清单与迁移建议", "",
               "| 旧键名 | 旧文件状态（仅非空检查） | 处理 |", "| --- | --- | --- |"]
    for key in old:
        if key in ("DEEPSEEK_API_KEY", "ALIYUN_API_KEY"):
            action = "用户在本地手工迁移至同名键；不复制到对话，未自动复制，有效性待真实验证"
        elif SENSITIVE.search(key):
            action = "不迁移旧远端账号/口令；应用引用本机 Compose 凭据"
        elif key in ("APP_RAG__EMBEDDING__PROVIDER", "APP_RAG__EMBEDDING__MODEL"):
            action = "按当前云方案配置；模型可用性/端点/区域待验证，不复制旧值"
        else:
            action = "不复制旧远端地址；用本地 APP_ 地址/端口；普通键是否被脚本读取待 M1 盘点"
        report.append(f"| `{key}` | {'非空，非有效性验证' if old[key] else '空'} | {action} |")
    report += ["", "## 缺失字段与接线任务（M1-1.3）", "",
               "1. MySQLSettings.user/password 与 RedisSettings.password 已存在，客户端也消费，不能按旧说明重复新增；字段名是 USER 而非 USERNAME。", "2. CheckpointSettings 缺独立 password，build_redis_checkpointer 当前取 database.redis.password；需要改为独立 Stack 口令。不能让两套 Compose 密码互相覆盖来规避问题。", "3. Settings 缺 auth 分组，APP_AUTH__JWT_SECRET 当前只是预留键；M1-1.3 补配置模型与必要校验，JWT/角色鉴权在 M5 实现。", "4. YamlPlaceholderSource 只解析 YAML 中实际存在的 ${VAR}；旧 config.yaml 仅使用 DEEPSEEK_API_KEY/ALIYUN_API_KEY，不存在 MYSQL_HOST、REDIS_HOST、MILVUS_HOST 等占位符，不能假设普通键会覆盖地址。", "5. 旧 get_settings 注入非 APP_ dotenv 键再加载 Settings；直接 Settings() 与 get_settings() 的占位符行为需测试。", "6. 保留 DeepSeek 推理/对话与 DashScope 嵌入/精排方向；模型与 base_url 为设计配置，不宣称外部可用。尚未迁入业务源码或安装依赖，本步不运行业务启动/建表/图。", "", "## 用户本地操作", "", "PowerShell 工作目录：D:\\learning\\factoring-marketing-agent。用本地编辑器打开根目录 .env，仅填写 DEEPSEEK_API_KEY、ALIYUN_API_KEY；保留 APP_ 同源引用及现有中间件凭据，不把 .env 覆盖为模板。", "", "```powershell", "Set-Location -LiteralPath 'D:\\learning\\factoring-marketing-agent'", ".\\.venv\\Scripts\\python.exe deploy\\verify_env.py --report", "docker compose --env-file .env -f deploy/docker/compose.yaml config --quiet", "git check-ignore .env", "```", "", "成功输出：映射相等、状态表更新、Compose 静默验证退出 0、Git 返回 .env。Key 非空仍不证明云服务可用，真实验证留到后续步骤。"]
    (ROOT / "docs/环境变量对照表.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Wrote masked docs/环境变量对照表.md")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Known controlled errors contain only key names. Never dump parser input.
        if isinstance(exc, ValueError):
            print("FAIL: " + str(exc))
        else:
            print("FAIL: " + type(exc).__name__ + "; details suppressed to protect credentials")
        raise SystemExit(1)
