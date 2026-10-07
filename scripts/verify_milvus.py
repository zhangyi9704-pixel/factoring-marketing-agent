"""Milvus 连通性验证脚本（TDD §11 验证）。

用法：
    python scripts/verify_milvus.py [--host 8.133.229.0] [--port 19530] [--skip-create]

验证内容：
    1. gRPC 端口(19530)连通 + 服务端版本
    2. metrics 端口(9091) HTTP 探测（版本信息，非致命）
    3. 现有 Collection 列表
    4. ensure_collection() 创建/校验 factoring_knowledge（TDD §11.2 schema）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import get_settings  # noqa: E402
from app.rag.milvus_client import VectorStore, VectorStoreError  # noqa: E402


def probe_metrics(host: str, port: int = 9091) -> None:
    """探测 9091 metrics 端口（Milvus 版本查询端口），失败不影响主流程。"""
    try:
        import httpx

        resp = httpx.get(f"http://{host}:{port}/metrics", timeout=5)
        version_lines = [
            line for line in resp.text.splitlines() if "version" in line.lower()
        ][:3]
        print(f"[OK] metrics 端口 {port} HTTP {resp.status_code}")
        for line in version_lines:
            print(f"     {line.strip()}")
    except Exception as exc:
        print(f"[WARN] metrics 端口 {port} 探测失败（不影响主功能）: {type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Milvus 连通性验证")
    parser.add_argument("--host", default=None, help="缺省使用 config.yaml 中的 host")
    parser.add_argument("--port", type=int, default=None, help="缺省使用 config.yaml 中的 port")
    parser.add_argument("--skip-create", action="store_true",
                        help="只验证连通，不创建 Collection")
    args = parser.parse_args()

    settings = get_settings()
    vd = settings.rag.vector_database
    host = args.host or vd.host
    port = args.port or vd.port

    print("=" * 60)
    print("Milvus 连通性验证")
    print("=" * 60)
    print(f"目标地址   : {host}:{port}")
    print(f"Collection : {vd.collection}（dim={settings.rag.embedding.dimension}）")
    print("-" * 60)

    store = VectorStore(settings, uri=f"http://{host}:{port}")

    try:
        version = store.ping()
        print(f"[OK] 连接成功，服务端版本: {version}")
    except VectorStoreError as exc:
        print(f"[FAIL] {exc}")
        print("       排查建议：")
        print("       1) 服务器上 Milvus 是否运行: docker ps | grep milvus")
        print("       2) 阿里云安全组是否放行 19530（建议仅白名单本机 IP）")
        print("       3) 未部署则按 TDD §16.1 docker-compose 部署 Milvus Standalone")
        probe_metrics(host)
        return 1

    probe_metrics(host)

    try:
        collections = store.client.list_collections()
        print(f"[OK] 现有 Collection（{len(collections)} 个）: {', '.join(collections) or '<无>'}")
    except Exception as exc:
        print(f"[FAIL] 列出 Collection 失败: {exc}")
        return 1

    if args.skip_create:
        print("[SKIP] 指定 --skip-create，未执行 Collection 创建")
        print("-" * 60)
        print("结论: 连通性验证通过")
        return 0

    try:
        status = store.ensure_collection()
        label = "已存在，校验通过" if status == "exists" else "创建成功"
        print(f"[OK] Collection '{vd.collection}' {label}")
        desc = store.client.describe_collection(vd.collection)
        fields = [f.get("name") for f in desc.get("fields", [])]
        print(f"     字段（{len(fields)} 个）: {', '.join(fields)}")
    except Exception as exc:
        print(f"[FAIL] ensure_collection 失败: {exc}")
        return 1

    print("-" * 60)
    print("结论: 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
