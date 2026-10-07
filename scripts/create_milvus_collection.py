"""创建/校验 Milvus factoring_knowledge Collection（TDD §11.2，幂等）。

用法：
    python scripts/create_milvus_collection.py
    # 远程地址来自 .env / config.yaml（rag.milvus），也可用环境变量覆盖：
    APP_RAG__MILVUS__HOST=8.133.229.0 python scripts/create_milvus_collection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import get_settings  # noqa: E402
from app.rag.milvus_client import VectorStore  # noqa: E402


def main() -> int:
    settings = get_settings()
    store = VectorStore(settings)
    version = store.ping()
    status = store.ensure_collection()
    print(f"[OK] Milvus {version} · collection={store.collection_name!r} -> {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
