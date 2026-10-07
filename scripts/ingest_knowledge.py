r"""旧基线知识入库 CLI（项目根目录 PowerShell）。

    .\.venv\Scripts\python.exe scripts/ingest_knowledge.py --source all

流程：loader → parent_child chunker → 云 embedding → Milvus + MySQL。
全部切片验证后按 doc_id/version 更新。
版本化 upsert，失败补偿；输出不打印原始异常。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.rag.ingest import (  # noqa: E402
    STATUS_FAILED,
    STATUS_INDEXED,
    STATUS_SKIPPED,
    KnowledgeIngestor,
)

MARKS = {STATUS_INDEXED: "[OK]  ", STATUS_FAILED: "[FAIL]", STATUS_SKIPPED: "[SKIP]"}


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库入库（loader→chunker→embedder→Milvus+MySQL）")
    parser.add_argument(
        "--source", choices=["all", "product", "industry", "compliance", "process"],
        help="只入库指定来源（缺省全量）",
    )
    parser.add_argument("--file", type=Path, help="入库单个带元数据头的文件")
    args = parser.parse_args()

    logging.disable(logging.CRITICAL)
    ingestor = KnowledgeIngestor()
    print("=" * 60)
    print("知识库入库流水线（TDD §11.1）")
    print("=" * 60)
    print(f"[Milvus] {ingestor.store.ping()} · collection "
          f"{ingestor.store.collection_name!r} -> {ingestor.store.ensure_collection()}")
    if ingestor.embedder.provider == "local":
        print("[Model] BGE-M3 首次运行需下载/加载权重，请耐心等待…")
    else:
        print(f"[Embedding] provider={ingestor.embedder.provider}")

    if args.file:
        if not args.source or args.source == "all":
            parser.error("--file 必须同时指定 --source")
        cfg = ingestor.settings.knowledge.sources[args.source]
        results = [ingestor.ingest_file(args.file, source=args.source, priority=cfg.priority)]
    elif args.source and args.source != "all":
        results = ingestor.ingest_source(args.source)
    else:
        results = ingestor.ingest_all().results

    for r in results:
        line = f"{MARKS[r.status]} {r.source}/{r.doc_name} chunks={r.chunk_cnt}"
        if r.error:
            line += " | error_category=ingest_failed"
        print(line)

    indexed = sum(r.status == STATUS_INDEXED for r in results)
    failed = sum(r.status == STATUS_FAILED for r in results)
    skipped = sum(r.status == STATUS_SKIPPED for r in results)
    print("-" * 60)
    print(f"结论: indexed={indexed} failed={failed} skipped={skipped} "
          f"(Milvus 总行数={ingestor.store.count()})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
