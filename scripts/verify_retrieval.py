"""在线检索验证脚本（TDD §11.3 E2E）。

用法：
    python scripts/verify_retrieval.py --query "正向保理的融资比例是多少"
    python scripts/verify_retrieval.py --query "..." --top-k 3 --show-full

走真实链路：EmbeddingService（.env 决定 provider，当前 dashscope）
→ Milvus hybrid_search（RRF k=60）→ parent 回溯 → 精排 top5 → 优先级排序。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.rag.retriever import HybridRetriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="知识库在线检索验证（hybrid + RRF + parent 回溯）")
    parser.add_argument("--query", required=True, help="检索问题")
    parser.add_argument("--top-k", type=int, default=None, help="hybrid 召回条数（缺省 config top_k=10）")
    parser.add_argument("--show-full", action="store_true", help="打印 parent 全文（缺省截断 120 字）")
    parser.add_argument("--industry", choices=["all", "基建", "医药", "电商"])
    args = parser.parse_args()

    retriever = HybridRetriever()
    print("=" * 60)
    print("在线检索流水线（TDD §11.3）")
    print("=" * 60)
    print(f"[Milvus] {retriever.store.ping()} · collection {retriever.store.collection_name!r}")
    print(f"[Embedding] provider={retriever.embedder.provider}")
    print(f"[Query] {args.query!r}")
    print("-" * 60)

    result = retriever.retrieve(args.query, top_k=args.top_k, industry=args.industry)
    data = result.to_dict()

    if data["missing"]:
        print(f"[MISS] action={data['action']}")
        print(f"[MISS] message={data['message']}")
        print("结论: 空结果 → human_review 分支（已记录待补充知识条目日志）")
        return 0

    for i, item in enumerate(result.items, 1):
        note = " · 仅供参考" if item.reference_only else ""
        print(f"[{i}] {item.source}(priority={item.priority}) score={item.score:.6f}{note}")
        print(f"    doc={item.doc_name} chunk={item.chunk_id[:12]}… child={item.content[:50]}…")
        print(f"    doc_id={item.doc_id} version={item.version} effective_date={item.effective_date} status={item.status} category={item.category} industry={item.industry} topic={item.topic}")
        context = item.parent_content or "(parent 缺失)"
        if args.show_full:
            print(f"    parent={context}")
        else:
            print(f"    parent={context[:120]}{'…' if len(context) > 120 else ''}")
    print("-" * 60)
    print(f"结论: 命中 {len(result.items)} 条（RRF k="
          f"{retriever.settings.rag.retrieval.hybrid.rrf_k}，rerank top_n="
          f"{retriever.settings.rag.retrieval.rerank.top_n}，云精排）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
