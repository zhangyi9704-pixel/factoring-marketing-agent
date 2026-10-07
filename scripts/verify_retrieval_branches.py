"""本地只读检索四问；保存脱敏结果，不运行/重放任何业务图。"""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rag.retriever import HybridRetriever

QUERIES = [
    ("保理费率由哪些部分构成，哪些因素影响费率？", "rag", None),
    ("暗保理需要付款人确权吗？", "conflict", None),
    ("冰岛海底光缆抢修保理的专用准入条款是什么？", "missing", "insufficient_evidence"),
    ("帮我预测明天天气", "missing", "out_of_scope"),
]


def main():
    retriever = HybridRetriever()
    results, calibration = [], []
    for question, expected, reason in QUERIES:
        result = retriever.retrieve(question, industry="all")
        data = result.to_dict()
        results.append(data)
        print(f"mode={result.mode}; reason={result.reason}; items={len(result.items)}; degraded={result.rerank_degraded}")
        assert result.mode == expected and (reason is None or result.reason == reason)
        assert not result.rerank_degraded
        if not reason or reason == "insufficient_evidence":
            dense, sparse = retriever.embedder.embed([question])
            hits = retriever.store.hybrid_search(dense[0], sparse[0], 10,
                filter_expr='chunk_type == "child" and status in ["active", "conflicted"] and industry in ["all"]')
            calibration += [dict(query=question, doc_id=h["doc_id"], retrieval_score=h["score"],
                relevant=h["doc_id"] in ({"KB-PD-003"} if expected == "rag" else
                    {"KB-PD-005A", "KB-PD-005B"} if expected == "conflict" else set())) for h in hits]
    positive = [r["retrieval_score"] for r in calibration if r["relevant"]]
    negative = [r["retrieval_score"] for r in calibration if not r["relevant"]]
    separable = bool(positive and negative and min(positive) > max(negative))
    def metrics(threshold):
        tp = sum(r["relevant"] and r["retrieval_score"] >= threshold for r in calibration)
        fp = sum(not r["relevant"] and r["retrieval_score"] >= threshold for r in calibration)
        return dict(threshold=threshold, tp=tp, fp=fp, fn=len(positive)-tp,
                    precision=tp/(tp+fp) if tp+fp else None, recall=tp/len(positive) if positive else 0)
    grid = [metrics(t) for t in (.016, .025, .03, .0321, .0323, .033)]
    report = dict(queries=results, fallback_calibration=dict(samples=calibration, metrics=grid,
        metric="Milvus RRF(k=60), dense/sparse each20 → fused10", separable=separable,
        threshold=retriever.settings.rag.retrieval.fallback_min_score,
        decision="tiny acceptance set only; prioritise zero pilot false positives; lower recall accepted; full labelled calibration belongs to M3-3.5",
        rerank_min_score=retriever.settings.rag.retrieval.rerank.min_score,
        production_fallback_min_score=retriever.settings.rag.retrieval.fallback_min_score))
    target = Path("docs/validation/3.3-retrieval-live.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] four branches; RRF separable={separable}; artifact={target}")


if __name__ == "__main__":
    main()
