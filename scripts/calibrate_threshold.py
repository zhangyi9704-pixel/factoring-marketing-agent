"""只读云检索采样，按评分来源回放阈值；留出集不参与选择。"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from collections import Counter
from types import SimpleNamespace
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import get_settings
from app.rag.retriever import HybridRetriever
from app.rag.revision import revision

SCORES = ROOT / "docs/validation/threshold-calibration-scores.json"
RESULT = ROOT / "docs/validation/threshold-calibration-results.json"
REPORT = ROOT / "docs/阈值校准记录.md"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_labels(path, split):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row.get("split") != split or row.get("industry") not in {"基建", "医药", "电商"}:
            raise ValueError("invalid split or industry")
        if row.get("kind") not in {"true", "blind", "conflict", "missing"}:
            raise ValueError("invalid kind")
        mode = {"true": "rag", "blind": "missing", "conflict": "conflict", "missing": "missing"}[row["kind"]]
        if row.get("expected_mode") != mode or not row.get("query"):
            raise ValueError("invalid expected mode/query")
        items = row.get("expected_items")
        if not isinstance(items, list) or any(set(i) != {"doc_id", "version"} for i in items):
            raise ValueError("invalid expected identities")
        if mode != "missing" and not items or mode == "missing" and items:
            raise ValueError("expected identities disagree with branch")
        if mode == "conflict" and len({(i["doc_id"], i["version"]) for i in items}) < 2:
            raise ValueError("conflict needs two identities")
    return rows


def validate_sets(calibration, holdout):
    all_rows = calibration + holdout
    if len({r["id"] for r in all_rows}) != len(all_rows):
        raise ValueError("duplicate IDs")
    normalized = [re.sub(r"[\W_]+", "", r["query"]).casefold() for r in all_rows]
    if len(set(normalized)) != len(normalized):
        raise ValueError("duplicate queries or split leakage")
    for industry in ("基建", "医药", "电商"):
        counts = Counter(r["kind"] for r in calibration if r["industry"] == industry)
        if any(counts[k] < n for k, n in {"true": 5, "blind": 3, "conflict": 1, "missing": 1}.items()):
            raise ValueError("insufficient calibration labels per industry")
        if not all(any(r["industry"] == industry and r["kind"] == k for r in holdout)
                   for k in ("true", "blind", "conflict", "missing")):
            raise ValueError("insufficient holdout coverage")


def validate_source_versions(rows):
    """从权威知识资产核对身份、版本与行业，不用检索表现反推标签。"""
    sources = {}
    for category in ("product", "process", "industry", "compliance"):
        for path in (ROOT / "data" / category).glob("*.md"):
            front = path.read_text(encoding="utf-8").split("---", 2)[1]
            metadata = yaml.safe_load(front)
            sources[(metadata["doc_id"], metadata["version"])] = metadata
    for row in rows:
        for expected in row["expected_items"]:
            metadata = sources.get((expected["doc_id"], expected["version"]))
            if not metadata or metadata["industry"] not in {row["industry"], "all"} or metadata["status"] == "archived":
                raise ValueError(f"label source/version mismatch: {row['id']}")


def context(settings):
    files = sorted(p for category in ("product", "process", "industry", "compliance", "samples")
                   for p in (ROOT / "data" / category).glob("*.md"))
    cfg = settings.rag.retrieval
    return dict(embedding_model=settings.rag.embedding.model,
        embedding_dimension=settings.rag.embedding.dimension,
        rerank_model=cfg.rerank.model, rerank_provider=cfg.rerank.provider,
        collection=settings.rag.vector_database.collection, knowledge_revision=revision(),
        knowledge_files={str(p.relative_to(ROOT)): digest(p) for p in files},
        labels={name: digest(ROOT / f"data/eval/{name}.jsonl") for name in ("calibration", "holdout")},
        sampling_labels={name: hashlib.sha256(json.dumps([
            {k: row[k] for k in ("id", "query", "industry", "split")}
            for row in load_labels(ROOT / f"data/eval/{name}.jsonl", name)], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            for name in ("calibration", "holdout")},
        recall_k=cfg.recall_k, fused_top_k=cfg.top_k, rerank_top_n=cfg.rerank.top_n,
        rrf_k=cfg.hybrid.rrf_k, sparse_encoder=cfg.hybrid.sparse)


def same_sampling_context(previous, current):
    # 标签订正不改变外部请求；保留采样时完整标签哈希并在结果中记录新标签哈希。
    return {k: v for k, v in previous.items() if k != "labels"} == {k: v for k, v in current.items() if k != "labels"}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def collect(rows, settings, provenance, resume=False):
    working = settings.model_copy(deep=True)
    working.rag.retrieval.rerank.min_score = 0.0
    retriever = HybridRetriever(working)
    recorded = []
    if resume and SCORES.exists():
        prior = json.loads(SCORES.read_text(encoding="utf-8"))
        if not same_sampling_context(prior["context"], provenance):
            raise ValueError("score provenance changed; cannot resume")
        recorded = prior["queries"]
    done = {r["id"] for r in recorded}
    for row in rows:
        if row["id"] in done:
            continue
        if HybridRetriever.out_of_scope(row["query"]):
            raw, ranked = [], []
            reason = "out_of_scope"
        else:
            expr = 'chunk_type == "child" and status in ["active", "conflicted"]'
            expr += f' and industry in ["all", "{row["industry"]}"]'
            dense, sparse = retriever.embedder.embed([row["query"]])
            hits = retriever.store.hybrid_search(dense[0], sparse[0], provenance["fused_top_k"], filter_expr=expr)
            if any(h.get("industry") not in {"all", row["industry"]} or h.get("status") not in {"active", "conflicted"} for h in hits):
                raise ValueError("store returned incompatible metadata")
            raw = [retriever._to_item(h) for h in hits]
            ranked, degraded = retriever.reranker.rerank_with_status(row["query"], copy.deepcopy(raw),
                top_n=provenance["fused_top_k"])
            if degraded:
                raise RuntimeError("rerank unavailable: stop; do not mix score sources")
            reason = None
        def compact(item):
            return {key: getattr(item, key) for key in ("doc_id", "version", "topic", "status", "retrieval_score", "rerank_score")}
        recorded.append(dict(id=row["id"], reason=reason, candidate_count=len(raw),
            retrieval=[compact(i) for i in raw], rerank=[compact(i) for i in ranked]))
        if context(settings) != provenance:
            raise ValueError("knowledge/config changed during sampling")
        write_json(SCORES, dict(context=provenance, date=datetime.now(timezone.utc).isoformat(), queries=recorded))
        print(f"sample {len(recorded)}/{len(rows)} {row['id']} candidates={len(raw)}", flush=True)
    return recorded


def predict(sample, source, threshold, top_n):
    if sample.get("reason") == "out_of_scope":
        return "missing", set()
    field = "rerank_score" if source == "rerank" else "retrieval_score"
    items = sorted(sample[source], key=lambda i: i[field], reverse=True)[:top_n]
    items = [i for i in items if i[field] >= threshold]
    if not items:
        return "missing", set()
    topics = HybridRetriever.conflict_topics([SimpleNamespace(**i) for i in items])
    if topics:
        return "conflict", {(i["doc_id"], i["version"]) for i in items if i["topic"] in topics}
    return "rag", {(i["doc_id"], i["version"]) for i in items}


def evaluate(rows, samples, source, threshold, top_n):
    by_id = {s["id"]: s for s in samples}
    details, branches = [], Counter()
    true_ok = blind_false = branch_ok = 0
    for row in rows:
        mode, identities = predict(by_id[row["id"]], source, threshold, top_n)
        expected = {(i["doc_id"], i["version"]) for i in row["expected_items"]}
        correct = mode == row["expected_mode"] and expected.issubset(identities)
        branches[mode] += 1
        true_ok += row["kind"] == "true" and correct
        blind_false += row["kind"] == "blind" and mode != "missing"
        branch_ok += correct
        details.append(dict(id=row["id"], mode=mode, correct=correct,
                            returned=[dict(doc_id=d, version=v) for d, v in sorted(identities)]))
    ntrue = sum(r["kind"] == "true" for r in rows)
    nblind = sum(r["kind"] == "blind" for r in rows)
    return dict(threshold=threshold, recall=true_ok / ntrue if ntrue else 0,
        false_recall=blind_false / nblind if nblind else 0,
        branch_accuracy=branch_ok / len(rows), branches={k: branches[k] for k in ("rag", "conflict", "missing")}, details=details)


def recommend(rows, samples, source, top_n):
    # 网格只来自校准分数，留出分数不得影响候选阈值或选择。
    ids = {r["id"] for r in rows}
    field = "rerank_score" if source == "rerank" else "retrieval_score"
    scores = [i[field] for s in samples if s["id"] in ids for i in s[source]]
    grid = set([i / 40 for i in range(41)] if source == "rerank" else [i / 2000 for i in range(71)])
    grid.update(round(s, 6) for s in scores)
    grid.update(round(s, 6) + .000001 for s in scores)
    table = [evaluate(rows, samples, source, t, top_n) for t in sorted(grid)]
    feasible = [m for m in table if m["recall"] >= .9 and m["false_recall"] <= .1]
    selected = max(feasible, key=lambda m: (m["branch_accuracy"], -m["false_recall"], m["recall"], m["threshold"])) if feasible else None
    return table, selected


def render_report(result, labels):
    lines = ["# 阈值校准记录", "", f"日期：{result['date']}。标注状态：{result['review_status']}。本次执行修改配置：{result['applied']}；当前配置匹配建议：{result.get('configuration_matches_recommendation', False)}。", "",
        "校准30条：每行业5真命中、3盲查、1冲突、1缺失；留出15条独立问句，不参与阈值选择。召回率=期望分支及doc_id/version均正确的真命中数/真命中总数；误召率=进入rag或conflict的盲查数/盲查总数。缺失和冲突另计分支正确率。", "",
        "人工标注依据与所有问句见[复核表](阈值校准标注复核表.md)。两集合问句去标点归一化后无重复；留出是问句级分离，冲突知识对复用，不宣称文档级独立或大样本泛化。", "",
        "模型与知识快照、候选数见[原始分数](validation/threshold-calibration-scores.json)。没有混用RRF与精排分数；外部精排不可用时停止采样，不用降级分数冒充精排。", "", "```json", json.dumps(result["context"], ensure_ascii=False, indent=2), "```", ""]
    selected_summary = result["sources"]["rerank"]["selected"]
    held_summary = result["sources"]["rerank"]["holdout"]
    if selected_summary:
        lines[4:4] = [f"结论：建议精排阈值 **{selected_summary['threshold']}**；校准召回{selected_summary['recall']:.1%}/误召{selected_summary['false_recall']:.1%}；留出召回{held_summary['recall']:.1%}/误召{held_summary['false_recall']:.1%}、分支正确率{held_summary['branch_accuracy']:.1%}。", "",
            "选取规则：先满足校准集召回≥90%、盲查误召≤10%，再最大化分支正确率、降低误召、提高召回，最后取较高阈值；留出集完全不参与选择。", ""]
    for source, data in result["sources"].items():
        lines += [f"## {source} 独立评分校准", "", "| 阈值 | 真命中召回 | 盲查误召 | 分支正确率 | rag/conflict/missing |", "| --- | --- | --- | --- | --- |"]
        for m in data["table"]:
            b = m["branches"]
            lines.append(f"| {m['threshold']:.6f} | {m['recall']:.1%} | {m['false_recall']:.1%} | {m['branch_accuracy']:.1%} | {b['rag']}/{b['conflict']}/{b['missing']} |")
        selected = data["selected"]
        lines += ["", f"建议阈值：{selected['threshold'] if selected else '无可行阈值'}。仅按校准集选取，不用留出表现调参。", ""]
        if data["holdout"]:
            h = data["holdout"]
            lines += [f"留出：召回{h['recall']:.1%}，误召{h['false_recall']:.1%}，分支正确率{h['branch_accuracy']:.1%}。", "",
                      "| 行业 | 召回 | 误召 | 分支正确率 |", "| --- | --- | --- | --- |"]
            for industry, m in data["per_industry"].items():
                lines.append(f"| {industry} | {m['recall']:.1%} | {m['false_recall']:.1%} | {m['branch_accuracy']:.1%} |")
    lines += ["", "## 操作与结论", "", "工作目录和PowerShell：", "", "```powershell",
        "Set-Location -LiteralPath 'D:\\learning\\factoring-marketing-agent'",
        ".\\.venv\\Scripts\\python.exe scripts\\calibrate_threshold.py",
        ".\\.venv\\Scripts\\python.exe scripts\\calibrate_threshold.py --reuse-scores",
        "# 人工复核且留出全部正确后才采用：",
        ".\\.venv\\Scripts\\python.exe scripts\\calibrate_threshold.py --reuse-scores --apply", "```", "",
        "首条只读查询Milvus及调用已授权的云嵌入/精排，不执行业务图、不入库；每问保存分数便于--resume。第二条基于同一模型/知识/问句快照回放，无云调用，指标可重复；标签订正保留前后哈希。第三条采用rerank.min_score，要求人工复核状态verified、校准达标、留出分支和文档版本全部正确。服务内配置缓存需重新加载后生效。", "",
        "RRF降级采用独立报告；若无可行阈值，--apply将fallback_min_score设null，精排故障时按无依据处理，不能继续把小样本阈值当成达标校准。无可行阈值需改进召回/知识覆盖，不伪造达标。M4队列及评估评分不在本步范围。",
        "", "版本标签订正：初稿holdout-INF-02/KB-PR-003、holdout-PHA-01/KB-PD-007、holdout-PHA-02/KB-PR-002、holdout-ECO-02/KB-PR-005误写v1.0，权威Markdown实际v1.2。2026-10-06用户补充确认四处修正；问句、doc_id、期望分支未改。初始结果保存在threshold-calibration-initial-labels.json；原始采样标签哈希与当前标签哈希分别留档。"]
    live_path = ROOT / "docs/validation/threshold-calibration-live.json"
    if live_path.exists():
        live = json.loads(live_path.read_text(encoding="utf-8"))
        if live["threshold"] == result.get("configured_rerank_min_score"):
            lines += ["", "## 真实检索留出验收", "", f"日期：{live['date']}；阈值{live['threshold']}；{sum(r['correct'] for r in live['results'])}/{len(live['results'])}符合期望分支及文档版本。", "",
                "PowerShell（同一项目根目录）：`.\\.venv\\Scripts\\python.exe -u scripts\\calibrate_threshold.py --verify-live`。通过真正HybridRetriever召回、精排、阈值、冲突和父块回溯；无业务图重放。"]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-scores", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--verify-live", action="store_true")
    args = parser.parse_args()
    calibration = load_labels(ROOT / "data/eval/calibration.jsonl", "calibration")
    holdout = load_labels(ROOT / "data/eval/holdout.jsonl", "holdout")
    validate_sets(calibration, holdout)
    validate_source_versions(calibration + holdout)
    settings = get_settings()
    provenance = context(settings)
    if args.verify_live:
        retriever = HybridRetriever(settings)
        results = []
        for row in holdout:
            actual = retriever.retrieve(row["query"], industry=row["industry"])
            identities = {(item.doc_id, item.version) for item in actual.items}
            expected = {(i["doc_id"], i["version"]) for i in row["expected_items"]}
            correct = (actual.mode == row["expected_mode"] and expected.issubset(identities)
                       and not actual.rerank_degraded and actual.reason != "retrieval_unavailable")
            results.append(dict(id=row["id"], correct=correct, result=actual.to_dict()))
            print(f"verify {row['id']} mode={actual.mode} expected={row['expected_mode']} correct={correct}", flush=True)
        write_json(ROOT / "docs/validation/threshold-calibration-live.json", dict(
            context=provenance, threshold=settings.rag.retrieval.rerank.min_score,
            date=datetime.now(timezone.utc).isoformat(), results=results))
        if not all(r["correct"] for r in results):
            raise ValueError("live held-out verification failed; inspect archived results")
        print("[OK] live holdout all expected branches and identities")
        return
    if args.reuse_scores:
        saved = json.loads(SCORES.read_text(encoding="utf-8"))
        if not same_sampling_context(saved["context"], provenance):
            raise ValueError("score provenance mismatch; resample")
        samples = saved["queries"]
        if {s["id"] for s in samples} != {r["id"] for r in calibration + holdout}:
            raise ValueError("incomplete score snapshot")
    else:
        samples = collect(calibration + holdout, settings, provenance, args.resume)
    reviewed = all(r.get("review_status") == "verified" for r in calibration + holdout)
    result = dict(date=datetime.now(timezone.utc).isoformat(), context=provenance,
                  review_status="verified" if reviewed else "pending", applied=False, sources={})
    if args.reuse_scores:
        result["sampled_label_hashes"] = saved["context"]["labels"]
    for source in ("rerank", "retrieval"):
        table, selected = recommend(calibration, samples, source, provenance["rerank_top_n"])
        held = evaluate(holdout, samples, source, selected["threshold"], provenance["rerank_top_n"]) if selected else None
        industries = {ind: evaluate([r for r in holdout if r["industry"] == ind], samples, source,
                      selected["threshold"], provenance["rerank_top_n"]) for ind in ("基建", "医药", "电商")} if selected else {}
        result["sources"][source] = dict(table=table, selected=selected, holdout=held, per_industry=industries)
        print(f"{source}: suggested={selected['threshold'] if selected else None}; holdout={held['branch_accuracy'] if held else None}")
        print("threshold recall false_recall rag/conflict/missing")
        for m in table:
            if m is selected or m["threshold"] in {0, .5, .7, .8, .9, 1, .03, .032, .0325, .033}:
                b = m["branches"]
                print(f"{m['threshold']:.6f} {m['recall']:.3f} {m['false_recall']:.3f} {b['rag']}/{b['conflict']}/{b['missing']}")
    if args.apply:
        chosen = result["sources"]["rerank"]
        if not reviewed or not chosen["selected"] or chosen["holdout"]["branch_accuracy"] != 1:
            write_json(RESULT, result)
            render_report(result, calibration + holdout)
            raise ValueError("cannot apply: human review/calibration/holdout gate not satisfied")
        cfg_path = ROOT / "config.yaml"
        text = cfg_path.read_text(encoding="utf-8")
        new, count = re.subn(r"(\n      min_score:\s*\n        )[^\n]+", lambda m: m[1] + str(chosen["selected"]["threshold"]), text)
        if count != 1:
            raise ValueError("unexpected config path")
        if result["sources"]["retrieval"]["selected"] is None:
            new, count = re.subn(r"(\n    fallback_min_score: )[^\n]+", r"\g<1>null", new)
            if count != 1:
                raise ValueError("unexpected fallback config path")
            result["fallback_disabled"] = True
        cfg_path.write_text(new, encoding="utf-8")
        result["applied"] = True
    result["configured_rerank_min_score"] = (result["sources"]["rerank"]["selected"]["threshold"]
        if args.apply else settings.rag.retrieval.rerank.min_score)
    result["configured_fallback_min_score"] = None if result.get("fallback_disabled") else settings.rag.retrieval.fallback_min_score
    result["configuration_matches_recommendation"] = bool(result["sources"]["rerank"]["selected"] and
        result["configured_rerank_min_score"] == result["sources"]["rerank"]["selected"]["threshold"])
    write_json(RESULT, result)
    render_report(result, calibration + holdout)
    print(f"[OK] calibration archived; review={result['review_status']}; applied={result['applied']}")


if __name__ == "__main__":
    main()
