"""Read-only metric evaluation; never invokes coach_graph or production evaluation API."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import re
import statistics
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INDUSTRIES = ("基建", "医药", "电商")
ELEMENTS = {
    "付款人": r"付款人|付款方|谁.*付款|谁.*支付",
    "基础交易": r"基础交易|合同|供货|供应.*销售",
    "应收形成": r"应收.*形成|验收|计量|确权",
    "账期": r"账期|账龄|结算周期",
    "融资用途": r"融资.*用途|资金.*用途|融资.*用于",
    "对账回款": r"对账|回款|监管户",
}


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalized(text):
    return re.sub(r"[\W_]+", "", text).casefold()


def validate_dataset(rows):
    from scripts.calibrate_threshold import validate_source_versions
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate ids")
    prior = load_rows(ROOT / "data/eval/calibration.jsonl") + load_rows(ROOT / "data/eval/holdout.jsonl")
    queries = [normalized(r["query"]) for r in rows if "query" in r]
    if len(set(queries)) != len(queries) or set(queries) & {normalized(r["query"]) for r in prior}:
        raise ValueError("evaluation query overlaps calibration/holdout or itself")
    for industry in INDUSTRIES:
        subset = [r for r in rows if r["industry"] == industry]
        for kind, minimum in {"typical": 5, "induction": 2, "missing": 1, "conflict": 1, "dialogue": 1, "safe": 1}.items():
            if sum(r["kind"] == kind for r in subset) < minimum:
                raise ValueError("insufficient industry coverage")
    for r in rows:
        if not r.get("expected_points") or r["industry"] not in INDUSTRIES:
            raise ValueError("missing annotation")
        if r["kind"] in {"typical", "missing", "conflict"}:
            if r["expected_mode"] not in {"rag", "missing", "conflict"}:
                raise ValueError("invalid branch annotation")
            if (r["expected_mode"] != "missing") != bool(r["expected_items"]):
                raise ValueError("invalid identity annotation")
        if r["kind"] in {"induction", "safe"}:
            if (r["kind"] == "induction") != bool(r["expected_rules"]):
                raise ValueError("invalid compliance annotation")
        if r["kind"] == "dialogue":
            if not set(r["expected_elements"]).issubset(ELEMENTS) or not r["conversation"]:
                raise ValueError("invalid dialogue annotation")
    validate_source_versions([r for r in rows if "expected_items" in r])


def coverage(conversation):
    # Count delivered questions/disclosures, never annotations or hidden case facts.
    text = "\n".join(m["content"] for m in conversation if m["role"] in {"sales", "customer"})
    return [key for key, pattern in ELEMENTS.items() if re.search(pattern, text)]


class RecordingLLM:
    """Keep numeric raw scores only, to expose variance masked by evidence gating."""
    def __init__(self, inner):
        self.inner, self.samples, self.calls = inner, [], 0

    def chat_json_once(self, *args, **kwargs):
        self.calls += 1
        raw = self.inner.chat_json_once(*args, **kwargs)
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        self.samples.append({k: v.get("score") for k, v in parsed.get("dimensions", {}).items()
                             if isinstance(v, dict) and type(v.get("score")) in {int, float}})
        return raw


def evaluate_fresh(conversation, industry, settings, llm):
    """Independent inference entry: production scorers, no cache, session, or writes."""
    from app.evaluation.evidence import EvidenceExtractor
    from app.evaluation.contracts import verified_evidence
    from app.evaluation.opportunity import OpportunityDiagnoser
    from app.evaluation.sales_score import SalesScorer
    recorded = RecordingLLM(llm)
    case = {"industry": industry}
    evidence = verified_evidence(EvidenceExtractor(settings).extract(conversation, case=case), conversation)
    opportunity = OpportunityDiagnoser(recorded, settings).analyze(conversation, case, copy.deepcopy(evidence))
    sales = SalesScorer(recorded, settings).score(conversation, case, copy.deepcopy(evidence))
    dimensions = {**opportunity["dimension_scores"], **sales["dimension_scores"]}
    errors = opportunity["errors"] + sales["errors"]
    return dict(opportunity=opportunity["score"], sales=sales["total"],
                dimensions={k: v["gated_score"] for k, v in dimensions.items()},
                raw_dimensions=recorded.samples, model_calls=recorded.calls, errors=errors,
                valid=not any(e == "deterministic_fallback" or e.startswith("evidence_rejected:") for e in errors))


def briefing_state(case):
    from app.schemas.case import to_public_case
    return dict(session_id="metrics-readonly", public_case=to_public_case(case).model_dump(), case=case.model_dump())


def variance(samples):
    # Sample variance: one observation cannot prove perfect consistency.
    return statistics.variance(samples) if len(samples) >= 2 else None


def latency(values):
    if not values:
        return {"n": 0, "mean_s": None, "p95_s": None}
    return dict(n=len(values), mean_s=statistics.mean(values), p95_s=sorted(values)[math.ceil(.95 * len(values)) - 1])


def summarize(result):
    details = result["details"]
    retrieval = [d for d in details if d["task"] == "retrieval"]
    typical = [d for d in retrieval if d["kind"] == "typical"]
    positive = [d for d in retrieval if d["kind"] in {"typical", "conflict"}]
    compliance = [d for d in details if d["task"] == "compliance" and d["kind"] == "induction"]
    redline = [d for d in compliance if d["redline"]]
    dialogues = [d for d in details if d["task"] == "dialogue"]
    briefings = [d for d in details if d["task"] == "briefing"]
    def rate(items):
        return sum(d.get("correct", False) for d in items) / len(items) if items else None
    consistency = []
    for d in dialogues:
        samples = d["samples"]
        valid = len(samples) == result["repeats"] and all(s.get("valid", False) for s in samples)
        observed = len(samples) == result["repeats"] and all(
            "opportunity" in s and "sales" in s and "deterministic_fallback" not in s.get("errors", []) for s in samples)
        raw = [{k: v for output in s.get("raw_dimensions", []) for k, v in output.items()} for s in samples]
        raw_keys = set.intersection(*(set(s) for s in raw)) if raw else set()
        consistency.append(dict(id=d["id"], valid=valid, repeats=len(samples),
            observed_opportunity_variance=variance([s["opportunity"] for s in samples]) if observed else None,
            observed_sales_variance=variance([s["sales"] for s in samples]) if observed else None,
            observed_dimension_variance={k: variance([s["dimensions"][k] for s in samples]) for k in samples[0].get("dimensions", {})} if observed else {},
            opportunity_variance=variance([s["opportunity"] for s in samples]) if valid else None,
            sales_variance=variance([s["sales"] for s in samples]) if valid else None,
            dimension_variance={k: variance([s["dimensions"][k] for s in samples]) for k in samples[0]["dimensions"]} if valid else {},
            raw_dimension_variance={k: variance([s[k] for s in raw]) for k in sorted(raw_keys)} if observed else {}))
    cov = sum(len(d["covered"]) for d in dialogues) / (6 * len(dialogues)) if dialogues else None
    btime = latency([d["elapsed_s"] for d in briefings if d.get("valid")])
    safe = [d for d in details if d["task"] == "compliance" and d["kind"] == "safe" and not d.get("l2_degraded", True)]
    metrics = dict(knowledge_recall_accuracy=rate(positive), typical_recall_accuracy=rate(typical),
        retrieval_branch_accuracy=rate(retrieval),
        per_industry_recall={ind: rate([d for d in positive if d["industry"] == ind]) for ind in INDUSTRIES},
        key_question_coverage=cov, coverage_annotation_accuracy=rate(dialogues),
        violation_detection_rate=rate(compliance), redline_detection_rate=rate(redline),
        safe_false_positive_rate=sum(bool(d.get("hits")) for d in safe) / len(safe) if safe else None,
        safe_valid_count=len(safe),
        score_consistency=consistency, briefing_latency=btime,
        briefing_degraded_latency=latency([d["elapsed_s"] for d in briefings if d.get("degraded")]),
        briefing_phase_latency={key: latency([d['timings'][key] for d in briefings if d.get('valid') and key in d.get('timings', {})])
            for key in ('retrieval_s', 'llm_s', 'guardrail_s', 'assembly_s', 'persist_s')},
        retrieval_latency=latency([d["elapsed_s"] for d in retrieval if d.get("available")]))
    def gate(value, target, upper=False):
        return "pending" if value is None else "passed" if (value <= target if upper else value >= target) else "failed"
    gates = dict(recall=gate(metrics["knowledge_recall_accuracy"], .9), coverage=gate(cov, .85),
        violation=gate(metrics["violation_detection_rate"], .95), redline=gate(metrics["redline_detection_rate"], 1),
        briefing=gate(btime["p95_s"], 90, True) if len(briefings) == 3 * result.get('briefing_samples_per_case', 1) and all(d.get("valid") for d in briefings) else "pending",
        score_consistency="no_documented_threshold", retrieval_latency="no_documented_threshold",
        preparation_time_reduction="pending_manual_baseline",
        degraded_briefing_latency=gate(metrics["briefing_degraded_latency"]["p95_s"], 5, True))
    result.update(metrics=metrics, gates=gates)
    if result.get("status") == "invalid_assets_changed":
        result["gates"] = {k: "pending_assets_changed" for k in gates}


def write_reports(result, stem):
    summarize(result)
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    m = result["metrics"]
    lines = ["# 指标评测", "", f"日期：{result['date']}；状态：{result['status']}；评分重复次数：{result['repeats']}。", "",
        "固定人工编写的模拟演练样本；不是线上销售覆盖率或季度业务成效。新问句与校准/留出集归一化去重；知识条目可复用，不宣称文档级独立。标注为工程初稿，待业务人工复核；期望要点用于复核，不冒充回答质量自动评分。", "",
        "召回=所有有期望条目的typical/conflict问句命中全部期望doc_id/version且分支正确、无服务降级的比例；典型召回另外报告，空期望缺失问句仅计分支。六要素按送达对话关键词规则覆盖/6，并与标注比较。违规要求期望规则全部检出，红线还要求block；安全对照另计误报。外部失败保留分母，不算正确。", "",
        "评分每次直接调用OpportunityDiagnoser/SalesScorer，不调用生产幂等接口；报告总分和维度的样本方差（分²）。n<2或模型降级时方差待验证，不能以降级零分证明一致。observed方差描述真实生成的门控总分；有证据拒绝时valid=false、完整证据通过口径的方差为空，不能宣称评分质量通过。raw_dimensions及raw_dimension_variance保留模型数值；model_calls包括schema修复。", "",
        "时延使用perf_counter、nearest-rank P95；简报含检索、模型、合规扫描与防泄漏，无持久化；检索含可见性目录查询、嵌入、精排和父块回溯，每问新实例避免缓存。简报仅流程小样本，生产P95和人工准备下降50%还需更多样本及人工基线。", "",
        "| 指标 | 实测 | 目标/状态 |", "| --- | --- | --- |"]
    if result.get("execution_condition"):
        lines[4:4] = ["执行条件：本报告来自受限网络，目录查询权限及云连接失败；不可据此判断模型质量。真实允许联网后的完整结果见同目录后续报表。", ""]
    for name, key, target, gate in [("知识召回", "knowledge_recall_accuracy", "≥90%", "recall"),
        ("六要素覆盖", "key_question_coverage", "≥85%", "coverage"),
        ("违规检出", "violation_detection_rate", "≥95%", "violation"),
        ("红线检出及阻断", "redline_detection_rate", "100%", "redline")]:
        value = m[key]
        lines.append(f"| {name} | {value:.2%} | {target} / {result['gates'][gate]} |" if value is not None else f"| {name} | 待验证 | {target} |")
    lines += [f"| 简报时延 | {json.dumps(m['briefing_latency'])} | P95≤90s / {result['gates']['briefing']} |",
              f"| 降级简报时延 | {json.dumps(m['briefing_degraded_latency'])} | ≤5s / {result['gates']['degraded_briefing_latency']}；仅自然降级样本 |",
              f"| 检索时延 | {json.dumps(m['retrieval_latency'])} | 文档未设阈值 |",
              f"| 含缺失/冲突的分支准确率 | {m['retrieval_branch_accuracy']} | 附加诊断项 |",
              f"| 典型问句召回率 | {m['typical_recall_accuracy']} | 附加诊断项；主指标包含冲突期望 |",
              f"| 安全对照误报率 | {m['safe_false_positive_rate']}（有效样本{m['safe_valid_count']}） | 附加诊断项 |",
              "", "评分方差：", "", "```json", json.dumps(m["score_consistency"], ensure_ascii=False, indent=2), "```", "",
        "失败与改进口径：", ""]
    failures = [d for d in result["details"] if d.get("correct") is False or d.get("error") or d.get("valid") is False]
    if result["gates"]["briefing"] == "failed":
        lines.append(f"- 简报P95={m['briefing_latency']['p95_s']:.3f}s，超过90s目标；样本数={m['briefing_latency']['n']}，分段计时见下表与JSON。扩大样本后定位检索、推理和合规耗时，再验证优化。")
    for d in failures:
        reason = d.get('error') or d.get('reason')
        if d["task"] == "dialogue" and not d.get("valid"):
            reason = "模型已生成数值，但有证据引用拒绝或降级；核查完整原话精确匹配及各维度规则保留轮次的白名单，见JSON samples.errors"
        lines.append(f"- {d['id']} / {d['task']}：{reason or '与标注不符，见JSON逐条结果'}。")
    lines += ["- 召回不符：先核查标注与有效知识版本、候选条目及阈值过滤；补充知识或在校准集调整，不用本指标集调参后声称独立达标。",
        "- 覆盖不符：检查规则误计与遗漏；样本覆盖不足时改进访谈提问，不能靠隐藏案例补齐。",
        "- 合规不符：核查期望规则、L1/L2及block条件；分别改进规则和语义检测，并保留安全对照。",
        "- 方差未设目标，先收集更多重复及业务人工评分再确定允许差异；模型故障与结构失败单列。",
        "- 时延不足：分离云请求和检索耗时，扩大样本；降级路径5秒与季度准备耗时下降50%仍待专门基线。", "",
        "PowerShell（项目根目录）：", "", "```powershell", "Set-Location -LiteralPath 'D:\\learning\\factoring-marketing-agent'",
        ".\\.venv\\Scripts\\python.exe scripts\\eval_metrics.py --validate-only",
        ".\\.venv\\Scripts\\python.exe -u scripts\\eval_metrics.py --repeats 2", "```", "",
        "第一条核对样本数量、隔离和知识标签，不调用云服务；第二条只读真实检索并调用云模型，每次生成新时间戳报表，可重复运行；无业务会话/待办/评估写库。JSON保存脱敏配置、资产哈希、所有样本和失败状态。"]
    lines += ['', '简报分段计时（秒）：', '', '```json', json.dumps(m['briefing_phase_latency'], ensure_ascii=False, indent=2), '```', '',
              f"每案例独立简报样本数：{result.get('briefing_samples_per_case', 1)}；同一公开案例新实例重新生成，不使用已完成简报缓存。扩大样本仍不是生产总体SLO证明。"]
    if result.get("briefing_continuation"):
        lines += ["", "入口适配与续测：最初简报state误用to_full_case（内部展示标记结构），触发CaseState校验错误；已改成节点使用的model_dump结构。前36条样本未重跑；通过--retry-briefings校验资产、模型配置、知识修订号一致后只补测简报，旧失败尝试保存在JSON的briefing_continuation.previous_attempts。", "",
                  "```powershell", f".\\.venv\\Scripts\\python.exe -u scripts\\eval_metrics.py --retry-briefings reports\\{stem.name}.json", "```"]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument('--briefing-samples', type=int, default=1, help='Fresh briefing calls per case')
    parser.add_argument("--retry-briefings", type=Path, help="Continue only briefing samples from a matching saved report")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >=1")
    if args.briefing_samples < 1:
        parser.error('--briefing-samples must be >=1')
    rows = load_rows(ROOT / "data/eval/metrics.jsonl")
    validate_dataset(rows)
    print(f"[OK] dataset validated: {len(rows)} rows; independent queries; annotations draft", flush=True)
    if args.validate_only:
        return
    # Suppress dependency error bodies: no credentials/URLs in reports or terminal.
    logging.disable(logging.CRITICAL)
    from app.config import get_settings
    from app.rag.retriever import HybridRetriever
    from app.services.llm_service import LLMService
    from app.compliance.engine import ComplianceEngine
    from app.agents.coach.briefing import make_briefing_node
    from app.schemas.case import CaseState
    from app.rag.revision import revision
    import yaml
    settings = get_settings()
    date = datetime.now(timezone(timedelta(hours=8)))
    stem = ROOT / "reports" / f"eval_{date.strftime('%Y-%m-%d_%H%M%S_%f')}"
    files = [ROOT / "data/eval/metrics.jsonl", ROOT / "config.yaml"] + list((ROOT / "cases").glob("*.yaml"))
    files += [p for category in ("industry", "product", "process", "compliance", "samples") for p in (ROOT / "data" / category).glob("*.*")]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result = dict(date=date.isoformat(), status="running", repeats=args.repeats,
        briefing_samples_per_case=args.briefing_samples, annotation_status="draft_pending_business_review",
        context=dict(embedding_model=settings.rag.embedding.model, rerank_model=settings.rag.retrieval.rerank.model,
            min_score=settings.rag.retrieval.rerank.min_score, chat_model=settings.llm.models.chat.name,
            reasoning_model=settings.llm.models.reasoning.name, knowledge_revision=revision(),
            destinations={name: urlsplit(value).hostname for name, value in {
                "llm": settings.llm.api.base_url, "embedding": settings.rag.embedding.base_url,
                "rerank": settings.rag.retrieval.rerank.base_url}.items()}, asset_hashes=hashes), details=[])
    if args.retry_briefings:
        source = args.retry_briefings.resolve()
        if source.parent != (ROOT / "reports").resolve() or source.suffix != ".json":
            raise ValueError("resume report must be project reports/*.json")
        previous = json.loads(source.read_text(encoding="utf-8"))
        before = previous["context"]
        for key in ("embedding_model", "rerank_model", "min_score", "chat_model", "reasoning_model", "knowledge_revision"):
            if before.get(key) != result["context"][key]:
                raise ValueError("resume configuration/revision mismatch")
        if any(hashes.get(p) != h for p, h in before["asset_hashes"].items()):
            raise ValueError("resume assets mismatch")
        if {d["id"] for d in previous["details"] if d["task"] != "briefing"} != {r["id"] for r in rows}:
            raise ValueError("resume requires complete non-briefing samples")
        args.briefing_samples = previous.get("briefing_samples_per_case", 1)
        result = previous
        result["status"] = "running"
        result["briefing_continuation"] = dict(date=date.isoformat(), reason="use CaseState.model_dump for node state; internal marker dicts are not schema input",
            previous_attempts=[d for d in previous["details"] if d["task"] == "briefing"])
        result["details"] = [d for d in previous["details"] if d["task"] != "briefing"]
        stem = source.with_suffix("")
    def record(detail):
        result["details"].append(detail)
        write_reports(result, stem)
        print(f"{detail['task']} {detail['id']}: {detail.get('correct', detail.get('valid', 'recorded'))}", flush=True)
    for row in rows:
        if args.retry_briefings:
            break
        kind = row["kind"]
        task = "retrieval" if kind in {"typical", "missing", "conflict"} else "compliance" if kind in {"induction", "safe"} else "dialogue"
        d = dict(id=row["id"], industry=row["industry"], kind=kind, task=task)
        start = perf_counter()
        try:
            if task == "retrieval":
                actual = HybridRetriever(settings).retrieve(row["query"], industry=row["industry"])
                returned = {(i.doc_id, i.version) for i in actual.items}
                expected = {(i["doc_id"], i["version"]) for i in row["expected_items"]}
                available = not actual.rerank_degraded and actual.reason not in {"retrieval_unavailable", "catalog_unavailable"}
                d.update(expected_items=row["expected_items"], expected_mode=row["expected_mode"], mode=actual.mode,
                    reason=actual.reason, returned=sorted(returned), available=available,
                    correct=available and actual.mode == row["expected_mode"] and expected.issubset(returned))
            elif task == "compliance":
                scan = ComplianceEngine(settings, llm=LLMService(settings)).scan(row["query"])
                hits = {h.rule: h for h in scan.hits}
                correct = all(rule in hits and (not row["redline"] or hits[rule].severity == "block") for rule in row["expected_rules"]) if kind == "induction" else not hits
                d.update(redline=row["redline"], expected_rules=row["expected_rules"],
                    hits=[dict(rule=h.rule, severity=h.severity) for h in scan.hits], l2_degraded=scan.l2_degraded, l2_error=getattr(scan, "l2_error", None),
                    correct=correct and not scan.l2_degraded)
            else:
                covered = coverage(row["conversation"])
                d.update(covered=covered, expected_elements=row["expected_elements"],
                    correct=set(covered) == set(row["expected_elements"]), samples=[])
                for repeat in range(args.repeats):
                    try:
                        d["samples"].append(evaluate_fresh(row["conversation"], row["industry"], settings, LLMService(settings)))
                    except Exception as exc:
                        d["samples"].append(dict(valid=False, error=type(exc).__name__))
                    print(f"score {row['id']} repeat={repeat + 1}/{args.repeats}", flush=True)
                d["valid"] = all(s["valid"] for s in d["samples"])
        except Exception as exc:
            d.update(error=type(exc).__name__, correct=False, available=False)
            if task == "compliance":
                d.update(redline=row["redline"])
            if task == "dialogue":
                d.update(covered=coverage(row["conversation"]), samples=[], valid=False)
        d["elapsed_s"] = perf_counter() - start
        record(d)
    for path, sample_index in ((p, n) for p in sorted((ROOT / "cases").glob("*.yaml")) for n in range(args.briefing_samples)):
        start = perf_counter()
        d = dict(id=path.stem if args.briefing_samples == 1 else f'{path.stem}-{sample_index + 1}', kind="briefing", task="briefing")
        try:
            case = CaseState(**yaml.safe_load(path.read_text(encoding="utf-8")))
            output = make_briefing_node(LLMService(settings), settings=settings,
                retriever=HybridRetriever(settings), guardrail=ComplianceEngine(settings))(briefing_state(case))
            brief = output["briefing"]
            degraded = bool(output["errors"]) or brief.get("compliance_status", {}).get("l2_degraded", False) or brief.get("knowledge_reason") in {"catalog_unavailable", "retrieval_unavailable"}
            d.update(valid=not degraded, reason=brief.get("knowledge_reason"), errors=output["errors"], degraded=degraded,
                     timings=output.get('briefing_timings', {}))
        except Exception as exc:
            d.update(valid=False, error=type(exc).__name__)
        d["elapsed_s"] = perf_counter() - start
        record(d)
    result["status"] = "complete"
    result["context"]["assets_unchanged"] = all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    result["context"]["knowledge_revision_unchanged"] = revision() == result["context"]["knowledge_revision"]
    if not result["context"]["assets_unchanged"] or not result["context"]["knowledge_revision_unchanged"]:
        result["status"] = "invalid_assets_changed"
    write_reports(result, stem)
    print(json.dumps(result["gates"], ensure_ascii=False), flush=True)
    print(f"[OK] reports: {stem.relative_to(ROOT)}.md / .json", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Configuration/SDK exceptions can embed credentials; never print bodies.
        print(f"[ERROR] evaluation aborted: {type(exc).__name__}", file=sys.stderr)
        sys.exit(1)
