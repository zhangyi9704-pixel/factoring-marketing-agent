"""2.3：20文件与M2共35条核对；--live只读查双库和底层行业过滤。"""
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")
from app.config import Settings
from app.rag.loader import KnowledgeLoader
from app.rag.chunker import ParentChildChunker

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_IDS = {f"KB-{prefix}-{i:03}" for prefix in ("INF", "PHA", "ECO", "CMP") for i in range(1, 6)}


def verify(live=False):
    manifest = json.loads((ROOT / "docs/validation/2.3-content-manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(Path(manifest["source_path"]).read_bytes()).hexdigest() == manifest["source_sha256"], "source_changed"
    files = {str(p.relative_to(ROOT)) for category in ("industry", "compliance")
             for p in (ROOT / "data" / category).glob("*.md")}
    assert files == {e["path"] for e in manifest["entries"]} and len(files) == 20, "file_inventory"
    settings = Settings(_env_file=None)
    chunker = ParentChildChunker.from_settings(settings)
    counts = {c: {"files": 0, "parents": 0, "children": 0} for c in ("industry", "compliance")}
    industry_counts = {i: 0 for i in ("基建", "医药", "电商", "all")}
    entries = []
    for expected in manifest["entries"]:
        path = ROOT / expected["path"]
        body, meta = KnowledgeLoader().load_document(path)
        assert meta == expected["metadata"], "metadata_changed"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["file_sha256"], "file_changed"
        assert hashlib.sha256(body.encode()).hexdigest() == expected["body_sha256"], "body_changed"
        prefix = meta["doc_id"].split("-")[1]
        assert meta["industry"] == {"INF": "基建", "PHA": "医药", "ECO": "电商", "CMP": "all"}[prefix]
        assert meta["version"] == "KB-v1.0" and meta["effective_date"] == "2026-09-05" and meta["status"] == "active"
        chunks = chunker.split(body, doc_name=path.name, source=meta["source"],
                               priority=settings.knowledge.sources[meta["category"]].priority)
        count = counts[meta["category"]]
        count["files"] += 1
        count["parents"] += sum(c["chunk_type"] == "parent" for c in chunks)
        count["children"] += sum(c["chunk_type"] == "child" for c in chunks)
        industry_counts[meta["industry"]] += 1
        entries.append({"file": expected["path"], "metadata": meta, "chunks": len(chunks)})
    assert {e["metadata"]["doc_id"] for e in entries} == EXPECTED_IDS
    assert all(n == 5 for n in industry_counts.values())
    for old in manifest["legacy_preserved"]:
        assert hashlib.sha256((ROOT / "archive" / old["preserved_path"]).read_bytes()).hexdigest() == old["sha256"]
    # 四知识目录是M2正式条目范围；legacy/samples和规则文件分别计数。
    m2 = []
    for category, expected_count in {"product": 9, "process": 6, "industry": 15, "compliance": 5}.items():
        paths = list((ROOT / "data" / category).glob("*.md"))
        assert len(paths) == expected_count, "m2_category_count"
        for path in paths:
            _, meta = KnowledgeLoader().load_document(path)
            assert meta["category"] == category
            m2.append(meta)
    assert len({meta["doc_id"] for meta in m2}) == len(m2) == 35
    assert not {".yaml", ".yml"} & KnowledgeLoader.SUPPORTED_SUFFIXES, "yaml_must_be_excluded"
    result = {"offline": "passed", "counts": counts, "industry_labels": industry_counts,
              "doc_ids": sorted(EXPECTED_IDS), "m2_knowledge_documents": 35,
              "entries": entries, "rules": {"yaml_excluded_from_ingest": True,
              "file_exists": (ROOT / "data/compliance/rules.yaml").is_file(),
                "independent_loading": "pending_step_2.5"}}
    if result["rules"]["file_exists"]:
        from app.guardrail.rules import load_compliance_rules
        rule_set = load_compliance_rules()
        result["rules"].update(independent_loading="passed", rule_count=len(rule_set.rules), rule_version=rule_set.version)
    if live:
        from sqlalchemy import select
        from app.db.mysql import get_mysql_database
        from app.models import KnowledgeDoc
        from app.rag.milvus_client import VectorStore
        store = VectorStore()
        assert store.collection_name == "factoring_knowledge_v2"
        assert store.ensure_collection() == "exists"
        all_ids = {meta["doc_id"] for meta in m2}
        with get_mysql_database().session_scope() as session:
            docs = list(session.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id.in_(all_ids))))
            assert len(docs) == 35, "mysql_m2_count"
            by_id = {doc.doc_id: doc for doc in docs}
            for meta in m2:
                assert all(getattr(by_id[meta["doc_id"]], key) == value for key, value in meta.items()), "mysql_m2_metadata"
            for entry in entries:
                meta = entry["metadata"]
                rows = store.snapshot_doc(meta["doc_id"])
                assert len(rows) == by_id[meta["doc_id"]].chunk_cnt == entry["chunks"], "chunk_count"
                assert len({r["id"] for r in rows}) == len(rows), "duplicate_ids"
                assert all(all(r[key] == value for key, value in meta.items()) for r in rows), "milvus_metadata"
                parents = {r["id"] for r in rows if r["chunk_type"] == "parent"}
                assert all(r["parent_id"] in parents for r in rows if r["chunk_type"] == "child"), "parent_backtrace"
        probes = []
        for industry, prefix in (("基建", "INF"), ("医药", "PHA"), ("电商", "ECO"), ("all", "CMP")):
            category = "compliance" if prefix == "CMP" else "industry"
            expr = f'category == "{category}" and industry == "{industry}" and chunk_type == "child" and status == "active"'
            rows = store.client.query(collection_name=store.collection_name, filter=expr,
                                      output_fields=["doc_id", "category", "industry", "version", "effective_date"], consistency_level="Strong")
            assert {r["doc_id"] for r in rows} == {f"KB-{prefix}-{i:03}" for i in range(1, 6)} and len(rows) == 5, "industry_filter"
            assert all(r["industry"] == industry for r in rows), "cross_industry_leak"
            probes.append({"filter": expr, "children": len(rows), "rows": rows})
        rule_ids = [f"CMP-{i:02}" for i in range(1, 11)]
        excluded = store.count('doc_name == "rules.yaml" or doc_id in ' + json.dumps(rule_ids))
        assert excluded == 0, "rule_entities_in_vector_store"
        m2_count = store.count('doc_id in ' + json.dumps(sorted(all_ids)))
        assert m2_count == sum(doc.chunk_cnt for doc in docs) == 70, "m2_chunk_count"
        old_count = int(store.client.query(collection_name="factoring_knowledge", filter='chunk_id != ""', output_fields=["count(*)"])[0]["count(*)"])
        assert old_count == 26
        result.update(stores="passed", mysql_new_documents=20, mysql_m2_documents=35,
                      milvus_new_chunks=sum(e["chunks"] for e in entries), milvus_m2_chunks=m2_count,
                      collection_total_rows=store.count(), old_collection_rows=old_count,
                      rule_vector_rows=excluded, filter_probes=probes)
        (ROOT / "docs/validation/2.3-content-acceptance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "entries"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="只读检查MySQL/Milvus和底层过滤，无云模型调用")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    try:
        verify(args.live)
    except Exception as exc:
        print("[FAIL] industry/compliance validation:" + type(exc).__name__ + (":" + str(exc) if isinstance(exc, AssertionError) else ""))
        raise SystemExit(1)
