"""2.2：离线文件清单核对；--live 验证双库存储及两条云检索。"""
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
EXPECTED_IDS = {f"KB-PD-{i:03}" for i in (1, 2, 3, 4, 6, 7, 8)} | {
    "KB-PD-005A", "KB-PD-005B"} | {f"KB-PR-{i:03}" for i in range(1, 7)}


def verify(live=False):
    manifest = json.loads((ROOT / "docs/validation/2.2-content-manifest.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(Path(manifest["source_path"]).read_bytes()).hexdigest() == manifest["source_sha256"], "source_changed"
    actual_files = {str(p.relative_to(ROOT)) for category in ("product", "process")
                    for p in (ROOT / "data" / category).glob("*.md")}
    assert actual_files == {e["path"] for e in manifest["entries"]}, "file_inventory_mismatch"
    assert len(actual_files) == 15
    chunker = ParentChildChunker.from_settings(Settings(_env_file=None))
    counts = {"product": {"files": 0, "parents": 0, "children": 0},
              "process": {"files": 0, "parents": 0, "children": 0}}
    entries = []
    for expected in manifest["entries"]:
        path = ROOT / expected["path"]
        body, meta = KnowledgeLoader().load_document(path)
        assert meta == expected["metadata"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["file_sha256"]
        assert hashlib.sha256(body.encode()).hexdigest() == expected["body_sha256"]
        chunks = chunker.split(body, doc_name=path.name, source=meta["source"], priority=2 if meta["category"] == "product" else 4)
        count = counts[meta["category"]]
        count["files"] += 1
        count["parents"] += sum(c["chunk_type"] == "parent" for c in chunks)
        count["children"] += sum(c["chunk_type"] == "child" for c in chunks)
        entries.append({"file": expected["path"], "metadata": meta, "chunks": len(chunks)})
    assert {e["metadata"]["doc_id"] for e in entries} == EXPECTED_IDS
    conflict = [e["metadata"] for e in entries if e["metadata"]["status"] == "conflicted"]
    assert len(conflict) == 2 and len({e["topic"] for e in conflict}) == 1
    assert {e["version"] for e in conflict} == {"KB-v1.0", "KB-v1.2"}
    for legacy in manifest["legacy_preserved"]:
        assert hashlib.sha256((ROOT / "archive" / legacy["preserved_path"]).read_bytes()).hexdigest() == legacy["sha256"]
    result = {"offline": "passed", "counts": counts, "doc_ids": sorted(EXPECTED_IDS), "entries": entries}
    if live:
        from sqlalchemy import select
        from app.db.mysql import get_mysql_database
        from app.models import KnowledgeDoc
        from app.rag.milvus_client import VectorStore
        from app.rag.retriever import HybridRetriever
        store = VectorStore()
        assert store.collection_name == "factoring_knowledge_v2"
        assert store.ensure_collection() == "exists"
        with get_mysql_database().session_scope() as session:
            docs = list(session.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id.in_(EXPECTED_IDS))))
            assert len(docs) == 15, "mysql_document_count"
            by_id = {doc.doc_id: doc for doc in docs}
            for entry in entries:
                meta = entry["metadata"]
                row = by_id[meta["doc_id"]]
                assert all(getattr(row, key) == value for key, value in meta.items()), "mysql_metadata"
                rows = store.snapshot_doc(meta["doc_id"])
                assert len(rows) == row.chunk_cnt == entry["chunks"], "chunk_count"
                assert len({r["id"] for r in rows}) == len(rows), "duplicate_chunk"
                assert all(all(r[key] == value for key, value in meta.items()) for r in rows), "milvus_metadata"
                parents = {r["id"] for r in rows if r["chunk_type"] == "parent"}
                assert all(r["parent_id"] in parents for r in rows if r["chunk_type"] == "child"), "parent_backtrace"
        retriever = HybridRetriever(store=store)
        queries = []
        for query, required in [("暗保理到底要不要付款人确认？", {"KB-PD-005A", "KB-PD-005B"}),
                                ("有效初访需要核实哪六要素？", {"KB-PR-002"})]:
            retrieved = retriever.retrieve(query)
            hits = [item.to_dict() for item in retrieved.items]
            assert required <= {item["doc_id"] for item in hits}, "expected_query_hits"
            assert all(item["version"] and item["effective_date"] for item in hits), "query_version_missing"
            queries.append({"query": query, "required_ids": sorted(required),
                            "hits": [{k: item[k] for k in ("doc_id", "version", "effective_date", "status", "source", "topic", "score")} for item in hits]})
        result.update(stores="passed", mysql_documents=15,
                      milvus_chunks=sum(e["chunks"] for e in entries), collection_total_rows=store.count(), queries=queries,
                      old_collection_rows=int(store.client.query(collection_name="factoring_knowledge", filter='chunk_id != ""', output_fields=["count(*)"])[0]["count(*)"]))
        assert result["old_collection_rows"] == 26
        (ROOT / "docs/validation/2.2-content-acceptance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "entries"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="只读检查双库，并调用两次云嵌入/精排检索")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    try:
        verify(args.live)
    except Exception as exc:
        print("[FAIL] product/process validation:" + type(exc).__name__)
        raise SystemExit(1)
