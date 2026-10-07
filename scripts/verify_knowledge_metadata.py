"""真实云嵌入/精排 + MySQL/Milvus 样例验收，不执行业务图。"""
import json
import logging
import sys
from pathlib import Path
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")
from app.rag.ingest import KnowledgeIngestor
from app.rag.retriever import HybridRetriever
from app.models import KnowledgeDoc


def main():
    logging.disable(logging.CRITICAL)
    ingestor = KnowledgeIngestor()
    assert ingestor.store.collection_name == "factoring_knowledge_v2", "collection_must_be_v2"
    ingestor.store.ensure_collection()
    path = Path(__file__).resolve().parents[1] / "data/samples/KB-PD-SAMPLE-001.md"
    _, metadata = ingestor.loader.load_document(path)
    old_count = ingestor.store.client.query(collection_name="factoring_knowledge", filter='chunk_id != ""', output_fields=["count(*)"])[0]["count(*)"]
    result = ingestor.ingest_file(path, source="product", priority=2)
    assert result.status == "indexed", "sample_ingest_failed:" + (result.error or "")
    rows = ingestor.store.snapshot_doc(metadata["doc_id"])
    assert len(rows) == result.chunk_cnt
    assert all(all(row[key] == value for key, value in metadata.items()) for row in rows)
    with ingestor.session_factory() as session:
        row = session.scalars(select(KnowledgeDoc).where(KnowledgeDoc.doc_id == metadata["doc_id"], KnowledgeDoc.version == metadata["version"])).one()
        assert all(getattr(row, key) == value for key, value in metadata.items())
    # 独立入库命令验证幂等，不重跑有写库副作用的业务图。
    again = ingestor.ingest_file(path, source="product", priority=2)
    assert again.status == "indexed" and len(ingestor.store.snapshot_doc(metadata["doc_id"])) == result.chunk_cnt
    retrieval = HybridRetriever(settings=ingestor.settings, store=ingestor.store).retrieve("保理融资如何核实真实应收账款与交易背景？", industry="基建")
    matching = [item.to_dict() for item in retrieval.items if item.doc_id == metadata["doc_id"]]
    assert matching, "sample_retrieval_missing"
    assert all(matching[0][key] == value for key, value in metadata.items())
    assert matching[0]["parent_content"]
    after = ingestor.store.client.query(collection_name="factoring_knowledge", filter='chunk_id != ""', output_fields=["count(*)"])[0]["count(*)"]
    assert old_count == after == 26
    report = {"status": "passed", "collection": ingestor.store.collection_name,
              "metadata": metadata, "chunks": result.chunk_cnt, "same_version_no_duplicates": True,
              "retrieval_echo": {key: matching[0][key] for key in metadata}, "old_collection_rows": after}
    target = Path(__file__).resolve().parents[1] / "docs/validation/2.1-metadata-acceptance.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # 不打印服务异常、连接串或模型响应；细节由离线测试验证。
        print("[FAIL] knowledge metadata acceptance:" + type(exc).__name__)
        raise SystemExit(1)
