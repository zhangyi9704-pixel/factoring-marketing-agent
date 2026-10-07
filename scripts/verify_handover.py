"""M7 owned backup/isolated restore; never drops source data or overwrites targets."""
import argparse
import copy
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import shutil
import sys
from uuid import uuid4
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ["docker", "compose", "--env-file", str(ROOT / ".env"), "-f", str(ROOT / "deploy/docker/compose.yaml")]

def docker(*args):
    result = subprocess.run(COMPOSE + list(args), capture_output=True, text=True, encoding="utf8", errors="replace")
    if result.returncode:
        raise RuntimeError("docker_operation_failed_output_withheld")
    return result.stdout

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))

def sql_fingerprints(engine):
    from sqlalchemy import inspect, text
    result = {}
    with engine.connect() as conn:
        for table in sorted(inspect(engine).get_table_names()):
            if not re.fullmatch(r"[a-zA-Z0-9_]+", table):
                raise RuntimeError("unexpected_table_identifier")
            rows = sorted(canonical(dict(r)) for r in conn.execute(text(f"SELECT * FROM `{table}`")).mappings())
            result[table] = dict(rows=len(rows), sha256=hashlib.sha256("\n".join(rows).encode()).hexdigest())
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", help="new hexadecimal run identifier; default random")
    args = parser.parse_args()
    run = args.run or uuid4().hex[:12]
    if not re.fullmatch(r"[a-f0-9]{12}", run):
        parser.error("run must be 12 lowercase hex characters")
    logging.disable(logging.CRITICAL)
    from app.config import get_settings
    from app.db.mysql import MySQLDatabase
    from app.rag.milvus_client import VectorStore
    from sqlalchemy import text
    settings = get_settings()
    database = "factoring_m7_" + run
    collection = "factoring_m7_" + run
    backup = ROOT / "backups" / ("m7_" + run)
    backup.mkdir(parents=True, exist_ok=False)
    report_path = ROOT / "docs/validation" / ("m7-handover-" + run + ".json")
    report = dict(run=run, database=database, collection=collection, backup=str(backup.relative_to(ROOT)), status="running", stages={})
    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    save()
    try:
        source_db = MySQLDatabase(settings)
        before = sql_fingerprints(source_db.engine)
        source_name = settings.database.mysql.database
        if source_name != "factoring_coach":
            raise RuntimeError("unexpected_source_database")
        remote = "/tmp/m7_" + run + ".sql"
        docker("exec", "-T", "mysql", "sh", "-c", 'umask 077; MYSQL_PWD="$MYSQL_ROOT_PASSWORD" mysqldump -uroot --single-transaction --routines --triggers --no-tablespaces factoring_coach > ' + remote)
        sql_path = backup / "mysql.sql"
        docker("cp", "mysql:" + remote, str(sql_path))
        if sql_path.stat().st_size == 0:
            raise RuntimeError("empty_backup")
        report["stages"]["mysqldump"] = dict(status="passed", bytes=sql_path.stat().st_size, sha256=hashlib.sha256(sql_path.read_bytes()).hexdigest())
        save()
        # CREATE without IF NOT EXISTS refuses pre-existing targets. No DROP/DELETE.
        docker("exec", "-T", "mysql", "sh", "-c", 'MYSQL_PWD="$MYSQL_ROOT_PASSWORD" mysql -uroot -e "CREATE DATABASE ' + database + ' CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci; GRANT ALL PRIVILEGES ON ' + database + '.* TO \'$MYSQL_USER\'@\'%\';"')
        docker("exec", "-T", "mysql", "sh", "-c", 'MYSQL_PWD="$MYSQL_ROOT_PASSWORD" mysql -uroot ' + database + ' < ' + remote)
        restored_settings = copy.deepcopy(settings)
        restored_settings.database.mysql.database = database
        after = sql_fingerprints(MySQLDatabase(restored_settings).engine)
        report["stages"]["mysql_restore"] = dict(status="passed" if before == after else "failed", tables=after, source_hashes_equal=before == after)
        save()
        if before != after:
            raise RuntimeError("mysql_restore_fingerprint_mismatch")
        source = VectorStore(settings)
        fields = [f.name for f in source.build_schema(source.dimension).fields]
        iterator = source.client.query_iterator(collection_name=source.collection_name, batch_size=200, filter='id != ""', output_fields=fields, consistency_level="Strong")
        rows = []
        try:
            while batch := iterator.next():
                rows.extend(batch)
        finally:
            iterator.close()
        rows.sort(key=lambda r: r["id"])
        vector_path = backup / "milvus-logical.json"
        vector_path.write_text(canonical(rows), encoding="utf8")
        (backup/"milvus-schema.json").write_text(canonical(dict(version=source.ping(),dimension=source.dimension,schema=source.client.describe_collection(collection_name=source.collection_name))),encoding="utf8")
        restored_settings.rag.vector_database.collection = collection
        target = VectorStore(restored_settings)
        if target.client.has_collection(collection):
            raise RuntimeError("restore_collection_already_exists")
        target.ensure_collection()
        loaded = json.loads(vector_path.read_text(encoding="utf8"))
        for row in loaded:
            row["sparse_vector"] = {int(k): v for k, v in row["sparse_vector"].items()}
        target.insert(loaded)
        target.client.flush(collection_name=collection)
        restored = target.client.query(collection_name=collection, filter='id != ""', output_fields=fields, limit=max(1,len(rows)+1), consistency_level="Strong")
        restored.sort(key=lambda r: r["id"])
        equal = canonical(rows) == canonical(restored)
        report["stages"]["milvus_logical_restore"] = dict(status="passed" if equal else "failed", source_rows=len(rows), restored_rows=len(restored), sha256=hashlib.sha256(vector_path.read_bytes()).hexdigest(), all_fields_equal=equal)
        save()
        if not equal:
            raise RuntimeError("vector_restore_fingerprint_mismatch")
        tracked = subprocess.run(["git", "status", "--short", "--", "data", "cases", "config.yaml", "requirements.lock.txt"],capture_output=True,text=True,encoding="utf8")
        commit = subprocess.run(["git", "rev-parse", "HEAD"],capture_output=True,text=True,encoding="utf8")
        assets = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for parent in ("data", "cases") for p in (ROOT/parent).rglob("*") if p.is_file()}
        (backup/"knowledge-source-manifest.json").write_text(json.dumps(dict(commit=commit.stdout.strip() if not commit.returncode else None,git_status=tracked.stdout.splitlines(),sha256=assets),ensure_ascii=False,indent=2),encoding="utf8")
        archive=backup/"knowledge-source";archive.mkdir(exist_ok=False)
        for name,digest in assets.items():
            src=ROOT/name
            if hashlib.sha256(src.read_bytes()).hexdigest()!=digest:raise RuntimeError("knowledge_source_changed")
            dest=archive/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
            if hashlib.sha256(dest.read_bytes()).hexdigest()!=digest:raise RuntimeError("knowledge_archive_mismatch")
        for name in ("config.yaml","requirements.lock.txt",".python-version"):shutil.copy2(ROOT/name,archive/name)
        report["stages"]["knowledge_source"] = dict(status="passed", files=len(assets), dirty=bool(tracked.stdout.strip()), committed=not bool(tracked.stdout.strip()),archive_files=len(assets),hash_verified_archive=str(archive.relative_to(ROOT)))
        report["stages"]["physical_volume_snapshot"] = dict(status="not_executed_no_downtime", volumes=["factoring_"+v for v in ("milvus_data","etcd_data","minio_data")], requirement="same stopped-service window; logical restore does not prove physical-volume recovery")
        report["status"] = "passed_logical_restore"
        save()
        print(json.dumps(dict(run=run,status=report["status"],database=database,collection=collection,report=str(report_path.relative_to(ROOT))),ensure_ascii=False))
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = type(exc).__name__
        save()
        print("[FAIL] handover: " + type(exc).__name__ + "; details withheld; inspect saved stage states, no automatic write replay")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
