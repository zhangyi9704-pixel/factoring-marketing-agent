"""MySQL catalog is the commit boundary for controlled knowledge publication.

The exact allowlist also retrieves still-effective old chunks if a publisher
crashed after changing their vector status but before committing MySQL.
"""
import hashlib
import json
from sqlalchemy import select
from app.db.mysql import get_mysql_database
from app.models import KnowledgeDoc
from app.rag.integrity import scalar_hash


class KnowledgeVisibility:
    def __init__(self, session_factory=None):
        self.factory = session_factory or get_mysql_database().session_scope

    def catalog(self):
        with self.factory() as db:
            rows = db.scalars(select(KnowledgeDoc)).all()
            versions = {(r.doc_id,r.version):r.status for r in rows if r.doc_id and r.version
                        and r.status in {"active","conflicted"}}
            # Preserve the baseline's explicitly indexed legacy documents.
            legacy = {r.doc_name for r in rows if not r.doc_id and r.status == "indexed"}
            manifests = {(r.doc_id, r.version): r.chunk_manifest for r in rows if r.chunk_manifest
                         and (r.doc_id, r.version) in versions}
            hashes = sorted((r.doc_id, r.version, r.content_hash or "", r.ingestion_operation_id or "") for r in rows
                            if (r.doc_id, r.version) in versions)
        stamp = hashlib.sha256(json.dumps([sorted((a,b,s) for (a,b),s in versions.items()),sorted(legacy), hashes],
                                         ensure_ascii=False).encode()).hexdigest()
        return versions, legacy, stamp, manifests

    @staticmethod
    def expression(catalog):
        versions, legacy, _ = catalog[:3]
        clauses = [f'(doc_id == {json.dumps(doc)} and version == {json.dumps(version)})' for doc,version in versions]
        clauses += [f'(doc_id == "" and doc_name == {json.dumps(name)} and status in ["active", "conflicted"])' for name in legacy]
        return "(" + " or ".join(clauses) + ")" if clauses else 'doc_id == "__no_effective_knowledge__"'

    @staticmethod
    def filter(rows,catalog):
        versions, legacy, _ = catalog[:3]
        manifests = catalog[3] if len(catalog) > 3 else {}
        result = []
        for row in rows:
            status = versions.get((row.get("doc_id"),row.get("version")))
            if status:
                manifest = manifests.get((row.get("doc_id"), row.get("version")))
                if manifest is not None and manifest.get(row.get("chunk_id")) != scalar_hash(row):
                    continue
                result.append(dict(row,status=status))
            elif not row.get("doc_id") and row.get("doc_name") in legacy and row.get("status") in {"active","conflicted"}:
                result.append(row)
        return result
