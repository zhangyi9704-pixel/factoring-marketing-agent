"""Canonical hashes for commit acknowledgement and publication recovery."""
import hashlib
import json
import struct

from app.rag.loader import METADATA_FIELDS


def document_hash(metadata, body):
    payload = [dict(metadata), body.strip()]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def scalar_hash(row):
    # Vector status changes before the SQL commit; catalog supplies that status.
    fields = ("chunk_id", "parent_id", "chunk_type", "doc_name", "priority",
              "content", "chunk_index", *(key for key in METADATA_FIELDS if key != "status"))
    return hashlib.sha256(json.dumps({key: row.get(key) for key in fields}, ensure_ascii=False,
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def vector_hash(rows, *, status=None):
    # Milvus stores FLOAT_VECTOR / sparse values as float32. Normalize both the
    # submitted and queried values to that representation before comparison.
    def f32(value):
        return struct.unpack("!f", struct.pack("!f", float(value)))[0]
    canonical = []
    fields = ("id", "chunk_id", "parent_id", "chunk_type", "doc_name", "priority",
              "content", "chunk_index", *METADATA_FIELDS)
    for row in rows:
        item = {key: row.get(key) for key in fields}
        if status is not None:
            item["status"] = status
        item["embedding"] = [f32(value) for value in row["embedding"]]
        item["sparse_vector"] = sorted((int(key), f32(value)) for key, value in row["sparse_vector"].items())
        canonical.append(item)
    canonical.sort(key=lambda item: item["id"])
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()
