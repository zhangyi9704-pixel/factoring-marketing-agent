"""跨进程发布代号；缓存只能复用同一个已提交知识版本。"""
from pathlib import Path
import os
import uuid

PATH = Path(__file__).resolve().parents[2] / ".runtime/knowledge_revision"


def revision() -> str:
    return PATH.read_text(encoding="ascii") if PATH.exists() else "initial"


def invalidate() -> None:
    PATH.parent.mkdir(exist_ok=True)
    token = uuid.uuid4().hex
    temporary = PATH.with_name("knowledge_revision." + token)
    temporary.write_text(token, encoding="ascii")
    os.replace(temporary, PATH)
