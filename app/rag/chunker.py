"""parent_child 切分（TDD §11.1：parent 1200 / child 300 / overlap 50）。

规则：
    - 以句子/换行边界贪心装箱，尽量不截断语义单元（中文句末标点 。！？；
      与半角 !?; 及换行视为边界）；
    - 超长单句按字符硬切；
    - child 在句子边界滑动并携带 overlap 尾部前缀，保证跨 child 语义连续；
    - 检索命中 child 后按 parent_id 回溯 parent 上下文（TDD §11.1）；
    - chunk_id = md5(doc_name|source|kind|序号) 前 32 位：确定性 ID，
      同一文档重复入库生成相同 ID，配合"先删后插"实现幂等。
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Optional

# 句末边界：句号/叹号/问号/分号（中英文）与换行之后切分
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")


class ParentChildChunker:
    """parent_child 切分器（纯函数，无 I/O，可离线单测）。"""

    def __init__(self, parent_size: int = 1200, child_size: int = 300, overlap: int = 50) -> None:
        if child_size <= 0:
            raise ValueError("child_size 必须为正")
        if not 0 <= overlap < child_size:
            raise ValueError("overlap 必须满足 0 <= overlap < child_size")
        if parent_size < child_size:
            raise ValueError("parent_size 必须 >= child_size")
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

    @classmethod
    def from_settings(cls, settings) -> "ParentChildChunker":
        chunk = settings.rag.chunk
        return cls(
            parent_size=chunk.parent_size,
            child_size=chunk.child_size,
            overlap=chunk.overlap,
        )

    # -------------------------------------------------
    # 对外入口
    # -------------------------------------------------

    def split(
        self,
        text: str,
        *,
        doc_name: str,
        source: str,
        priority: int = 3,
    ) -> list[dict[str, Any]]:
        """切分全文，返回 chunk 行列表（TDD §11.2 标量字段，不含向量）。

        Returns:
            [{"chunk_id", "parent_id", "chunk_type", "source",
              "doc_name", "priority", "content"}, ...]
            parent 行的 parent_id 为空字符串；child 行指向所属 parent。
        """
        parents = self._pack_parents(text)
        chunks: list[dict[str, Any]] = []
        for p_idx, p_text in enumerate(parents):
            parent_id = self._chunk_id(doc_name, source, f"parent-{p_idx}")
            chunks.append(self._row(
                chunk_id=parent_id, parent_id="", chunk_type="parent",
                content=p_text, doc_name=doc_name, source=source, priority=priority,
            ))
            for c_idx, c_text in enumerate(self._children(p_text)):
                chunks.append(self._row(
                    chunk_id=self._chunk_id(doc_name, source, f"child-{p_idx}-{c_idx}"),
                    parent_id=parent_id, chunk_type="child",
                    content=c_text, doc_name=doc_name, source=source, priority=priority,
                ))
        return chunks

    # -------------------------------------------------
    # 切分算法
    # -------------------------------------------------

    def _sentences(self, text: str, size: int) -> list[str]:
        """切出句子单元；超长单句按 size 字符硬切。"""
        units: list[str] = []
        for s in _SENTENCE_SPLIT.split(text.replace("\r\n", "\n")):
            if not s.strip():
                continue
            if len(s) > size:
                units.extend(s[i:i + size] for i in range(0, len(s), size))
            else:
                units.append(s)
        return units

    def _pack_parents(self, text: str) -> list[str]:
        """句子贪心装箱为 parent（不超过 parent_size）。"""
        packed: list[str] = []
        buf = ""
        for unit in self._sentences(text, self.parent_size):
            if buf and len(buf) + len(unit) > self.parent_size:
                packed.append(buf)
                buf = unit
            else:
                buf += unit
        if buf:
            packed.append(buf)
        return [p.strip() for p in packed if p.strip()]

    def _children(self, parent_text: str) -> list[str]:
        """parent 内句子级滑窗生成 child（携带 overlap 尾部前缀）。"""
        size, overlap = self.child_size, self.overlap
        units = self._sentences(parent_text, size)
        children: list[str] = []
        cur = ""
        for unit in units:
            if cur and len(cur) + len(unit) > size:
                children.append(cur)
                tail = cur[-overlap:] if overlap > 0 else ""
                cur = unit if len(tail) + len(unit) > size else tail + unit
            else:
                cur += unit
        if cur:
            children.append(cur)
        return children

    # -------------------------------------------------
    # 工具
    # -------------------------------------------------

    @staticmethod
    def _chunk_id(doc_name: str, source: str, kind: str) -> str:
        raw = f"{doc_name}|{source}|{kind}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:32]

    @staticmethod
    def _row(*, chunk_id: str, parent_id: str, chunk_type: str, content: str,
             doc_name: str, source: str, priority: int) -> dict[str, Any]:
        return {
            "chunk_id": chunk_id,
            "parent_id": parent_id,
            "chunk_type": chunk_type,
            "source": source,
            "doc_name": doc_name,
            "priority": priority,
            "content": content,
        }
