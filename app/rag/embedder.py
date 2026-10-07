"""向量化服务（dense 1024 + lexical sparse，TDD §11.4）。

三种 provider（config rag.embedding.provider）：
    - local：进程内 FlagEmbedding 加载 BGE-M3（需 ≥4GB 空闲内存，
      本机开发环境不适用）；
    - http：调用自建 embedding 服务（TDD §11.4 方案 A，服务脚本
      scripts/embed_server.py）；
    - dashscope：阿里云托管 embedding（OpenAI 兼容协议，
      text-embedding-v4 等，key 走 ${ALIYUN_API_KEY}）。

稀疏向量来源（与 TDD §11.3 的差异，已记录在 TDD §11.1）：
    - local / http：BGE-M3 自带 lexical weights（Milvus 2.4.4 无
      内置 BM25 Function）；
    - dashscope：托管 API 仅返回稠密向量，稀疏信号由本地
      lexical_sparse_texts()（中文字符 bigram + 次线性 TF）生成，
      入库与检索端使用同一函数即可对齐。

三种 provider 返回同一契约：dense 已 L2 归一化（与 Milvus IP 度量
配合等价于余弦），sparse 为 {index: weight}。
构造参数 model / http_client 可注入假实现用于单测。
"""
from __future__ import annotations

import math
import os
import zlib
from collections import Counter
from typing import Any, Optional

import numpy as np

from app.config import Settings, get_settings
from app.services.external_calls import ExternalCalls, ExternalCallError, timeout_for, check_cancelled

# dashscope /embeddings 单请求输入条数上限
DASHSCOPE_MAX_BATCH = 10
# 词法稀疏编码的哈希维度空间
LEXICAL_ENCODER_ID = "lexical_bigram_crc32_v1"
LEXICAL_DIMS = 65536

DASHSCOPE_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class EmbeddingError(RuntimeError):
    """向量化错误。"""


def _bigrams(text: str) -> list[str]:
    """中文字符 bigram（去空白；单字符文本退化为自身）。"""
    cleaned = "".join(ch for ch in text if not ch.isspace())
    if not cleaned:
        return []
    if len(cleaned) == 1:
        return [cleaned]
    return [cleaned[i:i + 2] for i in range(len(cleaned) - 1)]


def lexical_sparse_texts(texts: list[str]) -> list[dict[int, float]]:
    """本地词法稀疏编码（bigram + 次线性 TF，crc32 定维，进程无关确定性）。

    作为托管 embedding API（无稀疏输出）的 sparse_vector 来源；
    检索端（TDD §11.3）编码 query 时必须调用同一函数。
    """
    result: list[dict[int, float]] = []
    for text in texts:
        counts = Counter(_bigrams(text))
        vec: dict[int, float] = {}
        for gram, cnt in counts.items():
            index = zlib.crc32(gram.encode("utf-8")) % LEXICAL_DIMS
            vec[index] = round(vec.get(index, 0.0) + 1.0 + math.log(cnt), 6)
        result.append(vec)
    return result


class EmbeddingService:
    """Embedding 封装（local / http / dashscope，批量、L2 归一化）。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        model: Any = None,
        batch_size: int = 16,
        http_client: Any = None,
    ) -> None:
        self.settings = settings or get_settings()
        embedding_cfg = self.settings.rag.embedding
        if self.settings.rag.retrieval.hybrid.sparse not in {LEXICAL_ENCODER_ID, "BM25"}:
            raise EmbeddingError("稀疏编码器配置与现有索引不一致，请先迁移索引；不能混用编码器。")
        self.provider = embedding_cfg.provider
        self.url = embedding_cfg.url
        self.timeout = embedding_cfg.timeout
        self.api_key = embedding_cfg.api_key
        self.base_url = embedding_cfg.base_url or DASHSCOPE_DEFAULT_BASE_URL
        self.api_model = embedding_cfg.model      # dashscope 模式下的模型 ID
        self.batch_size = batch_size
        self.dimension = embedding_cfg.dimension
        self._model = model            # local 模式：依赖注入假模型
        self.calls = ExternalCalls(self.settings)
        self._http = http_client       # 远端模式：依赖注入 httpx.Client（MockTransport）

    # -------------------------------------------------
    # local 模式：进程内模型（懒加载）
    # -------------------------------------------------

    @property
    def model(self):
        if self._model is None:
            # 国内网络默认走 hf-mirror（已设置则不覆盖）
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            try:
                import torch
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:
                raise EmbeddingError(
                    "local provider requires optional FlagEmbedding/torch; "
                    "use dashscope for the default cloud environment"
                ) from exc

            self._model = BGEM3FlagModel(
                self.settings.rag.embedding.model,
                use_fp16=torch.cuda.is_available(),
            )
        return self._model

    # -------------------------------------------------
    # 对外入口
    # -------------------------------------------------

    def embed(
        self, texts: list[str]
    ) -> tuple[list[list[float]], list[dict[int, float]]]:
        """批量向量化（空串/纯空白替换为单空格）。

        Returns:
            (dense, sparse)：dense 为归一化向量列表；sparse 为
            {index: weight} 列表。
        """
        if not texts:
            return [], []
        if self.provider == "http":
            return self._embed_http(texts)
        if self.provider == "local":
            return self._embed_local(texts)
        if self.provider == "dashscope":
            return self._embed_dashscope(texts)
        raise EmbeddingError(f"未知 embedding provider: {self.provider!r}")

    # -------------------------------------------------
    # local 实现
    # -------------------------------------------------

    def _embed_local(self, texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
        dense_all: list[list[float]] = []
        sparse_all: list[dict[int, float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[start:start + self.batch_size]]
            out = self.model.encode(
                batch,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            dense = np.asarray(out["dense_vecs"], dtype=np.float32)
            dense = self._validate_dense(dense, len(batch))
            norms = np.linalg.norm(dense, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # 零向量保护，避免除零
            dense_all.extend((dense / norms).tolist())
            for weights in out["lexical_weights"]:
                sparse_all.append(self._clean_sparse(weights))
        return dense_all, sparse_all

    # -------------------------------------------------
    # http 实现（TDD §11.4 方案 A：独立服务）
    # -------------------------------------------------

    def _embed_http(self, texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
        if not self.url:
            raise EmbeddingError("embedding provider=http 但未配置 rag.embedding.url")
        if self._http is None:
            import httpx
            self._http = httpx.Client(timeout=timeout_for(self.settings, self.timeout))

        dense_all: list[list[float]] = []
        sparse_all: list[dict[int, float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[start:start + self.batch_size]]
            try:
                payload = self.calls.post(self._http, f"{self.url.rstrip('/')}/embed",
                                          operation="embedding_http", json={"texts": batch})
            except ExternalCallError as exc:
                raise EmbeddingError("embedding 请求失败: " + str(exc)) from None
            try:
                dense_rows = payload["dense"]
                sparse_rows = payload["sparse"]
            except (KeyError, TypeError) as exc:
                raise EmbeddingError(f"embedding 服务响应缺少字段: {exc}") from exc
            if len(dense_rows) != len(batch) or len(sparse_rows) != len(batch):
                raise EmbeddingError(
                    f"embedding 服务返回数量不符: 期望 {len(batch)}, "
                    f"实际 dense={len(dense_rows)} sparse={len(sparse_rows)}"
                )
            dense_all.extend(dense_rows)
            sparse_all.extend(self._clean_sparse(s) for s in sparse_rows)
        self._validate_dimension(dense_all)
        return dense_all, sparse_all

    # -------------------------------------------------
    # dashscope 实现（阿里云托管，OpenAI 兼容协议）
    # -------------------------------------------------

    def _embed_dashscope(self, texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
        if not self.api_key:
            raise EmbeddingError(
                "embedding provider=dashscope 但未配置 api_key（ALIYUN_API_KEY）"
            )
        if self._http is None:
            import httpx
            self._http = httpx.Client(timeout=timeout_for(self.settings, self.timeout))

        dense_all: list[list[float]] = []
        batch_limit = min(self.batch_size, DASHSCOPE_MAX_BATCH)
        for start in range(0, len(texts), batch_limit):
            batch = [t if t.strip() else " " for t in texts[start:start + batch_limit]]
            body: dict[str, Any] = {"model": self.api_model, "input": batch}
            # Model defaults differ; always request the collection's configured dimension.
            body["dimensions"] = self.dimension
            try:
                payload = self.calls.post(
                    self._http, f"{self.base_url.rstrip('/')}/embeddings", operation="embedding",
                    headers={"Authorization": f"Bearer {self.api_key}"}, json=body)
                rows = sorted(payload["data"], key=lambda d: d.get("index", 0))
                if len(rows) != len(batch):
                    raise EmbeddingError("embedding 返回数量不符")
            except ExternalCallError as exc:
                raise EmbeddingError("embedding 请求失败: " + str(exc)) from None
            except (KeyError, TypeError) as exc:
                raise EmbeddingError("embedding API 响应缺少 data 字段") from None
            dense_all.extend(row["embedding"] for row in rows)
        self._validate_dimension(dense_all)
        # 托管 API 无稀疏输出：本地 bigram 词法编码（检索端用同一函数）
        return dense_all, lexical_sparse_texts(texts)

    @staticmethod
    def _api_error(resp) -> str:
        """提取 OpenAI 兼容错误体的 message（best-effort）。"""
        try:
            err = resp.json().get("error", {})
            return str(err.get("message") or resp.text[:200])
        except Exception:
            return resp.text[:200]

    # -------------------------------------------------
    # 校验工具
    # -------------------------------------------------

    def _validate_dense(self, dense: np.ndarray, expected_rows: int) -> np.ndarray:
        if dense.ndim != 2 or dense.shape[0] != expected_rows:
            raise EmbeddingError(f"dense 输出形状异常: {dense.shape}")
        if dense.shape[1] != self.dimension:
            raise EmbeddingError(
                f"向量维度 {dense.shape[1]} 与配置 {self.dimension} 不一致"
            )
        return dense

    def _validate_dimension(self, dense_rows: list[list[float]]) -> None:
        for i, row in enumerate(dense_rows):
            if len(row) != self.dimension:
                raise EmbeddingError(
                    f"第 {i} 条向量维度 {len(row)} 与配置 {self.dimension} 不一致"
                )

    @staticmethod
    def _clean_sparse(weights: dict) -> dict[int, float]:
        return {int(k): float(v) for k, v in weights.items() if float(v) != 0.0}
