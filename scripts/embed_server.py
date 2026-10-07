"""BGE-M3 embedding 独立服务（TDD §11.4 方案 A）。

部署在云服务器（8.133.229.0:8100），应用通过 HTTP 调用，避免
多 worker / 低内存环境各自加载模型。自包含单文件，不依赖项目 app 包。

启动（服务器）：
    HF_ENDPOINT=https://hf-mirror.com nohup python embed_server.py \
        --port 8100 >> /root/embed_server.log 2>&1 &

接口：
    GET  /health           → {"status": "ok", "model": "BAAI/bge-m3"}
    POST /embed {"texts": ["…"]}
                          → {"dense": [[1024 个 float（已 L2 归一化）]],
                             "sparse": [{"token_id": weight, …}]}
"""
from __future__ import annotations

import argparse
import sys

# ---------- 参数 ----------

parser = argparse.ArgumentParser(description="BGE-M3 embedding HTTP 服务")
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8100)
parser.add_argument("--model", default="BAAI/bge-m3")
parser.add_argument("--batch-size", type=int, default=16)
args = argparse.Namespace(host="0.0.0.0", port=8100, model="BAAI/bge-m3", batch_size=16)

import numpy as np  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

MODEL = None  # Optional legacy service: importing the module never loads weights.


def load_model():
    global MODEL
    if MODEL is None:
        try:
            import torch
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "Legacy local service requires optional FlagEmbedding/torch; "
                "the project defaults to cloud dashscope"
            ) from exc
        MODEL = BGEM3FlagModel(args.model, use_fp16=torch.cuda.is_available())
    return MODEL

app = FastAPI(title="bge-m3-embedding", version="1.0")


class EmbedRequest(BaseModel):
    texts: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model": args.model}


@app.post("/embed")
def embed(req: EmbedRequest):
    texts = [t if t.strip() else " " for t in req.texts]
    out = load_model().encode(
        texts,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = np.asarray(out["dense_vecs"], dtype=np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dense = dense / norms
    sparse = [
        {int(k): float(v) for k, v in w.items() if float(v) != 0.0}
        for w in out["lexical_weights"]
    ]
    return {"dense": dense.tolist(), "sparse": sparse}


if __name__ == "__main__":
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import uvicorn
    load_model()
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
