"""知识库注入接口（TDD §15：POST /knowledge/ingest）。

复用 app/rag/ingest.py::KnowledgeIngestor（loader→chunker→embedder→Milvus→
t_knowledge_doc 全链条已就绪，CLI scripts/ingest_knowledge.py 已验证），这里
只补 HTTP 层：上传单个文档 → 临时落盘 → 触发入库 → 返回单文档结果。
"""
from __future__ import annotations

import tempfile
from functools import partial
from starlette.concurrency import run_in_threadpool
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.rag.ingest import STATUS_FAILED, STATUS_INDEXED, KnowledgeIngestor
from app.schemas.knowledge import IngestDocResult, SearchRequest, SearchResponse
from app.api.deps import require_admin, get_current_user
from app.api.v1.pending import get_pending_service

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

# config knowledge.sources 允许的来源（TDD §11.1）
SUPPORTED_SOURCES = ("product", "industry", "compliance", "process")


from functools import lru_cache


@lru_cache(maxsize=1)
def get_retriever():
    from app.rag.retriever import HybridRetriever
    return HybridRetriever()


@router.post("/search", response_model=SearchResponse, summary="知识检索三分支")
async def search_knowledge(request: SearchRequest, retriever=Depends(get_retriever),
                           user=Depends(get_current_user), pending=Depends(get_pending_service)):
    result = await run_in_threadpool(partial(retriever.retrieve, request.query,
        top_k=request.top_k, industry=request.industry))
    from app.services.escalation import register_retrieval
    result = await run_in_threadpool(partial(register_retrieval, result, user_id=user.id, service=pending))
    return result.to_dict()


def get_ingestor() -> KnowledgeIngestor:
    """KnowledgeIngestor 构建依赖（测试用 dependency_overrides 覆盖）。"""
    return KnowledgeIngestor()


@router.post("/ingest", response_model=IngestDocResult, dependencies=[Depends(require_admin)],
             summary="知识库文档注入",
             description="上传单个文档（md/txt/pdf/docx）触发入库流水线"
                         "（loader→chunker→embedder→Milvus→t_knowledge_doc），"
                         "返回该文档的入库结果。")
async def ingest_doc(
    file: UploadFile = File(..., description="文档文件"),
    source: str = Query(..., description="来源：product/industry/compliance/process"),
    ingestor: KnowledgeIngestor = Depends(get_ingestor),
) -> IngestDocResult:
    if source not in SUPPORTED_SOURCES:
        raise HTTPException(
            status_code=400,
            detail=f"未知来源: {source}（可选 {'/'.join(SUPPORTED_SOURCES)}）",
        )
    source_cfg = (getattr(ingestor.settings.knowledge, "sources", {}) or {}).get(source)
    if source_cfg is None:
        raise HTTPException(status_code=400, detail=f"config 未定义来源: {source}")

    tmp = await _save_upload(file)
    try:
        result = await run_in_threadpool(partial(
            ingestor.ingest_file, tmp, source=source, priority=source_cfg.priority,
        ))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"入库失败: {type(exc).__name__}") from exc
    finally:
        try:
            tmp.unlink(missing_ok=True)
            tmp.parent.rmdir()  # 仅删除本次创建的空临时目录
        except OSError:
            pass

    if result.status != STATUS_INDEXED:
        raise HTTPException(
            status_code=500,
            detail=f"入库未成功（{result.status}）",
        )
    return IngestDocResult(
        doc_name=result.doc_name, source=result.source,
        status=result.status, chunk_cnt=result.chunk_cnt, error=result.error,
    )


async def _save_upload(file: UploadFile) -> Path:
    """把上传内容写成临时文件（保留原扩展名，loader 按后缀分流）。"""
    data = await file.read()
    name = Path(file.filename or "knowledge.md").name
    directory = Path(tempfile.mkdtemp(prefix="factoring-upload-"))
    tmp = directory / name
    try:
        tmp.write_bytes(data)
    except Exception:
        directory.rmdir()
        raise
    return tmp


__all__ = ["router", "get_ingestor", "SUPPORTED_SOURCES"]

from app.api.deps import require_reviewer
from app.schemas.knowledge_feedback import DraftConfirmation
from app.services.knowledge_feedback import KnowledgeFeedback
from app.services.domain_errors import FeedbackConflict, FeedbackValidation, FeedbackUnavailable


def get_feedback_service():
    return KnowledgeFeedback()


def feedback_call(operation):
    try:
        result = operation()
    except PermissionError:
        raise HTTPException(403, "sales_scope_forbidden") from None
    except FeedbackConflict as exc:
        raise HTTPException(409, str(exc)) from None
    except FeedbackValidation as exc:
        raise HTTPException(422, str(exc)) from None
    except FeedbackUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    if result is None:
        raise HTTPException(404, "draft_not_found")
    return result


@router.get("/drafts/{draft_id}")
def get_draft(draft_id: str, actor=Depends(require_reviewer), service=Depends(get_feedback_service)):
    return feedback_call(lambda: service.get(draft_id, actor))


@router.post("/drafts/{draft_id}/confirm")
def confirm_draft(draft_id: str, payload: DraftConfirmation, actor=Depends(require_admin),
                  service=Depends(get_feedback_service)):
    return feedback_call(lambda: service.confirm(draft_id, actor, payload.content_hash))


@router.get("/drafts/{draft_id}/audit")
def draft_audit(draft_id: str, actor=Depends(require_reviewer), service=Depends(get_feedback_service)):
    return feedback_call(lambda: service.history(draft_id, actor))


@router.post("/drafts/{draft_id}/publish")
def publish_draft(draft_id: str, actor=Depends(require_admin), service=Depends(get_feedback_service)):
    return feedback_call(lambda: service.publish(draft_id, actor))


@router.post("/drafts/{draft_id}/recover")
def recover_draft(draft_id: str, actor=Depends(require_admin), service=Depends(get_feedback_service)):
    return feedback_call(lambda: service.recover(draft_id, actor))
