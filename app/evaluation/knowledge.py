"""读取有效知识的版本快照，不调用模型，不读取案例内部事实。"""
from pathlib import Path
from app.config import PROJECT_ROOT
from app.models import KnowledgeDoc, KnowledgeDraft
from app.rag.integrity import document_hash
from app.rag.loader import KnowledgeLoader
from app.evaluation.contracts import SourceSnapshot, CoachingItem
from app.evaluation.common import DIM_LABELS


def active_sources(session_factory, industry):
    with session_factory() as session:
        rows = session.query(KnowledgeDoc).filter(KnowledgeDoc.industry.in_(['all', industry]),
            KnowledgeDoc.status == 'active').all()
        allowed = {(r.doc_id, r.version) for r in rows}
        snapshots = []
        supplied = set()
        for row in rows:
            if row.content_snapshot is None or not row.content_hash:
                continue  # Legacy metadata is considered below; never guess a body.
            metadata = {key: getattr(row, key) for key in
                        ('doc_id', 'category', 'industry', 'source', 'version', 'effective_date', 'status', 'topic')}
            # Feedback staging committed the same body with staged status; the
            # final publication changes only catalog status, not the body hash.
            matches = any(document_hash(dict(metadata, status=status), row.content_snapshot) == row.content_hash
                          for status in (row.status, 'staged'))
            if not matches:
                raise ValueError('committed_knowledge_snapshot_hash_mismatch')
            snapshots.append(SourceSnapshot.model_validate({key: metadata[key] for key in
                ('doc_id', 'version', 'source', 'industry', 'status')} | {'content': row.content_snapshot}).model_dump())
            supplied.add((row.doc_id, row.version))
        # Older feedback publications predate document body snapshots. Read
        # their durable human-authored SQL body, never an editable local path.
        drafts = session.query(KnowledgeDraft).filter(KnowledgeDraft.status == 'published').all()
        for draft in drafts:
            metadata = draft.metadata_json
            identity = (metadata['doc_id'], metadata['version'])
            if identity in allowed and identity not in supplied:
                from app.services.knowledge_feedback import KnowledgeFeedback
                if KnowledgeFeedback.digest(metadata, draft.content) != draft.content_hash:
                    raise ValueError('published_draft_hash_mismatch')
                snapshots.append(SourceSnapshot.model_validate({key: metadata[key] for key in
                    ('doc_id', 'version', 'source', 'industry')} | {'status': 'active', 'content': draft.content}).model_dump())
                supplied.add(identity)
    if allowed <= supplied:
        return snapshots
    # Compatibility for pre-migration baseline entries. New ingestions always
    # use the committed snapshot and cannot change through a source-file edit.
    for category in ('product', 'process', 'industry', 'compliance'):
        for path in (Path(PROJECT_ROOT) / 'data' / category).glob('*.md'):
            content, meta = KnowledgeLoader().load_document(path)
            identity = (meta['doc_id'], meta['version'])
            if identity in allowed and identity not in supplied and meta['status'] == 'active':
                snapshots.append(SourceSnapshot.model_validate({key: meta[key] for key in
                    ('doc_id', 'version', 'source', 'industry', 'status')} | {'content': content}).model_dump())
                supplied.add(identity)
    return snapshots


def coaching(evidence, sources):
    """事实是验证原话，建议是核实问题；不宣称客户存在未披露风险。"""
    result = []
    for dim, items in evidence.items():
        for item in items:
            ids = {'compliance': {'KB-CMP-001', 'KB-CMP-003'},
                'risk_awareness': {'KB-PR-003'}, 'risk_identification': {'KB-PR-003'},
                'requirement_discovery': {'KB-PR-002'}, 'business_understanding': {'KB-PR-002'}}.get(dim, {'KB-PR-002', 'KB-PR-003'})
            references = [s for s in sources if s['doc_id'] in ids]
            result.append(CoachingItem(dimension=dim, turn_id=item['turn'], original_text=item['quote_sales'],
                issue=f'{DIM_LABELS[dim]}：需把问题、客户答复和待核实事项逐项区分。',
                better_version='请确认这项信息的依据和目前尚待核实的部分；资料核验后再提交独立审批，不预先承诺额度、费率或时效。',
                training_suggestion=f'下一次练习围绕{DIM_LABELS[dim]}问一个具体问题，复述实际答复，再确认待核实事项。',
                sources=references).model_dump())
    return result
