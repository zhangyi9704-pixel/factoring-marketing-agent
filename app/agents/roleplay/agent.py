"""教练 Agent 门面（单 Agent 架构，技术设计文档（新）§6）。

LangGraph 单图八节点（graphs/coach_graph.py：briefing + roleplay 循环）
+ Redis checkpoint（thread_id = session_id）。对外入口：
    - start_session：创建会话 → briefing（访前简报，§7）→ 图跑到
      sales_input 的 interrupt 挂起，返回 public_case + 简报 + 客户开场白；
    - send_message：Command(resume=话术) 续跑一轮，返回客户回复
      （guardrail 命中以规则名提醒，内部字段一律不出 Agent 边界）；
    - get_status / finish_session：checkpoint 快照与循环出口。
"""
from __future__ import annotations

import logging
from typing import Any, Optional
from contextlib import nullcontext
from uuid import uuid4

from app.config import Settings, get_settings
from app.schemas.case import assert_no_internal_leak

logger = logging.getLogger(__name__)


class RoleplayAgentError(RuntimeError):
    """roleplay 会话错误（API 层翻译 404/409）。"""


class RoleplayAgent:
    """教练 Agent 门面（briefing 走 reasoning、customer_simulator 走 chat）。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        llm: Any = None,
        case_service: Any = None,
        guardrail: Any = None,
        session_service: Any = None,
        retriever: Any = None,
        checkpointer: Any = None,
        request_store: Any = None,
        creation_store: Any = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._injected = {
            "llm": llm, "case_service": case_service,
            "guardrail": guardrail, "session_service": session_service,
            "retriever": retriever,
        }
        self._llm = llm
        self._case_service = case_service
        self._guardrail = guardrail
        self._session_service = session_service
        self._retriever = retriever
        self._checkpointer = checkpointer
        self._graph = None
        self._request_store = request_store
        self._creation_store = creation_store
        self._bindings = {}

    # -------------------------------------------------
    # 懒加载依赖（真实组件首次使用时才构造，便于测试注入替身）
    # -------------------------------------------------

    @property
    def llm(self):
        if self._llm is None:
            from app.services.llm_service import LLMService
            self._llm = LLMService(self.settings)
        return self._llm

    @property
    def case_service(self):
        if self._case_service is None:
            from app.services.case_service import CaseService
            self._case_service = CaseService(self.settings)
        return self._case_service

    @property
    def guardrail(self):
        if self._guardrail is None:
            from app.guardrail.detector import GuardrailService
            from app.compliance.engine import ComplianceEngine
            self._guardrail = ComplianceEngine(self.settings, llm=self.llm)
        return self._guardrail

    @property
    def session_service(self):
        if self._session_service is None:
            from app.services.session_service import SessionService
            self._session_service = SessionService(self.settings)
        return self._session_service

    @property
    def checkpointer(self):
        if self._checkpointer is None:
            from app.graphs.coach_graph import build_redis_checkpointer
            self._checkpointer = build_redis_checkpointer(self.settings)
        return self._checkpointer

    @property
    def graph(self):
        if self._graph is None:
            from app.graphs.coach_graph import build_coach_graph
            self._graph = build_coach_graph(
                self.llm,
                case_service=self.case_service,
                guardrail=self.guardrail,
                settings=self.settings,
                session_service=self.session_service,
                retriever=self.retriever,
                checkpointer=self.checkpointer,
            )
        return self._graph

    @property
    def retriever(self):
        if self._retriever is None:
            from app.services.public_retrieval import get_public_retriever
            self._retriever = get_public_retriever(self.settings)
        return self._retriever

    def _config(self, session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    @property
    def requests(self):
        if self._request_store is None:
            from app.services.message_requests import DatabaseRequests, MemoryRequests
            if self._injected['session_service'] is not None and not hasattr(self.session_service, 'session_factory'):
                self._request_store = MemoryRequests()
            else:
                from app.db.mysql import get_mysql_database
                self._request_store = DatabaseRequests(get_mysql_database())
        return self._request_store

    @property
    def creations(self):
        if self._creation_store is None:
            from app.services.message_requests import DatabaseCreations, MemoryCreations
            if self._injected['session_service'] is not None and not hasattr(self.session_service, 'session_factory'):
                self._creation_store = MemoryCreations()
            else:
                from app.db.mysql import get_mysql_database
                self._creation_store = DatabaseCreations(get_mysql_database())
        return self._creation_store

    def _binding(self, sid):
        if hasattr(self.session_service, 'get_session'):
            row = self.session_service.get_session(sid)
        else:
            row = self._bindings.get(sid)  # 显式替身测试路径
        if row is None:
            raise RoleplayAgentError('会话不存在: ' + sid)
        return row

    # -------------------------------------------------
    # 会话入口
    # -------------------------------------------------

    def start_session(
        self, case_id: str, user_id: int, *, session_id: Optional[str] = None, request_id=None,
    ) -> dict:
        """One creation ID owns one initialization; retries never rerun an uncertain graph."""
        from app.services.message_requests import MessageConflict, MemoryCreations
        rid = request_id or uuid4().hex  # Legacy internal callers; HTTP sends a durable ID.
        try:
            with self.creations.lock(f'create:{user_id}:{rid}'):
                existing = self.creations.get(user_id, rid)
                if existing:
                    if existing['case_id'] != case_id:
                        raise RoleplayAgentError('creation_request_id_conflict')
                    if existing['status'] == 'completed':
                        return existing['result']
                    if existing['status'] == 'failed':
                        raise RoleplayAgentError('session_initialization_failed')
                    if existing['status'] == 'aborted':
                        raise RoleplayAgentError('session_initialization_aborted')
                    return self._recover_creation(user_id, rid, existing)
                # Preflight before creating SQL rows, checkpoints or model work.
                if self.case_service.get_case(case_id) is None:
                    raise RoleplayAgentError('案例不存在: ' + case_id)
                sid = session_id or f'sess_{uuid4().hex}'
                if hasattr(self.session_service, 'get_session') and self.session_service.get_session(sid):
                    raise RoleplayAgentError('session_already_exists')
                self.creations.begin(user_id, rid, case_id, sid)
                self._bindings[sid] = dict(session_id=sid, user_id=user_id, case_id=case_id, status='initializing')
                try:
                    if isinstance(self.creations, MemoryCreations):
                        self.session_service.create_session(sid, user_id, case_id)
                    values = self.graph.invoke(dict(session_id=sid, user_id=user_id, case_id=case_id), self._config(sid))
                    if values.get('finish_reason') == 'error':
                        raise RoleplayAgentError('session_initialization_failed')
                    result = self._created_result(sid, case_id, user_id, values)
                    self.creations.complete(user_id, rid, result)
                    self._bindings[sid]['status'] = 'active'
                    return result
                except Exception:
                    # Check a lost acknowledgement before failing an already committed creation.
                    saved = self.creations.get(user_id, rid)
                    if saved and saved['status'] == 'completed':
                        return saved['result']
                    self.creations.fail(user_id, rid)
                    self._bindings[sid]['status'] = 'failed'
                    raise RoleplayAgentError('session_initialization_failed') from None
        except MessageConflict:
            raise RoleplayAgentError('creation_busy') from None

    def _recover_creation(self, uid, rid, existing):
        """Finalize only a fully persisted initialization; never replay partial nodes."""
        sid = existing['session_id']
        row = self._binding(sid)
        snapshot = self.graph.get_state(self._config(sid))
        values = snapshot.values if snapshot else {}
        matching = (row.get('status') == 'initializing' and values and tuple(snapshot.next) == ('sales_input',) and values.get('initialized')
            and values.get('turn') == row.get('turn_count', 0) == 0
            and all(values.get(k) == row.get(k) for k in ('session_id', 'user_id', 'case_id'))
            and isinstance(values.get('briefing'), dict))
        if matching and hasattr(self.session_service, 'list_messages'):
            matching = (self._visible_messages(values.get('conversation', [])) ==
                self._visible_messages(self.session_service.list_messages(sid))
                and self.session_service.get_briefing(sid) == values['briefing'])
        if not matching:
            self.creations.fail(uid, rid)
            raise RoleplayAgentError('session_initialization_failed')
        result = self._created_result(sid, row['case_id'], uid, values)
        self.creations.complete(uid, rid, result)
        return result

    def _created_result(self, session_id, case_id, user_id, result):
        public_dict = result.get('public_case')
        if public_dict is not None:
            assert_no_internal_leak(public_dict)          # 双视角二次校验（§19.4）
        briefing = result.get("briefing")
        if briefing is not None:
            assert_no_internal_leak(briefing)             # 简报同口径二次校验（§7.3）
        opening = next(
            (m for m in result.get("conversation") or [] if m.get("turn") == 0), None,
        )
        return {
            "session_id": session_id, "case_id": case_id, "user_id": user_id,
            "public_case": public_dict, "briefing": briefing, "opening": opening,
            "turn": result.get("turn", 0),
            "knowledge_mode": result.get("knowledge_mode", "missing"),
        }

    def send_message(self, session_id: str, content: str, *, request_id=None, expected_turn=None) -> dict:
        """注入销售话术续跑一轮（interrupt 处 resume）。"""
        for event in self.send_message_stream(session_id, content, request_id=request_id, expected_turn=expected_turn):
            if event['type'] == 'done':
                return {k: v for k, v in event.items() if k != 'type'}
        raise RoleplayAgentError('message_incomplete')

    def send_message_stream(self, session_id: str, content: str, *, request_id=None, expected_turn=None):
        """send_message 的流式版本（生成器）。

        yield {"type": "delta", "text": ...}（customer_reply 节点经 custom
        stream 通道上抛的回复增量）→ 最后 yield {"type": "done", ...}
        （与 send_message 返回值同构）。
        """
        from app.services.message_requests import payload_hash, replay, MessageConflict
        from app.evaluation.jobs import stream_job
        self._binding(session_id)
        rid = request_id or uuid4().hex  # 旧内部调用兼容；HTTP/前端固定显式ID
        base = expected_turn
        if base is None:
            existing = self.requests.get(session_id, rid)
            base = existing['expected_turn'] if existing else (self._snapshot(session_id).values.get('turn', 0) if self._snapshot(session_id) else 0)
        fingerprint = payload_hash(content, base)
        try:
            existing = self.requests.get(session_id, rid)
            if existing and existing['status'] == 'completed':
                yield {'type': 'done', **replay(existing, fingerprint)}
                return
            if existing and existing['payload_hash'] != fingerprint:
                raise MessageConflict('request_id_conflict')
            # 相同输入共享任务，客户端断开不会取消生产者；跨进程由DB锁及账本兜底。
            key = ('roleplay', id(self.requests) if type(self.requests).__name__ == 'MemoryRequests' else 'mysql', session_id, rid, fingerprint)
            yield from stream_job(key, lambda: self._message_owned(session_id, rid, content, base, fingerprint))
        except MessageConflict as exc:
            raise RoleplayAgentError(str(exc)) from None

    def _message_owned(self, sid, rid, content, base, fingerprint):
        from langgraph.types import Command
        from app.services.message_requests import replay, MessageConflict
        from app.schemas.session import TurnResponse
        with self.requests.lock(sid):
            existing = self.requests.get(sid, rid)
            if existing:
                yield {'type': 'done', **replay(existing, fingerprint)}
                return
            self._require_active(sid)
            snapshot = self._snapshot(sid)
            if snapshot.values.get('turn', 0) != base:
                raise MessageConflict('stale_turn')
            if self.requests.unfinished(sid):
                raise MessageConflict('session_interrupted')
            self.requests.begin(sid, rid, fingerprint, base)
            try:
                command = Command(resume=content, update=dict(active_request_id=rid,
                    active_payload_hash=fingerprint, active_expected_turn=base))
                for payload in self.graph.stream(command, self._config(sid), stream_mode='custom'):
                    if isinstance(payload, dict) and payload.get('kind') == 'customer_delta':
                        yield {'type': 'delta', 'text': str(payload.get('text') or '')}
                values = self.graph.get_state(self._config(sid)).values or {}
                result = TurnResponse(**self._turn_result(values), request_id=rid).model_dump()
                if hasattr(self.requests, 'prepare_result'):  # In-memory substitutes only.
                    self.requests.prepare_result(sid, rid, result)
                self.requests.complete(sid, rid, result)
            except Exception:
                self.requests.fail(sid, rid)
                raise RoleplayAgentError('message_execution_failed') from None
            yield {'type': 'done', **result}

    @staticmethod
    def _turn_result(result: dict) -> dict:
        """图终态 → TurnResponse 字段映射（流式/非流式共用）。"""
        from app.services.message_requests import turn_result
        return turn_result(result)

    @staticmethod
    def _visible_messages(messages):
        return [{k: m[k] for k in ('role', 'turn', 'content')} for m in messages]

    def reconcile_session(self, sid, rid, *, actor_id, action='reconcile', reason=''):
        """Owner/admin authorized by the route; named lock excludes live producers."""
        from app.schemas.session import TurnResponse
        from app.services.message_requests import MessageConflict
        try:
            with self.requests.lock(sid):
                row = self._binding(sid)
                if action == 'abort':
                    creation = self.creations.for_session(sid)
                    creation_lock = (self.creations.lock(f"create:{creation['user_id']}:{creation['request_id']}")
                                     if creation else nullcontext())
                    # Initialization uses a different lock from ordinary turns;
                    # take both before ending it so a live initializer cannot revive it.
                    with creation_lock:
                        row = self._binding(sid)
                        if row.get('status') == 'finished':
                            raise RoleplayAgentError('finished_session_cannot_abort')
                        if hasattr(self.session_service, 'abort_session'):
                            self.session_service.abort_session(sid, actor_id, reason)
                        else:
                            self._bindings[sid]['status'] = 'aborted'
                            self.requests.abort(sid)
                            self.creations.abort(sid)
                    return {'action': 'aborted', 'status': self.get_status(sid), 'result': None}
                record = self.requests.get(sid, rid)
                if record is None:
                    raise RoleplayAgentError('request_not_found')
                if record['status'] == 'completed':
                    return {'action': 'reconciled', 'status': self.get_status(sid), 'result': record['result']}
                snapshot = self._snapshot(sid)  # Identity + SQL turn must match.
                values = snapshot.values if snapshot else {}
                if (not values or tuple(snapshot.next) not in (('sales_input',), ())
                    or values.get('active_request_id') != rid
                    or values.get('active_payload_hash') != record['payload_hash']
                    or values.get('active_expected_turn') != record['expected_turn']
                    or record['status'] not in {'running', 'failed'}):
                    raise RoleplayAgentError('reconciliation_requires_abort')
                expected = TurnResponse(**self._turn_result(values), request_id=rid).model_dump()
                if record.get('result') != expected:
                    raise RoleplayAgentError('reconciliation_requires_abort')
                if hasattr(self.session_service, 'list_messages'):
                    if self._visible_messages(values.get('conversation', [])) != self._visible_messages(self.session_service.list_messages(sid)):
                        raise RoleplayAgentError('reconciliation_requires_abort')
                self.requests.reconcile(sid, rid, expected, actor_id)
                return {'action': 'reconciled', 'status': self.get_status(sid), 'result': expected}
        except MessageConflict as exc:
            raise RoleplayAgentError(str(exc)) from None

    def finish_session(self, session_id: str) -> dict:
        """结束演练：仍在等待输入则注入 [结束]，幂等返回最终状态。"""
        from app.agents.roleplay.state_judge import END_TOKEN
        if self._binding(session_id).get('status') in {'aborted', 'failed'}:
            return self.get_status(session_id)
        snapshot = self._snapshot(session_id)
        if snapshot is None:
            raise RoleplayAgentError(f"会话不存在: {session_id}")
        if snapshot.next:
            self.send_message(session_id, END_TOKEN, request_id='finish', expected_turn=None)
        return self.get_status(session_id)

    def get_status(self, session_id: str) -> dict:
        """checkpoint 快照（resume 端点数据源）。"""
        row = self._binding(session_id)
        if row.get('status') in {'aborted', 'failed'}:
            messages = self.session_service.list_messages(session_id) if hasattr(self.session_service, 'list_messages') else []
            return dict(session_id=session_id, case_id=row['case_id'], turn=row.get('turn_count', 0),
                finished=True, finish_reason='operator_aborted' if row['status'] == 'aborted' else 'initialization_failed',
                waiting_input=False, needs_human_review=True, interrupted=False,
                messages=self._visible_messages(messages))
        snapshot = self._snapshot(session_id)
        if snapshot is None:
            raise RoleplayAgentError(f"会话不存在: {session_id}")
        values = snapshot.values or {}
        return {
            "session_id": session_id,
            "case_id": values.get("case_id", ""),
            "turn": values.get("turn", 0),
            "finished": bool(values.get("finished")),
            "finish_reason": values.get("finish_reason", ""),
            "waiting_input": tuple(snapshot.next) == ('sales_input',),
            "needs_human_review": bool(values.get("needs_human_review")),
            "recovery_degraded": type(self.checkpointer).__module__.startswith('langgraph.checkpoint.memory'),
            "interrupted": bool(snapshot.next) and tuple(snapshot.next) != ('sales_input',) or self.requests.unfinished(session_id),
            "recovery_request_id": self.requests.first_unfinished(session_id),
            "messages": [{k: m[k] for k in ('role', 'turn', 'content')} for m in values.get('conversation', [])],
        }

    # -------------------------------------------------
    # 内部
    # -------------------------------------------------

    def _snapshot(self, session_id: str):
        row = self._binding(session_id)
        snapshot = self.graph.get_state(self._config(session_id))
        if snapshot and snapshot.values:
            if any(snapshot.values.get(k) != row.get(k) for k in ('session_id', 'case_id', 'user_id')):
                raise RoleplayAgentError('checkpoint_binding_mismatch')
            if 'turn_count' in row and snapshot.values.get('turn', 0) != row['turn_count']:
                raise RoleplayAgentError('checkpoint_persistence_mismatch')
        return snapshot if (snapshot and snapshot.values) else None

    def _require_active(self, session_id: str) -> None:
        row = self._binding(session_id)
        if row.get('status') in {'aborted', 'failed', 'initializing'}:
            raise RoleplayAgentError('session_not_active')
        snapshot = self._snapshot(session_id)
        if snapshot is None:
            raise RoleplayAgentError(f"会话不存在: {session_id}")
        if not snapshot.next:
            raise RoleplayAgentError("会话已结束，无法继续发送消息")
        if tuple(snapshot.next) != ('sales_input',):
            raise RoleplayAgentError('checkpoint_interrupted; manual_review_required')
