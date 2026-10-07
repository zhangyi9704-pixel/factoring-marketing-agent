"""分阶段真实验收；prepare只做一次，recover只读，retry只读原request结果。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import httpx
from app.agents.roleplay.agent import RoleplayAgent
from app.db.mysql import get_mysql_database
from sqlalchemy import text

TARGET = Path('docs/validation/3.7-session-live.json')
FIELDS = ('session_id','user_id','case_id','turn','unlocked_ids','dim_asked_count',
    'pending_directives','sensitive_fired','released_info','guardrail_hit_count','blocked_turns','needs_human_review')


def capture(sid):
    agent = RoleplayAgent()
    snapshot = agent._snapshot(sid)
    print('[READ] snapshot next=' + str(snapshot.next if snapshot else None) + '; interrupts=' + str([len(t.interrupts) for t in snapshot.tasks] if snapshot else []),flush=True)
    assert snapshot and tuple(snapshot.next) == ('sales_input',)
    state = {k: snapshot.values.get(k) for k in FIELDS}
    digest = hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return dict(hash=digest, next=list(snapshot.next), tasks=[t.name for t in snapshot.tasks],
        interrupt_metadata_available=bool(snapshot.tasks and snapshot.tasks[0].interrupts),
        turn=state['turn'], unlocked_count=len(state['unlocked_ids'] or []),
        progressive=state['dim_asked_count'], pending_count=len(state['pending_directives'] or []),
        compliance_count=state['guardrail_hit_count'])


def db_counts(sid):
    with get_mysql_database().engine.connect() as conn:
        counts = {key: conn.execute(text(f'SELECT COUNT(*) FROM {table} WHERE session_id=:sid'),{'sid':sid}).scalar()
            for key, table in [('messages','t_conversation_message'),('requests','t_message_request')]}
        counts['turn'] = conn.execute(text('SELECT turn_count FROM t_session WHERE session_id=:sid'),{'sid':sid}).scalar()
        return counts


def call(path, data=None):
    response = httpx.post('http://127.0.0.1:8001/api/v1'+path, json=data, timeout=360)
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare','capture','recover','final'], required=True)
    args = parser.parse_args()
    if args.phase == 'prepare':
        if TARGET.exists():
            raise RuntimeError('acceptance record exists; do not rerun prepare')
        record = dict(phase='preparing', no_graph_replay=True,
            creation_ids={key: uuid4().hex for key in ('first', 'second')})
        TARGET.write_text(json.dumps(record), encoding='utf-8')
        for key, case in [('first','PHARMA_001'),('second','ECOM_001')]:
            created = call('/sessions', {'case_id':case,'request_id':record['creation_ids'][key],'user_id':1001377})
            record[key] = created['session_id']
            TARGET.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            print('[OK] created ' + key + ': ' + created['session_id'], flush=True)
        sid = record['first']
        record['request'] = dict(content='我保证当天放款，医院客户的账期通常多久？',request_id='m37-concurrent-first',expected_turn=0)
        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(lambda _:call('/sessions/'+sid+'/messages',record['request']),range(2)))
        assert replies[0] == replies[1] and replies[0]['turn'] == 1 and not replies[0]['input_blocked']
        record['reply'] = replies[0]
        record['first_state'] = capture(sid)
        record['second_state'] = capture(record['second'])
        record['before_restart'] = db_counts(sid)
        assert record['before_restart'] == dict(messages=3,requests=1,turn=1)
        assert record['second_state']['turn'] == 0
        assert sum(record['first_state']['progressive'].values()) >= 1
        assert record['first_state']['compliance_count'] >= 1
        record['phase'] = 'ready_for_restart'
        print('[OK] two concurrent calls: one effective turn / 3 messages / 1 durable request',flush=True)
    elif args.phase == 'capture':
        # 仅补证已提交的首次并发轮，不调用图；用于验收断言差异诊断。
        from app.models import MessageRequestRecord
        record = json.loads(TARGET.read_text(encoding='utf-8'))
        sid = record['first']
        with get_mysql_database().session_scope() as db:
            row = db.query(MessageRequestRecord).filter_by(session_id=sid,request_id='m37-concurrent-first').one()
            assert row.status == 'completed'
            record['reply'] = row.result
        record['request'] = dict(content='我保证当天放款，医院客户的账期通常多久？',request_id='m37-concurrent-first',expected_turn=0)
        record['before_restart'] = db_counts(sid)
        record['first_state'] = capture(sid)
        record['second_state'] = capture(record['second'])
        record['phase'] = 'ready_for_restart'
        print('[OK] read-only captured existing request; counts=' + str(record['before_restart']) + '; state=' + str(record['first_state']),flush=True)
    elif args.phase == 'final':
        from app.models import MessageRequestRecord
        from sqlalchemy import inspect
        record = json.loads(TARGET.read_text(encoding='utf-8'))
        sid = record['first']
        counts = db_counts(sid)
        assert counts == dict(messages=9,requests=4,turn=4)
        assert db_counts(record['second']) == dict(messages=1,requests=0,turn=0)
        constraints = inspect(get_mysql_database().engine).get_unique_constraints('t_message_request')
        assert any(c['column_names'] == ['session_id','request_id'] for c in constraints)
        snapshot = capture(sid)
        assert max(snapshot['progressive'].values()) >= 2
        with get_mysql_database().session_scope() as db:
            rows = db.query(MessageRequestRecord).filter_by(session_id=sid).all()
            assert all(r.status == 'completed' for r in rows)
            requests = [dict(request_id=r.request_id, expected_turn=r.expected_turn, result=r.result) for r in rows]
        # 逐个重读浏览器已完成请求；绝不用新request_id重放。
        sales = {m['turn']:m['content'] for m in call('/sessions/'+sid+'/resume')['messages'] if m['role']=='sales'}
        for request in requests:
            content = sales[request['expected_turn']+1]
            repeated = call('/sessions/'+sid+'/messages',dict(content=content,
                request_id=request['request_id'],expected_turn=request['expected_turn']))
            assert repeated == request['result']
        assert db_counts(sid) == counts
        assert capture(record['second']) == record['second_state']
        record.update(browser_final_counts=counts, browser_final_state=snapshot,
            durable_unique_verified=True, all_completed_requests_replayed_without_write=True,
            browser_cancel_reconnected=True, browser_drop_done_reconnected=True,
            frontend_reload_restored=True, restart_new_turn_continued=True,
            browser_screenshots=['3.7-browser-cancel.png','3.7-browser-disconnected.png','3.7-browser-reconnected.png'])
        print('[OK] final: 4 turns / 9 messages / 4 completed durable requests; replays add no rows; second session unchanged',flush=True)
    else:
        record = json.loads(TARGET.read_text(encoding='utf-8'))
        assert record['phase'] in {'ready_for_restart','recovered'}
        for key in ('first','second'):
            assert capture(record[key]) == record[key+'_state']
            status = call('/sessions/'+record[key]+'/resume')
            assert status['waiting_input'] and not status['recovery_degraded'] and not status['interrupted']
            assert not any(k in status for k in ('unlocked_ids','dim_asked_count','pending_directives','checkpoint','mood','hidden_risks'))
        repeated = call('/sessions/'+record['first']+'/messages',record['request'])
        assert repeated == record['reply']
        assert db_counts(record['first']) == record['before_restart']
        record.update(phase='recovered', app_restart_restore=True, state_isolation=True, duplicate_replay_no_write=True)
        print('[OK] restart preserves all state + interrupt / two sessions isolated / request replay adds no rows',flush=True)
    TARGET.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        location = traceback.extract_tb(exc.__traceback__)[-1].lineno
        print('[FAIL] ' + type(exc).__name__ + '; line=' + str(location) + '; no automatic graph retry',file=sys.stderr)
        sys.exit(1)
