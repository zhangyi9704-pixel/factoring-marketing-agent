"""复用已结束且尚无评估的脱敏真实演练，只计算图外复盘；绝不重放图。"""
from pathlib import Path
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import threading
from unittest.mock import patch
from sqlalchemy import text
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.db.mysql import get_mysql_database
from app.evaluation.service import EvaluationService
from app.services.llm_service import LLMService
from app.services.session_service import SessionService
from app.services.user_score_service import UserScoreService

TARGET = Path('docs/validation/evaluation-upgrade-live.json')


def count_rows(sid):
    with get_mysql_database().engine.connect() as conn:
        return conn.execute(text('SELECT COUNT(*) FROM t_evaluation WHERE session_id=:sid'), {'sid': sid}).scalar()


def deny(*args, **kwargs):
    raise AssertionError('unexpected model call on read/board')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', default='sess_40519c6f0906')
    parser.add_argument('--read-only', action='store_true')
    parser.add_argument('--output', type=Path, default=TARGET)
    args = parser.parse_args()
    session = SessionService().get_session(args.session)
    assert session and session['status'] == 'finished'
    before = count_rows(args.session)
    counter, mutex = Counter(), threading.Lock()
    if not args.read_only:
        if before:
            raise RuntimeError('report already exists: use --read-only; no recomputation')
        class Counting(LLMService):
            def chat(self, *args, **kwargs):
                with mutex:
                    counter[kwargs.get('model_role', 'reasoning')] += 1
                return super().chat(*args, **kwargs)
        llm = Counting()
        services = [EvaluationService(llm=llm), EvaluationService(llm=llm)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            reports = list(pool.map(lambda service: service.get(args.session), services))
        assert reports[0] == reports[1]
        report = reports[0]
        print('[OK] concurrent first evaluation completed', flush=True)
    else:
        assert before == 1
        with patch.object(LLMService, 'chat', deny), patch.object(LLMService, 'chat_json', deny), patch.object(LLMService, 'chat_json_once', deny):
            report = EvaluationService().get(args.session)
    after = count_rows(args.session)
    assert after == 1 and report['schema_version'] == '2.0'
    messages = SessionService().list_messages(args.session)
    for evidence in report['evidence']:
        assert any(m['role'] == 'sales' and m['turn'] == evidence['turn'] and m['content'] == evidence['quote_sales'] for m in messages)
    assert all(s['status'] == 'active' and s['version'] for s in report['sources'])
    with patch.object(LLMService, 'chat', deny), patch.object(LLMService, 'chat_json', deny), patch.object(LLMService, 'chat_json_once', deny):
        repeated = EvaluationService().get(args.session)
        reconnected = list(EvaluationService().run_stream(args.session))
        assert repeated == report and reconnected == [{'type': 'done', 'report': report}]
        board = UserScoreService()
        profile = board.scores(session['user_id'])
        assert profile['evaluated_sessions'] >= 1
        assert list(board.scores_stream(session['user_id']))[-1]['type'] == 'done'
    assert count_rows(args.session) == after
    record = dict(session_id=args.session, before=before, after=after,
        invocation_counts=dict(counter), no_graph_replay=True, verified_sales_evidence=True,
        repeat_report_same=True, sse_reconnect_same=True, dashboard_model_calls=0,
        report=report, fresh_process_read=args.read_only)
    if args.read_only and args.output.exists():
        prior = json.loads(args.output.read_text(encoding='utf-8'))
        assert prior['report'] == report
        prior['fresh_process_read'] = True
        record = prior
    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"[OK] evaluation rows={after}; evidence={len(report['evidence'])}; sources={len(report['sources'])}; degraded={report['degraded']}; repeat/SSE/board passed")


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('FAIL: ' + type(exc).__name__ + '; no graph or evaluation retry', file=sys.stderr)
        sys.exit(1)
