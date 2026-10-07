const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), vm = require('node:vm'), path = require('node:path');

function runtime(storage = new Map()) {
  const listeners = new Map(); let sequence = 0;
  const ctx = vm.createContext({Headers, Event, FormData, AbortController, TextDecoder, URLSearchParams, console,
    crypto: {randomUUID: () => `fixture-${++sequence}`},
    localStorage: {getItem: k => storage.get(k), setItem: (k,v) => storage.set(k,v), removeItem: k => storage.delete(k)},
    window: {addEventListener: (k,v) => listeners.set(k,v), dispatchEvent: e => listeners.get(e.type)?.(e),
      Vue: {createApp: () => ({mount: () => {}})}}, fetch: async () => new Response('{}')});
  for (const file of ['auth.js', 'review.js', 'app.js']) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../frontend', file), 'utf8'), ctx);
  }
  const run = code => vm.runInContext(code, ctx);
  run('globalThis.state=App.data(); Object.assign(state,App.methods); state.auth.user={id:1,role:"sales"}; state.userId="1"; state.brief.caseId=state.drill.caseId="INFRA_001"; CoachAuth.save("fixture")');
  return {ctx, run, storage};
}
const created = {session_id: 'sid', case_id: 'INFRA_001', turn: 0, public_case: {}, briefing: {}, opening: {content: '公开开场'}};
const history = [{role: 'customer', turn: 0, content: '公开开场'}, {role: 'sales', turn: 1, content: '核实'}];
const status = {session_id: 'sid', case_id: 'INFRA_001', turn: 1, messages: history,
  finished: false, interrupted: false, needs_human_review: false, finish_reason: ''};

for (const [section, method] of [['brief', 'startBriefing'], ['drill', 'startSession']]) {
  test(`${section}: unknown creation outcome survives reload with the same ID`, async () => {
    const first = runtime(), payloads = [];
    first.ctx.fetch = async (_, options) => {payloads.push(JSON.parse(options.body)); throw new TypeError('offline disconnect');};
    await first.run(`state.${method}()`);
    assert.match(first.run('state.notice'), /失败/);
    const saved = first.storage.get(`coach_create_1_${section}`);
    assert.ok(saved);
    const reloaded = runtime(first.storage);
    reloaded.ctx.fetch = async (url, options) => {
      assert.equal(url, '/api/v1/sessions');
      assert.equal(options.headers.get('Authorization'), 'Bearer fixture');
      payloads.push(JSON.parse(options.body)); return new Response(JSON.stringify(created));
    };
    await reloaded.run(`state.${method}()`);
    assert.deepEqual(payloads[1], payloads[0]);
    assert.equal(reloaded.storage.has(`coach_create_1_${section}`), false);
    assert.equal(reloaded.run(`state.${section}.session.session_id`), 'sid');
  });
  test(`${section}: a second click while initialization waits sends no extra POST`, async () => {
    const r = runtime(); let calls = 0, reply;
    r.ctx.fetch = () => {calls++; return new Promise(resolve => reply = resolve);};
    const pending = r.run(`state.${method}()`);
    await r.run(`state.${method}()`);
    assert.equal(calls, 1);
    reply(new Response(JSON.stringify(created))); await pending;
    assert.equal(r.run(`state.${section}.loading`), false);
  });
}

test('creation IDs isolate owners and cases; explicit reset starts a new operation', () => {
  const r = runtime();
  const original = JSON.parse(r.run('JSON.stringify(state.creationPayload("brief"))'));
  assert.equal(r.run('state.creationPayload("brief").request_id'), original.request_id);
  r.run('state.brief.caseId="PHARMA_001"');
  assert.notEqual(r.run('state.creationPayload("brief").request_id'), original.request_id);
  r.run('state.auth.user.id=2');
  const secondOwner = r.run('state.creationPayload("brief").request_id');
  assert.ok(r.storage.has('coach_create_1_brief') && r.storage.has('coach_create_2_brief'));
  r.run('state.resetBriefing()');
  assert.notEqual(r.run('state.creationPayload("brief").request_id'), secondOwner);
});

test('busy creation retains retry ID and confirmed initialization failure clears it', async () => {
  const r = runtime(), payloads = [];
  r.ctx.fetch = async (_, options) => {
    payloads.push(JSON.parse(options.body));
    return new Response(JSON.stringify({detail: payloads.length === 1 ? 'creation_busy' : 'session_initialization_failed'}), {status: 409});
  };
  await r.run('state.startSession()'); assert.ok(r.storage.has('coach_create_1_drill'));
  await r.run('state.startSession()'); assert.deepEqual(payloads[0], payloads[1]);
  assert.equal(r.storage.has('coach_create_1_drill'), false);
});

test('resume exposes the durable interrupted ID and reconcile restores history without replay', async () => {
  const r = runtime(), calls = []; r.run('state.drill.resumeId="sid";state.loadEvaluation=()=>{throw Error("unexpected score")};');
  r.ctx.fetch = async (url, options) => {
    calls.push([url, options]);
    return new Response(JSON.stringify(url.endsWith('/resume')
      ? {...status, interrupted: true, recovery_request_id: 'durable-request'}
      : {action: 'reconciled', status, result: {turn: 1}}));
  };
  await r.run('state.restoreSession()'); assert.equal(r.run('state.drill.recoveryRequestId'), 'durable-request');
  await r.run('state.reconcileMessage()');
  assert.equal(calls.length, 2);
  assert.equal(calls[1][0], '/api/v1/sessions/sid/reconcile');
  assert.equal(JSON.parse(calls[1][1].body).request_id, 'durable-request');
  assert.deepEqual(JSON.parse(r.run('JSON.stringify(state.drill.messages)')), history);
  assert.equal(r.run('state.drill.interrupted'), false);
  assert.equal(r.run('state.drill.recoveryRequestId'), null);
});

test('explicit abort from a restored ID retains history and never starts evaluation', async () => {
  const r = runtime(); let payload, calls = 0;
  r.run('state.drill.resumeId="sid";state.drill.evaluation={old:true};state.loadEvaluation=()=>{throw Error("unexpected score")};');
  r.ctx.fetch = async (_, options) => {
    calls++; payload = JSON.parse(options.body);
    return new Response(JSON.stringify({action: 'aborted', status: {...status, finished: true, needs_human_review: true, finish_reason: 'operator_aborted'}}));
  };
  await r.run('state.reconcileMessage("abort")');
  assert.equal(calls, 1); assert.equal(payload.action, 'abort'); assert.ok(payload.reason);
  assert.deepEqual(JSON.parse(r.run('JSON.stringify(state.drill.messages)')), history);
  assert.equal(r.run('state.drill.finished'), true); assert.equal(r.run('state.drill.evaluation'), null);
  assert.match(r.run('state.notice'), /历史记录保留/);
});

test('unverified reconciliation retains the original pending request and does not abort automatically', async () => {
  const r = runtime(); let calls = 0;
  r.run('state.drill.session={session_id:"sid"};state.drill.pending={request_id:"original",content:"核实",expected_turn:0};localStorage.setItem("coach_pending_sid",JSON.stringify(state.drill.pending))');
  r.ctx.fetch = async () => {calls++; return new Response('{"detail":"reconciliation_requires_abort"}', {status: 409});};
  await r.run('state.reconcileMessage()');
  assert.equal(calls, 1); assert.equal(r.run('state.drill.pending.request_id'), 'original');
  assert.ok(r.storage.has('coach_pending_sid')); assert.match(r.run('state.notice'), /明确终止/);
});

test('restoring an aborted session removes stale pending data and never loads a score', async () => {
  const r = runtime();
  r.run('state.drill.resumeId="sid";state.drill.evaluation={old:true};localStorage.setItem("coach_pending_sid",JSON.stringify({request_id:"old"}));state.loadEvaluation=()=>{throw Error("unexpected score")};');
  r.ctx.fetch = async () => new Response(JSON.stringify({...status, finished: true, finish_reason: 'operator_aborted'}));
  await r.run('state.restoreSession()');
  assert.equal(r.run('state.drill.pending'), null); assert.equal(r.run('state.drill.evaluation'), null);
  assert.equal(r.storage.has('coach_pending_sid'), false); assert.match(r.run('state.notice'), /已终止/);
});

test('new drill resets only a finished local view and creates a new request ID', () => {
  const r = runtime();
  const previousId = r.run('state.creationPayload("drill").request_id');
  r.run('state.drill.session={session_id:"sid"};state.drill.resumeId="sid";state.drill.pending={request_id:"old"};localStorage.setItem("coach_pending_sid","{}");localStorage.setItem("coach_session_1","sid");localStorage.setItem("coach_session_2","another-owner")');
  r.run('state.resetDrill()'); assert.equal(r.run('state.drill.session.session_id'), 'sid');
  r.run('state.drill.finished=true;state.resetDrill()');
  assert.equal(r.run('state.drill.session'), null); assert.equal(r.run('state.drill.pending'), null);
  assert.equal(r.run('state.drill.resumeId'), ''); assert.equal(r.run('state.drill.caseId'), 'INFRA_001');
  assert.equal(r.storage.has('coach_session_1'), false); assert.equal(r.storage.has('coach_pending_sid'), false);
  assert.equal(r.storage.get('coach_session_2'), 'another-owner');
  assert.notEqual(r.run('state.creationPayload("drill").request_id'), previousId);
});

test('late creation acknowledgement after logout cannot restore the previous session', async () => {
  const r = runtime();
  r.run('CoachAuth.token=""'); await r.run('App.mounted.call(state)');
  r.run('CoachAuth.save("fixture");state.auth.user={id:1,role:"sales"};state.userId="1"');
  let reply; r.ctx.fetch = () => new Promise(resolve => reply = resolve);
  const pending = r.run('state.startSession()');
  r.run('CoachAuth.clear("logout")'); reply(new Response(JSON.stringify(created))); await pending;
  assert.equal(r.run('state.drill.session'), null); assert.equal(r.run('state.view'), 'login');
  assert.equal(r.run('state.notice'), '已退出登录');
});
