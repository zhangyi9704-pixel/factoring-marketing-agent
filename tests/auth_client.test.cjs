const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

function runtime() {
  const storage = new Map(), listeners = new Map();
  const context = vm.createContext({Headers, Event, AbortController, TextDecoder, FormData, console,
    localStorage: {getItem: k => storage.get(k), setItem: (k,v) => storage.set(k,v), removeItem: k => storage.delete(k)},
    window: {addEventListener: (k,v) => listeners.set(k,v), dispatchEvent: e => listeners.get(e.type)?.(e),
      location: {reload: () => {}}, Vue: {createApp: () => ({mount: () => {}})}}, fetch: async () => new Response('{}'),
  });
  for (const file of ['auth.js','review.js','app.js','sse.js']) {
    vm.runInContext(fs.readFileSync(path.join(__dirname,'../frontend',file),'utf8'), context);
  }
  return {context, storage, run: code => vm.runInContext(code,context)};
}

test('JSON, multipart and SSE share bearer identity', async () => {
  const r = runtime();
  r.run('CoachAuth.save("test-token")');
  assert.equal(r.run('CoachAuth.headers().get("Authorization")'), 'Bearer test-token');
  let seen;
  r.context.fetch = async (_,opts) => { seen = opts; return new Response('{}'); };
  await r.run('api("/auth/me")');
  assert.equal(seen.headers.get('Authorization'), 'Bearer test-token');
  r.run('globalThis.state = App.data(); Object.assign(state,App.methods)');
  r.run('state.auth.user = {id:1,role:"admin"}');
  r.context.file = new Blob(['test'], {type:'text/plain'});
  r.run('state.$refs = {file:{files:[file]}}');
  await r.run('state.uploadDoc()');
  assert.equal(seen.headers.get('Authorization'), 'Bearer test-token');
  assert.equal(seen.headers.get('Content-Type'), null); // Browser must generate boundary.
  r.context.fetch = async (_,opts) => {
    seen = opts;
    return new Response('event: done\ndata: {}\n\n', {headers: {'Content-Type':'text/event-stream'}});
  };
  await r.run('streamSSE("/sessions/owned/messages/stream", {}, () => {})');
  assert.equal(seen.headers.get('Authorization'), 'Bearer test-token');
});

test('401 clears token and previous user data before login', async () => {
  const r = runtime();
  r.run('globalThis.state = App.data(); Object.assign(state, App.methods)');
  await r.run('App.mounted.call(state)');
  r.run('CoachAuth.save("expired"); state.auth.user = {id:1}; state.brief.session = {session_id:"a"}; state.drill.messages = ["private"]; state.dashboard.score = {private:true}');
  r.context.fetch = async () => new Response('{"detail":"invalid_credentials"}', {status:401});
  await assert.rejects(r.run('api("/auth/me")'));
  assert.equal(r.run('CoachAuth.token'), '');
  assert.equal(r.storage.has('coach_token'), false);
  assert.equal(r.run('state.view'), 'login');
  assert.equal(r.run('state.brief.session'), null);
  assert.equal(r.run('state.drill.messages.length'), 0);
  assert.equal(r.run('state.dashboard.score'), null);
});

test('frontend identity follows authenticated user', () => {
  const r = runtime();
  assert.equal(r.run('App.methods.currentUserId.call({auth:{user:{id:7}},userId:"999"})'), 7);
});

test('403 uses one message and keeps login for JSON and SSE', async () => {
  const r = runtime();
  r.run('CoachAuth.save("valid")');
  r.context.fetch = async () => new Response('{"detail":"role_forbidden"}', {status:403});
  await assert.rejects(r.run('api("/knowledge/ingest")'), /没有权限/);
  await assert.rejects(r.run('streamSSE("/users/2/scores/stream", {}, () => {})'), /没有权限/);
  assert.equal(r.run('CoachAuth.token'), 'valid');
});

test('SSE 401 clears token and sends the same expiry message', async () => {
  const r = runtime();
  r.run('CoachAuth.save("expired")');
  r.context.fetch = async () => new Response('{"detail":"invalid_credentials"}', {status:401});
  await assert.rejects(r.run('streamSSE("/users/1/scores/stream", {}, () => {})'), /登录已失效/);
  assert.equal(r.run('CoachAuth.token'), '');
});

test('role gates distinguish maintenance, review and personal dashboard', () => {
  const r = runtime();
  for (const [role,maintenance,review,label] of [['sales',false,false,'个人成绩'],['coach',false,true,'主管看板'],['admin',true,true,'主管看板']]) {
    r.context.role = role;
    r.run('globalThis.userState = {auth:{user:{role}}}');
    assert.equal(r.run('App.computed.canMaintainKnowledge.call(userState)'), maintenance);
    assert.equal(r.run('App.computed.canReview.call(userState)'), review);
    assert.equal(r.run('App.computed.dashboardLabel.call(userState)'), label);
  }
});

test('late JSON response cannot restore previous user data after logout', async () => {
  const r = runtime();
  r.run('CoachAuth.save("old")');
  let finish;
  r.context.fetch = () => new Promise(resolve => { finish = resolve; });
  const pending = r.run('api("/cases")');
  r.run('CoachAuth.clear("logout"); CoachAuth.save("new")');
  finish(new Response('[{"private":"old-user"}]'));
  await assert.rejects(pending, /登录状态已变化/);
  assert.equal(r.run('CoachAuth.token'), 'new');
});

test('late 401 from previous login cannot sign out new user', async () => {
  const r = runtime();
  r.run('CoachAuth.save("old")');
  let finish;
  r.context.fetch = () => new Promise(resolve => { finish = resolve; });
  const pending = r.run('api("/cases")');
  r.run('CoachAuth.save("new")');
  finish(new Response('{}', {status:401}));
  await assert.rejects(pending, /登录状态已变化/);
  assert.equal(r.run('CoachAuth.token'), 'new');
});

test('logout removes both token copies and shows signed-out prompt', async () => {
  const r = runtime();
  r.run('globalThis.state = App.data(); Object.assign(state,App.methods)');
  await r.run('App.mounted.call(state)');
  r.run('CoachAuth.save("valid"); state.logout()');
  assert.equal(r.run('CoachAuth.token'), '');
  assert.equal(r.storage.has('coach_token'), false);
  assert.equal(r.run('state.view'), 'login');
  assert.equal(r.run('state.notice'), '已退出登录');
});

test('login keeps token in memory and storage and clears entered password', async () => {
  const r = runtime();
  r.run('globalThis.state = App.data(); Object.assign(state,App.methods); state.auth.username="sales_demo"; state.auth.password="fixture-password"');
  r.context.fetch = async url => new Response(url.endsWith('/auth/login') ?
    '{"access_token":"fixture-token","user":{"id":7,"username":"sales_demo","role":"sales"}}' : '[]');
  await r.run('state.login()');
  assert.equal(r.run('CoachAuth.token'), 'fixture-token');
  assert.equal(r.storage.get('coach_token'), 'fixture-token');
  assert.equal(r.run('state.auth.password'), '');
  assert.equal(r.run('state.userId'), '7');
  assert.equal(r.run('state.view'), 'brief');
});

test('Headers objects are preserved and token never enters URL', async () => {
  const r = runtime();
  r.run('CoachAuth.save("fixture-token")');
  let seenUrl, seenHeaders;
  r.context.fetch = async (url,opts) => { seenUrl=url; seenHeaders=opts.headers; return new Response('{}'); };
  await r.run('api("/auth/me", {headers:new Headers({"X-Test":"preserved"})})');
  assert.equal(seenHeaders.get('X-Test'), 'preserved');
  assert.equal(seenHeaders.get('Authorization'), 'Bearer fixture-token');
  assert.equal(seenUrl, '/api/v1/auth/me');
  assert.equal(seenUrl.includes('fixture-token'), false);
});
