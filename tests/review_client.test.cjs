const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs'), vm = require('node:vm'), path = require('node:path');
function runtime() {
  const listeners = new Map(), storage = new Map();
  const ctx = vm.createContext({Headers,Event,FormData,AbortController,TextDecoder,URLSearchParams,console,
    localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},
    window:{addEventListener:(k,v)=>listeners.set(k,v),dispatchEvent:e=>listeners.get(e.type)?.(e),Vue:{createApp:()=>({mount:()=>{}})}},
    fetch:async()=>new Response('[]')});
  for (const file of ['auth.js','review.js','app.js']) vm.runInContext(fs.readFileSync(path.join(__dirname,'../frontend',file),'utf8'),ctx);
  const run = code=>vm.runInContext(code,ctx);
  run('globalThis.state=App.data(); Object.assign(state,App.methods); state.auth.user={id:2,role:"coach"}; CoachAuth.save("fixture")');
  for(const key of ['canReview','reviewOwned']) Object.defineProperty(ctx.state,key,{get:()=>run('App.computed.'+key+'.call(state)')});
  return {ctx,run};
}
test('review filters use UTC and no token in URL; page boundary preserves 25 rows',async()=>{
  const r=runtime();let url,options;
  r.ctx.fetch=async(u,o)=>{url=u;options=o;return new Response(JSON.stringify(Array.from({length:26},(_,i)=>({id:i+1}))));};
  r.run('state.review.filters.type="conflict"; state.review.filters.since="2026-10-06T08:00"');
  assert.equal(await r.run('state.loadReview(25)'),true);
  assert.equal(r.run('state.review.items.length'),25);assert.equal(r.run('state.review.hasNext'),true);
  const query=new URL('http://fixture'+url).searchParams;
  assert.equal(query.get('offset'),'25');assert.equal(query.get('type'),'conflict');assert.match(query.get('since'),/Z$/);
  assert.equal(options.headers.get('Authorization'),'Bearer fixture');assert.equal(query.has('token'),false);
});
test('invalid time range does not issue a request',async()=>{
  const r=runtime();let calls=0;r.ctx.fetch=async()=>{calls++;return new Response('[]');};
  r.run('state.review.filters.since="2026-10-07T00:00"; state.review.filters.until="2026-10-06T00:00"');
  await r.run('state.loadReview()');assert.equal(calls,0);assert.match(r.run('state.review.error'),/开始时间/);
});
test('claim and close target selected queue, do not accept client assignee, refresh list',async()=>{
  const r=runtime(),calls=[];let row={id:7,status:'pending',context:{},queue:'compliance'};
  r.ctx.fetch=async(u,o)=>{calls.push([u,o]); if(o.method==='PATCH'){const p=JSON.parse(o.body);row={...row,...p,assignee:2};return new Response(JSON.stringify(row));}return new Response('[]');};
  r.run('state.review.queue="compliance"; state.review.selected={id:7,status:"pending",queue:"compliance",context:{}}');
  await r.run('state.updateReview("processing")');
  assert.equal(calls[0][0],'/api/v1/compliance/pending/7');assert.equal(JSON.parse(calls[0][1].body).assignee,undefined);
  assert.match(r.run('state.review.feedback'),/领取成功.*列表已刷新/);
  r.run('state.review.resolution="  核查完成  "');await r.run('state.updateReview("resolved")');
  assert.equal(JSON.parse(calls[2][1].body).resolution,'核查完成');assert.match(r.run('state.review.feedback'),/待办已关闭/);
});
test('empty conclusion and another claimant cannot submit close',async()=>{
  const r=runtime();let calls=0;r.ctx.fetch=async()=>{calls++;return new Response('{}');};
  r.run('state.review.selected={id:1,status:"processing",assignee:2,queue:"knowledge"}; state.review.resolution="  "');
  await r.run('state.updateReview("resolved")');assert.match(r.run('state.review.error'),/填写处理结论/);
  r.run('state.review.resolution="done"; state.review.selected.assignee=3');await r.run('state.updateReview("dismissed")');assert.equal(calls,0);
});
test('409 is readable, no PATCH retry; sales cannot enter',async()=>{
  const r=runtime();let calls=0;r.ctx.fetch=async()=>{calls++;return new Response('{"detail":"pending_state_or_assignee_conflict"}',{status:409});};
  r.run('state.review.selected={id:1,status:"pending",queue:"knowledge"}');await r.run('state.updateReview("processing")');
  assert.equal(calls,1);assert.equal(r.run('state.review.selected'),null);assert.match(r.run('state.review.error'),/其他人领取/);
  r.run('state.auth.user.role="sales"; state.view="brief"');await r.run('state.enterReview()');assert.equal(r.run('state.view'),'brief');assert.equal(calls,1);
});
test('successful write with failed refresh is accurately reported',async()=>{
  const r=runtime();r.ctx.fetch=async(_,o)=>o.method==='PATCH'?new Response('{"id":1,"status":"processing","assignee":2}'):new Response('{}',{status:503});
  r.run('state.review.selected={id:1,status:"pending",queue:"knowledge"}');await r.run('state.updateReview("processing")');
  assert.match(r.run('state.review.feedback'),/领取成功.*刷新失败/);assert.equal(r.run('state.review.selected.status'),'processing');
});
test('401 clears queue and conclusion; late replies cannot restore previous user data',async()=>{
  const r=runtime();r.run('CoachAuth.token=""');await r.run('App.mounted.call(state)');
  r.run('CoachAuth.save("fixture")');
  r.run('state.review.selected={id:9};state.review.resolution="private"');
  r.ctx.fetch=async()=>new Response('{}',{status:401});await r.run('state.loadReview()');
  assert.equal(r.run('state.view'),'login');assert.equal(r.run('state.review.selected'),null);assert.equal(r.run('state.review.resolution'),'');
  r.run('state.auth.user={id:2,role:"coach"};CoachAuth.save("new")');let reply;
  r.ctx.fetch=()=>new Promise(resolve=>reply=resolve);const pending=r.run('state.loadReview()');
  r.run('CoachAuth.clear("logout")');reply(new Response('[{"id":99}]'));await pending;
  assert.equal(r.run('state.review.items.length'),0);
});
test('newer list wins and queue switch cancels old details',async()=>{
  const r=runtime();let reply;
  r.ctx.fetch=()=>new Promise(resolve=>reply=resolve);const first=r.run('state.loadReview()');
  r.ctx.fetch=async()=>new Response('[{"id":2}]');await r.run('state.loadReview()');reply(new Response('[{"id":1}]'));await first;
  assert.equal(r.run('state.review.items[0].id'),2);
  r.ctx.fetch=()=>new Promise(resolve=>reply=resolve);const detail=r.run('state.selectReview({id:2})');
  r.ctx.fetch=async()=>new Response('[]');r.run('state.review.queue="compliance"');await r.run('state.changeReviewQueue()');
  reply(new Response('{"id":2,"question":"old"}'));await detail;
  assert.equal(r.run('state.review.selected'),null);assert.equal(r.run('state.review.detailLoading'),false);
});

test('conflict closure saves human draft and never confirms or publishes automatically',async()=>{
  const r=runtime(),calls=[];
  r.run('state.review.selected={id:2,status:"processing",assignee:2,queue:"knowledge",type:"conflict",context:{items:[{doc_id:"K",version:"v1"},{doc_id:"K",version:"v2"}]}};state.review.resolution="核实";Object.assign(state.review.publication,{action:"adopt",adoptedKey:"1",content:"人工正文",source:"核实来源",topic:"主题"})');
  r.ctx.fetch=async(u,o)=>{calls.push([u,o]);if(o.method==='PATCH')return new Response('{"id":2,"status":"resolved","knowledge_draft":{"id":"abc","status":"draft"}}');if(u.includes('/drafts/'))return new Response('{"id":"abc","status":"draft"}');return new Response('[]');};
  await r.run('state.updateReview("resolved")');
  assert.deepEqual(JSON.parse(calls[0][1].body).feedback.adopted,{doc_id:'K',version:'v2'});
  assert.equal(r.run('state.review.draft.status'),'draft');
  assert.equal(calls.some(([u])=>u.endsWith('/confirm')||u.endsWith('/publish')),false);
});
test('only admin with both checks confirms; publication errors never retry',async()=>{
  const r=runtime(),calls=[];r.run('state.review.draft={id:"abc",status:"draft",content_hash:"hash"}');
  r.ctx.fetch=async(u,o)=>{calls.push([u,o]);return new Response('{}',{status:503});};
  await r.run('state.publishReviewDraft("confirm")');assert.equal(calls.length,0);
  r.run('state.auth.user.role="admin";state.review.contentConfirmed=true');
  await r.run('state.publishReviewDraft("confirm")');assert.equal(calls.length,0);
  r.run('state.review.sourceConfirmed=true');await r.run('state.publishReviewDraft("confirm")');
  assert.equal(calls.length,1);assert.match(r.run('state.review.error'),/勿自动重试/);
  assert.equal(calls[0][1].headers.get('Authorization'),'Bearer fixture');
  assert.deepEqual(JSON.parse(calls[0][1].body),{content_hash:'hash',content_confirmed:true,source_confirmed:true});
});
test('late confirmation after logout cannot restore a private draft',async()=>{
  const r=runtime();r.run('CoachAuth.token=""');await r.run('App.mounted.call(state)');
  r.run('CoachAuth.save("fixture");state.auth.user={id:3,role:"admin"};state.review.draft={id:"abc"};state.review.contentConfirmed=true;state.review.sourceConfirmed=true');
  let reply;r.ctx.fetch=()=>new Promise(resolve=>reply=resolve);const call=r.run('state.publishReviewDraft("confirm")');
  r.run('CoachAuth.clear("logout")');reply(new Response('{"id":"abc","status":"confirmed"}'));await call;
  assert.equal(r.run('state.review.draft'),null);assert.equal(r.run('state.view'),'login');
});

test('admin recovery is explicit, never publishes automatically and keeps unknown outcomes',async()=>{
  const r=runtime(),calls=[];
  r.run('state.review.draft={id:"abc",status:"reconcile_required"}');
  r.ctx.fetch=async(u,o)=>{calls.push([u,o]);return new Response('{"id":"abc","status":"confirmed"}');};
  await r.run('state.publishReviewDraft("recover")');assert.equal(calls.length,0);
  r.run('state.auth.user.role="admin"');await r.run('state.publishReviewDraft("recover")');
  assert.equal(calls.length,1);assert.equal(calls[0][0],'/api/v1/knowledge/drafts/abc/recover');
  assert.match(r.run('state.review.feedback'),/另行发布/);
  r.run('state.review.draft.status="publishing"');
  r.ctx.fetch=async(u,o)=>{calls.push([u,o]);return new Response('{}',{status:503});};
  await r.run('state.publishReviewDraft("recover")');
  assert.equal(calls.length,2);assert.equal(r.run('state.review.draft.status'),'publishing');
  assert.match(r.run('state.review.error'),/勿自动重试/);
});

test('admin release and transfer compare original assignee and require a reason',async()=>{
  const r=runtime(),calls=[];
  r.run('state.review.selected={id:7,status:"processing",assignee:4,queue:"compliance"};state.review.transferReason="离岗"');
  r.ctx.fetch=async(u,o)=>{calls.push([u,o]);if(o.method==='POST')return new Response('{"id":7,"status":"pending","assignee":null}');return new Response('[]');};
  await r.run('state.reassignReview()');assert.equal(calls.length,0);
  r.run('state.auth.user.role="admin"');await r.run('state.reassignReview()');
  assert.equal(calls[0][0],'/api/v1/compliance/pending/7/reassign');
  assert.deepEqual(JSON.parse(calls[0][1].body),{expected_assignee:4,assignee_id:null,reason:'离岗'});
  assert.match(r.run('state.review.feedback'),/已回收/);
  r.run('state.review.selected={id:8,status:"processing",assignee:5,queue:"knowledge"};state.review.transferReason="调整分工";state.review.transferAssignee="9"');
  await r.run('state.reassignReview()');
  assert.deepEqual(JSON.parse(calls[2][1].body),{expected_assignee:5,assignee_id:9,reason:'调整分工'});
  r.run('state.review.transferReason="";state.review.selected.status="processing"');
  await r.run('state.reassignReview()');assert.equal(calls.length,4);
});

test('stale admin reassignment does not overwrite or retry',async()=>{
  const r=runtime();let calls=0;
  r.run('state.auth.user.role="admin";state.review.selected={id:1,status:"processing",assignee:4,queue:"knowledge"};state.review.transferReason="回收"');
  r.ctx.fetch=async()=>{calls++;return new Response('{}',{status:409});};
  await r.run('state.reassignReview()');
  assert.equal(calls,1);assert.equal(r.run('state.review.selected'),null);assert.match(r.run('state.review.error'),/不自动重试/);
});
