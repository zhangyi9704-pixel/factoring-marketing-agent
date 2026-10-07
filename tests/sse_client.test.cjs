const {test} = require('node:test');
const assert = require('node:assert/strict');
const {SSEParser, streamSSE} = require('../frontend/sse.js');

test('every UTF-8 byte split, CRLF, multiline data, all six events', () => {
  const events = [], parser = new SSEParser((...a) => events.push(a));
  const source = ':heartbeat\r\nevent: delta\r\ndata: {\r\ndata: "text": "中文医院"}\r\n\r\n' +
    ['stage','section','profile','done','error'].map(e => `event: ${e}\ndata: {}\n\n`).join('');
  const decoder = new TextDecoder('utf-8', {fatal:true});
  for (const byte of new TextEncoder().encode(source)) parser.push(decoder.decode(Uint8Array.of(byte),{stream:true}));
  parser.push(decoder.decode());
  assert.equal(events[0][1].text, '中文医院');
  assert.deepEqual(events.map(e => e[0]), ['delta','stage','section','profile','done','error']);
});

function mock(parts, inspect) {
  global.fetch = async (url, opts) => {
    if (inspect) inspect(opts);
    return new Response(new ReadableStream({start(c) {
      for (const s of parts) c.enqueue(new TextEncoder().encode(s));
      c.close();
    }}), {headers:{'Content-Type':'text/event-stream'}});
  };
}

test('done completes, supplied authentication headers preserved', async () => {
  mock(['event: delta\ndata: {"text":"片段"}\n\n','event: done\ndata: {"turn":1}\n\n'], o => {
    assert.equal(o.headers.get('Authorization'), 'Bearer test-only');
  });
  const events = [];
  await streamSSE('/test', {headers:{Authorization:'Bearer test-only'}}, e => events.push(e));
  assert.deepEqual(events, ['delta','done']);
});
test('partial response without done is incomplete', async () => {
  mock(['event: delta\ndata: {"text":"部分回复"}\n\n']);
  await assert.rejects(streamSSE('/test', {}, () => {}), /未收到完成确认/);
});
test('error is terminal failure, not completed', async () => {
  mock(['event: error\ndata: {"detail":"session_busy"}\n\n']);
  await assert.rejects(streamSSE('/test', {}, () => {}), /session_busy/);
});
test('AbortController signal reaches fetch and cancels', async () => {
  const controller = new AbortController();
  global.fetch = (url, opts) => new Promise((resolve,reject) => {
    assert.equal(opts.signal, controller.signal);
    opts.signal.addEventListener('abort', () => reject(new DOMException('cancelled','AbortError')));
  });
  const pending = streamSSE('/test', {signal:controller.signal}, () => {});
  controller.abort();
  await assert.rejects(pending, {name:'AbortError'});
});
