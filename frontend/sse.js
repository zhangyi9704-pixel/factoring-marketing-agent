"use strict";

// 增量行解析：保留跨chunk CRLF、UTF-8由TextDecoder负责、多行data按换行拼接。
class SSEParser {
  constructor(onEvent) { this.onEvent = onEvent; this.buffer = ""; this.event = "message"; this.data = []; }
  push(text) {
    this.buffer += text;
    for (;;) {
      const i = this.buffer.search(/[\r\n]/);
      if (i < 0 || (this.buffer[i] === "\r" && i === this.buffer.length - 1)) return;
      const line = this.buffer.slice(0, i);
      const n = this.buffer[i] === "\r" && this.buffer[i + 1] === "\n" ? 2 : 1;
      this.buffer = this.buffer.slice(i + n);
      if (!line) {
        if (this.data.length) this.onEvent(this.event, JSON.parse(this.data.join("\n")));
        this.event = "message"; this.data = [];
      } else if (!line.startsWith(":")) {
        const colon = line.indexOf(":");
        const key = colon < 0 ? line : line.slice(0, colon);
        let value = colon < 0 ? "" : line.slice(colon + 1);
        if (value.startsWith(" ")) value = value.slice(1);
        if (key === "event") this.event = value;
        if (key === "data") this.data.push(value);
      }
    }
  }
}

async function streamSSE(path, opts = {}, onEvent) {
  const headers = new Headers(opts.headers || {});
  if (!headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const auth = typeof CoachAuth !== "undefined" ? CoachAuth : null;
  const generation = auth && auth.generation;
  const res = auth ? await auth.request(path, {...opts, headers}) : await fetch("/api/v1" + path, {...opts, headers});
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  if (!res.body) throw new Error("响应流不可用，本轮未完成");
  const reader = res.body.getReader(), decoder = new TextDecoder("utf-8", {fatal: true});
  let terminal = false;
  const parser = new SSEParser((event, data) => {
    if (auth && generation !== auth.generation) throw new Error("登录状态已变化，请重新操作");
    if (terminal) return;
    if (event === "error") throw new Error(data.detail || "本轮失败");
    onEvent(event, data);
    if (event === "done") terminal = true;
  });
  try {
    while (!terminal) {
      const {done, value} = await reader.read();
      if (done) { parser.push(decoder.decode()); break; }
      parser.push(decoder.decode(value, {stream: true}));
    }
    if (!terminal) throw new Error("网络断开，未收到完成确认；请重连核对原请求");
  } finally {
    await reader.cancel().catch(() => {});
    reader.releaseLock();
  }
}

if (typeof module !== "undefined") module.exports = {SSEParser, streamSSE};
