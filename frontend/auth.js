"use strict";
const CoachAuth = {
  token: localStorage.getItem("coach_token") || "",
  generation: 0,
  headers(headers = {}) {
    const result = new Headers(headers);
    if (this.token) result.set("Authorization", "Bearer " + this.token);
    return result;
  },
  save(token) { this.generation++; this.token = token; localStorage.setItem("coach_token", token); },
  clear(reason = "expired") {
    this.generation++;
    this.token = "";
    localStorage.removeItem("coach_token");
    const event = new Event("coach-unauthorized");
    event.reason = reason;
    window.dispatchEvent(event);
  },
  async request(path, opts = {}) {
    const generation = this.generation;
    const headers = this.headers(opts.headers);
    const res = await fetch("/api/v1" + path, {...opts, headers});
    if (generation !== this.generation) throw new Error("登录状态已变化，请重新操作");
    if (!res.ok) {
      let message = res.statusText || "请求失败";
      if (res.status === 401) {
        this.clear();
        message = "登录已失效，请重新登录";
      } else if (res.status === 403) {
        message = "没有权限执行此操作";
      } else {
        try {
          const body = await res.json();
          if (typeof body.detail === "string") message = body.detail;
        } catch (_) { /* keep safe status text */ }
      }
      const error = new Error(message);
      error.status = res.status;
      throw error;
    }
    return res;
  },
  async json(path, opts = {}) {
    const generation = this.generation;
    const headers = new Headers(opts.headers || {});
    if (!headers.has("Content-Type") && !(opts.body instanceof FormData)) {
      headers.set("Content-Type", "application/json");
    }
    const res = await this.request(path, {...opts, headers});
    const body = res.status === 204 ? null : await res.json();
    if (generation !== this.generation) throw new Error("登录状态已变化，请重新操作");
    return body;
  },
};
