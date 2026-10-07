"use strict";
const CoachReview = {
  types: {missing: "知识缺失", conflict: "知识冲突", out_of_scope: "超出知识范围", compliance: "合规升级"},
  statuses: {pending: "待领取", processing: "处理中", resolved: "已解决", dismissed: "已忽略"},
  state() {
    return {queue: "knowledge", filters: {type: "", status: "pending", since: "", until: ""},
      items: [], selected: null, resolution: "", loading: false, detailLoading: false, saving: false,
      offset: 0, hasNext: false, listSeq: 0, detailSeq: 0, feedback: "", error: "",
      transferAssignee: "", transferReason: "",
      publication: this.publication(), draft: null, contentConfirmed: false, sourceConfirmed: false};
  },
  publication() {
    const today = new Date(), date = [today.getFullYear(), String(today.getMonth()+1).padStart(2,"0"), String(today.getDate()).padStart(2,"0")].join("-");
    return {action: "", content: "", source: "", category: "product", industry: "all",
      topic: "", effective_date: date, doc_id: "", adoptedKey: ""};
  },
  payload(review) {
    const p = review.publication;
    if (!p || !p.action) return null;
    const {adoptedKey, ...body} = p;
    if (!body.doc_id) delete body.doc_id;
    if (review.selected.type === "conflict") {
      const item = (review.selected.context.items || [])[Number(adoptedKey)];
      if (adoptedKey === "" || !item) throw new Error("请选择采纳的原始版本");
      body.adopted = {doc_id: item.doc_id, version: item.version};
    }
    if (![body.content, body.source, body.topic].every(s => s.trim())) throw new Error("请填写知识正文、核实来源及主题");
    return body;
  },
  params(review, offset) {
    const p = new URLSearchParams({offset: String(offset), limit: "26"});
    for (const key of ["type", "status"]) if (review.filters[key]) p.set(key, review.filters[key]);
    for (const key of ["since", "until"]) {
      if (review.filters[key]) {
        const date = new Date(review.filters[key]);
        if (Number.isNaN(date.getTime())) throw new Error("请选择有效时间");
        p.set(key, date.toISOString()); // datetime-local is local; API filters in UTC.
      }
    }
    if (p.has("since") && p.has("until") && p.get("since") > p.get("until")) {
      throw new Error("开始时间不能晚于结束时间");
    }
    return p.toString();
  },
  time(value) {
    if (!value) return "—";
    const date = new Date(/Z$|[+-]\d\d:\d\d$/.test(value) ? value : value + "Z");
    return Number.isNaN(date.getTime()) ? value : date.toLocaleString("zh-CN", {hour12: false});
  },
  source(item) { return [item.doc_name || item.doc_id || item.source || "来源未记录", item.version || "版本未记录", item.effective_date || "生效日未记录"].join(" · "); },
  error(error) {
    return ({401: "登录已失效，请重新登录", 403: "没有权限处理此待办或查看该销售的事项",
      404: "待办不存在，请刷新列表", 409: "状态已变化或已由其他人领取，请刷新后查看",
      422: "请检查筛选条件，并填写非空处理结论", 500: "请求未完成，请刷新核对待办状态",
      503: "服务暂不可用，请稍后核对待办状态"})[error.status]
      || (error.name === "TypeError" ? "网络连接失败，请检查服务后刷新核对" : error.message || "操作失败，请刷新核对");
  },
};
