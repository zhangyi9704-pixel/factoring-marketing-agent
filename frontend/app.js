"use strict";

const API_BASE = "/api/v1";

// 评分维度英文键 → 中文标签（与后端 app/evaluation/common.py::DIM_LABELS 保持一致）
const DIM_LABELS = {
  transaction_authenticity: "交易真实性",
  receivable_quality: "应收质量",
  payer_confirmation: "付款来源",
  financing_need: "融资需求",
  risk_identification: "风险识别",
  requirement_discovery: "需求挖掘",
  business_understanding: "业务理解",
  risk_awareness: "风险意识",
  compliance: "合规意识",
};

async function api(path, opts = {}) {
  return CoachAuth.json(path, opts);
}

// streamSSE/SSEParser 位于sse.js，六种既有事件协议保持兼容。

const App = {
  data() {
    return {
      view: "login",
      auth: { username: "", password: "", user: null, loading: false },
      notice: "",
      userId: "",
      brief: {
        cases: [], caseId: "", session: null, publicCase: null,
        briefing: null, loading: false, createPending: null,
      },
      drill: {
        cases: [], caseId: "", session: null, publicCase: null,
        messages: [], message: "", finished: false,
        needsHumanReview: false, l2Degraded: false, evaluation: null, loading: false,
        evalStages: [],          // 流式评估阶段进度（stage/label/status/summary）
        pending: null, controller: null, resumeId: "", interrupted: false, recoveryDegraded: false,
        createPending: null, recoveryRequestId: null,
      },
      dashboard: { userId: "1001", score: null, loading: false },
      knowledge: { source: "product", result: null, loading: false, query: "", industry: "all", search: null, searching: false },
      review: CoachReview.state(),
    };
  },

  computed: {
    canMaintainKnowledge() { return this.auth.user && this.auth.user.role === "admin"; },
    canReview() { return this.auth.user && ["coach", "admin"].includes(this.auth.user.role); },
    reviewOwned() { return this.review.selected && this.auth.user && Number(this.review.selected.assignee) === this.auth.user.id; },
    dashboardLabel() { return this.auth.user && this.auth.user.role === "sales" ? "个人成绩" : "主管看板"; },
    levelDistText() {
      const dist = this.dashboard.score && this.dashboard.score.level_distribution;
      if (!dist) return "";
      return Object.entries(dist).map(([k, v]) => `${k} ${v} 个`).join("、");
    },
  },

  // 刷新先向后端确认保存的token；登录后才加载案例和恢复本人会话。
  async mounted() {
    window.addEventListener("coach-unauthorized", (event) => {
      if (this.drill.controller) this.drill.controller.abort();
      this.auth.user = null; this.userId = ""; this.view = "login";
      this.notice = event.reason === "logout" ? "已退出登录" : "登录已失效，请重新登录";
      this.brief.session = null; this.brief.publicCase = null; this.brief.briefing = null;
      this.drill.session = null; this.drill.publicCase = null; this.drill.messages = [];
      this.drill.pending = null; this.drill.evaluation = null; this.drill.resumeId = "";
      this.drill.evalStages = []; this.dashboard.score = null;
      this.knowledge.result = null; this.knowledge.search = null;
      this.auth.password = ""; this.dashboard.userId = "";
      this.drill.message = ""; this.knowledge.query = "";
      this.brief.loading = false; this.drill.loading = false; this.dashboard.loading = false;
      this.knowledge.loading = false; this.knowledge.searching = false;
      this.review = CoachReview.state();
    });
    if (!CoachAuth.token) return;
    try { this.auth.user = await api("/auth/me"); } catch (e) {
      if (e.status !== 401) this.notice = "无法恢复登录，请稍后刷新重试";
      return;
    }
    this.userId = String(this.auth.user.id); this.dashboard.userId = this.userId;
    this.view = "brief";
    await this.loadCases();
    const sid = localStorage.getItem("coach_session_" + this.userId);
    if (sid) { this.drill.resumeId = sid; this.restoreSession(); }
  },

  methods: {
    async login() {
      this.auth.loading = true;
      try {
        const result = await api("/auth/login", {method: "POST", body: JSON.stringify({
          username: this.auth.username, password: this.auth.password,
        })});
        CoachAuth.save(result.access_token); this.auth.user = result.user;
        this.userId = String(result.user.id); this.dashboard.userId = this.userId;
        this.auth.password = ""; this.notice = ""; this.view = "brief";
        await this.loadCases();
        const sid = localStorage.getItem("coach_session_" + this.userId);
        if (sid) { this.drill.resumeId = sid; await this.restoreSession(); }
      } catch (_) { this.notice = "登录失败，请核对账号、密码及账号状态"; }
      finally { this.auth.password = ""; this.auth.loading = false; }
    },
    logout() { CoachAuth.clear("logout"); },
    pretty(obj) { return JSON.stringify(obj, null, 2); },

    dimLabel(key) { return DIM_LABELS[key] || key; },

    reviewType(value) { return CoachReview.types[value] || value; },
    reviewStatus(value) { return CoachReview.statuses[value] || value; },
    reviewTime(value) { return CoachReview.time(value); },
    reviewSource(item) { return CoachReview.source(item); },
    async enterReview() {
      if (!this.canReview) { this.notice = "没有权限执行此操作"; return; }
      this.view = "review"; this.notice = "";
      await this.loadReview();
    },
    async changeReviewQueue() {
      this.review.filters.type = "";
      this.review.selected = null; this.review.resolution = "";
      this.review.detailSeq++;
      this.review.detailLoading = false;
      this.review.feedback = "";
      await this.loadReview(0);
    },
    async loadReview(offset = 0) {
      if (!this.canReview || this.review.saving) return false;
      const state = this.review, seq = ++state.listSeq, generation = CoachAuth.generation;
      state.loading = true; state.error = "";
      // Old rows cannot be selected with a newly chosen queue/filter.
      state.items = [];
      try {
        const params = CoachReview.params(state, offset), queue = state.queue;
        const items = await api("/" + queue + "/pending?" + params);
        if (this.review !== state || seq !== state.listSeq || generation !== CoachAuth.generation) return false;
        state.items = items.slice(0, 25); state.hasNext = items.length > 25; state.offset = offset;
        return true;
      } catch (e) {
        if (this.review === state && seq === state.listSeq && generation === CoachAuth.generation) state.error = CoachReview.error(e);
        return false;
      } finally { if (this.review === state && seq === state.listSeq) state.loading = false; }
    },
    async selectReview(item) {
      if (!this.canReview || this.review.saving || this.review.loading) return;
      const state = this.review, seq = ++state.detailSeq, generation = CoachAuth.generation, queue = state.queue;
      state.detailLoading = true; state.selected = null; state.resolution = ""; state.error = ""; state.feedback = "";
      state.publication = CoachReview.publication(); state.draft = null; state.contentConfirmed = false; state.sourceConfirmed = false;
      state.transferAssignee = ""; state.transferReason = "";
      try {
        const result = await api("/" + queue + "/pending/" + item.id);
        if (this.review !== state || seq !== state.detailSeq || generation !== CoachAuth.generation) return;
        state.selected = {...result, queue}; state.resolution = result.resolution || "";
        if (result.knowledge_draft) {
          const draft = await api("/knowledge/drafts/" + result.knowledge_draft.id);
          if (this.review === state && seq === state.detailSeq && generation === CoachAuth.generation) state.draft = draft;
        }
      } catch (e) {
        if (this.review === state && seq === state.detailSeq && generation === CoachAuth.generation) state.error = CoachReview.error(e);
      } finally { if (this.review === state && seq === state.detailSeq) state.detailLoading = false; }
    },
    async updateReview(status) {
      const state = this.review, item = state.selected;
      if (!this.canReview || state.saving || state.loading || !item) return;
      if (status === "processing" ? item.status !== "pending" : item.status !== "processing" || !this.reviewOwned) return;
      if (status !== "processing" && !state.resolution.trim()) { state.error = "请先填写处理结论"; return; }
      if (state.resolution.length > 10000) { state.error = "处理结论不能超过10000字"; return; }
      const generation = CoachAuth.generation;
      state.saving = true; state.error = ""; state.feedback = "";
      try {
        const feedback = status === "resolved" && item.queue === "knowledge" ? CoachReview.payload(state) : null;
        const result = await api("/" + item.queue + "/pending/" + item.id, {method: "PATCH", body: JSON.stringify({
          status, ...(status === "processing" ? {} : {resolution: state.resolution.trim()}),
          ...(feedback ? {feedback} : {}),
        })});
        if (this.review !== state || generation !== CoachAuth.generation) return;
        state.selected = {...result, queue: item.queue}; state.resolution = result.resolution || "";
        if (result.knowledge_draft) {
          const draft = await api("/knowledge/drafts/" + result.knowledge_draft.id);
          if (this.review !== state || generation !== CoachAuth.generation) return;
          state.draft = draft;
        }
        state.saving = false;
        const refreshed = await this.loadReview(0);
        if (this.review === state && generation === CoachAuth.generation) {
          state.feedback = (status === "processing" ? "领取成功，请填写处理结论" : "待办已关闭，处理结论已保存")
            + (refreshed ? "，列表已刷新" : "；列表刷新失败，请点击刷新核对");
        }
      } catch (e) {
        if (this.review === state && generation === CoachAuth.generation) {
          state.error = CoachReview.error(e);
          // No automatic PATCH retry, including an unknown outcome after disconnect.
          if (e.status === 409) { state.selected = null; state.resolution = ""; }
        }
      } finally { if (this.review === state) state.saving = false; }
    },

    async reassignReview() {
      const state = this.review, item = state.selected, generation = CoachAuth.generation;
      if (!item || item.status !== "processing" || state.saving || state.loading || !this.auth.user || this.auth.user.role !== "admin") return;
      const target = state.transferAssignee.trim(), reason = state.transferReason.trim();
      if (!reason || reason.length > 1000) { state.error = "请填写1000字以内的回收或转派理由"; return; }
      const assignee = target ? Number(target) : null;
      if (target && (!/^\d+$/.test(target) || !Number.isSafeInteger(assignee) || assignee <= 0)) {
        state.error = "目标处理人编号须为正整数；留空表示回收待办"; return;
      }
      state.saving = true; state.error = ""; state.feedback = "";
      try {
        const result = await api("/" + item.queue + "/pending/" + item.id + "/reassign", {method: "POST",
          body: JSON.stringify({expected_assignee: item.assignee, assignee_id: assignee, reason})});
        if (this.review !== state || generation !== CoachAuth.generation) return;
        state.selected = {...result, queue: item.queue}; state.transferReason = ""; state.transferAssignee = "";
        state.saving = false;
        const refreshed = await this.loadReview(0);
        if (this.review === state && generation === CoachAuth.generation) {
          state.feedback = (assignee === null ? "待办已回收，可重新领取" : "待办已转派") + (refreshed ? "，列表已刷新" : "；请刷新核对列表");
        }
      } catch (e) {
        if (this.review === state && generation === CoachAuth.generation) {
          state.error = CoachReview.error(e) + "；请刷新核对，不自动重试";
          if (e.status === 409) state.selected = null;
        }
      } finally { if (this.review === state) state.saving = false; }
    },

    async publishReviewDraft(operation) {
      const state = this.review, draft = state.draft, generation = CoachAuth.generation;
      if (!draft || state.saving || !this.auth.user || this.auth.user.role !== "admin") return;
      if (!["confirm", "publish", "recover"].includes(operation)) return;
      if (operation === "recover" && !["failed", "reconcile_required", "publishing"].includes(draft.status)) return;
      if (operation === "confirm" && (!state.contentConfirmed || !state.sourceConfirmed)) return;
      state.saving = true; state.error = ""; state.feedback = "";
      try {
        const result = await api("/knowledge/drafts/" + draft.id + "/" + operation, {method: "POST",
          ...(operation === "confirm" ? {body: JSON.stringify({content_hash: draft.content_hash,
            content_confirmed: true, source_confirmed: true})} : {})});
        if (this.review !== state || generation !== CoachAuth.generation) return;
        state.draft = result;
        state.feedback = operation === "confirm" ? "正文与来源确认已保存，请另行发布"
          : operation === "recover" ? (result.status === "published" ? "核对完成，知识版本已发布" : "核对与恢复完成；请再次核实状态后另行发布")
          : "入库验证成功，新版本已生效，旧版本已归档";
      } catch (e) {
        if (this.review === state && generation === CoachAuth.generation) state.error = CoachReview.error(e) + "；请重新选择待办核对发布状态，勿自动重试";
      } finally { if (this.review === state) state.saving = false; }
    },

    currentUserId() {
      const id = this.auth.user && this.auth.user.id;
      if (!id || id <= 0) {
        this.notice = "请先登录";
        return null;
      }
      return id;
    },

    async enterBrief() {
      this.view = "brief";
      await this.loadCases();
    },

    async enterDrill() {
      this.view = "drill";
      await this.loadCases();
    },

    // ---------- 案例简报（访前简报随会话创建产出） ----------
    async loadCases() {
      try {
        const cases = await api("/cases");
        this.brief.cases = cases;
        this.drill.cases = cases;
        if (cases.length) {
          if (!this.brief.caseId) this.brief.caseId = cases[0].case_id;
          if (!this.drill.caseId) this.drill.caseId = cases[0].case_id;
        }
      } catch (e) {
        this.notice = "加载案例失败：" + e.message;
      }
    },

    creationPayload(section) {
      const state = this[section], owner = this.currentUserId();
      const key = "coach_create_" + owner + "_" + section;
      let pending = state.createPending;
      if (!pending) {
        try { pending = JSON.parse(localStorage.getItem(key) || "null"); } catch (_) { pending = null; }
      }
      if (!pending || pending.owner !== owner || pending.case_id !== state.caseId) {
        pending = {owner, case_id: state.caseId, request_id: crypto.randomUUID()};
      }
      state.createPending = pending;
      localStorage.setItem(key, JSON.stringify(pending));
      return {case_id: pending.case_id, request_id: pending.request_id};
    },
    clearCreation(section) {
      this[section].createPending = null;
      localStorage.removeItem("coach_create_" + this.currentUserId() + "_" + section);
    },
    async startBriefing() {
      if (this.brief.loading) return;
      const userId = this.currentUserId();
      if (!userId) return;
      if (!this.brief.caseId) {
        this.notice = "请先选择案例";
        return;
      }
      const generation = CoachAuth.generation;
      this.brief.loading = true;
      try {
        const s = await api("/sessions", {
          method: "POST",
          body: JSON.stringify(this.creationPayload("brief")),
        });
        if (generation !== CoachAuth.generation) return;
        this.brief.session = s;
        this.brief.publicCase = s.public_case;
        this.brief.briefing = s.briefing;
        this.clearCreation("brief");
        this.notice = "";
      } catch (e) {
        if (generation !== CoachAuth.generation) return;
        if (["session_initialization_failed", "session_initialization_aborted"].includes(e.message) || e.status === 404) this.clearCreation("brief");
        this.notice = "生成简报失败：" + e.message;
      } finally {
        if (generation === CoachAuth.generation) this.brief.loading = false;
      }
    },

    resetBriefing() {
      if (this.brief.loading) return;
      this.clearCreation("brief");
      this.brief.session = null;
      this.brief.publicCase = null;
      this.brief.briefing = null;
    },

    enterDrillFromBrief() {
      if (!this.brief.session) return;
      this.applySession(this.brief.session);
      this.view = "drill";
    },

    // ---------- 情景演练 ----------
    applySession(s) {
      this.drill.session = s;
      this.drill.publicCase = s.public_case;
      this.drill.messages = s.opening
        ? [{ role: "customer", turn: 0, content: s.opening.content }]
        : [];
      this.drill.finished = false;
      this.drill.evaluation = null;
      this.drill.needsHumanReview = false;
      this.drill.pending = null;
      this.drill.interrupted = false;
      this.drill.recoveryRequestId = null;
      this.drill.recoveryDegraded = false;
      localStorage.setItem("coach_user", this.userId);
      localStorage.setItem("coach_session_" + this.userId, s.session_id);
    },

    resetDrill() {
      if (this.drill.loading || !this.drill.finished) return;
      const owner = this.currentUserId();
      if (!owner) return;
      const sid = this.drill.session?.session_id || this.drill.resumeId;
      if (sid) localStorage.removeItem("coach_pending_" + sid);
      localStorage.removeItem("coach_session_" + owner);
      this.clearCreation("drill");
      Object.assign(this.drill, {session: null, publicCase: null, messages: [], message: "", finished: false,
        needsHumanReview: false, l2Degraded: false, evaluation: null, evalStages: [], pending: null,
        controller: null, resumeId: "", interrupted: false, recoveryDegraded: false, recoveryRequestId: null});
      this.notice = "可以选择案例，开始新的演练。";
    },

    async restoreSession() {
      if (this.drill.loading || !this.drill.resumeId) return;
      const generation = CoachAuth.generation;
      this.drill.loading = true;
      try {
        const s = await api("/sessions/" + encodeURIComponent(this.drill.resumeId) + "/resume", {method: "POST"});
        if (generation !== CoachAuth.generation) return;
        this.drill.session = {session_id: s.session_id, case_id: s.case_id, turn: s.turn};
        this.drill.messages = s.messages || [];
        this.drill.finished = s.finished;
        this.drill.interrupted = s.interrupted;
        this.drill.recoveryRequestId = s.recovery_request_id;
        this.drill.recoveryDegraded = s.recovery_degraded;
        this.drill.needsHumanReview = s.needs_human_review;
        this.drill.evaluation = null; this.drill.evalStages = [];
        this.drill.pending = JSON.parse(localStorage.getItem("coach_pending_" + s.session_id) || "null");
        const terminal = ["operator_aborted", "initialization_failed"].includes(s.finish_reason);
        if (terminal) {
          this.drill.pending = null; this.drill.recoveryRequestId = null;
          localStorage.removeItem("coach_pending_" + s.session_id);
        }
        localStorage.setItem("coach_user", this.userId);
        localStorage.setItem("coach_session_" + this.userId, s.session_id);
        this.view = "drill";
        this.notice = terminal ? "会话已终止，历史记录保留。"
          : this.drill.pending ? "上次请求未确认完成，请重连核对本轮。" : "已从检查点恢复会话。";
      } catch (e) { if (generation === CoachAuth.generation) this.notice = "恢复失败：" + e.message; }
      finally { if (generation === CoachAuth.generation) this.drill.loading = false; }
    },

    cancelMessage() { if (this.drill.controller) this.drill.controller.abort(); },
    reconnectMessage() { return this.sendMessage(true); },
    async reconcileMessage(action = "reconcile") {
      const sid = this.drill.session?.session_id || this.drill.resumeId;
      if (this.drill.loading || !sid) return;
      const generation = CoachAuth.generation;
      this.drill.loading = true;
      try {
        const result = await api("/sessions/" + encodeURIComponent(sid) + "/reconcile", {
          method: "POST", body: JSON.stringify({action,
            request_id: this.drill.pending?.request_id || this.drill.recoveryRequestId || "operator-abort",
            reason: action === "abort" ? "用户明确终止无法核对的演练" : "用户核对中断请求的已持久化结果"}),
        });
        if (generation !== CoachAuth.generation) return;
        const state = result.status;
        this.drill.session = {session_id: sid, case_id: state.case_id, turn: state.turn};
        this.drill.messages = state.messages || [];
        this.drill.finished = state.finished;
        this.drill.interrupted = state.interrupted;
        this.drill.needsHumanReview = state.needs_human_review;
        this.drill.recoveryDegraded = !!state.recovery_degraded;
        this.drill.pending = null; this.drill.recoveryRequestId = null;
        localStorage.removeItem("coach_pending_" + sid);
        this.notice = result.action === "aborted" ? "会话已明确终止，历史记录保留。" : "已核对完成，可以继续演练。";
        if (result.action === "aborted") { this.drill.evaluation = null; this.drill.evalStages = []; }
        if (state.finished && !["operator_aborted", "initialization_failed"].includes(state.finish_reason)) await this.loadEvaluation();
      } catch (e) {
        if (generation !== CoachAuth.generation) return;
        this.notice = "核对未完成：" + e.message + "。可重试核对，或明确终止本次演练。";
      } finally { if (generation === CoachAuth.generation) this.drill.loading = false; }
    },

    async startSession() {
      if (this.drill.loading) return;
      const userId = this.currentUserId();
      if (!userId) return;
      if (!this.drill.caseId) {
        this.notice = "请先选择案例";
        return;
      }
      const generation = CoachAuth.generation;
      this.drill.loading = true;
      try {
        const s = await api("/sessions", {
          method: "POST",
          body: JSON.stringify(this.creationPayload("drill")),
        });
        if (generation !== CoachAuth.generation) return;
        this.applySession(s);
        this.clearCreation("drill");
        this.notice = "";
      } catch (e) {
        if (generation !== CoachAuth.generation) return;
        if (["session_initialization_failed", "session_initialization_aborted"].includes(e.message) || e.status === 404) this.clearCreation("drill");
        this.notice = "创建会话失败：" + e.message;
      } finally {
        if (generation === CoachAuth.generation) this.drill.loading = false;
      }
    },

    async sendMessage(reconnect = false) {
      reconnect = reconnect === true; // Vue表单传入SubmitEvent不是重连标记
      if (this.drill.loading || !this.drill.session || (!reconnect && (this.drill.finished || this.drill.pending || this.drill.interrupted))) return;
      const text = reconnect ? this.drill.pending?.content : this.drill.message.trim();
      if (!text) return;
      if (text.length > 4000) { this.notice = "话术最多4000字，请缩短后发送。"; return; }
      const sid = this.drill.session.session_id;
      if (!reconnect) this.drill.pending = {content: text, request_id: crypto.randomUUID(), expected_turn: this.drill.session.turn || 0};
      localStorage.setItem("coach_pending_" + sid, JSON.stringify(this.drill.pending));
      this.drill.loading = true;
      this.drill.controller = new AbortController();
      // 重连先移除已从历史回读的同轮正文，done后使用服务端历史去重。
      if (reconnect) this.drill.messages = this.drill.messages.filter(m => m.turn <= this.drill.pending.expected_turn);
      // 先占位两条消息：销售原文 + 客户流式占位（delta 逐段填充）
      const salesIdx = this.drill.messages.length;
      this.drill.messages.push({ role: "sales", turn: null, content: text, warnings: [] });
      this.drill.messages.push({ role: "customer", turn: null, content: "", streaming: true });
      const custIdx = salesIdx + 1;
      const rollback = (detail) => {
        this.drill.messages.splice(salesIdx, 2);          // 失败回滚占位
        this.notice = "本轮未完成确认：" + detail + "；请重连核对原请求。";
      };
      try {
        await streamSSE(
          "/sessions/" + this.drill.session.session_id + "/messages/stream",
          { method: "POST", body: JSON.stringify(this.drill.pending), signal: this.drill.controller.signal },
          (ev, d) => {
            if (ev === "delta") {
              this.drill.messages[custIdx].content += d.text;
            } else if (ev === "done") {
              if (d.request_id !== this.drill.pending.request_id) throw new Error("请求编号不匹配");
              this.drill.session.turn = d.turn;
              this.drill.pending = null;
              localStorage.removeItem("coach_pending_" + sid);
              this.drill.interrupted = false;
              this.notice = "本轮已完成。";
              this.drill.l2Degraded = !!d.l2_degraded;
              if (d.input_blocked) {
                this.drill.messages.splice(salesIdx, 2);
                this.notice = "话术已拦截：" + (d.guardrail_events || [])
                  .filter(e => e.severity === "block")
                  .map(e => e.compliant_response || e.rule).join("；");
                this.drill.needsHumanReview = d.needs_human_review || this.drill.needsHumanReview;
                return;
              }
              const sales = this.drill.messages[salesIdx];
              const cust = this.drill.messages[custIdx];
              sales.turn = d.turn;
              sales.warnings = d.guardrail_warnings || [];
              cust.turn = d.turn;
              cust.content = d.reply;                     // 终帧权威内容
              cust.streaming = false;
              this.drill.finished = d.finished || this.drill.finished;
              this.drill.needsHumanReview =
                d.needs_human_review || this.drill.needsHumanReview;
              if (this.drill.finished) this.loadEvaluation();
            } else if (ev === "error") {
              rollback(d.detail || "未知错误");
            }
          },
        );
        this.drill.message = "";
      } catch (e) {
        rollback(e.name === "AbortError" ? "已取消接收，服务端结果待核对" : e.message);
      } finally {
        this.drill.loading = false;
        this.drill.controller = null;
      }
    },

    async finishSession() {
      if (!this.drill.session || this.drill.loading || this.drill.pending) return;
      this.drill.loading = true;
      try {
        const result = await api("/sessions/" + this.drill.session.session_id + "/finish", { method: "POST" });
        this.drill.finished = result.finished;
        if (result.finished && !["operator_aborted", "initialization_failed"].includes(result.finish_reason)) await this.loadEvaluation();
      } catch (e) {
        this.notice = "结束演练失败：" + e.message;
      } finally {
        this.drill.loading = false;
      }
    },

    async loadEvaluation() {
      if (!this.drill.session) return;
      this.drill.evaluation = null;
      this.drill.evalStages = [];
      const STAGE_OF_SECTION = {
        opportunity: "opportunity", sales_score: "sales",
        missing_questions: "missing", next_actions: "next_action",
      };
      const sectionSummary = (key, data) => {
        if (key === "opportunity") return `${data.score} 分 · ${data.level}`;
        if (key === "sales_score") return `总分 ${data.total}`;
        if (key === "missing_questions") return `遗漏 ${data.length} 项`;
        if (key === "next_actions") return `行动 ${data.length} 条`;
        return "";
      };
      try {
        await streamSSE(
          "/sessions/" + this.drill.session.session_id + "/evaluation/stream", {},
          (ev, d) => {
            if (ev === "stage") {
              const found = this.drill.evalStages.find(s => s.stage === d.stage);
              if (found) found.status = d.status;
              else this.drill.evalStages.push({
                stage: d.stage, label: d.label, status: d.status, summary: "",
              });
            } else if (ev === "section") {
              const stageKey = STAGE_OF_SECTION[d.key];
              const found = this.drill.evalStages.find(s => s.stage === stageKey);
              if (found) {
                found.status = "done";
                found.summary = sectionSummary(d.key, d.data);
              }
            } else if (ev === "done") {
              this.drill.evaluation = d.report;
              this.drill.evalStages.forEach(stage => { stage.status = "done"; });
            } else if (ev === "error") {
              this.notice = "复盘加载失败：" + (d.detail || "未知错误");
            }
          },
        );
      } catch (e) {
        this.notice = "复盘加载失败：" + e.message;
      }
    },

    // ---------- 主管看板 ----------
    enterDashboard() {
      this.view = "dashboard";
      if (this.userId) this.dashboard.userId = String(parseInt(this.userId, 10) || this.userId);
    },

    async loadScore() {
      this.dashboard.loading = true;
      this.dashboard.score = null;
      const targetId = this.auth.user && this.auth.user.role === "sales" ? this.auth.user.id : this.dashboard.userId;
      try {
        await streamSSE("/users/" + targetId + "/scores/stream", {}, (ev, d) => {
          if (ev === "profile") {
            this.dashboard.score = d;                     // 画像先到先渲染
            this.dashboard.score.coaching = {};
            this.notice = "";
          } else if (ev === "section") {
            if (this.dashboard.score) {
              this.dashboard.score.coaching[d.key] = d.data;  // 五段逐段填充
            }
          } else if (ev === "error") {
            this.notice = "查询失败：" + (d.detail || "未知错误");
          }
        });
      } catch (e) {
        this.notice = "查询失败：" + e.message;
      } finally {
        this.dashboard.loading = false;
      }
    },

    // ---------- 知识库 ----------
    async searchKnowledge() {
      this.knowledge.searching = true;
      this.knowledge.search = null;
      try {
        this.knowledge.search = await api("/knowledge/search", {
          method: "POST", body: JSON.stringify({query: this.knowledge.query, industry: this.knowledge.industry})
        });
        this.notice = "";
      } catch (e) { this.notice = "检索失败：" + e.message; }
      finally { this.knowledge.searching = false; }
    },
    async uploadDoc() {
      if (!this.auth.user || this.auth.user.role !== "admin") {
        this.notice = "没有权限执行此操作";
        return;
      }
      const file = this.$refs.file && this.$refs.file.files[0];
      if (!file) {
        this.notice = "请选择文件";
        return;
      }
      this.knowledge.loading = true;
      const fd = new FormData();
      fd.append("file", file);
      try {
        this.knowledge.result = await api("/knowledge/ingest?source=" + encodeURIComponent(this.knowledge.source),
          {method: "POST", body: fd});
        this.notice = "";
      } catch (e) {
        this.notice = "入库失败：" + e.message;
      } finally {
        this.knowledge.loading = false;
      }
    },
  },
};

if (window.Vue) {
  window.Vue.createApp(App).mount("#app");
} else {
  console.error("Vue 未加载（需外网访问 unpkg CDN）");
}
