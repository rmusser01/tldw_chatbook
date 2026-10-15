// Chatbook mobile companion — /m (ADR-234 Phase 1b).
//
// Vanilla, dependency-free client over the /api/v1/mobile REST table and
// the /api/v1/mobile/stream WebSocket. Every dynamic string is rendered
// via textContent/createElement — never innerHTML — so wire text is always
// inert. Wire shapes match the Task 4/5 contracts exactly; unknown error
// codes and frame types are shown/ignored, never fatal.
"use strict";

(() => {
  const API_BASE = "/api/v1/mobile";
  const STREAM_PATH = "/api/v1/mobile/stream";
  const CSRF_HEADER = "X-Chatbook-CSRF";
  const SYNC_INTERVAL_MS = 12000; // cross-surface completion sync (spec 4.4)
  const RECONNECT_BACKOFF_MS = [1000, 2000, 5000, 15000];
  const CLOSE_AUTH_EXPIRED = 4401;
  const DRAFT_PREFIX = "chatbook-mobile-draft:";

  const csrfMeta = document.querySelector('meta[name="chatbook-csrf"]');
  const csrfToken = csrfMeta ? csrfMeta.content : "";

  const $ = (id) => document.getElementById(id);
  const els = {
    views: { list: $("view-list"), chat: $("view-chat"), approvals: $("view-approvals") },
    backButton: $("back-button"),
    statusLine: $("status-line"),
    sessionList: $("session-list"),
    sessionsEmpty: $("sessions-empty"),
    newChatButton: $("new-chat-button"),
    newChatForm: $("new-chat-form"),
    newChatTitle: $("new-chat-title"),
    chatTitle: $("chat-title"),
    transcript: $("transcript"),
    chatEmpty: $("chat-empty"),
    composer: $("composer"),
    composerInput: $("composer-input"),
    sendButton: $("send-button"),
    stopButton: $("stop-button"),
    approvalsButton: $("approvals-button"),
    approvalsBadge: $("approvals-badge"),
    approvalsList: $("approvals-list"),
    approvalsEmpty: $("approvals-empty"),
  };

  const state = {
    sessions: [],
    openSessionId: null,
    messages: [], // wire messages of the open session, in order
    messageIds: new Set(),
    cursor: null,
    rowsById: new Map(), // message id -> rendered <li>
    pendingUserTexts: [], // optimistic drafts awaiting their real wire row
    activeRuns: new Set(), // session ids with a run in flight
    approvals: [],
    ws: null,
    wsAttempt: 0,
    statusTimer: 0,
  };

  // --- API helper — {"error": code} bodies become thrown codes            ---

  function sessionUrl(sessionId, suffix, query) {
    return (
      API_BASE +
      "/conversations/" +
      encodeURIComponent(sessionId) +
      suffix +
      (query || "")
    );
  }

  async function api(method, path, body) {
    const init = {
      method,
      credentials: "same-origin",
      headers: { [CSRF_HEADER]: csrfToken },
    };
    if (body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    let response;
    try {
      response = await fetch(path, init);
    } catch (networkError) {
      throw "network_error";
    }
    let data = null;
    try {
      data = await response.json();
    } catch (parseError) {
      data = null;
    }
    if (!response.ok) {
      const code = data && typeof data.error === "string" ? data.error : null;
      throw code || "http_" + response.status;
    }
    return data;
  }

  function showError(code) {
    els.statusLine.textContent = "Error: " + String(code);
    els.statusLine.hidden = false;
    clearTimeout(state.statusTimer);
    state.statusTimer = setTimeout(() => {
      els.statusLine.hidden = true;
    }, 6000);
  }

  // --- Views                                                              ---

  function showView(name) {
    for (const [key, section] of Object.entries(els.views)) {
      section.hidden = key !== name;
    }
    els.backButton.hidden = name === "list";
  }

  // --- Session list                                                       ---

  async function refreshSessions() {
    let data;
    try {
      data = await api("GET", API_BASE + "/conversations");
    } catch (code) {
      showError(code);
      return;
    }
    state.sessions = Array.isArray(data.sessions) ? data.sessions : [];
    renderSessionList();
  }

  function renderSessionList() {
    els.sessionList.replaceChildren();
    for (const session of state.sessions) {
      const row = document.createElement("li");
      row.dataset.sessionId = String(session.id);
      const title = document.createElement("span");
      title.className = "session-title";
      title.textContent = session.title || "Untitled chat";
      row.append(title);
      if (session.has_folder_tools) {
        const badge = document.createElement("span");
        badge.className = "folder-badge";
        badge.textContent = "folder tools";
        row.append(badge);
      }
      row.addEventListener("click", () => openSession(String(session.id)));
      els.sessionList.append(row);
    }
    els.sessionsEmpty.hidden = state.sessions.length > 0;
  }

  async function createSession(title) {
    let data;
    try {
      data = await api("POST", API_BASE + "/conversations", {
        title: title || undefined,
      });
    } catch (code) {
      showError(code);
      return;
    }
    els.newChatForm.hidden = true;
    els.newChatTitle.value = "";
    await refreshSessions();
    if (data && data.session && data.session.id) {
      openSession(String(data.session.id));
    }
  }

  // --- Transcript                                                         ---

  function toolNameOf(msg) {
    // Compact tool rows come from what the wire actually carries (verified
    // live: role "tool" rows whose content holds the call, e.g. "⚙ fs_list")
    // or optional tool metadata if a future wire adds it — never invented.
    if (typeof msg.tool_name === "string" && msg.tool_name) return msg.tool_name;
    if (msg.tool && typeof msg.tool.name === "string" && msg.tool.name) {
      return msg.tool.name;
    }
    if (msg.role === "tool" && typeof msg.content === "string") {
      return msg.content.replace(/^[⚙]\s*/, "").trim() || null;
    }
    return null;
  }

  function renderMessage(msg) {
    const li = document.createElement("li");
    const toolName = toolNameOf(msg);
    if (toolName) {
      li.className = "role-tool";
      li.textContent = "tool: " + toolName;
    } else {
      const role = typeof msg.role === "string" ? msg.role : "assistant";
      li.className = "role-" + (role === "user" ? "user" : "assistant");
      li.textContent = typeof msg.content === "string" ? msg.content : "";
      if (typeof msg.status === "string") {
        li.classList.add("status-" + msg.status);
      }
    }
    state.rowsById.set(msg.id, li);
    return li;
  }

  function appendMessage(msg) {
    if (state.messageIds.has(msg.id)) return;
    // Reconcile an optimistic user bubble with its real wire row (the
    // non-blocking send persists the row a tick after its 202).
    if (msg.role === "user" && state.pendingUserTexts[0] === msg.content) {
      state.pendingUserTexts.shift();
      const optimistic = els.transcript.querySelector("li[data-optimistic]");
      if (optimistic) optimistic.remove();
    }
    state.messageIds.add(msg.id);
    state.messages.push(msg);
    els.transcript.append(renderMessage(msg));
    els.chatEmpty.hidden = true;
    scrollTranscript();
  }

  function appendOptimisticUser(text) {
    const li = document.createElement("li");
    li.className = "role-user";
    li.dataset.optimistic = "1";
    li.textContent = text;
    state.pendingUserTexts.push(text);
    els.transcript.append(li);
    els.chatEmpty.hidden = true;
    scrollTranscript();
  }

  function dropOptimisticUsers() {
    state.pendingUserTexts = [];
    for (const row of els.transcript.querySelectorAll("li[data-optimistic]")) {
      row.remove();
    }
  }

  function scrollTranscript() {
    const region = els.transcript; // the transcript is the scroll container
    region.scrollTop = region.scrollHeight;
  }

  function mergeMessages(incoming) {
    for (const msg of incoming) appendMessage(msg);
    if (incoming.length > 0) {
      state.cursor = incoming[incoming.length - 1].id;
    }
  }

  async function loadTranscript(sessionId) {
    let data;
    try {
      data = await api("GET", sessionUrl(sessionId, "/messages"));
    } catch (code) {
      showError(code);
      return;
    }
    state.messages = [];
    state.messageIds = new Set();
    state.rowsById = new Map();
    state.pendingUserTexts = [];
    els.transcript.replaceChildren();
    mergeMessages(Array.isArray(data.messages) ? data.messages : []);
    state.cursor = data.cursor || state.cursor;
    els.chatEmpty.hidden = state.messages.length > 0;
    scrollTranscript();
  }

  async function syncOpenSession() {
    const sessionId = state.openSessionId;
    if (!sessionId) return;
    const query = state.cursor
      ? "?after=" + encodeURIComponent(state.cursor)
      : "";
    let data;
    try {
      data = await api("GET", sessionUrl(sessionId, "/messages", query));
    } catch (code) {
      if (code !== "invalid_cursor") showError(code);
      return; // a vanished cursor falls back to the next full load
    }
    mergeMessages(Array.isArray(data.messages) ? data.messages : []);
    if (data.cursor) state.cursor = data.cursor;
  }

  function openSession(sessionId) {
    state.openSessionId = sessionId;
    state.cursor = null;
    els.chatTitle.textContent =
      (state.sessions.find((s) => String(s.id) === sessionId) || {}).title ||
      "Chat";
    restoreDraft(sessionId);
    updateComposerState();
    showView("chat");
    loadTranscript(sessionId);
  }

  // --- Composer                                                           ---

  function draftKey(sessionId) {
    return DRAFT_PREFIX + sessionId;
  }

  function saveDraft() {
    const sessionId = state.openSessionId;
    if (!sessionId) return;
    const text = els.composerInput.value;
    if (text) sessionStorage.setItem(draftKey(sessionId), text);
    else sessionStorage.removeItem(draftKey(sessionId));
  }

  function restoreDraft(sessionId) {
    els.composerInput.value = sessionStorage.getItem(draftKey(sessionId)) || "";
    autoGrow();
  }

  function autoGrow() {
    const input = els.composerInput;
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 160) + "px";
  }

  function updateComposerState() {
    const runActive =
      state.openSessionId && state.activeRuns.has(state.openSessionId);
    els.sendButton.disabled = Boolean(runActive);
    els.stopButton.hidden = !runActive;
  }

  async function sendDraft() {
    const sessionId = state.openSessionId;
    const text = els.composerInput.value.trim();
    if (!sessionId || !text || state.activeRuns.has(sessionId)) return;
    try {
      await api("POST", sessionUrl(sessionId, "/messages"), { text });
    } catch (code) {
      showError(code);
      return;
    }
    els.composerInput.value = "";
    sessionStorage.removeItem(draftKey(sessionId));
    autoGrow();
    // The 202 returns before the turn runs (non-blocking send): show the
    // user bubble now; the real row reconciles it on the next sync.
    appendOptimisticUser(text);
    await syncOpenSession();
  }

  async function stopRun() {
    const sessionId = state.openSessionId;
    if (!sessionId) return;
    try {
      await api("POST", sessionUrl(sessionId, "/cancel"));
    } catch (code) {
      showError(code);
    }
  }

  // --- Approvals                                                          ---

  async function refreshApprovals() {
    let data;
    try {
      data = await api("GET", API_BASE + "/approvals");
    } catch (code) {
      showError(code);
      return;
    }
    state.approvals = Array.isArray(data.approvals) ? data.approvals : [];
    renderApprovals();
  }

  function renderApprovals() {
    els.approvalsList.replaceChildren();
    for (const round of state.approvals) renderApprovalCard(round);
    els.approvalsEmpty.hidden = state.approvals.length > 0;
    els.approvalsBadge.hidden = state.approvals.length === 0;
    els.approvalsBadge.textContent = String(state.approvals.length);
  }

  function prettyArgs(arguments_) {
    try {
      return JSON.stringify(arguments_, null, 2);
    } catch (stringifyError) {
      return String(arguments_);
    }
  }

  function renderApprovalCard(round) {
    const card = document.createElement("li");
    const calls = Array.isArray(round.calls) ? round.calls : [];
    for (const call of calls) {
      const block = document.createElement("div");
      block.className = "approval-call";
      const name = document.createElement("p");
      name.className = "approval-name";
      name.textContent = "tool: " + (call.llm_name || "unknown tool");
      const args = document.createElement("pre");
      args.className = "approval-args";
      args.textContent = prettyArgs(call.arguments);
      const options = document.createElement("div");
      options.className = "approval-options";
      for (const option of call.options || []) {
        const button = document.createElement("button");
        button.type = "button";
        button.className =
          "option-button" + (option === "deny" ? " option-deny" : "");
        button.textContent = String(option);
        button.addEventListener("click", () =>
          decide(round, call, String(option), card)
        );
        options.append(button);
      }
      block.append(name, args, options);
      card.append(block);
    }
    if (!calls.length) {
      const empty = document.createElement("p");
      empty.className = "approval-note";
      empty.textContent = "Approval round with no calls.";
      card.append(empty);
    }
    els.approvalsList.append(card);
  }

  async function decide(round, call, option, card) {
    const callId = typeof call.call_id === "string" ? call.call_id : "";
    let result;
    try {
      result = await api(
        "POST",
        API_BASE + "/approvals/" + encodeURIComponent(String(round.round_id)),
        { decisions: { [callId]: option } }
      );
    } catch (code) {
      showError(code);
      return;
    }
    if (result && result.ok) {
      card.remove();
      await refreshApprovals();
    } else {
      const note = document.createElement("p");
      note.className = "approval-note";
      note.textContent = "Decision not delivered — the round may have expired.";
      card.append(note);
    }
  }

  // --- Live stream                                                        ---

  function streamUrl() {
    const scheme = location.protocol === "https:" ? "wss:" : "ws:";
    return scheme + "//" + location.host + STREAM_PATH;
  }

  function connectStream() {
    const socket = new WebSocket(streamUrl(), ["chatbook-v1", "csrf." + csrfToken]);
    state.ws = socket;
    socket.addEventListener("message", (frame) => {
      let event;
      try {
        event = JSON.parse(frame.data);
      } catch (parseError) {
        return;
      }
      if (event && event.type === "ping") {
        socket.send(JSON.stringify({ type: "pong" }));
        return;
      }
      handleEvent(event || {});
    });
    socket.addEventListener("open", () => {
      state.wsAttempt = 0;
      syncOpenSession(); // resync past anything missed while offline
      refreshApprovals();
    });
    socket.addEventListener("close", (close) => {
      if (state.ws !== socket) return;
      state.ws = null;
      if (close.code === CLOSE_AUTH_EXPIRED) {
        saveDraft();
        location.assign("/auth/login");
        return;
      }
      const delay = RECONNECT_BACKOFF_MS[
        Math.min(state.wsAttempt, RECONNECT_BACKOFF_MS.length - 1)
      ];
      state.wsAttempt += 1;
      setTimeout(connectStream, delay);
    });
    socket.addEventListener("error", () => {
      socket.close(); // the close handler owns the backoff/redirect policy
    });
  }

  function ensureStreamingAssistant(messageId) {
    const existing = state.messages.find((m) => m.id === messageId);
    if (existing) return existing;
    const placeholder = {
      id: messageId,
      role: "assistant",
      content: "",
      status: "streaming",
      created_at: null,
    };
    appendMessage(placeholder);
    return placeholder;
  }

  function setMessageRowClass(message) {
    const row = state.rowsById.get(message.id);
    if (row) {
      row.className =
        "role-assistant status-" +
        String(message.status).replace(/[^a-z_]/g, "");
    }
  }

  function handleEvent(event) {
    const sessionId = typeof event.session_id === "string" ? event.session_id : "";
    const payload = event.payload || {};
    switch (event.type) {
      case "run_started":
        state.activeRuns.add(sessionId);
        if (sessionId === state.openSessionId) updateComposerState();
        break;
      case "delta": {
        if (sessionId !== state.openSessionId) break; // live-render only the open chat
        const message = ensureStreamingAssistant(String(payload.message_id));
        const current = typeof message.content === "string" ? message.content : "";
        const offset = Number(payload.offset) || 0;
        const text = typeof payload.text === "string" ? payload.text : "";
        message.content =
          offset <= current.length
            ? current.slice(0, offset) + text
            : current + text;
        const row = state.rowsById.get(message.id);
        if (row) row.textContent = message.content;
        scrollTranscript();
        break;
      }
      case "message_done": {
        if (sessionId !== state.openSessionId) break;
        const message = state.messages.find(
          (m) => m.id === String(payload.message_id)
        );
        if (message) {
          message.status = payload.status || "complete";
          setMessageRowClass(message);
        }
        break;
      }
      case "run_completed":
      case "run_failed":
        state.activeRuns.delete(sessionId);
        if (sessionId === state.openSessionId) updateComposerState();
        if (
          event.type === "run_failed" &&
          payload.status === "send_refused" &&
          sessionId === state.openSessionId
        ) {
          // The engine refused the turn after its 202 (nothing persisted):
          // drop the optimistic bubble and surface the refusal.
          dropOptimisticUsers();
          showError("send_refused");
        }
        syncOpenSession(); // cross-surface completions land via this resync
        break;
      case "approval_requested":
        refreshApprovals();
        break;
      case "session_updated":
        refreshSessions();
        break;
      default:
        break; // unknown frame types are ignored, never fatal
    }
  }

  // --- Boot                                                               ---

  function wireUi() {
    els.backButton.addEventListener("click", () => {
      state.openSessionId = null;
      showView("list");
      refreshSessions();
    });
    els.newChatButton.addEventListener("click", () => {
      els.newChatForm.hidden = !els.newChatForm.hidden;
      if (!els.newChatForm.hidden) els.newChatTitle.focus();
    });
    els.newChatForm.addEventListener("submit", (submit) => {
      submit.preventDefault();
      createSession(els.newChatTitle.value.trim());
    });
    els.approvalsButton.addEventListener("click", () => {
      showView("approvals");
      refreshApprovals();
    });
    els.composer.addEventListener("submit", (submit) => {
      submit.preventDefault();
      sendDraft();
    });
    els.stopButton.addEventListener("click", stopRun);
    els.composerInput.addEventListener("input", () => {
      autoGrow();
      saveDraft();
    });
    document.addEventListener("visibilitychange", () => {
      if (document.visibilityState !== "visible") return;
      refreshSessions();
      syncOpenSession();
      refreshApprovals();
    });
    window.setInterval(() => {
      if (document.visibilityState === "visible") syncOpenSession();
    }, SYNC_INTERVAL_MS);
  }

  wireUi();
  showView("list");
  refreshSessions();
  refreshApprovals();
  connectStream();
})();
