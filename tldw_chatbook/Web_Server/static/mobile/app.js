// Shared Console phone view (ADR-244). Dynamic wire text is always inert.
"use strict";
(() => {
  const API_BASE = "/api/v1/mobile";
  // Native approval_review.APPROVAL_OPTIONS labels; values travel unchanged.
  const APPROVAL_LABELS = {approve_once: "Once", approve_session: "This session", allow_matching: "Always · these args", always_allow: "Always", deny: "Deny"};
  const $ = (id) => document.getElementById(id);
  const csrfToken = document.querySelector('meta[name="chatbook-csrf"]').content;
  const els = {
    views: {chats: $("view-chats"), chat: $("view-chat"), approvals: $("view-approvals"),
      library: $("view-library"), note: $("view-note"), item: $("view-item")},
    backButton: $("back-button"), tabBar: $("tab-bar"), tabChats: $("tab-chats"), tabLibrary: $("tab-library"),
    statusLine: $("status-line"), sessionList: $("chat-list"), sessionsEmpty: $("sessions-empty"), savedMore: $("saved-chats-more"),
    chatTitle: $("chat-title"), transcript: $("chat-transcript"), draft: $("chat-draft"),
    targetLabel: $("target-label"), targetState: $("target-state"), connection: $("connection-state"),
    send: $("send-command"), steer: $("steer-command"), redirect: $("redirect-command"), stop: $("stop-command"),
    commandStatus: $("command-status"), agents: $("agent-list"), decisions: $("chat-decisions"),
    approvalsBadge: $("approvals-badge"), approvalsList: $("approvals-list"), approvalsEmpty: $("approvals-empty"),
    libraryNotesButton: $("library-notes-button"), libraryMediaButton: $("library-media-button"),
    notesList: $("notes-list"), notesEmpty: $("notes-empty"), notesMoreButton: $("notes-more-button"),
    mediaList: $("media-list"), mediaEmpty: $("media-empty"), mediaMoreButton: $("media-more-button"),
    noteTitle: $("note-title"), noteMeta: $("note-meta"), noteContent: $("note-content"),
    itemTitle: $("item-title"), itemMeta: $("item-meta"), itemTranscripts: $("item-transcripts"),
  };
  const state = {
    selectedTarget: null, drafts: new Map(), outstandingCommands: new Map(), hostSelectionCommand: null, hostActionGeneration: 0, connectionState: "connecting",
    lastConfirmedUpdate: null, chats: [], chat: null, snapshot: null, ws: null, wsAttempt: 0,
    activeTab: "chats", view: "chats", available: false, hostAvailable: false, ownerAvailable: false, refreshing: false, refreshAgain: false,
    selectionRevision: 0, confirmationRevision: 0, targetConfirmed: false, targetMissing: false, historyLoading: false, renderedDecisions: [], answerabilitySent: null, hostBusy: false, history: null, readWorker: null, workerHistory: [], workerCursor: null, workerConversation: null, workerReading: false, workerResultOffset: null, workerResultRevision: null,
    savedPage: 0, savedHasMore: false, savedLoading: false,
    librarySection: "notes", notesPage: 0, notesHasMore: false, mediaPage: 0, mediaHasMore: false,
  };
  const targetFields = ["server_epoch", "console_id", "session_id", "incarnation_id", "turn_id", "run_id", "handle_id", "round_id"];
  function targetOf(row, work = false) {
    const target = {};
    for (const field of targetFields) {
      if (row[field] != null && (work || !["turn_id", "run_id", "handle_id", "round_id"].includes(field))) target[field] = row[field];
    }
    return target;
  }
  function draftKey(target) {
    return JSON.stringify([target.server_epoch, target.console_id, target.session_id, target.incarnation_id,
      target.handle_id || null, target.handle_id ? target.run_id : null]);
  }
  function draftFor(target) {
    const key = draftKey(target);
    if (!state.drafts.has(key)) state.drafts.set(key, {text: "", revision: 0});
    return state.drafts.get(key);
  }
  function captureSubmission(target, draft, expectedRevisions) {
    return Object.freeze({key: draftKey(target), revision: draft.revision, text: draft.text,
      target: Object.freeze({...target}), expected_revisions: Object.freeze({...expectedRevisions})});
  }
  function clearAcceptedDraft(submission) {
    const current = state.drafts.get(submission.key);
    if (current && current.revision === submission.revision && current.text === submission.text) {
      state.drafts.set(submission.key, {text: "", revision: current.revision + 1});
      if (state.selectedTarget && draftKey(state.selectedTarget) === submission.key) els.draft.value = "";
    }
  }
  async function api(method, path, body) {
    const init = {method, credentials: "same-origin", headers: {"X-Chatbook-CSRF": csrfToken}};
    if (body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    const response = await fetch(path, init);
    if (response.status === 401) { location.assign("/auth/login"); throw "auth_required"; }
    const data = await response.json();
    if (!response.ok) throw data.error || "http_" + response.status;
    return data;
  }
  function showError(code) {
    els.statusLine.textContent = "Unable to complete: " + String(code) + ". Refresh state or open terminal.";
    els.statusLine.hidden = false;
  }
  function node(tag, text, className) {
    const el = document.createElement(tag);
    if (text != null) el.textContent = String(text);
    if (className) el.className = className;
    return el;
  }
  function button(label, action, className = "header-button") {
    const el = node("button", label, className);
    el.type = "button";
    el.addEventListener("click", action);
    return el;
  }
  function showView(name) {
    state.view = name;
    for (const [key, section] of Object.entries(els.views)) section.hidden = key !== name;
    const root = name === "chats" || name === "library";
    els.backButton.hidden = root;
    els.tabBar.hidden = !root;
    for (const [tab, el] of [["chats", els.tabChats], ["library", els.tabLibrary]]) {
      el.classList.toggle("tab-active", state.activeTab === tab);
      if (state.activeTab === tab) el.setAttribute("aria-current", "page");
      else el.removeAttribute("aria-current");
    }
    publishAnswerability();
  }
  function showTab(tab) {
    state.selectionRevision++; state.targetConfirmed = false; state.historyLoading = false;
    state.activeTab = tab;
    showView(tab);
    if (tab === "library") showLibrarySection(state.librarySection);
    else refresh();
  }
  function workspaceLabel(row) {
    return row.scope_type === "global" ? "Global" : row.workspace_name || "Workspace name unavailable";
  }
  function statusLabel(row) {
    if (!row.live || !state.hostAvailable) return "Unavailable";
    if (row.pending_decision_count) return "Waiting · decision";
    const status = String(row.status || "idle");
    if (["complete", "completed", "finished", "stopped", "cancelled", "failed", "error"].includes(status)) return "Finished · " + status;
    if (status === "blocked" || status.includes("wait") || status.includes("approval")) return "Waiting · " + status;
    if (status === "idle") return "Ready";
    return status === "preparing" ? "Preparing" : "Running · " + status;
  }
  function renderChats() {
    const existing = new Map([...els.sessionList.children].map(li => [li.firstChild.dataset.sessionId, li]));
    for (const [index, row] of state.chats.entries()) {
      const key = String(row.session_id || row.conversation_id || row.id);
      let li = existing.get(key);
      if (!li) {
        li = node("li");
        const open = button("", () => {}, "chat-row"); open.dataset.sessionId = key;
        open.append(node("strong"), node("span", "", "detail-meta"), node("span", "", "detail-meta"), node("span", "", "attention"));
        li.append(open);
      }
      existing.delete(key);
      const open = li.firstChild;
      open.onclick = () => openChat(row);
      const values = [row.title || "Untitled chat", workspaceLabel(row) + " · " + statusLabel(row),
        "Latest activity: " + (row.last_modified || "unavailable"), "Needs a decision · " + (row.pending_decision_count || 0)];
      for (const [position, text] of values.entries()) {
        if (open.children[position].textContent !== text) open.children[position].textContent = text;
      }
      open.lastChild.hidden = !row.pending_decision_count;
      if (els.sessionList.children[index] !== li) els.sessionList.insertBefore(li, els.sessionList.children[index] || null);
    }
    for (const li of existing.values()) li.remove();
    els.sessionsEmpty.hidden = state.chats.length > 0;
    els.savedMore.hidden = state.ownerAvailable || !state.savedHasMore;
    els.savedMore.disabled = state.savedLoading;
  }
  async function snapshot(target, onPage) {
    const aggregate = {latest_message: null, chats: [], messages: [], fleet: [], decisions: [], recoveries: [], command_receipts: []};
    let cursor = null, receiptCursor = null, initialReset = false;
    for (let pages = 0; pages < 500; pages++) {
      const query = new URLSearchParams({protocol_version: "1", limit: "100"});
      if (target) query.set("target", JSON.stringify(target));
      if (cursor) query.set("cursor", cursor);
      if (receiptCursor) query.set("receipt_cursor", receiptCursor);
      const page = await api("GET", API_BASE + "/console/snapshot?" + query);
      if (pages && ((page.reset_required && !initialReset) || page.sequence !== aggregate.sequence || page.server_epoch !== aggregate.server_epoch || page.console_id !== aggregate.console_id)) throw "snapshot_changed";
      if (!pages) initialReset = Boolean(page.reset_required);
      if (pages && page.chats?.some(row => aggregate.chats.some(prior =>
        (row.session_id || row.conversation_id) === (prior.session_id || prior.conversation_id)))) throw "snapshot_changed";
      for (const section of ["chats", "messages", "fleet", "decisions", "recoveries", "command_receipts"]) {
        aggregate[section].push(...(page[section] || []));
      }
      Object.assign(aggregate, {server_epoch: page.server_epoch, console_id: page.console_id, sequence: page.sequence, execution_admitted: page.execution_admitted});
      aggregate.latest_message = page.latest_message || null;
      if (onPage) onPage(aggregate, page);
      cursor = page.next_cursor; receiptCursor = page.next_receipt_cursor;
      if (!cursor && !receiptCursor) return aggregate;
    }
    throw "snapshot_limit";
  }
  function applySnapshot(data) {
    state.snapshot = data;
    for (const receipt of data.command_receipts) applyReceipt(receipt);
    const selected = state.selectedTarget;
    if (data.server_epoch !== selected.server_epoch || data.console_id !== selected.console_id) { state.available = false; state.targetConfirmed = false; state.targetMissing = true; return; }
    const row = data.chats.find(row => row.live && draftKey(targetOf(row)) === draftKey({...selected, handle_id: null}));
    const worker = selected.handle_id && data.fleet.find(row => row.handle_id === selected.handle_id && row.run_id === selected.run_id);
    state.available = Boolean(row && (!selected.handle_id || worker));
    state.targetMissing = !state.available;
    if (row) state.chat = row;
    // A replacement owner never inherits an old draft or silently changes its target.
    if (state.available) state.selectedTarget = selected.handle_id ? targetOf(worker, true) : targetOf(row, true);
    renderChat();
  }
  function confirmTarget(data, page, selection, confirmation, socket) {
    if (selection !== state.selectionRevision || confirmation !== state.confirmationRevision || socket !== state.ws || socket?.readyState !== WebSocket.OPEN || page.execution_admitted !== true || !state.selectedTarget) return;
    const selected = state.selectedTarget;
    if (data.server_epoch !== selected.server_epoch || data.console_id !== selected.console_id) return;
    const row = data.chats.find(row => row.live && draftKey(targetOf(row)) === draftKey({...selected, handle_id: null}));
    const revisions = row?.expected_revisions;
    if (!row || !revisions || !["settings_revision", "generation_settings_revision", "context_policy_revision", "identity_revision", "conversation_binding_revision"].every(key => Number.isInteger(revisions[key]) && revisions[key] >= 0)) return;
    const worker = selected.handle_id && data.fleet.find(row => row.handle_id === selected.handle_id && row.run_id === selected.run_id && row.live !== false);
    if (selected.handle_id && !worker) { state.chat = row; state.snapshot = data; renderChat(); return; }
    state.targetMissing = false;
    state.chat = row; state.snapshot = data; state.hostAvailable = true; state.available = true; state.targetConfirmed = true;
    state.selectedTarget = selected.handle_id ? targetOf(worker, true) : targetOf(row, true);
    for (const receipt of data.command_receipts) applyReceipt(receipt);
    state.historyLoading = Boolean(page.next_cursor || page.next_receipt_cursor);
    state.lastConfirmedUpdate = new Date().toLocaleTimeString();
    renderChat(); setConnection("connected");
  }
  async function refresh() {
    if (state.refreshing) { state.refreshAgain = true; return; }
    state.refreshing = true;
    const selection = state.selectionRevision, confirmation = state.confirmationRevision, socket = state.ws;
    try {
      // Retained IDs reconcile even when changing transcript pages never reach EOF.
      await queryOutstanding();
      if (selection !== state.selectionRevision || confirmation !== state.confirmationRevision || socket !== state.ws) return;
      const hostActionGeneration = state.hostActionGeneration;
      const list = await snapshot(null);
      if (selection !== state.selectionRevision) return;
      state.ownerAvailable = true; state.hostAvailable = list.execution_admitted === true; state.available = true;
      state.chats = list.chats;
      if (state.ownerAvailable) { state.savedPage = 0; state.savedHasMore = false; }
      for (const receipt of list.command_receipts) applyReceipt(receipt);
      if (completeHostSelection(list, hostActionGeneration)) return;
      if (state.selectedTarget && state.view === "chat") {
        const current = list.chats.find(row => row.live && draftKey(targetOf(row)) === draftKey({...state.selectedTarget, handle_id: null}));
        if (current) {
          state.historyLoading = true;
          const detail = await snapshot(targetOf(current), (data, page) => confirmTarget(data, page, selection, confirmation, socket));
          if (selection !== state.selectionRevision || confirmation !== state.confirmationRevision || socket !== state.ws) return;
          state.historyLoading = false; applySnapshot(detail);
        } else { state.available = false; state.targetConfirmed = false; state.targetMissing = true; state.historyLoading = false; renderChat(); }
      }
      if (selection !== state.selectionRevision) return;
      state.lastConfirmedUpdate = new Date().toLocaleTimeString();
      setConnection(state.ws?.readyState === WebSocket.OPEN ? "connected" : "disconnected");
      els.statusLine.hidden = true;
    } catch (error) {
      if (error === "snapshot_changed" && state.targetConfirmed && confirmation === state.confirmationRevision && socket === state.ws && socket?.readyState === WebSocket.OPEN) {
        state.historyLoading = true; setConnection("connected");
        state.refreshAgain = false; setTimeout(refresh, 1000); return;
      }
      state.ownerAvailable = false; state.hostAvailable = false; state.available = false;
      setConnection("uncertain");
      if (!state.savedPage) await loadSavedChats();
      if (error !== "console_unavailable") showError(error);
    } finally {
      state.refreshing = false;
      renderChats(); renderDecisions(); updateControls();
      if (state.refreshAgain) { state.refreshAgain = false; refresh(); }
    }
  }
  async function loadSavedChats() {
    if (state.savedLoading || state.ownerAvailable) return;
    state.savedLoading = true; renderChats();
    const page = state.savedPage + 1;
    try {
      const saved = await api("GET", API_BASE + "/conversations?page=" + page + "&per_page=50");
      if (state.ownerAvailable) return;
      if (saved.page !== page || !Array.isArray(saved.items) || typeof saved.has_more !== "boolean") throw "invalid_history_page";
      const known = new Set(state.chats.map(row => row.conversation_id));
      for (const row of saved.items) {
        if (!known.has(row.id)) {
          state.chats.push({...row, conversation_id: row.id, live: false}); known.add(row.id);
        }
      }
      state.savedPage = saved.page; state.savedHasMore = saved.has_more;
    } catch (error) { showError(error); }
    finally { state.savedLoading = false; renderChats(); }
  }
  function setConnection(value) {
    if (value !== "connected") { state.targetConfirmed = false; state.targetMissing = false; state.confirmationRevision++; }
    state.connectionState = value;
    els.connection.textContent = (value === "connected" ? "Connected" : value === "connecting" ? "Connecting…" : value === "uncertain" ? "State uncertain · controls paused" : "Disconnected · draft kept") +
      (state.historyLoading && value === "connected" ? " · loading history" : "") + (state.lastConfirmedUpdate ? " · confirmed " + state.lastConfirmedUpdate : " · no confirmed update");
    updateControls(); publishAnswerability(); renderLatest();
  }
  function chooseTarget(target) {
    state.selectionRevision++;
    state.selectedTarget = Object.freeze({...target}); state.targetConfirmed = false; state.targetMissing = false;
    els.transcript.replaceChildren(); renderLatest();
    state.history = null; state.readWorker = null;
    if (state.workerConversation !== state.chat?.conversation_id) {
      $("worker-history").replaceChildren(); $("worker-history-state").textContent = "Loading worker history…"; $("workers-more").hidden = true;
    }
    if (!$("agent-panel").hidden) loadWorkers(true);
    $("worker-detail").hidden = true; els.transcript.hidden = false;
    els.draft.value = draftFor(target).text;
    state.available = false;
    showView("chat"); updateControls();
    refresh();
  }
  async function openChat(row) {
    if (row.live && state.ownerAvailable) { state.chat = row; chooseTarget(targetOf(row, true)); return; }
    if (!row.conversation_id) { showError("console_unavailable"); return; }
    // Saved history remains readable even when execution has no live owner.
    state.selectionRevision++;
    const selection = state.selectionRevision;
    state.selectedTarget = null; state.available = false; state.chat = row; state.snapshot = null;
    state.history = row; state.readWorker = null; $("worker-detail").hidden = true; els.transcript.hidden = false; els.draft.value = ""; showView("chat"); renderChat();
    try {
      const messages = [];
      let offset = 0, charStart = 0, revision = null;
      for (let page = 0; page < 500; page++) {
        const query = new URLSearchParams({message_offset: offset, message_limit: 1, max_chars: 2000, char_start: charStart});
        const history = await api("GET", API_BASE + "/conversations/" + encodeURIComponent(row.conversation_id) + "/messages?" + query);
        if (selection !== state.selectionRevision) return;
        const currentRevision = history.version + ":" + history.message_epoch;
        if (revision !== null && revision !== currentRevision) throw "history_changed";
        revision = currentRevision;
        const message = history.messages[0];
        if (message) messages.push({...message, message_id: message.id, role: message.sender, content: message.text, offset: charStart});
        if (message?.has_more) {
          if (!Number.isInteger(message.returned_chars) || message.returned_chars <= 0 || message.returned_chars > 2000) throw "invalid_history_page";
          charStart += message.returned_chars;
        } else {
          charStart = 0;
          if (history.has_more === false) { renderMessages(messages); return; }
          const next = history.next_message_offset;
          if (history.has_more !== true || !Number.isInteger(next) || next <= offset || next >= history.message_total) throw "invalid_history_page";
          offset = next;
        }
      }
      throw "history_limit";
    } catch (error) { showError(error); }
  }
  function completeHostSelection(list, hostActionGeneration) {
    const id = state.hostSelectionCommand, command = state.outstandingCommands.get(id);
    const intent = command?.hostSelection;
    if (!intent) return false;
    if (intent.revision !== state.selectionRevision) {
      state.hostSelectionCommand = null; return false;
    }
    // Only lists begun after this exact host-bound command was retained may settle it.
    if (hostActionGeneration !== intent.generation) return false;
    if (list.server_epoch !== intent.server_epoch || list.console_id !== intent.console_id) {
      state.hostSelectionCommand = null; return false;
    }
    const receipt = command.receipt;
    if (!receipt) return false;
    if (receipt.command_id !== id || receipt.action !== intent.action || receipt.server_epoch !== intent.server_epoch || receipt.console_id !== intent.console_id || ["refused", "expired", "outcome_unknown"].includes(receipt.state)) {
      state.hostSelectionCommand = null; return false;
    }
    if (!["applied", "accepted"].includes(receipt.state) || !receipt.native_id) return false;
    const row = list.chats.find(row => row.live && row.server_epoch === intent.server_epoch && row.console_id === intent.console_id && row.session_id === receipt.native_id);
    if (!row) return false;
    // Complete only the original explicit action; normal detail confirmation still gates controls.
    state.hostSelectionCommand = null;
    state.chat = row; chooseTarget(targetOf(row, true));
    return true;
  }
  async function hostCommand(action, payload) {
    if (state.hostBusy || !navigator.onLine || (state.lastConfirmedUpdate && state.connectionState !== "connected")) return;
    const revision = state.selectionRevision;
    state.hostSelectionCommand = null;
    state.hostBusy = true; updateControls();
    try {
      const {host} = await api("POST", API_BASE + "/console/host", {protocol_version: 1});
      await submit(action, host, payload, null, {revision, action, server_epoch: host.server_epoch, console_id: host.console_id});
      await refresh();
    } catch (error) { showError(error); }
    finally { state.hostBusy = false; updateControls(); }
  }
  function renderMessages(messages) {
    const combined = new Map();
    for (const message of messages) {
      const current = combined.get(message.message_id) || {...message, content: ""};
      // Server offsets count Unicode code points; UTF-16 JS indices would corrupt emoji chunks.
      const points = Array.from(current.content);
      if ((message.offset || 0) > points.length) throw "message_gap";
      current.content = points.slice(0, message.offset || 0).join("") + (message.content || "");
      combined.set(message.message_id, current);
    }
    const opened = new Set([...els.transcript.querySelectorAll("details[open]")].map(el => el.dataset.messageId));
    const atBottom = els.transcript.scrollHeight - els.transcript.scrollTop - els.transcript.clientHeight < 40;
    const position = els.transcript.scrollTop;
    state.renderedBodies = combined;
    els.transcript.replaceChildren();
    for (const message of combined.values()) {
      const row = node("li", null, "role-" + (message.role === "user" ? "user" : message.role === "tool" ? "tool" : "assistant"));
      row.dataset.messageId = message.message_id;
      if (message.role === "tool") {
        const details = node("details"); details.dataset.messageId = message.message_id; details.open = opened.has(message.message_id);
        details.append(node("summary", "Tool activity"), node("pre", message.content, "tool-content")); row.append(details);
      } else row.append(node("span", message.role === "user" ? "You" : "Assistant", "message-role"), node("div", message.content, "message-body"));
      els.transcript.append(row);
    }
    if (!combined.size) els.transcript.append(node("li", "No messages yet.", "empty-note"));
    els.transcript.scrollTop = atBottom ? els.transcript.scrollHeight : position;
  }
  function renderLatest() {
    const excerpt = state.snapshot?.latest_message, target = state.selectedTarget;
    let shown = Boolean(state.targetConfirmed && target && !target.handle_id && !state.readWorker && excerpt &&
      ["server_epoch", "console_id", "session_id", "incarnation_id"].every(key => excerpt[key] === target[key]));
    if (shown && !state.historyLoading) {
      const body = state.renderedBodies?.get(excerpt.message_id)?.content;
      if (body !== undefined && Array.from(body).slice(excerpt.offset, excerpt.offset + Array.from(excerpt.content).length).join("") === excerpt.content) shown = false;
    }
    $("latest-output").hidden = !shown;
    if (!shown) return;
    $("latest-output-label").textContent = "Latest output excerpt · " + excerpt.role + " · history loading";
    if ($("latest-output-content").textContent !== excerpt.content) {
      $("latest-output-content").textContent = excerpt.content;
      $("latest-output").scrollTop = $("latest-output").scrollHeight;
    }
  }
  function renderChat() {
    const chat = state.chat;
    els.chatTitle.textContent = chat?.title || "Chat unavailable";
    if (state.snapshot && state.selectedTarget) renderMessages(state.snapshot.messages);
    else els.transcript.replaceChildren();
    const items = [];
    if (chat?.live) items.push({key: "main", label: "Main agent · " + statusLabel(chat), action: () => chooseTarget(targetOf(chat, true))});
    for (const worker of state.snapshot?.fleet || []) {
      if (items.some(item => item.key === worker.handle_id)) continue;
      items.push({key: worker.handle_id, runId: worker.run_id, label: "Worker " + worker.handle_id + " · " + worker.status,
        action: () => worker.live === false ? openWorker({id: worker.run_id, task: worker.task, status: worker.status}) : chooseTarget(targetOf(worker, true))});
    }
    const existing = new Map([...els.agents.children].map(li => [li.firstChild.dataset.agentKey, li]));
    for (const [index, item] of items.entries()) {
      let li = existing.get(item.key);
      if (!li) { li = node("li"); const control = node("button", "", "header-button"); control.type = "button"; control.dataset.agentKey = item.key; li.append(control); }
      existing.delete(item.key);
      const control = li.firstChild;
      if (control.textContent !== item.label) control.textContent = item.label;
      if (control.dataset.runId !== (item.runId || "")) control.dataset.runId = item.runId || "";
      control.onclick = item.action;
      if (els.agents.children[index] !== li) els.agents.insertBefore(li, els.agents.children[index] || null);
    }
    for (const li of existing.values()) li.remove();
    renderDecisions(); updateControls(); renderLatest();
  }
  function setDisabled(control, value) {
    const disabled = Boolean(value);
    if (control.disabled !== disabled) control.disabled = disabled;
  }
  function updateControls() {
    const target = state.selectedTarget;
    const worker = Boolean(target?.handle_id);
    const unknown = [...state.outstandingCommands.values()].some(row => row.receipt?.state === "outcome_unknown" || row.unknown);
    const liveWorker = target?.handle_id && state.snapshot?.fleet.find(row => row.handle_id === target.handle_id && row.run_id === target.run_id);
    if (unknown) els.connection.textContent = "State uncertain · unresolved command outcome · open terminal" + (state.lastConfirmedUpdate ? " · confirmed " + state.lastConfirmedUpdate : "");
    const ready = state.connectionState === "connected" && state.targetConfirmed && state.available && target && !unknown && (!target.handle_id || liveWorker?.live !== false);
    const draft = target ? draftFor(target) : {text: "", revision: 0};
    const pending = [...state.outstandingCommands.values()].some(row => row.submission && row.submission.key === (target && draftKey(target)) && row.submission.revision === draft.revision && !["accepted", "queued", "applied", "refused", "expired"].includes(row.receipt?.state));
    setDisabled(els.send, !ready || !draft.text.trim() || pending);
    const sendLabel = worker ? "Send guidance" : "Send";
    if (els.send.textContent !== sendLabel) els.send.textContent = sendLabel;
    els.steer.hidden = worker; els.redirect.hidden = worker;
    setDisabled(els.steer, !ready || !target?.run_id || !draft.text.trim() || pending);
    setDisabled(els.redirect, els.steer.disabled);
    setDisabled(els.stop, !ready || !(target?.run_id || target?.turn_id));
    setDisabled($("new-chat-button"), unknown || state.hostBusy || !navigator.onLine || (state.lastConfirmedUpdate && state.connectionState !== "connected"));
    setDisabled(els.draft, !target);
    $("composer").classList.toggle("readonly", Boolean(state.readWorker || state.history));
    els.targetLabel.textContent = state.readWorker ? (state.chat?.title || "Chat") + " · Worker " + state.readWorker.id + " · History (read only)" : state.history ? "Saved history · " + (state.chat?.title || "Chat") : target ?
      (state.chat?.title || "Chat") + " · " + (worker ? "Worker " + target.handle_id + " · run " + target.run_id : "Main agent") : "Choose a chat";
    els.targetState.textContent = state.readWorker ? "Recorded status: " + state.readWorker.status + ". Historical records do not grant live controls." : state.history ? "Read-only history. Open this saved chat to control its native agent." :
      state.targetMissing && target ? "Original target unavailable · draft retained here. Choose a current chat explicitly." : !state.targetConfirmed && target ? "Target not confirmed · controls paused." : state.chat ? statusLabel(state.chat) : "";
    if (liveWorker) {
      $("worker-detail").hidden = false; $("worker-detail").classList.add("live-activity");
      $("worker-activity").textContent = (liveWorker.agent || "Worker") + " · " + (liveWorker.task || "Activity unavailable") + " · queued guidance: " + (liveWorker.queued_steering ?? "unavailable") + " · undelivered: " + (liveWorker.undelivered_steering ?? "unavailable");
      $("worker-result").textContent = liveWorker.activity || "No result published yet.";
      $("worker-result-more").hidden = true;
    }
    if (state.history && !state.readWorker) {
      const open = button("Open saved chat", () => hostCommand("open", {conversation_id: state.history.conversation_id}));
      open.disabled = unknown || state.hostBusy || !navigator.onLine || (state.lastConfirmedUpdate && state.connectionState !== "connected");
      els.targetState.append(open);
    }
    const relevant = [...state.outstandingCommands.values()].filter(row => target && row.receipt?.target?.session_id === target.session_id && (row.receipt.target.handle_id || null) === (target.handle_id || null));
    els.commandStatus.textContent = relevant.map(row => receiptLabel(row.receipt, row.unknown)).join(" · ");
    for (const control of els.decisions.querySelectorAll("button, select, input, textarea")) setDisabled(control, !ready || state.view !== "chat");
  }
  function receiptLabel(receipt, unknown) {
    if (unknown) return "Response lost · checking command ID; no replay";
    if (!receipt) return "Submitting…";
    const action = receipt.action.replaceAll("_", " ");
    if (receipt.state === "pending") return action + ": Preparing · not accepted yet";
    if (receipt.action.includes("stop") && receipt.state === "accepted") return "Stop requested · physical exit not confirmed";
    if (receipt.state === "queued") return action + ": queued · consumption not confirmed";
    if (receipt.state === "outcome_unknown") return action + ": outcome unknown · open terminal";
    return action + ": " + receipt.state + (receipt.reason ? " · " + receipt.reason : "");
  }
  function applyReceipt(receipt) {
    const prior = state.outstandingCommands.get(receipt.command_id) || {};
    // Preparing cannot retract a later native acceptance already observed on another request.
    if (receipt.state === "pending" && prior.receipt && prior.receipt.state !== "pending") return;
    prior.receipt = receipt; prior.unknown = false;
    state.outstandingCommands.set(receipt.command_id, prior);
    if (prior.submission && ["accepted", "queued", "applied"].includes(receipt.state)) clearAcceptedDraft(prior.submission);
    updateControls();
  }
  async function submit(action, target, payload, submission, hostSelection = null) {
    const ticket = await api("POST", API_BASE + "/commands/tickets", {protocol_version: 1, action, target, payload});
    // Retain ID before the effect request. A lost ACK can only query this ID.
    state.outstandingCommands.set(ticket.command_id, {submission, unknown: false, hostSelection});
    if (hostSelection && hostSelection.revision === state.selectionRevision) {
      hostSelection.generation = ++state.hostActionGeneration;
      state.hostSelectionCommand = ticket.command_id;
    }
    updateControls();
    try {
      const receipt = await api("POST", API_BASE + "/commands", {protocol_version: 1, ticket: ticket.ticket, payload});
      applyReceipt(receipt); return receipt;
    } catch (error) {
      state.outstandingCommands.get(ticket.command_id).unknown = true;
      setConnection("uncertain"); await queryOutstanding(); showError(error); return null;
    }
  }
  async function queryOutstanding() {
    for (const [id, command] of state.outstandingCommands) {
      if (!command.unknown && ["applied", "refused", "expired"].includes(command.receipt?.state)) continue;
      try { applyReceipt(await api("GET", API_BASE + "/commands/" + encodeURIComponent(id))); }
      catch (error) { command.unknown = true; updateControls(); }
    }
  }
  const submitting = new Set();
  async function command(action) {
    const target = state.selectedTarget;
    const control = action === "send" ? els.send : action === "steer" ? els.steer : action === "redirect" ? els.redirect : els.stop;
    if (!target || control.disabled) return;
    const submission = captureSubmission(target, draftFor(target), state.chat.expected_revisions);
    const key = submission.key + ":" + submission.revision + ":" + action;
    if (submitting.has(key)) return;
    submitting.add(key);
    try {
      const nativeAction = target.handle_id ? action === "stop" ? "subagent_stop" : "subagent_guidance" : action;
      const payload = action === "stop" ? {} : action === "send" && !target.handle_id ?
        {text: submission.text, expected_revisions: submission.expected_revisions} : {text: submission.text};
      await submit(nativeAction, submission.target, payload, action === "stop" ? null : submission);
    } catch (error) { showError(error); }
    finally { submitting.delete(key); updateControls(); }
  }
  function publishAnswerability() {
    if (state.ws?.readyState !== WebSocket.OPEN) return;
    const answerable = !state.readWorker && state.view === "chat" && state.targetConfirmed && state.available && state.connectionState === "connected" && ![...state.outstandingCommands.values()].some(row => row.unknown || row.receipt?.state === "outcome_unknown");
    const signature = JSON.stringify([state.renderedDecisions, answerable]);
    // A stale-round reset must not provoke another identical rejected withdrawal.
    if (state.answerabilitySent?.socket === state.ws && state.answerabilitySent.signature === signature) return;
    state.answerabilitySent = {socket: state.ws, signature};
    for (const target of state.renderedDecisions) state.ws.send(JSON.stringify({type: "answerable", protocol_version: 1, target, answerable}));
  }
  function renderDecisions() {
    const rows = state.snapshot?.decisions || [];
    const signature = JSON.stringify([state.chat && targetOf(state.chat), rows.map(({answerable, ...round}) => round), state.snapshot?.recoveries, state.chats.map(row => [row.session_id, row.pending_decision_count])]);
    if (state.decisionSignature === signature) { publishAnswerability(); return; }
    state.decisionSignature = signature;
    // Withdraw old claims before replacing controls or changing the target.
    state.answerabilitySent = null;
    if (state.ws?.readyState === WebSocket.OPEN) for (const target of state.renderedDecisions) state.ws.send(JSON.stringify({type: "answerable", protocol_version: 1, target, answerable: false}));
    state.renderedDecisions = [];
    els.decisions.replaceChildren(); els.approvalsList.replaceChildren();
    const count = state.chats.reduce((total, row) => total + (row.pending_decision_count || 0), 0);
    els.approvalsBadge.hidden = !count; els.approvalsBadge.textContent = String(count);
    for (const row of state.chats.filter(row => row.pending_decision_count && row.session_id !== state.chat?.session_id)) {
      const overview = node("li"); overview.append(button((row.title || "Chat") + " · " + row.pending_decision_count + " pending decisions", () => openChat(row))); els.approvalsList.append(overview);
    }
    for (const round of rows) {
      const target = {...targetOf(state.chat), run_id: round.run_id, round_id: round.round_id};
      const card = node("details", null, "decision"); card.open = true;
      card.append(node("summary", "Decision · " + round.kind));
      const overview = node("li"); overview.append(button((state.chat?.title || "Chat") + " · " + round.kind, () => showView("chat"))); els.approvalsList.append(overview);
      const supported = round.supported && ["approval", "question"].includes(round.kind);
      if (!supported) { card.append(node("p", round.reason || "This review requires the terminal.")); const link = node("a", "Open terminal"); link.href = "/"; card.append(link); }
      else {
        const answers = [];
        for (const call of round.calls || []) {
          const label = node("label", call.llm_name || call.tool_name || "Tool");
          const select = node("select"); select.setAttribute("aria-label", "Decision for " + label.textContent);
          for (const option of call.options || []) { const el = node("option", APPROVAL_LABELS[option] || option); el.value = option; select.append(el); }
          label.append(select); card.append(label, node("pre", call.arguments_summary || "", "approval-args"));
          answers.push({call, select});
        }
        for (const question of round.questions || []) {
          const fieldset = node("fieldset"); fieldset.append(node("legend", question.question));
          const options = [];
          for (const option of question.options) {
            const label = node("label", option.label + (option.description ? " · " + option.description : ""));
            const input = node("input"); input.type = question.multiSelect ? "checkbox" : "radio"; input.name = round.round_id + ":" + answers.length; input.value = option.label; label.prepend(input); fieldset.append(label); options.push(input);
          }
          const text = node("textarea"); text.setAttribute("aria-label", "Additional answer for " + question.question); text.maxLength = 500;
          fieldset.append(text); card.append(fieldset); answers.push({question, options, text});
        }
        const decide = button("Submit decision", async () => {
          if (decide.disabled) return;
          const answer = round.kind === "approval" ? {decisions: Object.fromEntries(answers.map(({call, select}) => [call.call_id || call.llm_name, select.value]))} :
            {answers: answers.map(({question, options, text}) => ({question: question.question, selected: options.filter(input => input.checked).map(input => input.value), other_text: text.value || null, unanswered: false}))};
          decide.disabled = true;
          try { await submit("decision", target, {kind: round.kind, answer}); await refresh(); }
          catch (error) { showError(error); }
        }, "primary-button");
        card.append(decide); state.renderedDecisions.push(target);
      }
      els.decisions.append(card);
    }
    for (const _recovery of state.snapshot?.recoveries || []) { const p = node("p", "Unsent turn needs attention. "); const link = node("a", "Open terminal"); link.href = "/"; p.append(link); els.decisions.append(p); }
    els.approvalsEmpty.hidden = els.approvalsList.children.length > 0;
    publishAnswerability();
  }
  function showLibrarySection(section) {
    state.librarySection = section;
    const notes = section === "notes";
    els.libraryNotesButton.setAttribute("aria-pressed", String(notes));
    els.libraryMediaButton.setAttribute("aria-pressed", String(!notes));
    els.notesList.hidden = !notes;
    els.mediaList.hidden = notes;
    // The loaders re-derive each section's empty note + Load-more state;
    // hide everything until their responses land.
    els.notesEmpty.hidden = true;
    els.mediaEmpty.hidden = true;
    els.notesMoreButton.hidden = true;
    els.mediaMoreButton.hidden = true;
    if (notes) {
      loadNotes(true);
    } else {
      loadMedia(true);
    }
  }

  async function loadNotes(reset) {
    const nextPage = reset ? 1 : state.notesPage + 1;
    let data;
    try {
      data = await api("GET", API_BASE + "/notes?page=" + nextPage);
    } catch (code) {
      showError(code);
      return;
    }
    state.notesPage = typeof data.page === "number" ? data.page : nextPage;
    state.notesHasMore = Boolean(data.has_more);
    const items = Array.isArray(data.items) ? data.items : [];
    if (reset) els.notesList.replaceChildren();
    for (const note of items) {
      els.notesList.append(renderNoteRow(note));
    }
    els.notesEmpty.hidden = els.notesList.children.length > 0;
    els.notesMoreButton.hidden = !state.notesHasMore;
  }

  function renderNoteRow(note) {
    const row = document.createElement("li");
    row.dataset.noteId = String(note.id);
    const title = document.createElement("span");
    title.className = "entry-title";
    title.textContent = note.title || "Untitled note";
    const preview = document.createElement("span");
    preview.className = "entry-preview";
    preview.textContent = typeof note.preview === "string" ? note.preview : "";
    row.append(title, preview);
    row.addEventListener("click", () => openNote(note));
    return row;
  }

  async function loadMedia(reset) {
    const nextPage = reset ? 1 : state.mediaPage + 1;
    let data;
    try {
      data = await api("GET", API_BASE + "/library?page=" + nextPage);
    } catch (code) {
      showError(code);
      return;
    }
    state.mediaPage = typeof data.page === "number" ? data.page : nextPage;
    state.mediaHasMore = Boolean(data.has_more);
    const items = Array.isArray(data.items) ? data.items : [];
    if (reset) els.mediaList.replaceChildren();
    for (const item of items) {
      els.mediaList.append(renderMediaRow(item));
    }
    els.mediaEmpty.hidden = els.mediaList.children.length > 0;
    els.mediaMoreButton.hidden = !state.mediaHasMore;
  }

  function renderMediaRow(item) {
    const row = document.createElement("li");
    row.dataset.mediaId = String(item.media_id);
    const title = document.createElement("span");
    title.className = "entry-title";
    title.textContent = item.title || "Untitled media";
    const type = document.createElement("span");
    type.className = "entry-type";
    type.textContent =
      typeof item.media_type === "string" && item.media_type
        ? item.media_type
        : "media";
    row.append(title, type);
    row.addEventListener("click", () => openMediaItem(item));
    return row;
  }

  function detailMeta(fields, keys) {
    const parts = [];
    for (const key of keys) {
      const value = fields ? fields[key] : null;
      if (value !== null && value !== undefined && String(value) !== "") {
        parts.push(key.replace(/_/g, " ") + ": " + String(value));
      }
    }
    return parts.join(" · ");
  }

  async function openNote(note) {
    showView("note");
    els.noteTitle.textContent = note.title || "Untitled note";
    els.noteMeta.textContent = "";
    els.noteContent.textContent = "";
    let data;
    try {
      data = await api(
        "GET",
        API_BASE + "/notes/" + encodeURIComponent(String(note.id))
      );
    } catch (code) {
      showError(code);
      return;
    }
    const detail = (data && data.note) || {};
    els.noteTitle.textContent = detail.title || note.title || "Untitled note";
    els.noteMeta.textContent = detailMeta(detail, [
      "created_at",
      "last_modified",
    ]);
    els.noteContent.textContent =
      typeof detail.content === "string" ? detail.content : "";
  }

  async function openMediaItem(item) {
    showView("item");
    els.itemTitle.textContent = item.title || "Untitled media";
    els.itemMeta.textContent = "";
    els.itemTranscripts.textContent = "";
    let data;
    try {
      data = await api(
        "GET",
        API_BASE + "/library/" + encodeURIComponent(String(item.media_id))
      );
    } catch (code) {
      showError(code);
      return;
    }
    const detail = (data && data.item) || {};
    els.itemTitle.textContent = detail.title || item.title || "Untitled media";
    els.itemMeta.textContent = detailMeta(detail, [
      "media_type",
      "media_id",
      "ingestion_date",
      "last_modified",
    ]);
    const transcripts = Array.isArray(detail.transcripts)
      ? detail.transcripts
      : [];
    const texts = transcripts
      .map((entry) =>
        entry && typeof entry.transcription === "string"
          ? entry.transcription
          : ""
      )
      .filter(Boolean);
    if (texts.length > 0) {
      els.itemTranscripts.textContent = texts.join("\n\n—\n\n");
    } else {
      els.itemTranscripts.textContent =
        transcripts.length > 0
          ? "No transcript text available."
          : "No transcripts yet.";
    }
  }

  async function loadWorkers(reset) {
    const conversation = state.chat?.conversation_id;
    if (!conversation) { $("worker-history-state").textContent = "No saved worker history for this chat yet."; return; }
    if (state.workerReading) return;
    state.workerReading = true;
    if (reset || state.workerConversation !== conversation) {
      state.workerConversation = conversation; state.workerHistory = []; state.workerCursor = null;
    }
    try {
      const data = await api("GET", API_BASE + "/conversations/" + encodeURIComponent(conversation) + "/workers" + (state.workerCursor ? "?cursor=" + encodeURIComponent(state.workerCursor) : ""));
      if (state.chat?.conversation_id !== conversation) return;
      state.workerHistory.push(...data.items); state.workerCursor = data.next_cursor;
      $("worker-history").replaceChildren();
      for (const worker of state.workerHistory) {
        const li = node("li"); li.append(button((worker.task || "Worker " + worker.id) + " · recorded " + worker.status, () => openWorker(worker))); $("worker-history").append(li);
      }
      $("worker-history-state").textContent = state.workerHistory.length ? "History is read only." : "No saved worker history.";
      $("workers-more").hidden = !state.workerCursor;
    } catch (error) { $("worker-history-state").textContent = "Worker history unavailable. Open terminal for details."; }
    finally { state.workerReading = false; if (state.chat?.conversation_id !== conversation && !$("agent-panel").hidden) loadWorkers(true); }
  }
  function openWorker(worker) {
    state.selectionRevision++; state.selectedTarget = null; state.available = false;
    $("worker-detail").classList.remove("live-activity");
    $("latest-output").hidden = true;
    state.readWorker = worker; state.history = null; state.workerResultOffset = 0; state.workerResultRevision = null;
    els.draft.value = ""; els.transcript.hidden = true; $("worker-detail").hidden = false;
    $("worker-activity").textContent = worker.task || "Historical worker result";
    $("worker-result").textContent = "Loading result…";
    $("worker-result-more").hidden = true;
    showView("chat"); updateControls(); loadWorkerResult(true);
  }
  async function loadWorkerResult(reset) {
    const worker = state.readWorker, conversation = state.chat?.conversation_id;
    const selection = state.selectionRevision;
    if (!worker || !conversation) return;
    $("worker-result-more").disabled = true;
    try {
      const result = await api("GET", API_BASE + "/conversations/" + encodeURIComponent(conversation) + "/workers/" + encodeURIComponent(worker.id) + "?char_start=" + state.workerResultOffset + "&max_chars=2000");
      if (selection !== state.selectionRevision) return;
      const revision = JSON.stringify([result.updated_at, result.total_chars, result.status]);
      if (!reset && state.workerResultRevision !== revision) throw "result_changed";
      state.workerResultRevision = revision;
      $("worker-result").textContent = !result.available ? "Result unavailable. Open terminal for details." : (reset ? "" : $("worker-result").textContent) + result.text;
      state.workerResultOffset = result.next_offset;
      $("worker-result-more").hidden = !result.has_more;
    } catch (error) {
      if (selection !== state.selectionRevision) return;
      $("worker-result").textContent = "Result unavailable or changed. Select this worker again or open terminal.";
      $("worker-result-more").hidden = true;
    } finally { $("worker-result-more").disabled = false; }
  }
  function connectStream() {
    const socket = new WebSocket((location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host + API_BASE + "/stream", ["chatbook-v1", "csrf." + csrfToken]);
    state.ws = socket;
    socket.addEventListener("open", () => { state.wsAttempt = 0; setConnection("connecting"); refresh(); });
    socket.addEventListener("message", frame => {
      if (state.ws !== socket) return;
      let event; try { event = JSON.parse(frame.data); } catch (_) { return; }
      if (event.type === "ping") { socket.send(JSON.stringify({type: "pong"})); return; }
      if (["console.changed", "reset"].includes(event.type)) { setConnection("uncertain"); refresh(); }
    });
    socket.addEventListener("close", event => {
      if (state.ws !== socket) return;
      state.ws = null; state.ownerAvailable = false; state.hostAvailable = false; setConnection("disconnected"); renderChats();
      if (event.code === 4401) { location.assign("/auth/login"); return; }
      const delay = [1000, 2000, 5000, 15000][Math.min(state.wsAttempt++, 3)];
      setTimeout(connectStream, delay);
    });
    socket.addEventListener("error", () => socket.close());
  }
  els.backButton.addEventListener("click", () => {
    if (!$("agent-panel").hidden) { $("agent-panel").hidden = true; $("agents-button").setAttribute("aria-expanded", "false"); return; }
    showTab(state.activeTab);
  });
  els.tabChats.addEventListener("click", () => showTab("chats"));
  els.savedMore.addEventListener("click", loadSavedChats);
  els.tabLibrary.addEventListener("click", () => showTab("library"));
  els.libraryNotesButton.addEventListener("click", () => showLibrarySection("notes"));
  els.libraryMediaButton.addEventListener("click", () => showLibrarySection("media"));
  els.notesMoreButton.addEventListener("click", () => loadNotes(false));
  els.mediaMoreButton.addEventListener("click", () => loadMedia(false));
  $("new-chat-button").addEventListener("click", () => hostCommand("create", {workspace_id: "global"}));
  $("approvals-button").addEventListener("click", () => showView("approvals"));
  $("agents-button").addEventListener("click", () => {
    $("agent-panel").hidden = !$("agent-panel").hidden;
    $("agents-button").setAttribute("aria-expanded", String(!$("agent-panel").hidden));
    if (!$("agent-panel").hidden) loadWorkers(true);
  });
  $("composer").addEventListener("submit", event => { event.preventDefault(); command("send"); });
  els.steer.addEventListener("click", () => command("steer"));
  els.redirect.addEventListener("click", () => command("redirect"));
  els.stop.addEventListener("click", () => command("stop"));
  $("workers-more").addEventListener("click", () => loadWorkers(false));
  $("worker-result-more").addEventListener("click", () => loadWorkerResult(false));
  els.draft.addEventListener("input", () => {
    if (!state.selectedTarget) return;
    const draft = draftFor(state.selectedTarget);
    state.drafts.set(draftKey(state.selectedTarget), {text: els.draft.value, revision: draft.revision + 1}); updateControls();
  });
  window.addEventListener("offline", () => setConnection("disconnected"));
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") refresh(); });
  // visualViewport tracks the keyboard's visible area without scrolling the whole page.
  function fitViewport() { document.body.style.height = (window.visualViewport?.height || window.innerHeight) + "px"; }
  window.visualViewport?.addEventListener("resize", fitViewport);
  window.addEventListener("resize", fitViewport); fitViewport();
  window.setInterval(() => { if (document.visibilityState === "visible") refresh(); }, 12000);
  showView("chats"); refresh(); connectStream();
})();
