(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const ui = {
    selector: byId("canvas-selector"), title: byId("canvas-title"), temporary: byId("temporary-badge"),
    connection: byId("connection-state"), followState: byId("follow-state"), revision: byId("revision-label"),
    provenance: byId("provenance"), frame: byId("canvas-preview"), loading: byId("loading-state"),
    pin: byId("pin-button"), follow: byId("follow-button"), source: byId("source-button"),
    copy: byId("copy-button"), download: byId("download-button"), reload: byId("reload-button"), close: byId("close-button"),
    notice: byId("notice"), noticeCopy: byId("notice-copy"), noticePrevious: byId("notice-previous"),
    noticeFollow: byId("notice-follow"), noticeDismiss: byId("notice-dismiss"),
    compatibility: byId("compatibility"), compatibilityCopy: byId("compatibility-copy"),
    scriptsDisabled: byId("scripts-disabled-button"), sourcePanel: byId("source-panel"),
    sourceClose: byId("source-close-button"), sourceView: byId("source-view"),
  };
  const basePath = location.pathname;
  const api = (path) => new URL(path, location.href).href;
  let csrf = "";
  let selection = null;
  let displayedRevisionId = "";
  let previousRevisionId = "";
  let latestMetadata = {};
  let lastEventId = "";
  let following = true;
  let closed = false;
  let pollTimer = null;
  let pendingPlan = null;
  let currentPort = null;
  let rendererReady = false;

  async function post(path, value, extraHeaders = {}) {
    const headers = {"Content-Type": "application/json", ...extraHeaders};
    if (csrf) headers["X-Canvas-CSRF"] = csrf;
    const response = await fetch(api(path), {method: "POST", headers, body: JSON.stringify(value), cache: "no-store"});
    if (!response.ok) throw new Error(`Canvas request failed: ${response.status}`);
    return response.json();
  }

  function setConnection(label, disconnected = false) {
    ui.connection.textContent = label;
    ui.connection.classList.toggle("state-disconnected", disconnected);
  }

  function setFollowing(value) {
    following = value;
    ui.followState.textContent = value ? "Following" : "Pinned";
    ui.followState.className = `state-badge ${value ? "state-follow" : "state-pinned"}`;
    ui.pin.hidden = !value;
    ui.follow.hidden = value;
  }

  function rememberCanvas(canvasId, title) {
    if (![...ui.selector.options].some((option) => option.value === canvasId)) {
      ui.selector.add(new Option(title || "Canvas", canvasId));
    }
    ui.selector.value = canvasId;
  }

  function applyMetadata(metadata = {}) {
    latestMetadata = {...latestMetadata, ...metadata};
    const title = typeof latestMetadata.title === "string" && latestMetadata.title ? latestMetadata.title : "Canvas";
    ui.title.value = title;
    rememberCanvas(selection.canvas_id, title);
    const sequence = Number.isInteger(latestMetadata.sequence) ? latestMetadata.sequence : "—";
    ui.revision.textContent = `Revision ${sequence}`;
    ui.temporary.hidden = latestMetadata.temporary !== true;
    const message = latestMetadata.origin_message_id || "unknown message";
    const turn = latestMetadata.origin_turn_id || "unknown turn";
    ui.provenance.textContent = `From ${message} · ${turn} · ${selection.revision_id}`;
  }

  function showNotice(copy, {previous = false, follow = false} = {}) {
    ui.noticeCopy.textContent = copy;
    ui.noticePrevious.hidden = !previous;
    ui.noticeFollow.hidden = !follow;
    ui.notice.hidden = false;
  }

  function dismissNotice() { ui.notice.hidden = true; }

  async function mintAction(action) {
    return (await post("api/actions", {action})).capability;
  }

  async function readSource() {
    const capability = await mintAction("source_read");
    const response = await fetch(api("api/source"), {headers: {Authorization: `CanvasCapability ${capability}`}, cache: "no-store"});
    if (!response.ok) throw new Error("Source is unavailable for this revision.");
    return response.text();
  }

  async function loadFrame({updated = false, scriptsDisabled = false} = {}) {
    rendererReady = false;
    pendingPlan = null;
    if (currentPort) currentPort.close();
    currentPort = null;
    ui.loading.hidden = false;
    ui.loading.textContent = "Preparing isolated preview…";
    const frame = await post("api/frame", {});
    const planResponse = await fetch(api("api/plan"), {cache: "no-store"});
    if (!planResponse.ok) throw new Error("Canvas render plan is unavailable.");
    const planPayload = await planResponse.json();
    const issues = Array.isArray(planPayload.compatibility_issues) ? planPayload.compatibility_issues : [];
    const {compatibility_issues: _shellOnlyIssues, ...rendererPlan} = planPayload;
    pendingPlan = rendererPlan;
    if (scriptsDisabled) pendingPlan.scripts = [];
    ui.compatibility.hidden = issues.length === 0;
    ui.compatibilityCopy.textContent = issues.map((issue) => issue.message).join(" ");
    ui.frame.src = frame.renderer_url;
    previousRevisionId = displayedRevisionId;
    displayedRevisionId = selection.revision_id;
    if (updated) showNotice("Updated · Undo / View previous", {previous: Boolean(previousRevisionId)});
    if (scriptsDisabled) showNotice("Opened with generated scripts disabled.");
  }

  function initializeRenderer() {
    if (!rendererReady || !pendingPlan || !ui.frame.contentWindow) return;
    const channel = new MessageChannel();
    const nonce = crypto.randomUUID();
    currentPort = channel.port1;
    channel.port1.onmessage = (event) => {
      const message = event.data;
      if (!message || typeof message !== "object") return;
      if (message.type === "canvas:execution-started") {
        channel.port1.postMessage({type: "canvas:execution-ack", nonce});
      }
      if (message.type === "canvas:status") {
        if (message.state === "ready") {
          ui.loading.hidden = true;
          setConnection("Connected");
        } else if (message.state === "failed") {
          ui.loading.textContent = message.message || "Canvas preview failed. Inspect source or reload.";
          ui.loading.hidden = false;
        }
      }
    };
    channel.port1.start();
    // A sandboxed renderer has an opaque receiving origin, so the browser
    // requires "*" here. Authority remains bound to this exact contentWindow,
    // the private MessagePort, the one-load nonce, and server-minted plan.
    ui.frame.contentWindow.postMessage({type: "canvas:init", nonce, plan: pendingPlan}, "*", [channel.port2]);
  }

  async function pollEvents() {
    if (closed) return;
    try {
      const headers = lastEventId ? {"Last-Event-ID": lastEventId} : {};
      const response = await fetch(api("api/events"), {headers, cache: "no-store"});
      if (!response.ok) throw new Error("event channel unavailable");
      const payload = await response.json();
      for (const event of payload.events || []) {
        lastEventId = event.event_id;
        if (event.kind === "disconnected") {
          setConnection("Disconnected", true);
          continue;
        }
        if (event.kind === "discarded") {
          showNotice("Draft update discarded");
          continue;
        }
        const changed = event.revision_id !== displayedRevisionId;
        selection = {canvas_id: event.canvas_id, revision_id: event.revision_id};
        applyMetadata(event.metadata);
        if (!changed) continue;
        if (following || event.kind === "selection_changed") {
          await loadFrame({updated: event.kind === "updated"});
        } else {
          showNotice("New version available", {follow: true});
        }
      }
      setConnection("Connected");
    } catch (_) {
      setConnection("Disconnected", true);
    } finally {
      if (!closed) pollTimer = window.setTimeout(pollEvents, 350);
    }
  }

  async function boot() {
    const params = new URLSearchParams(location.hash.slice(1));
    const bootstrap = params.get("boot");
    history.replaceState(null, "", basePath);
    if (!bootstrap) throw new Error("This Canvas link has expired. Reopen it from Chatbook.");
    const result = await post("api/boot", {bootstrap});
    csrf = result.csrf;
    selection = result.selection;
    rememberCanvas(selection.canvas_id, "Canvas");
    applyMetadata();
    await loadFrame();
    void pollEvents();
  }

  window.addEventListener("message", (event) => {
    if (!["null", location.origin].includes(event.origin) || event.source !== ui.frame.contentWindow || !event.data || event.data.type !== "canvas:renderer-ready") return;
    rendererReady = true;
    initializeRenderer();
  });

  ui.pin.addEventListener("click", () => setFollowing(false));
  ui.follow.addEventListener("click", async () => { setFollowing(true); dismissNotice(); await loadFrame({updated: displayedRevisionId !== selection.revision_id}); });
  ui.noticeFollow.addEventListener("click", () => ui.follow.click());
  ui.noticePrevious.addEventListener("click", () => { setFollowing(false); dismissNotice(); });
  ui.noticeDismiss.addEventListener("click", dismissNotice);
  ui.reload.addEventListener("click", () => loadFrame());
  ui.scriptsDisabled.addEventListener("click", () => loadFrame({scriptsDisabled: true}));
  ui.source.addEventListener("click", async () => {
    try {
      ui.sourceView.value = await readSource();
      ui.sourcePanel.hidden = false;
      ui.source.setAttribute("aria-expanded", "true");
      ui.sourceClose.focus();
    } catch (error) { showNotice(error.message); }
  });
  ui.sourceClose.addEventListener("click", () => { ui.sourcePanel.hidden = true; ui.source.setAttribute("aria-expanded", "false"); ui.source.focus(); });
  ui.copy.addEventListener("click", async () => {
    try {
      ui.sourceView.value = await readSource();
      ui.sourceView.select();
      const copied = document.execCommand("copy");
      showNotice(copied ? "Source copied · Running it outside Chatbook bypasses the Canvas sandbox." : "Source ready to copy in Inspect source.");
    } catch (error) { showNotice(error.message); }
  });
  ui.download.addEventListener("click", async () => {
    try {
      const capability = await mintAction("source_download");
      const response = await fetch(api("api/source-download"), {headers: {Authorization: `CanvasCapability ${capability}`}, cache: "no-store"});
      if (!response.ok) throw new Error("Source download is unavailable for this revision.");
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = "canvas-source.canvas.html.txt";
      document.body.append(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      showNotice("Source download requested as inert text.");
    } catch (error) { showNotice(error.message); }
  });
  ui.close.addEventListener("click", async () => {
    closed = true;
    if (pollTimer) clearTimeout(pollTimer);
    try { await post("api/close", {}); } catch (_) { /* already disconnected */ }
    ui.frame.removeAttribute("src");
    ui.loading.textContent = "Canvas closed. Reopen it from Chatbook.";
    ui.loading.hidden = false;
    setConnection("Disconnected", true);
  });
  ui.title.addEventListener("change", () => {
    const title = ui.title.value.trim() || "Canvas";
    ui.title.value = title;
    ui.selector.selectedOptions[0].textContent = title;
    showNotice("Title edited locally · the next Chatbook rename revision will preserve this value.");
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (!ui.sourcePanel.hidden) ui.sourceClose.click();
    else if (!ui.notice.hidden) dismissNotice();
  });

  boot().catch((error) => {
    setConnection("Disconnected", true);
    ui.loading.textContent = error.message || "Canvas could not connect. Reopen it from Chatbook.";
    ui.loading.hidden = false;
  });
})();
