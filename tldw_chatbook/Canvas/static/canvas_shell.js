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
    runnableDownload: byId("runnable-download-button"), bridgeDialog: byId("bridge-dialog"),
    bridgeHeading: byId("bridge-heading"), bridgeSummary: byId("bridge-summary"), bridgeKind: byId("bridge-kind"),
    bridgeTarget: byId("bridge-target"), bridgeFilenameRow: byId("bridge-filename-row"), bridgeFilename: byId("bridge-filename"),
    bridgeMimeRow: byId("bridge-mime-row"), bridgeMime: byId("bridge-mime"), bridgeSizeRow: byId("bridge-size-row"), bridgeSize: byId("bridge-size"),
    bridgeTextRegion: byId("bridge-text-region"), bridgeText: byId("bridge-complete-text"), bridgeRecovery: byId("bridge-recovery"),
    bridgeCancel: byId("bridge-cancel-button"), bridgeCopy: byId("bridge-copy-button"), bridgeRetry: byId("bridge-retry-button"),
    bridgeSwitch: byId("bridge-switch-button"), bridgeConfirm: byId("bridge-confirm-button"),
  };
  const basePath = location.pathname;
  const api = (path) => new URL(path, location.href).href;
  let csrf = "";
  let selection = null;
  let displayedRevisionId = "";
  let displayedMetadata = {};
  let latestRevisionId = "";
  let lastEventId = "";
  let following = true;
  let closed = false;
  let pollTimer = null;
  let pendingPlan = null;
  let currentPort = null;
  let rendererReady = false;
  let branchUnavailable = false;
  let pendingBridge = null;
  let cancellingBridge = false;

  async function post(path, value, extraHeaders = {}) {
    const headers = {"Content-Type": "application/json", ...extraHeaders};
    if (csrf) headers["X-Canvas-CSRF"] = csrf;
    const response = await fetch(api(path), {method: "POST", headers, body: JSON.stringify(value), cache: "no-store"});
    if (!response.ok) throw new Error(`Canvas request failed: ${response.status}`);
    return response.json();
  }

  async function postWithCapability(path, value, action) {
    const capability = await mintAction(action);
    return post(path, value, {Authorization: `CanvasCapability ${capability}`});
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

  function applyProjection(projection) {
    selection = projection.selection;
    displayedRevisionId = selection.revision_id;
    latestRevisionId = selection.revision_id;
    displayedMetadata = {...projection.metadata};
    ui.selector.replaceChildren(...(projection.options || []).map((option) => new Option(option.title, option.canvas_id)));
    ui.selector.value = selection.canvas_id;
    const title = typeof displayedMetadata.title === "string" && displayedMetadata.title ? displayedMetadata.title : "Canvas";
    ui.title.value = title;
    const sequence = Number.isInteger(displayedMetadata.sequence) ? displayedMetadata.sequence : "—";
    ui.revision.textContent = `Revision ${sequence}`;
    ui.temporary.hidden = displayedMetadata.temporary !== true;
    const message = displayedMetadata.origin_message_id || "unknown message";
    const turn = displayedMetadata.origin_turn_id || "unknown turn";
    ui.provenance.textContent = `From ${message} · ${turn} · ${selection.revision_id}`;
    setFollowing(projection.following === true);
  }

  async function navigate(action, values = {}, {updated = false, reload = true} = {}) {
    await cancelPendingBridge({restoreFocus: false});
    const previous = displayedRevisionId;
    const projection = await post("api/navigate", {action, ...values});
    applyProjection(projection);
    if (reload) await loadFrame({updated, previousRevisionId: previous});
  }

  function showNotice(copy, {previous = false, follow = false} = {}) {
    ui.noticeCopy.textContent = copy;
    ui.noticePrevious.hidden = !previous;
    ui.noticeFollow.hidden = !follow;
    ui.notice.hidden = false;
  }

  function dismissNotice() { ui.notice.hidden = true; }

  function setBackgroundForDialog(open) {
    for (const child of document.querySelector(".canvas-workbench").children) {
      if (child !== ui.bridgeDialog) child.inert = open || (!ui.sourcePanel.hidden && child !== ui.sourcePanel);
    }
  }

  function closeBridgeDialog({restoreFocus = true} = {}) {
    const pending = pendingBridge;
    if (!pending) return;
    if (pending.timer) clearTimeout(pending.timer);
    pendingBridge = null;
    ui.bridgeDialog.hidden = true;
    ui.bridgeRecovery.hidden = true;
    ui.bridgeRetry.hidden = true;
    ui.bridgeSwitch.hidden = true;
    ui.bridgeConfirm.hidden = false;
    setBackgroundForDialog(false);
    if (restoreFocus && pending.returnFocus?.isConnected) pending.returnFocus.focus();
    pending.request = null;
    pending.presentation = null;
    pending.source = null;
  }

  async function cancelPendingBridge({notifyServer = true, restoreFocus = true} = {}) {
    const pending = pendingBridge;
    if (!pending) return;
    const request = pending.request;
    const shouldNotify = notifyServer && pending.mode === "bridge" && request && pending.prepared;
    if (pendingBridge === pending) closeBridgeDialog({restoreFocus});
    if (shouldNotify) {
      cancellingBridge = true;
      try {
        await postWithCapability("api/bridge", {approved: false, request}, "bridge_confirm");
      } catch (_) { /* expiry or invalidation already cancelled process authority */ }
      finally { cancellingBridge = false; }
    }
  }

  function completeJson(value) {
    if (value === null || typeof value !== "object") return JSON.stringify(value);
    if (Array.isArray(value)) return `[${value.map(completeJson).join(",")}]`;
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${completeJson(value[key])}`).join(",")}}`;
  }

  function validateBridgeMessage(message) {
    if (Object.keys(message).sort().join(",") !== "kind,nonce,request_id,type,value" ||
        message.type !== "canvas:bridge-request" || typeof message.request_id !== "string" ||
        !message.request_id || message.request_id.length > 256 || !["submit", "download"].includes(message.kind)) {
      throw new Error("Canvas request was refused by the trusted shell.");
    }
    const request = {version: "canvas-v1", request_id: message.request_id, kind: message.kind, value: message.value};
    const encoded = message.kind === "submit" && typeof message.value === "string" ? message.value : completeJson(message.value);
    const limit = message.kind === "submit" ? 16 * 1024 : 13985108;
    if (typeof encoded !== "string" || new TextEncoder().encode(encoded).length > limit) {
      throw new Error("Canvas request exceeds its trusted-shell limit.");
    }
    return request;
  }

  function showBridgeDialog(pending, presentation) {
    pending.presentation = presentation;
    pending.prepared = true;
    ui.bridgeHeading.textContent = presentation.kind === "submit" ? "Send result to chat" : "Download generated file";
    ui.bridgeSummary.textContent = presentation.kind === "submit"
      ? "Confirm to replace the unchanged Chatbook composer with this unsent draft. Nothing is sent automatically."
      : "Confirm to download this passive generated file from the trusted Canvas shell.";
    ui.bridgeKind.textContent = presentation.kind === "submit" ? "Unsent draft" : "Passive file";
    ui.bridgeTarget.textContent = `Conversation ${presentation.conversation_id} · Canvas ${presentation.canvas_id} · Revision ${presentation.revision_id}`;
    ui.bridgeFilenameRow.hidden = presentation.filename === null;
    ui.bridgeMimeRow.hidden = presentation.mime_type === null;
    ui.bridgeSizeRow.hidden = presentation.byte_size === null;
    ui.bridgeFilename.textContent = presentation.filename || "";
    ui.bridgeMime.textContent = presentation.mime_type || "";
    ui.bridgeSize.textContent = presentation.byte_size === null ? "" : `${presentation.byte_size} bytes`;
    ui.bridgeTextRegion.hidden = presentation.complete_text === null;
    ui.bridgeText.tabIndex = presentation.complete_text === null ? -1 : 0;
    ui.bridgeText.value = presentation.complete_text || "";
    ui.bridgeCopy.hidden = presentation.complete_text === null;
    ui.bridgeConfirm.textContent = presentation.kind === "submit" ? "Send to composer" : "Download file";
    ui.bridgeDialog.inert = false;
    ui.bridgeDialog.hidden = false;
    setBackgroundForDialog(true);
    ui.bridgeCancel.focus();
    pending.timer = setTimeout(() => {
      if (pendingBridge !== pending) return;
      void cancelPendingBridge({notifyServer: false});
      showNotice("Canvas confirmation expired. Request it again from the preview.");
    }, 30_000);
  }

  async function prepareBridgeMessage(message) {
    if (cancellingBridge) {
      showNotice("The previous Canvas confirmation is still cancelling. Try again.");
      return;
    }
    if (pendingBridge) {
      ui.bridgeRecovery.textContent = "Another request was refused while this confirmation is pending.";
      ui.bridgeRecovery.hidden = false;
      return;
    }
    let request;
    try { request = validateBridgeMessage(message); }
    catch (error) { showNotice(error.message); return; }
    const pending = {mode: "bridge", request, presentation: null, prepared: false, returnFocus: ui.frame, timer: null, source: null};
    pendingBridge = pending;
    try {
      const presentation = await postWithCapability("api/bridge/prepare", {request}, "bridge_prepare");
      if (pendingBridge === pending) showBridgeDialog(pending, presentation);
    } catch (_) {
      if (pendingBridge === pending) closeBridgeDialog({restoreFocus: false});
      showNotice("Canvas request could not be confirmed. Reload the preview and try again.");
    }
  }

  function generatedBlob(request, presentation) {
    const value = request.value;
    if (presentation.mime_type.startsWith("image/")) {
      const binary = atob(value.data.split(",", 2)[1]);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
      return new Blob([bytes], {type: presentation.mime_type});
    }
    return new Blob([value.data], {type: `${presentation.mime_type};charset=utf-8`});
  }

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.append(link);
    try { link.click(); }
    finally {
      link.remove();
      URL.revokeObjectURL(url);
    }
  }

  function showBranchUnavailable() {
    void cancelPendingBridge({notifyServer: false, restoreFocus: false});
    branchUnavailable = true;
    rendererReady = false;
    pendingPlan = null;
    latestRevisionId = "";
    if (currentPort) currentPort.close();
    currentPort = null;
    ui.frame.src = "about:blank";
    ui.sourceView.value = "";
    ui.sourcePanel.hidden = true;
    ui.source.setAttribute("aria-expanded", "false");
    for (const child of document.querySelector(".canvas-workbench").children) child.inert = false;
    ui.loading.textContent = "Unavailable on this branch. Return to Chatbook and reopen this Canvas from a reachable transcript card.";
    ui.loading.hidden = false;
    ui.compatibility.hidden = true;
    dismissNotice();
    setConnection("Disconnected", true);
    for (const control of document.querySelectorAll(".canvas-toolbar button, .canvas-toolbar select, .canvas-toolbar input")) {
      if (control !== ui.close) control.disabled = true;
    }
    ui.close.focus();
  }

  async function mintAction(action) {
    return (await post("api/actions", {action})).capability;
  }

  async function readSource() {
    const capability = await mintAction("source_read");
    const response = await fetch(api("api/source"), {headers: {Authorization: `CanvasCapability ${capability}`}, cache: "no-store"});
    if (!response.ok) throw new Error("Source is unavailable for this revision.");
    return response.text();
  }

  async function loadFrame({updated = false, scriptsDisabled = false, previousRevisionId = ""} = {}) {
    await cancelPendingBridge({restoreFocus: false});
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
    if (updated) showNotice("Updated · View previous", {previous: Boolean(previousRevisionId || displayedMetadata.parent_revision_id)});
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
      if (message.nonce !== nonce) return;
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
          ui.compatibility.hidden = false;
          byId("compatibility-title").textContent = "Preview issue";
          ui.compatibilityCopy.textContent = "The generated script failed in the isolated runtime. You can retry without generated scripts.";
          ui.scriptsDisabled.hidden = false;
        }
      }
      if (message.type === "canvas:bridge-request") void prepareBridgeMessage(message);
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
          if (event.metadata?.notice === "unavailable_on_branch") showBranchUnavailable();
          else setConnection("Disconnected", true);
          continue;
        }
        if (event.kind === "discarded") {
          showNotice("Draft update discarded");
          continue;
        }
        const changed = event.revision_id !== displayedRevisionId;
        latestRevisionId = event.revision_id;
        if (!changed) continue;
        if (following) {
          await navigate("follow", {}, {updated: event.kind === "updated"});
        } else {
          showNotice("New version available", {follow: true});
        }
      }
      if (!branchUnavailable) setConnection("Connected");
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
    const projectionResponse = await fetch(api("api/state"), {cache: "no-store"});
    if (!projectionResponse.ok) throw new Error("Canvas state is unavailable.");
    applyProjection(await projectionResponse.json());
    await loadFrame();
    void pollEvents();
  }

  window.addEventListener("message", (event) => {
    if (!["null", location.origin].includes(event.origin) || event.source !== ui.frame.contentWindow || !event.data || event.data.type !== "canvas:renderer-ready") return;
    rendererReady = true;
    initializeRenderer();
  });

  ui.selector.addEventListener("change", () => navigate("select", {canvas_id: ui.selector.value}));
  ui.pin.addEventListener("click", () => navigate("pin", {}, {reload: false}));
  ui.follow.addEventListener("click", async () => { dismissNotice(); await navigate("follow", {}, {updated: displayedRevisionId !== latestRevisionId}); });
  ui.noticeFollow.addEventListener("click", () => ui.follow.click());
  ui.noticePrevious.addEventListener("click", async () => { dismissNotice(); await navigate("previous"); });
  ui.noticeDismiss.addEventListener("click", dismissNotice);
  ui.reload.addEventListener("click", () => loadFrame());
  ui.scriptsDisabled.addEventListener("click", () => loadFrame({scriptsDisabled: true}));
  ui.source.addEventListener("click", async () => {
    try {
      ui.sourceView.value = await readSource();
      ui.sourcePanel.hidden = false;
      for (const child of document.querySelector(".canvas-workbench").children) {
        if (child !== ui.sourcePanel) child.inert = true;
      }
      ui.source.setAttribute("aria-expanded", "true");
      ui.sourceClose.focus();
    } catch (error) { showNotice(error.message); }
  });
  ui.sourceClose.addEventListener("click", () => {
    ui.sourcePanel.hidden = true;
    for (const child of document.querySelector(".canvas-workbench").children) child.inert = false;
    ui.source.setAttribute("aria-expanded", "false");
    ui.source.focus();
  });
  ui.runnableDownload.addEventListener("click", async () => {
    if (pendingBridge) return;
    try {
      const source = ui.sourceView.value || await readSource();
      let title = (ui.title.value || "canvas").trim().replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^\.+|\.+$/g, "").replace(/\.html$/i, "") || "canvas";
      if (/^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(title)) title = "canvas";
      const pending = {mode: "runnable", request: null, presentation: null, prepared: true, returnFocus: ui.runnableDownload, timer: null, source};
      pendingBridge = pending;
      ui.bridgeHeading.textContent = "Run outside the Canvas sandbox?";
      ui.bridgeSummary.textContent = "This HTML runs outside Chatbook and bypasses Canvas zero-egress and sandbox protections.";
      ui.bridgeKind.textContent = "Runnable HTML";
      ui.bridgeTarget.textContent = `Canvas ${selection.canvas_id} · Revision ${displayedRevisionId}`;
      ui.bridgeFilenameRow.hidden = false;
      ui.bridgeFilename.textContent = `${title}.html`;
      ui.bridgeMimeRow.hidden = false;
      ui.bridgeMime.textContent = "text/html";
      ui.bridgeSizeRow.hidden = false;
      ui.bridgeSize.textContent = `${new TextEncoder().encode(source).length} bytes`;
      ui.bridgeTextRegion.hidden = false;
      ui.bridgeText.tabIndex = 0;
      ui.bridgeText.value = source;
      ui.bridgeCopy.hidden = false;
      ui.bridgeConfirm.textContent = "Download runnable HTML";
      ui.bridgeDialog.inert = false;
      ui.bridgeDialog.hidden = false;
      setBackgroundForDialog(true);
      ui.bridgeCancel.focus();
    } catch (error) { showNotice(error.message); }
  });
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
    await cancelPendingBridge({restoreFocus: false});
    try { await post("api/close", {}); } catch (_) { /* already disconnected */ }
    ui.frame.removeAttribute("src");
    ui.loading.textContent = "Canvas closed. Reopen it from Chatbook.";
    ui.loading.hidden = false;
    setConnection("Disconnected", true);
  });
  ui.bridgeCancel.addEventListener("click", () => void cancelPendingBridge());
  ui.bridgeCopy.addEventListener("click", () => {
    ui.bridgeText.select();
    ui.bridgeRecovery.textContent = document.execCommand("copy")
      ? "Content copied."
      : "Content is selected and ready to copy.";
    ui.bridgeRecovery.hidden = false;
  });
  ui.bridgeConfirm.addEventListener("click", async () => {
    const pending = pendingBridge;
    if (!pending) return;
    ui.bridgeConfirm.disabled = true;
    try {
      if (pending.mode === "runnable") {
        downloadBlob(new Blob([pending.source], {type: "text/html;charset=utf-8"}), ui.bridgeFilename.textContent);
        closeBridgeDialog();
        showNotice("Runnable HTML download requested outside Canvas protections.");
        return;
      }
      const result = await postWithCapability("api/bridge", {approved: true, request: pending.request}, "bridge_confirm");
      if (pendingBridge !== pending) return;
      if (result.status !== "confirmed") throw new Error("refused");
      if (pending.request.kind === "submit") {
        closeBridgeDialog();
        showNotice("Draft inserted · Review it in Chatbook before sending.");
      } else {
        const blob = generatedBlob(pending.request, pending.presentation);
        const filename = pending.presentation.filename;
        closeBridgeDialog();
        downloadBlob(blob, filename);
        showNotice("Generated file download requested.");
      }
    } catch (_) {
      if (pendingBridge !== pending) return;
      ui.bridgeRecovery.textContent = pending.request.kind === "submit"
        ? "The Chatbook draft changed. Nothing was inserted."
        : "The Canvas selection changed or this confirmation expired. Nothing was downloaded.";
      ui.bridgeRecovery.hidden = false;
      ui.bridgeRetry.hidden = false;
      ui.bridgeSwitch.hidden = pending.request.kind !== "submit";
      ui.bridgeConfirm.hidden = true;
      ui.bridgeCancel.focus();
    } finally {
      ui.bridgeConfirm.disabled = false;
    }
  });
  ui.bridgeRetry.addEventListener("click", async () => {
    const pending = pendingBridge;
    if (!pending || pending.mode !== "bridge") return;
    const message = {type: "canvas:bridge-request", request_id: pending.request.request_id, kind: pending.request.kind, value: pending.request.value};
    closeBridgeDialog({restoreFocus: false});
    await prepareBridgeMessage(message);
  });
  ui.bridgeSwitch.addEventListener("click", () => {
    showNotice("Return to the matching Chatbook conversation, then choose Retry or copy this result.");
  });
  ui.title.addEventListener("change", async () => {
    const previous = displayedMetadata.title || "Canvas";
    try {
      await navigate("rename", {title: ui.title.value});
      showNotice("Title saved as a new revision.");
    } catch (_) {
      ui.title.value = previous;
      showNotice("Title was not changed. Use 1–200 characters.");
    }
  });
  document.addEventListener("keydown", (event) => {
    if (!ui.bridgeDialog.hidden && event.key === "Tab") {
      const focusable = [...ui.bridgeDialog.querySelectorAll("button:not([hidden]):not(:disabled), textarea:not([hidden]):not([tabindex='-1'])")];
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    } else if (!ui.sourcePanel.hidden && event.key === "Tab") {
      const focusable = [...ui.sourcePanel.querySelectorAll("button, textarea")];
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
    if (event.key !== "Escape") return;
    if (!ui.bridgeDialog.hidden) void cancelPendingBridge();
    else if (!ui.sourcePanel.hidden) ui.sourceClose.click();
    else if (!ui.notice.hidden) dismissNotice();
  });

  boot().catch((error) => {
    setConnection("Disconnected", true);
    ui.loading.textContent = error.message || "Canvas could not connect. Reopen it from Chatbook.";
    ui.loading.hidden = false;
  });
})();
