(() => {
  "use strict";

  const workbench = document.getElementById("served-workbench");
  const region = document.getElementById("served-canvas-region");
  const frame = document.getElementById("served-canvas-frame");
  const state = document.getElementById("served-canvas-state");
  const openButton = document.getElementById("served-open-canvas");
  const closeButton = document.getElementById("served-close-canvas");
  const restartButton = document.getElementById("terminal-restart");
  const terminal = document.getElementById("terminal");
  let latest = null;
  let userClosed = false;
  let pollTimer = null;

  // The terminal websocket is always same-origin. The server-substituted
  // absolute URL (from public_url, default localhost) only matches the page
  // host when that default is right; loading the page via 127.0.0.1, a LAN
  // IP, or any other name crossed the session-cookie boundary and the
  // socket died at the handshake ("Terminal session ended"). textual.js
  // reads this attribute when it connects at window.onload, which always
  // runs after this deferred script, so the override wins.
  if (terminal) {
    terminal.dataset.sessionWebsocketUrl =
      (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws";
  }

  function setStatus(message) {
    if (state.textContent !== message) state.textContent = message;
  }

  function safeCanvasPath(value) {
    if (typeof value !== "string") return null;
    const url = new URL(value, location.origin);
    if (url.origin !== location.origin || !url.pathname.startsWith("/canvas/")) return null;
    return `${url.pathname}${url.search}${url.hash}`;
  }

  function showCanvas({focusControl = false} = {}) {
    if (!latest?.url) return;
    userClosed = false;
    if (frame.getAttribute("src") !== latest.url) frame.setAttribute("src", latest.url);
    region.hidden = false;
    workbench.classList.remove("terminal-only");
    openButton.hidden = true;
    closeButton.hidden = false;
    setStatus("Canvas connected");
    if (focusControl) closeButton.focus();
    window.dispatchEvent(new Event("resize"));
  }

  function hideCanvas({focusControl = false} = {}) {
    userClosed = true;
    region.hidden = true;
    workbench.classList.add("terminal-only");
    openButton.hidden = !latest?.url;
    closeButton.hidden = true;
    setStatus(latest?.url ? "Canvas hidden" : "Terminal only");
    if (focusControl && !openButton.hidden) openButton.focus();
    window.dispatchEvent(new Event("resize"));
  }

  function disableCanvas() {
    latest = null;
    userClosed = false;
    // A 404 from /canvas/api/session means Canvas is unavailable at the
    // server (kill switch off): it cannot become ready again without a
    // server restart and page reload, so the 1 Hz poll would only burn
    // requests forever.
    if (pollTimer !== null) {
      window.clearInterval(pollTimer);
      pollTimer = null;
    }
    frame.setAttribute("src", "about:blank");
    region.hidden = true;
    workbench.classList.add("terminal-only");
    openButton.hidden = true;
    closeButton.hidden = true;
    setStatus("Canvas disabled — restart Chatbook after re-enabling it");
    window.dispatchEvent(new Event("resize"));
  }

  function applyState(detail) {
    if (!detail || typeof detail !== "object") return;
    if (detail.status === "ready") {
      const url = safeCanvasPath(detail.url);
      if (!url) return;
      latest = {url, revision_id: detail.revision_id || ""};
      openButton.hidden = false;
      if (!userClosed) showCanvas();
      return;
    }
    if (detail.status === "terminal_only") {
      latest = null;
      userClosed = false;
      region.hidden = true;
      frame.removeAttribute("src");
      workbench.classList.add("terminal-only");
      openButton.hidden = true;
      closeButton.hidden = true;
      setStatus("Terminal only");
      return;
    }
    if (detail.status === "disconnected" || detail.status === "reconnecting") {
      region.hidden = true;
      workbench.classList.add("terminal-only");
      openButton.hidden = true;
      closeButton.hidden = true;
      setStatus("Canvas reconnecting");
      return;
    }
    if (!latest) {
      region.hidden = true;
      workbench.classList.add("terminal-only");
      openButton.hidden = true;
      closeButton.hidden = true;
      setStatus("Terminal only");
    }
  }

  async function refreshCanvasState() {
    try {
      const response = await fetch("/canvas/api/session", {cache: "no-store"});
      if (response.status === 404) {
        disableCanvas();
        return;
      }
      if (!response.ok) throw new Error("unavailable");
      applyState(await response.json());
    } catch (_) {
      applyState({status: latest ? "reconnecting" : "terminal_only"});
    }
  }

  window.addEventListener("chatbook:canvas-state", (event) => applyState(event.detail));
  window.addEventListener("message", (event) => {
    if (
      event.origin === location.origin &&
      event.source === frame.contentWindow &&
      event.data?.type === "chatbook:canvas-close"
    ) {
      hideCanvas({focusControl: true});
    }
  });
  openButton.addEventListener("click", () => showCanvas({focusControl: true}));
  closeButton.addEventListener("click", () => hideCanvas({focusControl: true}));
  restartButton.addEventListener("click", () => location.reload());
  refreshCanvasState();
  pollTimer = window.setInterval(refreshCanvasState, 1000);
  window.addEventListener("pagehide", () => window.clearInterval(pollTimer), {once: true});
})();
