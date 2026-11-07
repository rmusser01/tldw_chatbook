// Chatbook served device manager (ADR-233, task 4; QR pairing in task 5).
// Reads the per-session CSRF token from the page's meta tag and performs
// same-origin fetch POSTs carrying the CSRF header; the server never embeds
// secrets in this file, which is served with the shared static-asset cache
// policy.
(() => {
  "use strict";

  const csrfMeta = document.querySelector('meta[name="chatbook-csrf"]');
  const csrf = csrfMeta ? csrfMeta.content : "";

  async function post(url, body) {
    const response = await fetch(url, {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        "X-Chatbook-CSRF": csrf,
      },
      body: JSON.stringify(body || {}),
    });
    if (!response.ok) {
      throw new Error("Chatbook device request failed");
    }
    return response.json().catch(() => ({}));
  }

  // --- QR device pairing (task 5) ---------------------------------------
  //
  // "Add a device" POSTs the CSRF-protected /auth/devices/pair endpoint and
  // renders the returned one-use bootstrap URL as a QR code (vendored
  // qrcodegen.js) plus selectable fallback text. The URL's nonce expires
  // 120 s after issue, so a fresh one is requested after 100 s — the shown
  // code then always carries at least 20 s of validity.
  const PAIR_REFRESH_MILLISECONDS = 100 * 1000;
  const PAIR_QUIET_ZONE_MODULES = 4;
  const PAIR_CANVAS_TARGET_PIXELS = 256;
  let pairRefreshTimer = null;

  function renderPairQr(url) {
    const canvas = document.querySelector("canvas.device-pair-qr");
    const fallback = document.querySelector(".device-pair-url");
    const hint = document.querySelector(".device-pair-hint");
    if (fallback) {
      fallback.textContent = url;
    }
    const qrcodegen = window.qrcodegen;
    if (!canvas || !qrcodegen) {
      return;
    }
    const qr = qrcodegen.QrCode.encodeText(url, qrcodegen.QrCode.Ecc.MEDIUM);
    const moduleCount = qr.getModuleCount();
    const modules = moduleCount + PAIR_QUIET_ZONE_MODULES * 2;
    // Integer scaling only: fractional scales blur module edges.
    const scale = Math.max(2, Math.floor(PAIR_CANVAS_TARGET_PIXELS / modules));
    const border = PAIR_QUIET_ZONE_MODULES * scale;
    canvas.width = modules * scale;
    canvas.height = modules * scale;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#000000";
    for (let y = 0; y < moduleCount; y += 1) {
      for (let x = 0; x < moduleCount; x += 1) {
        if (qr.getModule(x, y)) {
          ctx.fillRect(border + x * scale, border + y * scale, scale, scale);
        }
      }
    }
    canvas.hidden = false;
    if (hint) {
      hint.hidden = false;
    }
  }

  async function requestPairUrl() {
    if (pairRefreshTimer) {
      clearTimeout(pairRefreshTimer);
      pairRefreshTimer = null;
    }
    const data = await post("/auth/devices/pair");
    if (typeof data.url !== "string" || !data.url) {
      throw new Error("Chatbook pairing response missing url");
    }
    renderPairQr(data.url);
    // A failed refresh (e.g. the session ended) stops the loop instead of
    // showing a stale code; the button stays available to retry.
    pairRefreshTimer = setTimeout(() => {
      requestPairUrl().catch(() => undefined);
    }, PAIR_REFRESH_MILLISECONDS);
  }

  document.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) {
      return;
    }
    const section = button.closest("section");
    const sessionId = section ? section.dataset.sessionId : "";
    if (button.classList.contains("device-pair")) {
      button.disabled = true;
      requestPairUrl()
        .catch(() => undefined)
        .finally(() => {
          button.disabled = false;
        });
      return;
    }
    if (button.classList.contains("device-signout")) {
      post("/auth/logout")
        .then(() => window.location.assign("/auth/login"))
        .catch(() => undefined);
      return;
    }
    // Header-level incident response (revoke-all also ends THIS session, so
    // navigating to the login page — a reload would 401).
    if (button.classList.contains("device-revoke-all")) {
      post("/auth/devices/revoke-all")
        .then(() => window.location.assign("/auth/login"))
        .catch(() => undefined);
      return;
    }
    if (!sessionId) {
      return;
    }
    if (button.classList.contains("device-revoke")) {
      post("/auth/devices/revoke", { session_id: sessionId })
        .then(() => window.location.reload())
        .catch(() => undefined);
      return;
    }
    if (button.classList.contains("device-rename")) {
      const input = section.querySelector('input[name="label"]');
      const label = input ? input.value.trim() : "";
      if (!label) {
        return;
      }
      post("/auth/devices/rename", { session_id: sessionId, label })
        .then(() => window.location.reload())
        .catch(() => undefined);
    }
  });
})();
