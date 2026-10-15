// Chatbook served device manager (ADR-233, task 4).
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

  document.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) {
      return;
    }
    const section = button.closest("section");
    const sessionId = section ? section.dataset.sessionId : "";
    if (button.classList.contains("device-signout")) {
      post("/auth/logout")
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
