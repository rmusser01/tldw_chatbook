// Chatbook served passkey login and registration (ADR-235).
// Loaded with defer by both the login page and the devices page. The
// server renders its passkey buttons hidden and this file reveals them
// only when the browser actually supports WebAuthn
// (window.PublicKeyCredential), so unsupported browsers never see a dead
// control. Wire format: the server sends WebAuthn dictionary fields
// (challenge, user id, allow/exclude-list ids) base64url-encoded strings,
// and expects the response fields as standard padded base64
// (attestationObject / authenticatorData / signature) plus plain UTF-8
// JSON text for clientDataJSON. Failures surface as inline text next to
// the button — never alert().
(() => {
  "use strict";

  const csrfMeta = document.querySelector('meta[name="chatbook-csrf"]');
  const csrf = csrfMeta ? csrfMeta.content : "";

  function base64UrlToBuffer(value) {
    const base64 = value.replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64 + "=".repeat((4 - (base64.length % 4)) % 4);
    const binary = atob(padded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
  }

  function bufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    for (let i = 0; i < bytes.length; i += 1) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }

  function bufferToUtf8(buffer) {
    return new TextDecoder().decode(buffer);
  }

  // Rebuild the server's options dictionary with BufferSource fields where
  // the WebAuthn dictionaries require them (challenge, user.id, and any
  // allow/exclude credential ids arrive as base64url strings).
  function preparePublicKey(options) {
    const prepared = Object.assign({}, options, {
      challenge: base64UrlToBuffer(options.challenge),
    });
    if (prepared.user && typeof prepared.user.id === "string") {
      prepared.user = Object.assign({}, prepared.user, {
        id: base64UrlToBuffer(prepared.user.id),
      });
    }
    const decodeEntry = (entry) =>
      Object.assign({}, entry, { id: base64UrlToBuffer(entry.id) });
    if (Array.isArray(prepared.allowCredentials)) {
      prepared.allowCredentials = prepared.allowCredentials.map(decodeEntry);
    }
    if (Array.isArray(prepared.excludeCredentials)) {
      prepared.excludeCredentials = prepared.excludeCredentials.map(decodeEntry);
    }
    return prepared;
  }

  function postJson(url, body, overrides) {
    return fetch(
      url,
      Object.assign(
        {
          method: "POST",
          credentials: "same-origin",
          headers: {
            "Content-Type": "application/json",
            "X-Chatbook-CSRF": csrf,
          },
          body: JSON.stringify(body || {}),
        },
        overrides || {}
      )
    );
  }

  function showMessage(text) {
    const message = document.querySelector(".passkey-message");
    if (message) {
      message.textContent = text;
      message.hidden = false;
    }
  }

  async function registerPasskey() {
    const labelInput = document.querySelector("input.passkey-label");
    const label = labelInput ? labelInput.value.trim() : "";
    const begin = await postJson("/auth/passkeys/register/begin", { label });
    if (!begin.ok) {
      showMessage("Passkey registration failed");
      return;
    }
    const payload = await begin.json();
    let credential;
    try {
      credential = await navigator.credentials.create({
        publicKey: preparePublicKey(payload.publicKey),
      });
    } catch (error) {
      showMessage("Passkey registration failed");
      return;
    }
    const complete = await postJson("/auth/passkeys/register/complete", {
      challenge_id: payload.challenge_id,
      client_data_json: bufferToUtf8(credential.response.clientDataJSON),
      attestation_object_b64: bufferToBase64(credential.response.attestationObject),
    });
    if (!complete.ok) {
      showMessage("Passkey registration failed");
      return;
    }
    window.location.reload();
  }

  async function loginPasskey() {
    const begin = await postJson("/auth/passkeys/login/begin", {});
    if (!begin.ok) {
      showMessage(
        begin.status === 503
          ? "Passkeys need a domain origin"
          : "Passkey sign-in failed"
      );
      return;
    }
    const payload = await begin.json();
    let assertion;
    try {
      assertion = await navigator.credentials.get({
        publicKey: preparePublicKey(payload.publicKey),
      });
    } catch (error) {
      showMessage("Passkey sign-in failed");
      return;
    }
    // redirect:"manual" exposes the login 303 as an opaque redirect instead
    // of silently following it into the app shell; success navigates there.
    const complete = await postJson(
      "/auth/passkeys/login/complete",
      {
        challenge_id: payload.challenge_id,
        client_data_json: bufferToUtf8(assertion.response.clientDataJSON),
        authenticator_data_b64: bufferToBase64(
          assertion.response.authenticatorData
        ),
        signature_b64: bufferToBase64(assertion.response.signature),
      },
      { redirect: "manual" }
    );
    if (complete.type === "opaqueredirect" || complete.ok) {
      window.location.assign("/");
      return;
    }
    showMessage("Passkey sign-in failed");
  }

  const supported = typeof window.PublicKeyCredential === "function";
  if (supported) {
    document
      .querySelectorAll("button.passkey-login, button.passkey-add")
      .forEach((button) => {
        button.hidden = false;
      });
  }

  document.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button || !supported) {
      return;
    }
    if (button.classList.contains("passkey-login")) {
      button.disabled = true;
      loginPasskey()
        .catch(() => showMessage("Passkey sign-in failed"))
        .finally(() => {
          button.disabled = false;
        });
      return;
    }
    if (button.classList.contains("passkey-add")) {
      button.disabled = true;
      registerPasskey()
        .catch(() => showMessage("Passkey registration failed"))
        .finally(() => {
          button.disabled = false;
        });
      return;
    }
    if (button.classList.contains("passkey-remove")) {
      const item = button.closest("li.passkey");
      const credentialId = item ? item.dataset.credentialId : "";
      if (!credentialId) {
        return;
      }
      postJson("/auth/passkeys/remove", { credential_id: credentialId })
        .then(() => window.location.reload())
        .catch(() => showMessage("Passkey removal failed"));
    }
  });
})();
