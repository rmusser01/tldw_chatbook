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
    bridgeReturn: byId("bridge-return-button"), bridgeConfirm: byId("bridge-confirm-button"),
    bridgeExpiry: byId("bridge-expiry"), bridgeExpiryStatus: byId("bridge-expiry-status"),
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
  let currentLoadNonce = "";
  let rendererReady = false;
  let branchUnavailable = false;
  let previewStopped = false;
  let pendingBridge = null;
  let cancellingBridge = false;

  async function post(path, value, extraHeaders = {}, signal = undefined) {
    const headers = {"Content-Type": "application/json", ...extraHeaders};
    if (csrf) headers["X-Canvas-CSRF"] = csrf;
    const response = await fetch(api(path), {method: "POST", headers, body: JSON.stringify(value), cache: "no-store", signal});
    if (!response.ok) throw new Error(`Canvas request failed: ${response.status}`);
    return response.json();
  }

  async function postWithCapability(path, value, action, signal = undefined) {
    const capability = await mintAction(action, signal);
    return post(path, value, {Authorization: `CanvasCapability ${capability}`}, signal);
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
    if (pending.countdownTimer) clearInterval(pending.countdownTimer);
    if (pending.prepareTimer) clearTimeout(pending.prepareTimer);
    if (pending.prepareAbort) pending.prepareAbort.abort();
    pendingBridge = null;
    ui.bridgeDialog.hidden = true;
    ui.bridgeRecovery.hidden = true;
    ui.bridgeRetry.hidden = true;
    ui.bridgeReturn.hidden = true;
    ui.bridgeConfirm.hidden = false;
    ui.bridgeExpiry.textContent = "";
    ui.bridgeExpiryStatus.textContent = "";
    setBackgroundForDialog(false);
    if (restoreFocus && pending.returnFocus?.isConnected) pending.returnFocus.focus();
    pending.request = null;
    pending.presentation = null;
    pending.source = null;
    pending.prepareAbort = null;
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

  const bridgeEncoder = new TextEncoder();
  const bridgeLimits = Object.freeze({submitBytes: 16 * 1024, downloadBytes: 10 * 1024 * 1024, downloadEncodedBytes: 13985108, jsonDepth: 16});
  const passiveDownloadTypes = Object.freeze({
    "text/plain": [".txt"], "text/csv": [".csv"], "application/json": [".json"],
    "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"], "image/gif": [".gif"], "image/webp": [".webp"],
  });

  function ownRecord(value, expectedKeys) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) return false;
    const keys = Reflect.ownKeys(value);
    return keys.length === expectedKeys.length && keys.every((key) =>
      typeof key === "string" && expectedKeys.includes(key));
  }

  function cloneBridgeJson(value, depth = 0, seen = new Set()) {
    if (depth > bridgeLimits.jsonDepth) throw new Error("Canvas request exceeds its trusted-shell depth limit.");
    if (value === null || typeof value === "string" || typeof value === "boolean") return value;
    if (typeof value === "number") {
      if (!Number.isFinite(value)) throw new Error("Canvas request contains a non-finite number.");
      return value;
    }
    if (!value || typeof value !== "object" || seen.has(value)) {
      throw new Error("Canvas request is not JSON-compatible.");
    }
    const prototype = Object.getPrototypeOf(value);
    if (!Array.isArray(value) && prototype !== Object.prototype && prototype !== null) {
      throw new Error("Canvas request is not JSON-compatible.");
    }
    seen.add(value);
    try {
      if (Array.isArray(value)) {
        const cloned = [];
        for (let index = 0; index < value.length; index += 1) {
          cloned.push(cloneBridgeJson(value[index], depth + 1, seen));
        }
        return cloned;
      }
      const cloned = Object.create(null);
      const keys = Reflect.ownKeys(value);
      if (keys.some((key) => typeof key !== "string")) throw new Error("Canvas request is not JSON-compatible.");
      for (const key of keys.sort()) cloned[key] = cloneBridgeJson(value[key], depth + 1, seen);
      return cloned;
    } finally {
      seen.delete(value);
    }
  }

  function normalizeBridgeNumber(token) {
    if (token.length > 64) throw new Error("Canvas number token exceeds its trusted-shell limit.");
    const negative = token.startsWith("-");
    const unsigned = negative ? token.slice(1) : token;
    const exponentIndex = unsigned.search(/[eE]/);
    const mantissa = exponentIndex === -1 ? unsigned : unsigned.slice(0, exponentIndex);
    const explicitExponent = exponentIndex === -1 ? 0n : BigInt(unsigned.slice(exponentIndex + 1));
    const decimalIndex = mantissa.indexOf(".");
    const fractionLength = decimalIndex === -1 ? 0 : mantissa.length - decimalIndex - 1;
    let digits = mantissa.replace(".", "").replace(/^0+/, "");
    if (!digits) return "0";
    const trailingZeros = digits.match(/0+$/)?.[0].length || 0;
    if (trailingZeros) digits = digits.slice(0, -trailingZeros);
    const exponent = explicitExponent - BigInt(fractionLength) + BigInt(trailingZeros);
    return `${negative ? "-" : ""}${digits}e${exponent}`;
  }

  function parseLosslessBridgeJson(source) {
    let index = 0;
    const skipWhitespace = () => {
      while (index < source.length && /[\t\n\r ]/.test(source[index])) index += 1;
    };
    const parseString = () => {
      const start = index;
      index += 1;
      while (index < source.length) {
        const character = source[index];
        if (character === '"') {
          index += 1;
          return JSON.parse(source.slice(start, index));
        }
        if (character === "\\") {
          index += 2;
        } else {
          if (character.charCodeAt(0) <= 0x1f) throw new Error("Canvas JSON string is invalid.");
          index += 1;
        }
      }
      throw new Error("Canvas JSON string is unterminated.");
    };
    const parseValue = (depth) => {
      if (depth > bridgeLimits.jsonDepth) throw new Error("Canvas JSON exceeds its trusted-shell depth limit.");
      skipWhitespace();
      const character = source[index];
      if (character === '"') return {kind: "string", value: parseString()};
      if (source.startsWith("null", index)) { index += 4; return {kind: "null"}; }
      if (source.startsWith("true", index)) { index += 4; return {kind: "boolean", value: true}; }
      if (source.startsWith("false", index)) { index += 5; return {kind: "boolean", value: false}; }
      if (character === "[") {
        index += 1;
        const items = [];
        skipWhitespace();
        if (source[index] === "]") { index += 1; return {kind: "array", items}; }
        while (true) {
          items.push(parseValue(depth + 1));
          skipWhitespace();
          if (source[index] === "]") { index += 1; return {kind: "array", items}; }
          if (source[index] !== ",") throw new Error("Canvas JSON array is invalid.");
          index += 1;
        }
      }
      if (character === "{") {
        index += 1;
        const entries = new Map();
        skipWhitespace();
        if (source[index] === "}") { index += 1; return {kind: "object", entries}; }
        while (true) {
          skipWhitespace();
          if (source[index] !== '"') throw new Error("Canvas JSON object key is invalid.");
          const key = parseString();
          if (entries.has(key)) throw new Error("Canvas JSON object has duplicate keys.");
          skipWhitespace();
          if (source[index] !== ":") throw new Error("Canvas JSON object is invalid.");
          index += 1;
          entries.set(key, parseValue(depth + 1));
          skipWhitespace();
          if (source[index] === "}") { index += 1; return {kind: "object", entries}; }
          if (source[index] !== ",") throw new Error("Canvas JSON object is invalid.");
          index += 1;
        }
      }
      const number = source.slice(index).match(/^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/);
      if (!number) throw new Error("Canvas JSON value is invalid.");
      index += number[0].length;
      return {kind: "number", value: normalizeBridgeNumber(number[0])};
    };
    const parsed = parseValue(0);
    skipWhitespace();
    if (index !== source.length) throw new Error("Canvas JSON has trailing content.");
    return parsed;
  }

  function losslessBridgeJsonEqual(leftSource, rightSource) {
    const compare = (left, right) => {
      if (left.kind !== right.kind) return false;
      if (["string", "boolean", "number"].includes(left.kind)) return left.value === right.value;
      if (left.kind === "null") return true;
      if (left.kind === "array") {
        return left.items.length === right.items.length &&
          left.items.every((item, index) => compare(item, right.items[index]));
      }
      if (left.entries.size !== right.entries.size) return false;
      for (const [key, value] of left.entries) {
        if (!right.entries.has(key) || !compare(value, right.entries.get(key))) return false;
      }
      return true;
    };
    return compare(parseLosslessBridgeJson(leftSource), parseLosslessBridgeJson(rightSource));
  }

  function rasterSignatureMatches(mimeType, decoded) {
    const byte = (index) => decoded.charCodeAt(index);
    if (mimeType === "image/png") {
      const png = [137, 80, 78, 71, 13, 10, 26, 10];
      return decoded.length >= png.length && png.every((value, index) => byte(index) === value);
    }
    if (mimeType === "image/jpeg") return decoded.length >= 3 && byte(0) === 255 && byte(1) === 216 && byte(2) === 255;
    if (mimeType === "image/gif") return decoded.startsWith("GIF87a") || decoded.startsWith("GIF89a");
    return mimeType === "image/webp" && decoded.length >= 12 && decoded.startsWith("RIFF") && decoded.slice(8, 12) === "WEBP";
  }

  function validateShellDownload(value) {
    if (!ownRecord(value, ["filename", "mime_type", "data"]) ||
        typeof value.filename !== "string" || typeof value.mime_type !== "string" || typeof value.data !== "string") {
      throw new Error("Canvas download request has an invalid schema.");
    }
    if (/[\x00-\x1f\x7f]/.test(value.filename)) throw new Error("Canvas download filename is unsafe.");
    const filename = value.filename.trim();
    if (!filename || bridgeEncoder.encode(filename).length > 255 || /[\\/<>:"|?*]/.test(filename) ||
        filename.startsWith(".") || filename.endsWith(".") || filename.endsWith(" ") ||
        /^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i.test(filename.split(".", 1)[0])) {
      throw new Error("Canvas download filename is unsafe.");
    }
    const extensions = passiveDownloadTypes[value.mime_type];
    if (!extensions || !extensions.some((extension) => filename.toLowerCase().endsWith(extension))) {
      throw new Error("Canvas download type is not passive or does not match its filename.");
    }
    let byteSize;
    let completeText = null;
    if (value.mime_type.startsWith("image/")) {
      const prefix = `data:${value.mime_type};base64,`;
      const encoded = value.data.slice(prefix.length);
      if (!value.data.startsWith(prefix) || encoded.length % 4 !== 0 ||
          !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(encoded)) {
        throw new Error("Canvas image download encoding is invalid.");
      }
      let decoded;
      try { decoded = atob(encoded); } catch (_) { throw new Error("Canvas image download encoding is invalid."); }
      byteSize = decoded.length;
      if (byteSize > bridgeLimits.downloadBytes) throw new Error("Canvas download exceeds its trusted-shell limit.");
      if (!rasterSignatureMatches(value.mime_type, decoded)) throw new Error("Canvas image download signature is invalid.");
    } else {
      if (value.data.startsWith("data:")) throw new Error("Canvas text download encoding is invalid.");
      byteSize = bridgeEncoder.encode(value.data).length;
      if (byteSize > bridgeLimits.downloadBytes) throw new Error("Canvas download exceeds its trusted-shell limit.");
      if (value.mime_type === "application/json") {
        try { JSON.parse(value.data); } catch (_) { throw new Error("Canvas JSON download is invalid."); }
      }
      completeText = value.data;
    }
    return {filename, mimeType: value.mime_type, byteSize, completeText};
  }

  function validateBridgeMessage(message) {
    if (!ownRecord(message, ["type", "nonce", "request_id", "kind", "value"]) ||
        message.type !== "canvas:bridge-request" || message.nonce !== currentLoadNonce ||
        typeof message.request_id !== "string" || !message.request_id ||
        bridgeEncoder.encode(message.request_id).length > 256 || !["submit", "download"].includes(message.kind)) {
      throw new Error("Canvas request was refused by the trusted shell.");
    }
    let value;
    if (message.kind === "submit" && typeof message.value === "string") {
      value = message.value;
      if (bridgeEncoder.encode(value).length > bridgeLimits.submitBytes) {
        throw new Error("Canvas request exceeds its trusted-shell limit.");
      }
    } else {
      value = cloneBridgeJson(message.value);
      const encoded = JSON.stringify(value);
      const limit = message.kind === "submit" ? bridgeLimits.submitBytes : bridgeLimits.downloadEncodedBytes;
      if (typeof encoded !== "string" || bridgeEncoder.encode(encoded).length > limit) {
        throw new Error("Canvas request exceeds its trusted-shell limit.");
      }
    }
    if (message.kind === "download") validateShellDownload(value);
    return {version: "canvas-v1", request_id: message.request_id, kind: message.kind, value};
  }

  function validatePreparationResponse(value, request) {
    const fields = ["request_id", "kind", "conversation_id", "canvas_id", "revision_id", "canvas_title", "revision_number", "complete_text", "filename", "mime_type", "byte_size", "expires_in_seconds"];
    if (!ownRecord(value, fields) || value.request_id !== request.request_id || value.kind !== request.kind ||
        value.canvas_id !== selection.canvas_id || value.revision_id !== displayedRevisionId ||
        value.canvas_title !== displayedMetadata.title || value.revision_number !== displayedMetadata.sequence ||
        typeof value.conversation_id !== "string" || !value.conversation_id ||
        typeof value.canvas_title !== "string" || !value.canvas_title ||
        !Number.isInteger(value.revision_number) || value.revision_number < 1 ||
        typeof value.expires_in_seconds !== "number" || !Number.isFinite(value.expires_in_seconds) ||
        value.expires_in_seconds <= 0 || value.expires_in_seconds > 300) {
      throw new Error("Canvas confirmation metadata was refused by the trusted shell.");
    }
    if (request.kind === "submit") {
      const completeText = value.complete_text;
      const completeBytes = typeof completeText === "string" && completeText.length <= bridgeLimits.submitBytes
        ? bridgeEncoder.encode(completeText).length
        : -1;
      if (completeBytes < 0 || completeBytes > bridgeLimits.submitBytes) {
        throw new Error("Canvas confirmation content did not match its request.");
      }
      let contentMatches = typeof request.value === "string" && completeText === request.value;
      if (typeof request.value !== "string" && typeof completeText === "string") {
        try {
          contentMatches = losslessBridgeJsonEqual(JSON.stringify(request.value), completeText);
        } catch (_) { contentMatches = false; }
      }
      if (!contentMatches || value.filename !== null || value.mime_type !== null ||
          value.byte_size !== completeBytes) {
        throw new Error("Canvas confirmation content did not match its request.");
      }
    } else {
      const download = validateShellDownload(request.value);
      if (value.complete_text !== download.completeText || value.filename !== download.filename ||
          value.mime_type !== download.mimeType || value.byte_size !== download.byteSize) {
        throw new Error("Canvas confirmation content did not match its request.");
      }
    }
    return value;
  }

  function updateBridgeCountdown(pending) {
    const seconds = Math.max(0, Math.ceil((pending.expiresAt - Date.now()) / 1000));
    const minutes = Math.floor(seconds / 60);
    ui.bridgeExpiry.textContent = `Review expires in ${minutes}:${String(seconds % 60).padStart(2, "0")}`;
    for (const threshold of [60, 10]) {
      if (pending.lastCountdownSeconds > threshold && seconds <= threshold && !pending.announcedExpiry.has(threshold)) {
        pending.announcedExpiry.add(threshold);
        ui.bridgeExpiryStatus.textContent =
          `Canvas confirmation expires in ${threshold === 60 ? "one minute" : "10 seconds"}.`;
      }
    }
    pending.lastCountdownSeconds = seconds;
  }

  function showBridgeRecovery(copy) {
    ui.bridgeRecovery.textContent = copy;
    ui.bridgeRecovery.hidden = false;
  }

  function showBridgeDialog(pending, presentation) {
    presentation = validatePreparationResponse(presentation, pending.request);
    if (pending.prepareTimer) clearTimeout(pending.prepareTimer);
    pending.prepareTimer = null;
    pending.prepareAbort = null;
    pending.presentation = presentation;
    pending.prepared = true;
    ui.bridgeHeading.textContent = presentation.kind === "submit" ? "Send result to chat" : "Download generated file";
    ui.bridgeSummary.textContent = presentation.kind === "submit"
      ? "Confirm to replace the unchanged Chatbook composer with this unsent draft. Nothing is sent automatically."
      : "Confirm to download this passive generated file from the trusted Canvas shell.";
    ui.bridgeKind.textContent = presentation.kind === "submit" ? "Unsent draft" : "Passive file";
    ui.bridgeTarget.textContent = `Conversation ${presentation.conversation_id} · Canvas “${presentation.canvas_title}” · Revision ${presentation.revision_number} · Canvas ID ${presentation.canvas_id} · Revision ID ${presentation.revision_id}`;
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
    ui.bridgeExpiryStatus.textContent = "";
    ui.bridgeDialog.inert = false;
    ui.bridgeDialog.hidden = false;
    setBackgroundForDialog(true);
    ui.bridgeCancel.focus();
    pending.expiresAt = Date.now() + presentation.expires_in_seconds * 1000;
    updateBridgeCountdown(pending);
    pending.countdownTimer = setInterval(() => updateBridgeCountdown(pending), 1000);
    pending.timer = setTimeout(() => {
      if (pendingBridge !== pending) return;
      void cancelPendingBridge({notifyServer: false});
      showNotice("Canvas confirmation expired. Request it again from the preview.");
    }, presentation.expires_in_seconds * 1000);
  }

  async function prepareBridgeMessage(message) {
    if (cancellingBridge) {
      showNotice("The previous Canvas confirmation is still cancelling. Try again.");
      return;
    }
    if (pendingBridge) {
      showBridgeRecovery("Another request was refused while this confirmation is pending.");
      return;
    }
    let request;
    try { request = validateBridgeMessage(message); }
    catch (error) { showNotice(error.message); return; }
    const prepareAbort = new AbortController();
    const pending = {
      mode: "bridge", request, presentation: null, prepared: false, returnFocus: ui.frame,
      timer: null, countdownTimer: null, prepareTimer: null, prepareAbort,
      source: null, frameNonce: message.nonce, expiresAt: null,
      lastCountdownSeconds: Number.POSITIVE_INFINITY, announcedExpiry: new Set(),
    };
    pendingBridge = pending;
    pending.prepareTimer = setTimeout(() => prepareAbort.abort(), 300_000);
    try {
      const presentation = await postWithCapability("api/bridge/prepare", {request}, "bridge_prepare", prepareAbort.signal);
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

  function stopPreview(message) {
    if (previewStopped) return;
    previewStopped = true;
    closed = true;
    if (pollTimer) clearTimeout(pollTimer);
    void cancelPendingBridge({notifyServer: false, restoreFocus: false});
    branchUnavailable = true;
    rendererReady = false;
    pendingPlan = null;
    latestRevisionId = "";
    if (currentPort) currentPort.close();
    currentPort = null;
    currentLoadNonce = "";
    ui.frame.src = "about:blank";
    ui.sourceView.value = "";
    ui.sourcePanel.hidden = true;
    ui.source.setAttribute("aria-expanded", "false");
    for (const child of document.querySelector(".canvas-workbench").children) child.inert = false;
    ui.loading.textContent = message;
    ui.loading.hidden = false;
    ui.compatibility.hidden = true;
    dismissNotice();
    setConnection("Disconnected", true);
    for (const control of document.querySelectorAll(".canvas-toolbar button, .canvas-toolbar select, .canvas-toolbar input")) {
      if (control !== ui.close) control.disabled = true;
    }
    ui.close.focus();
  }

  function showBranchUnavailable() {
    stopPreview("Unavailable on this branch. Return to Chatbook and reopen this Canvas from a reachable transcript card.");
  }

  function showDisconnected() {
    stopPreview("Canvas disconnected. Reopen it from Chatbook after Canvas is available.");
  }

  async function mintAction(action, signal = undefined) {
    return (await post("api/actions", {action}, {}, signal)).capability;
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
    currentLoadNonce = "";
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
    currentLoadNonce = nonce;
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
          else showDisconnected();
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
      showDisconnected();
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
    if (window.parent !== window) {
      window.parent.postMessage({type: "chatbook:canvas-close"}, location.origin);
    }
  });
  ui.bridgeCancel.addEventListener("click", () => void cancelPendingBridge());
  ui.bridgeCopy.addEventListener("click", () => {
    ui.bridgeText.select();
    showBridgeRecovery(document.execCommand("copy")
      ? "Content copied."
      : "Content is selected and ready to copy.");
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
      showBridgeRecovery(pending.request.kind === "submit"
        ? "The Chatbook draft changed. Nothing was inserted."
        : "The Canvas selection changed or this confirmation expired. Nothing was downloaded.");
      ui.bridgeRetry.hidden = false;
      ui.bridgeReturn.hidden = pending.request.kind !== "submit";
      ui.bridgeConfirm.hidden = true;
      ui.bridgeCancel.focus();
    } finally {
      ui.bridgeConfirm.disabled = false;
    }
  });
  ui.bridgeRetry.addEventListener("click", async () => {
    const pending = pendingBridge;
    if (!pending || pending.mode !== "bridge" || pending.frameNonce !== currentLoadNonce) return;
    const message = {
      type: "canvas:bridge-request",
      nonce: pending.frameNonce,
      request_id: `bridge-retry-${crypto.randomUUID()}`,
      kind: pending.request.kind,
      value: pending.request.value,
    };
    await cancelPendingBridge({notifyServer: false, restoreFocus: false});
    await prepareBridgeMessage(message);
  });
  ui.bridgeReturn.addEventListener("click", () => {
    if (!pendingBridge) return;
    closeBridgeDialog({restoreFocus: false});
    try { window.close(); } catch (_) { /* browser policy may refuse */ }
    setTimeout(() => {
      if (window.closed) return;
      showNotice("This browser could not return to Chatbook automatically. Return to the matching Chatbook conversation; the result was not inserted.");
      ui.frame.focus();
    }, 0);
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
