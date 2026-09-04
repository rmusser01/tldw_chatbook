/* Chatbook Canvas V1: generated code executes only inside this QuickJS worker. */

"use strict";

import {newQuickJSWASMModule} from "./quickjs-runtime.js";

export function startCanvasRuntimeWorker(self) {
const NATIVE_WORKER_SENTINEL = "native-worker-clean";
self.__canvasNativeWorkerSentinel = NATIVE_WORKER_SENTINEL;

const LIMITS = Object.freeze({
  runtimeMemoryBytes: 32 * 1024 * 1024,
  stackBytes: 512 * 1024,
  startupMilliseconds: 250,
  eventMilliseconds: 50,
  pendingJobs: 100,
  timers: 64,
  timerFiresPerSecond: 100,
  listeners: 500,
  patchesPerEvent: 1000,
  mutationsPerSecond: 2000,
  bridgeRequestsPerOperation: 16,
  bridgeRequestsPerSecond: 32,
  domNodes: 5000,
  scriptBytes: 256 * 1024,
  submitBytes: 16 * 1024,
  downloadBytes: 10 * 1024 * 1024,
  downloadEncodedBytes: Math.ceil((10 * 1024 * 1024) / 3) * 4 + 4096,
  jsonDepth: 16,
  hostMessageBytes: 15 * 1024 * 1024,
});

const encoder = new TextEncoder();
let quickJS = null;
let runtime = null;
let context = null;
let virtualControls = null;
let plan = null;
let prepared = false;
let executed = false;
let terminal = false;
let deadline = Number.POSITIVE_INFINITY;
let interrupted = false;
let internalOperation = 1_000_000;
const nativeTimers = new Map();
const mutationWindow = [];
const bridgeWindow = [];
const timerFireWindow = [];
const BUDGET_FAILURE_CODES = new Set([
  "bridge-limit", "bridge-rate-limit", "dom-limit", "job-limit",
  "listener-limit", "mutation-rate-limit", "patch-limit", "runtime-timeout",
  "timer-limit", "timer-rate-limit",
]);

function ownRecord(value, keys) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
}

function boundedString(value, limit) {
  return typeof value === "string" && encoder.encode(value).byteLength <= limit;
}

function boundedIdentifier(value) {
  return typeof value === "string" && value.length > 0 && boundedString(value, 256);
}

function validatePlan(value) {
  if (!ownRecord(value, ["runtime_profile", "source_identity", "root", "assets", "css_rules", "scripts"])) {
    throw new Error("render-plan-schema");
  }
  if (value.runtime_profile !== "canvas-v1" || !Array.isArray(value.scripts)) {
    throw new Error("runtime-profile");
  }
  let scriptBytes = 0;
  for (const script of value.scripts) {
    if (typeof script !== "string") throw new Error("script-type");
    scriptBytes += encoder.encode(script).byteLength;
    if (scriptBytes > LIMITS.scriptBytes) throw new Error("script-limit");
  }
  const stack = [value.root];
  const identifiers = new Set();
  let count = 0;
  while (stack.length) {
    const node = stack.pop();
    if (!ownRecord(node, ["node_id", "tag", "attributes", "text", "children"])) {
      throw new Error("node-schema");
    }
    if (!boundedIdentifier(node.node_id) || identifiers.has(node.node_id)) {
      throw new Error("node-identity");
    }
    identifiers.add(node.node_id);
    if (!Array.isArray(node.attributes) || !Array.isArray(node.children)) {
      throw new Error("node-collections");
    }
    count += 1;
    if (count > LIMITS.domNodes) throw new Error("dom-limit");
    stack.push(...node.children);
  }
}

function postFailure(code, message) {
  if (terminal) return;
  terminal = true;
  const boundedCode = boundedIdentifier(code) ? code : "runtime-error";
  const boundedMessage = boundedString(message, 4096)
    ? message
    : "Canvas execution failed inside the bounded virtual runtime.";
  postMessage({
    type: "failure",
    code: boundedCode,
    message: boundedMessage,
    native_worker_sentinel: self.__canvasNativeWorkerSentinel,
  });
}

function dumpEval(source, filename) {
  interrupted = false;
  const result = context.evalCode(source, filename);
  if (result.error) {
    const error = context.dump(result.error);
    result.error.dispose();
    const wrapped = new Error("QuickJS evaluation failed");
    wrapped.quickJSError = error;
    wrapped.interrupted = interrupted;
    throw wrapped;
  }
  const value = context.dump(result.value);
  result.value.dispose();
  return value;
}

function handleEval(source, filename) {
  interrupted = false;
  const result = context.evalCode(source, filename);
  if (result.error) {
    const error = context.dump(result.error);
    result.error.dispose();
    const wrapped = new Error("QuickJS evaluation failed");
    wrapped.quickJSError = error;
    wrapped.interrupted = interrupted;
    throw wrapped;
  }
  return result.value;
}

function callVirtualControl(name, values = []) {
  const callable = context.getProp(virtualControls, name);
  const arguments_ = values.map((value) => (
    typeof value === "number" ? context.newNumber(value) : context.newString(value)
  ));
  try {
    interrupted = false;
    const result = context.callFunction(callable, context.undefined, arguments_);
    if (result.error) {
      const error = context.dump(result.error);
      result.error.dispose();
      const wrapped = new Error("QuickJS control call failed");
      wrapped.quickJSError = error;
      wrapped.interrupted = interrupted;
      throw wrapped;
    }
    const value = context.dump(result.value);
    result.value.dispose();
    return value;
  } finally {
    for (const argument of arguments_) argument.dispose();
    callable.dispose();
  }
}

function executeJobs() {
  let count = 0;
  while (runtime.hasPendingJob()) {
    if (count >= LIMITS.pendingJobs) {
      const error = new Error("QuickJS pending-job limit exceeded");
      error.limitCode = "job-limit";
      throw error;
    }
    if (performance.now() > deadline) {
      const error = new Error("QuickJS operation deadline exceeded");
      error.limitCode = "runtime-timeout";
      throw error;
    }
    const result = runtime.executePendingJobs(1);
    if (result.error) {
      const owner = result.error.context || context;
      const value = owner.dump(result.error);
      result.error.dispose();
      const error = new Error("QuickJS pending job failed");
      error.quickJSError = value;
      throw error;
    }
    count += 1;
  }
}

function failureCode(error) {
  if (error?.limitCode) return error.limitCode;
  const quickJSMessage = error?.quickJSError?.message;
  if (BUDGET_FAILURE_CODES.has(quickJSMessage)) return quickJSMessage;
  if (error?.interrupted || interrupted || performance.now() > deadline) return "runtime-timeout";
  return "runtime-error";
}

function beginOperation(milliseconds) {
  deadline = performance.now() + milliseconds;
  callVirtualControl("beginOperation", [performance.now()]);
}

function drainTransaction() {
  const encoded = callVirtualControl("drainTransaction");
  if (typeof encoded !== "string" || encoder.encode(encoded).byteLength > LIMITS.hostMessageBytes) {
    const error = new Error("QuickJS transaction message exceeded its limit");
    error.limitCode = "patch-limit";
    throw error;
  }
  let transaction;
  try {
    transaction = JSON.parse(encoded);
  } catch (_) {
    throw new Error("QuickJS transaction was not valid JSON");
  }
  if (!ownRecord(transaction, ["patches", "bridges", "timers", "poison"])) {
    throw new Error("QuickJS transaction schema was invalid");
  }
  if (transaction.poison !== null) {
    const error = new Error("QuickJS resource budget was exhausted");
    error.limitCode = boundedIdentifier(transaction.poison) ? transaction.poison : "runtime-error";
    throw error;
  }
  if (!Array.isArray(transaction.patches) || transaction.patches.length > LIMITS.patchesPerEvent) {
    const error = new Error("QuickJS patch limit exceeded");
    error.limitCode = "patch-limit";
    throw error;
  }
  if (
    !Array.isArray(transaction.bridges) ||
    transaction.bridges.length > LIMITS.bridgeRequestsPerOperation ||
    !Array.isArray(transaction.timers)
  ) {
    throw new Error("QuickJS transaction list was invalid");
  }
  return transaction;
}

function enforceMutationRate(patchCount) {
  const now = performance.now();
  while (mutationWindow.length && mutationWindow[0].time <= now - 1000) {
    mutationWindow.shift();
  }
  const total = mutationWindow.reduce((sum, item) => sum + item.count, patchCount);
  if (total > LIMITS.mutationsPerSecond) {
    const error = new Error("QuickJS mutation-rate limit exceeded");
    error.limitCode = "mutation-rate-limit";
    throw error;
  }
  mutationWindow.push({time: now, count: patchCount});
}

function enforceBridgeRate(bridgeCount) {
  const now = performance.now();
  while (bridgeWindow.length && bridgeWindow[0].time <= now - 1000) {
    bridgeWindow.shift();
  }
  const total = bridgeWindow.reduce((sum, item) => sum + item.count, bridgeCount);
  if (total > LIMITS.bridgeRequestsPerSecond) {
    const error = new Error("QuickJS bridge-request rate exceeded");
    error.limitCode = "bridge-rate-limit";
    throw error;
  }
  bridgeWindow.push({time: now, count: bridgeCount});
}

function syncTimers(records) {
  if (!Array.isArray(records) || records.length > LIMITS.timers) {
    throw new Error("QuickJS timer record count was invalid");
  }
  const active = new Set();
  for (const record of records) {
    if (
      !ownRecord(record, ["id", "delay"]) || !Number.isSafeInteger(record.id) ||
      record.id <= 0 || !Number.isFinite(record.delay) || record.delay < 0 ||
      record.delay > 2_147_483_647 || active.has(record.id)
    ) throw new Error("QuickJS timer record was invalid");
    active.add(record.id);
    if (!nativeTimers.has(record.id)) {
      const handle = setTimeout(() => fireTimer(record.id), record.delay);
      nativeTimers.set(record.id, handle);
    }
  }
  for (const [identifier, handle] of nativeTimers) {
    if (!active.has(identifier)) {
      clearTimeout(handle);
      nativeTimers.delete(identifier);
    }
  }
}

function postTransaction(operationId, operationKind, transaction) {
  enforceMutationRate(transaction.patches.length);
  enforceBridgeRate(transaction.bridges.length);
  syncTimers(transaction.timers);
  postMessage({
    type: "transaction",
    operation_id: operationId,
    operation_kind: operationKind,
    patches: transaction.patches,
    bridges: transaction.bridges,
    native_worker_sentinel: self.__canvasNativeWorkerSentinel,
  });
}

function runStartup(operationId) {
  beginOperation(LIMITS.startupMilliseconds);
  for (let index = 0; index < plan.scripts.length; index += 1) {
    const source = `(function () { "use strict";\n${plan.scripts[index]}\n}).call(undefined);`;
    dumpEval(source, `canvas-generated-${index + 1}.js`);
    executeJobs();
  }
  postTransaction(operationId, "startup", drainTransaction());
}

function validateEvent(value) {
  if (!ownRecord(value, ["type", "target_id", "value", "checked", "key"])) return false;
  return (
    ["change", "click", "focus", "input", "keydown", "keyup", "reset", "submit"].includes(value.type) &&
    boundedIdentifier(value.target_id) &&
    (value.value === null || boundedString(value.value, 16 * 1024)) &&
    (value.checked === null || typeof value.checked === "boolean") &&
    (value.key === null || boundedString(value.key, 64))
  );
}

function runEvent(operationId, event) {
  beginOperation(LIMITS.eventMilliseconds);
  callVirtualControl("dispatch", [JSON.stringify(event)]);
  executeJobs();
  postTransaction(operationId, "event", drainTransaction());
}

function recordTimerFire() {
  const now = performance.now();
  while (timerFireWindow.length && timerFireWindow[0] <= now - 1000) timerFireWindow.shift();
  if (timerFireWindow.length >= LIMITS.timerFiresPerSecond) {
    const error = new Error("QuickJS timer-fire rate exceeded");
    error.limitCode = "timer-rate-limit";
    throw error;
  }
  timerFireWindow.push(now);
}

function fireTimer(identifier) {
  if (terminal || !executed) return;
  nativeTimers.delete(identifier);
  internalOperation += 1;
  try {
    recordTimerFire();
    beginOperation(LIMITS.eventMilliseconds);
    callVirtualControl("fireTimer", [identifier, performance.now()]);
    executeJobs();
    postTransaction(internalOperation, "timer", drainTransaction());
  } catch (error) {
    postFailure(failureCode(error), "Canvas timer exceeded the bounded virtual runtime.");
  }
}

const VIRTUAL_RUNTIME_SOURCE = String.raw`
(() => {
  "use strict";
  const HTML = "http://www.w3.org/1999/xhtml";
  const SVG = "http://www.w3.org/2000/svg";
  const MAX = Object.freeze({
    nodes: 5000,
    patches: 1000,
    listeners: 500,
    timers: 64,
    bridges: 16,
    submitBytes: 16384,
    downloadBytes: 10485760,
    downloadEncodedBytes: 13985108,
    jsonDepth: 16,
    consoleEntries: 100,
    consoleBytes: 16384,
  });
  const htmlTags = new Set([
    "a", "abbr", "address", "article", "aside", "b", "bdi", "bdo", "blockquote",
    "body", "br", "button", "caption", "cite", "code", "col", "colgroup", "data",
    "datalist", "dd", "del", "details", "dfn", "div", "dl", "dt", "em", "fieldset",
    "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
    "head", "header", "hr", "html", "i", "img", "input", "ins", "kbd", "label",
    "legend", "li", "main", "mark", "menu", "meta", "meter", "nav", "ol", "optgroup",
    "option", "output", "p", "pre", "progress", "q", "s", "samp", "section", "select",
    "small", "span", "strong", "sub", "summary", "sup", "table", "tbody", "td",
    "textarea", "tfoot", "th", "thead", "time", "title", "tr", "u", "ul", "var", "wbr",
  ]);
  const svgTags = new Set(["svg", "g", "circle", "ellipse", "line", "path", "polygon", "polyline", "rect", "text", "tspan"]);
  const globalAttributes = new Set(["class", "dir", "hidden", "id", "lang", "role", "style", "tabindex", "title"]);
  const htmlAttributes = {
    a: new Set(["href"]),
    button: new Set(["disabled", "name", "type", "value"]),
    col: new Set(["span"]), colgroup: new Set(["span"]), data: new Set(["value"]),
    del: new Set(["datetime"]), fieldset: new Set(["disabled", "name"]),
    form: new Set(["name", "novalidate"]),
    img: new Set(["alt", "decoding", "height", "loading", "width"]),
    input: new Set(["checked", "disabled", "form", "list", "max", "maxlength", "min", "minlength", "multiple", "name", "pattern", "placeholder", "readonly", "required", "size", "step", "type", "value"]),
    ins: new Set(["datetime"]), label: new Set(["for"]), li: new Set(["value"]),
    meta: new Set(["charset"]), meter: new Set(["high", "low", "max", "min", "optimum", "value"]),
    ol: new Set(["reversed", "start", "type"]), optgroup: new Set(["disabled", "label"]),
    option: new Set(["disabled", "label", "selected", "value"]), output: new Set(["for", "form", "name"]),
    progress: new Set(["max", "value"]), select: new Set(["disabled", "form", "multiple", "name", "required", "size"]),
    td: new Set(["colspan", "headers", "rowspan"]),
    textarea: new Set(["cols", "disabled", "form", "maxlength", "minlength", "name", "placeholder", "readonly", "required", "rows", "wrap"]),
    th: new Set(["abbr", "colspan", "headers", "rowspan", "scope"]), time: new Set(["datetime"]),
  };
  const svgAttributes = new Set(["cx", "cy", "d", "fill", "fill-opacity", "height", "points", "preserveAspectRatio", "r", "rx", "ry", "stroke", "stroke-dasharray", "stroke-dashoffset", "stroke-linecap", "stroke-linejoin", "stroke-opacity", "stroke-width", "text-anchor", "transform", "vector-effect", "viewBox", "width", "x", "x1", "x2", "y", "y1", "y2"]);
  const styleProperties = new Set([
    "align-content", "align-items", "align-self", "animation", "animation-delay", "animation-direction", "animation-duration", "animation-fill-mode", "animation-iteration-count", "animation-name", "animation-play-state", "animation-timing-function", "appearance", "aspect-ratio", "backdrop-filter", "backface-visibility", "background", "background-attachment", "background-blend-mode", "background-clip", "background-color", "background-image", "background-origin", "background-position", "background-repeat", "background-size", "block-size", "border", "border-block", "border-block-color", "border-block-end", "border-block-start", "border-block-style", "border-block-width", "border-bottom", "border-bottom-color", "border-bottom-left-radius", "border-bottom-right-radius", "border-bottom-style", "border-bottom-width", "border-collapse", "border-color", "border-inline", "border-inline-color", "border-inline-end", "border-inline-start", "border-inline-style", "border-inline-width", "border-left", "border-left-color", "border-left-style", "border-left-width", "border-radius", "border-right", "border-right-color", "border-right-style", "border-right-width", "border-spacing", "border-style", "border-top", "border-top-color", "border-top-left-radius", "border-top-right-radius", "border-top-style", "border-top-width", "border-width", "bottom", "box-shadow", "box-sizing", "break-after", "break-before", "break-inside", "caption-side", "caret-color", "clear", "color", "column-gap", "column-width", "columns", "display", "empty-cells", "filter", "flex", "flex-basis", "flex-direction", "flex-flow", "flex-grow", "flex-shrink", "flex-wrap", "float", "font", "font-family", "font-feature-settings", "font-kerning", "font-size", "font-stretch", "font-style", "font-variant", "font-weight", "gap", "grid", "grid-area", "grid-auto-columns", "grid-auto-flow", "grid-auto-rows", "grid-column", "grid-column-end", "grid-column-gap", "grid-column-start", "grid-gap", "grid-row", "grid-row-end", "grid-row-gap", "grid-row-start", "grid-template", "grid-template-areas", "grid-template-columns", "grid-template-rows", "height", "hyphens", "inline-size", "inset", "inset-block", "inset-block-end", "inset-block-start", "inset-inline", "inset-inline-end", "inset-inline-start", "isolation", "justify-content", "justify-items", "justify-self", "left", "letter-spacing", "line-height", "list-style", "list-style-position", "list-style-type", "margin", "margin-block", "margin-block-end", "margin-block-start", "margin-bottom", "margin-inline", "margin-inline-end", "margin-inline-start", "margin-left", "margin-right", "margin-top", "max-block-size", "max-height", "max-inline-size", "max-width", "min-block-size", "min-height", "min-inline-size", "min-width", "mix-blend-mode", "object-fit", "object-position", "opacity", "order", "outline", "outline-color", "outline-offset", "outline-style", "outline-width", "overflow", "overflow-wrap", "overflow-x", "overflow-y", "padding", "padding-block", "padding-block-end", "padding-block-start", "padding-bottom", "padding-inline", "padding-inline-end", "padding-inline-start", "padding-left", "padding-right", "padding-top", "perspective", "perspective-origin", "place-content", "place-items", "place-self", "pointer-events", "position", "resize", "right", "rotate", "row-gap", "scale", "scroll-behavior", "shape-rendering", "stroke", "stroke-dasharray", "stroke-dashoffset", "stroke-linecap", "stroke-linejoin", "stroke-opacity", "stroke-width", "table-layout", "text-align", "text-decoration", "text-decoration-color", "text-decoration-line", "text-decoration-style", "text-indent", "text-overflow", "text-shadow", "text-transform", "top", "transform", "transform-origin", "transform-style", "transition", "transition-delay", "transition-duration", "transition-property", "transition-timing-function", "translate", "unicode-bidi", "user-select", "vertical-align", "visibility", "white-space", "width", "word-break", "word-spacing", "writing-mode", "z-index", "fill", "fill-opacity",
  ]);
  const eventTypes = new Set(["change", "click", "focus", "input", "keydown", "keyup", "reset", "submit"]);
  const nodes = new Map();
  const timers = new Map();
  const logs = [];
  let root = null;
  let body = null;
  let nextNode = 1;
  let nextTimer = 1;
  let nextBridge = 1;
  let listenerCount = 0;
  let logBytes = 0;
  const safeJsonParse = JSON.parse.bind(JSON);
  const safeJsonStringify = JSON.stringify.bind(JSON);
  const safeNumber = Number;
  const safeNumberIsFinite = Number.isFinite.bind(Number);
  const safeMathFloor = Math.floor.bind(Math);
  const safeMathMax = Math.max.bind(Math);
  const safeMathMin = Math.min.bind(Math);
  const safeReflectApply = Reflect.apply;
  const safeArrayIsArray = Array.isArray;
  const safeArrayPushMethod = Array.prototype.push;
  const safeArrayPopMethod = Array.prototype.pop;
  const safeArrayIndexOfMethod = Array.prototype.indexOf;
  const safeStringCharCodeAtMethod = String.prototype.charCodeAt;
  const safeStringEndsWithMethod = String.prototype.endsWith;
  const safeStringSliceMethod = String.prototype.slice;
  const safeStringSplitMethod = String.prototype.split;
  const safeStringStartsWithMethod = String.prototype.startsWith;
  const safeStringToLowerCaseMethod = String.prototype.toLowerCase;
  const safeStringTrimMethod = String.prototype.trim;
  const safeRegExpTestMethod = RegExp.prototype.test;
  const safeMapForEachMethod = Map.prototype.forEach;
  const safeMapGetMethod = Map.prototype.get;
  const safeMapSetMethod = Map.prototype.set;
  const safeMapDeleteMethod = Map.prototype.delete;
  const safeMapSizeGetter = Object.getOwnPropertyDescriptor(Map.prototype, "size").get;
  const safeObjectCreate = Object.create;
  const safeObjectKeys = Object.keys.bind(Object);
  const safeObjectSetPrototypeOf = Object.setPrototypeOf;

  function apply(method, receiver, arguments_ = []) {
    return safeReflectApply(method, receiver, arguments_);
  }
  function stringEndsWith(value, suffix) { return apply(safeStringEndsWithMethod, value, [suffix]); }
  function stringSlice(value, start) { return apply(safeStringSliceMethod, value, [start]); }
  function stringStartsWith(value, prefix) { return apply(safeStringStartsWithMethod, value, [prefix]); }
  function stringToLowerCase(value) { return apply(safeStringToLowerCaseMethod, value); }
  function stringTrim(value) { return apply(safeStringTrimMethod, value); }
  function regexTest(pattern, value) { return apply(safeRegExpTestMethod, pattern, [value]); }
  function push(list, value) { apply(safeArrayPushMethod, list, [value]); }
  function pop(list) { return apply(safeArrayPopMethod, list); }
  function listIncludes(list, value) {
    return apply(safeArrayIndexOfMethod, list, [value]) !== -1;
  }
  function makeList() {
    const list = [];
    safeObjectSetPrototypeOf(list, null);
    return list;
  }
  function makeRecord() { return safeObjectCreate(null); }
  function ownRecord(value, keys) {
    if (value === null || typeof value !== "object" || safeArrayIsArray(value)) return false;
    const actual = safeObjectKeys(value);
    if (actual.length !== keys.length) return false;
    for (let index = 0; index < actual.length; index += 1) {
      if (!listIncludes(keys, actual[index])) return false;
    }
    return true;
  }
  function mapForEach(map, callback) { apply(safeMapForEachMethod, map, [callback]); }
  function mapGet(map, key) { return apply(safeMapGetMethod, map, [key]); }
  function mapSet(map, key, value) { apply(safeMapSetMethod, map, [key, value]); }
  function mapDelete(map, key) { apply(safeMapDeleteMethod, map, [key]); }
  function mapSize(map) { return apply(safeMapSizeGetter, map); }

  const state = makeRecord();
  state.patches = makeList();
  state.bridges = makeList();
  state.poison = null;
  state.now = 0;

  function poison(code) {
    if (state.poison === null) state.poison = code;
    throw new Error(code);
  }
  function utf8Length(value) {
    let count = 0;
    for (let index = 0; index < value.length; index += 1) {
      const code = apply(safeStringCharCodeAtMethod, value, [index]);
      if (code < 128) count += 1;
      else if (code < 2048) count += 2;
      else if (code >= 0xD800 && code <= 0xDBFF && index + 1 < value.length && apply(safeStringCharCodeAtMethod, value, [index + 1]) >= 0xDC00 && apply(safeStringCharCodeAtMethod, value, [index + 1]) <= 0xDFFF) { count += 4; index += 1; }
      else count += 3;
    }
    return count;
  }
  function emit(patch) {
    if (state.patches.length >= MAX.patches) poison("patch-limit");
    push(state.patches, patch);
  }
  function cloneJson(value, depth = 0, seen = makeList(), maxDepth = MAX.jsonDepth) {
    if (depth > maxDepth) throw new RangeError("Canvas bridge value exceeds its depth limit");
    if (value === null || typeof value === "string" || typeof value === "boolean") return value;
    if (typeof value === "number" && safeNumberIsFinite(value)) return value;
    if (!value || typeof value !== "object" || listIncludes(seen, value)) {
      throw new TypeError("Canvas bridge value must be JSON-compatible");
    }
    push(seen, value);
    const cloned = safeArrayIsArray(value) ? makeList() : makeRecord();
    if (safeArrayIsArray(value)) {
      for (let index = 0; index < value.length; index += 1) {
        push(cloned, cloneJson(value[index], depth + 1, seen, maxDepth));
      }
    } else {
      const keys = safeObjectKeys(value);
      for (let index = 0; index < keys.length; index += 1) {
        const key = keys[index];
        cloned[key] = cloneJson(value[key], depth + 1, seen, maxDepth);
      }
    }
    pop(seen);
    return cloned;
  }
  function validateJson(value, byteLimit) {
    const cloned = cloneJson(value);
    let encoded;
    try { encoded = safeJsonStringify(cloned); } catch (_) { throw new TypeError("Canvas bridge value must be JSON-compatible"); }
    if (encoded === undefined || utf8Length(encoded) > byteLimit) throw new RangeError("Canvas bridge value exceeds its byte limit");
    return cloned;
  }
  function allowedAttribute(node, name) {
    if (name === "data-canvas-asset") return false;
    if (name.startsWith("aria-") || name.startsWith("data-") || globalAttributes.has(name)) return true;
    return node.namespaceURI === SVG ? svgAttributes.has(name) : Boolean(htmlAttributes[node.localName]?.has(name));
  }
  function validateAttribute(node, name, value) {
    if (typeof name !== "string" || !name || typeof value !== "string" || utf8Length(value) > 16384) throw new TypeError("Invalid attribute");
    if (!allowedAttribute(node, name) || name.startsWith("on")) throw new TypeError("Attribute is outside Canvas V1");
    if (["href", "src", "action", "formaction", "target", "download"].includes(name)) {
      if (!(node.localName === "a" && name === "href" && /^#[^\s]+$/.test(value))) throw new TypeError("URL/navigation attributes are unavailable");
    }
    if (["fill", "stroke"].includes(name) && (value.includes("\\") || /url\s*\(/i.test(value))) {
      throw new TypeError("SVG resource paints are unavailable");
    }
  }
  function camelToKebab(value) { return value.replace(/[A-Z]/g, (match) => "-" + match.toLowerCase()); }

  class VirtualStyle {
    constructor(owner) { this.owner = owner; this.values = new Map(); }
    setProperty(name, value) {
      name = String(name).toLowerCase(); value = String(value);
      if (!styleProperties.has(name) || name.startsWith("--") || utf8Length(value) > 16384) throw new TypeError("Style is outside Canvas V1");
      this.values.set(name, value);
      emit({op: "set-style", node_id: this.owner.__id, name, value});
    }
    getPropertyValue(name) { return this.values.get(String(name).toLowerCase()) || ""; }
    removeProperty(name) {
      name = String(name).toLowerCase();
      if (!styleProperties.has(name)) throw new TypeError("Style is outside Canvas V1");
      const previous = this.values.get(name) || "";
      this.values.delete(name);
      emit({op: "remove-style", node_id: this.owner.__id, name});
      return previous;
    }
  }
  function styleProxy(owner) {
    const style = new VirtualStyle(owner);
    return new Proxy(style, {
      get(target, property) {
        if (property in target) {
          const value = target[property];
          return typeof value === "function" ? value.bind(target) : value;
        }
        return target.getPropertyValue(camelToKebab(String(property)));
      },
      set(target, property, value) { target.setProperty(camelToKebab(String(property)), value); return true; },
    });
  }

  class VirtualEvent {
    constructor(type, init = {}) {
      if (!eventTypes.has(String(type))) throw new TypeError("Event type is outside Canvas V1");
      this.type = String(type); this.bubbles = init.bubbles !== false; this.cancelable = true;
      this.defaultPrevented = false; this.target = null; this.currentTarget = null;
      this.key = init.key || null; this.__stopped = false;
    }
    preventDefault() { this.defaultPrevented = true; }
    stopPropagation() { this.__stopped = true; }
  }

  class VirtualNode {
    constructor(id, localName, namespaceURI, text = null) {
      this.__id = id; this.localName = localName; this.namespaceURI = namespaceURI;
      this.nodeType = localName === "#text" ? 3 : 1; this.parentNode = null;
      this.childNodes = []; this.attributes = new Map(); this.listeners = new Map();
      this.__text = text; this.__value = ""; this.__checked = false;
      this.__disabled = false; this.__selected = false; this.style = styleProxy(this);
    }
    get tagName() { return this.nodeType === 1 ? (this.namespaceURI === HTML ? this.localName.toUpperCase() : this.localName) : undefined; }
    get id() { return this.getAttribute("id") || ""; }
    set id(value) { this.setAttribute("id", String(value)); }
    get className() { return this.getAttribute("class") || ""; }
    set className(value) { this.setAttribute("class", String(value)); }
    get value() { return this.__value; }
    set value(value) { this.__value = String(value); emit({op: "set-property", node_id: this.__id, name: "value", value: this.__value}); }
    get checked() { return this.__checked; }
    set checked(value) { this.__checked = Boolean(value); emit({op: "set-property", node_id: this.__id, name: "checked", value: this.__checked}); }
    get disabled() { return this.__disabled; }
    set disabled(value) { this.__disabled = Boolean(value); emit({op: "set-property", node_id: this.__id, name: "disabled", value: this.__disabled}); }
    get selected() { return this.__selected; }
    set selected(value) { this.__selected = Boolean(value); emit({op: "set-property", node_id: this.__id, name: "selected", value: this.__selected}); }
    get textContent() {
      if (this.nodeType === 3) return this.__text;
      if (this.__text !== null) return this.__text;
      return this.childNodes.map((child) => child.textContent).join("");
    }
    set textContent(value) {
      value = String(value); if (utf8Length(value) > 524288) throw new RangeError("Text exceeds Canvas V1");
      for (const child of this.childNodes) child.parentNode = null;
      this.childNodes = []; this.__text = value;
      emit({op: "set-text", node_id: this.__id, value});
    }
    getAttribute(name) { name = String(name); return this.attributes.has(name) ? this.attributes.get(name) : null; }
    hasAttribute(name) { return this.attributes.has(String(name)); }
    setAttribute(name, value) {
      name = String(name); value = String(value); validateAttribute(this, name, value);
      if (name === "style") throw new TypeError("Use the Canvas style facade");
      this.attributes.set(name, value);
      if (name === "value") this.__value = value;
      if (["checked", "disabled", "selected"].includes(name)) this["__" + name] = true;
      emit({op: "set-attribute", node_id: this.__id, name, value});
    }
    removeAttribute(name) {
      name = String(name); if (!allowedAttribute(this, name)) throw new TypeError("Attribute is outside Canvas V1");
      this.attributes.delete(name); emit({op: "remove-attribute", node_id: this.__id, name});
    }
    appendChild(child) {
      if (!(child instanceof VirtualNode) || child === this) throw new TypeError("Invalid child");
      if (child.parentNode) child.parentNode.removeChild(child);
      child.parentNode = this; this.childNodes.push(child);
      emit({op: "append-child", node_id: this.__id, child_id: child.__id}); return child;
    }
    insertBefore(child, reference) {
      if (!(child instanceof VirtualNode) || (reference !== null && !(reference instanceof VirtualNode))) throw new TypeError("Invalid child/reference");
      if (reference !== null && reference.parentNode !== this) throw new Error("Reference is not a child");
      if (child.parentNode) child.parentNode.removeChild(child);
      const index = reference === null ? this.childNodes.length : this.childNodes.indexOf(reference);
      child.parentNode = this; this.childNodes.splice(index, 0, child);
      emit({op: "insert-before", node_id: this.__id, child_id: child.__id, reference_id: reference ? reference.__id : null}); return child;
    }
    removeChild(child) {
      const index = this.childNodes.indexOf(child); if (index < 0) throw new Error("Node is not a child");
      this.childNodes.splice(index, 1); child.parentNode = null;
      emit({op: "remove-child", node_id: this.__id, child_id: child.__id}); return child;
    }
    addEventListener(type, callback, options = false) {
      type = String(type);
      if (!eventTypes.has(type) || typeof callback !== "function") throw new TypeError("Listener is outside Canvas V1");
      const listeners = this.listeners.get(type) || [];
      if (listeners.some((entry) => entry.callback === callback)) return;
      if (listenerCount >= MAX.listeners) poison("listener-limit");
      listenerCount += 1; listeners.push({callback, once: Boolean(options && typeof options === "object" && options.once)});
      this.listeners.set(type, listeners);
    }
    removeEventListener(type, callback) {
      type = String(type); const listeners = this.listeners.get(type) || [];
      const next = listeners.filter((entry) => entry.callback !== callback);
      listenerCount -= listeners.length - next.length; this.listeners.set(type, next);
    }
    dispatchEvent(event) {
      if (!(event instanceof VirtualEvent)) throw new TypeError("Expected a Canvas Event");
      event.target = this; let current = this;
      while (current) {
        event.currentTarget = current;
        const listeners = [...(current.listeners.get(event.type) || [])];
        for (const entry of listeners) {
          entry.callback.call(current, event);
          if (entry.once) current.removeEventListener(event.type, entry.callback);
          if (event.__stopped) break;
        }
        if (event.__stopped || !event.bubbles) break;
        current = current.parentNode;
      }
      return !event.defaultPrevented;
    }
    click() { return this.dispatchEvent(new VirtualEvent("click")); }
    querySelector(selector) { return query(this, String(selector), true); }
    querySelectorAll(selector) { return query(this, String(selector), false); }
  }

  function matches(node, selector) {
    if (node.nodeType !== 1) return false;
    if (/^#[A-Za-z0-9_-]+$/.test(selector)) return node.id === selector.slice(1);
    if (/^\.[A-Za-z0-9_-]+$/.test(selector)) return node.className.split(/\s+/).includes(selector.slice(1));
    if (/^[A-Za-z][A-Za-z0-9-]*$/.test(selector)) return node.localName === selector.toLowerCase();
    throw new TypeError("Selector is outside Canvas V1");
  }
  function query(scope, selector, first) {
    const found = []; const stack = [...scope.childNodes].reverse();
    while (stack.length) {
      const node = stack.pop();
      if (matches(node, selector)) { if (first) return node; found.push(node); }
      stack.push(...[...node.childNodes].reverse());
    }
    return first ? null : Object.freeze(found);
  }
  function allocate(localName, namespaceURI, text = null, suppliedId = null) {
    if (nodes.size >= MAX.nodes) poison("dom-limit");
    const id = suppliedId || "virtual-" + nextNode++;
    if (nodes.has(id)) throw new Error("Duplicate Canvas node ID");
    const node = new VirtualNode(id, localName, namespaceURI, text); nodes.set(id, node); return node;
  }
  function installNode(record, inheritedNamespace) {
    const namespace = inheritedNamespace === SVG || record.tag === "svg" ? SVG : HTML;
    const node = allocate(record.tag, namespace, record.text, record.node_id);
    for (const [name, value] of record.attributes) {
      if (name === "data-canvas-asset") continue;
      node.attributes.set(name, value);
      if (name === "value") node.__value = value;
      if (["checked", "disabled", "selected"].includes(name)) node["__" + name] = true;
      if (name === "style") {
        for (const declaration of value.split(";")) {
          const separator = declaration.indexOf(":");
          if (separator > 0) node.style.values.set(declaration.slice(0, separator), declaration.slice(separator + 1));
        }
      }
    }
    for (const childRecord of record.children) {
      const child = installNode(childRecord, namespace); child.parentNode = node; node.childNodes.push(child);
    }
    return node;
  }

  class VirtualDocument {
    get documentElement() { return root; }
    get body() { return body; }
    get defaultView() { return null; }
    get cookie() { return ""; }
    set cookie(_) { throw new TypeError("Cookies are unavailable"); }
    getElementById(id) { for (const node of nodes.values()) if (node.id === String(id)) return node; return null; }
    querySelector(selector) { return matches(root, String(selector)) ? root : query(root, String(selector), true); }
    querySelectorAll(selector) { const found = query(root, String(selector), false); return matches(root, String(selector)) ? Object.freeze([root, ...found]) : found; }
    createElement(tag) {
      tag = String(tag).toLowerCase(); if (!htmlTags.has(tag)) throw new TypeError("Element is outside Canvas V1");
      const node = allocate(tag, HTML); emit({op: "create-element", node_id: node.__id, tag, namespace: HTML}); return node;
    }
    createElementNS(namespace, tag) {
      namespace = String(namespace); tag = String(tag);
      if (namespace !== SVG || !svgTags.has(tag)) throw new TypeError("Namespaced element is outside Canvas V1");
      const node = allocate(tag, SVG); emit({op: "create-element", node_id: node.__id, tag, namespace: SVG}); return node;
    }
    createTextNode(value) {
      value = String(value); if (utf8Length(value) > 524288) throw new RangeError("Text exceeds Canvas V1");
      const node = allocate("#text", HTML, value); emit({op: "create-text", node_id: node.__id, value}); return node;
    }
    addEventListener(type, callback, options) { root.addEventListener(type, callback, options); }
    removeEventListener(type, callback) { root.removeEventListener(type, callback); }
  }

  function schedule(callback, delay, repeat, args) {
    if (typeof callback !== "function") throw new TypeError("Timer callback must be a function");
    if (mapSize(timers) >= MAX.timers) poison("timer-limit");
    delay = safeNumber(delay); if (!safeNumberIsFinite(delay) || delay < 0) delay = 0;
    delay = safeMathMin(safeMathFloor(delay), 2147483647);
    const id = nextTimer++;
    mapSet(timers, id, {callback, args, delay, due: state.now + delay, repeat});
    return id;
  }
  function cancelTimer(id) { mapDelete(timers, safeNumber(id)); }
  function bridge(kind, value) {
    if (state.bridges.length >= MAX.bridges) poison("bridge-limit");
    let cloned;
    if (kind === "submit" && typeof value === "string") {
      if (utf8Length(value) > MAX.submitBytes) throw new RangeError("Canvas bridge value exceeds its byte limit");
      cloned = value;
    } else {
      const limit = kind === "submit" ? MAX.submitBytes : MAX.downloadEncodedBytes;
      cloned = validateJson(value, limit);
    }
    if (kind === "download") validateDownload(cloned);
    const record = makeRecord();
    record.request_id = "bridge-" + nextBridge++;
    record.kind = kind;
    record.value = cloned;
    push(state.bridges, record);
  }
  function validateDownload(value) {
    if (!ownRecord(value, ["filename", "mime_type", "data"]) ||
        typeof value.filename !== "string" || typeof value.mime_type !== "string" || typeof value.data !== "string") {
      throw new TypeError("Canvas download request has an invalid schema");
    }
    const filename = stringTrim(value.filename);
    if (!filename || utf8Length(filename) > 255 || regexTest(/[\\/\x00-\x1f\x7f<>:"|?*]/, filename) ||
        stringStartsWith(filename, ".") || stringEndsWith(filename, ".") || stringEndsWith(filename, " ")) {
      throw new TypeError("Canvas download filename is unsafe");
    }
    const stem = stringToLowerCase(apply(safeStringSplitMethod, filename, [".", 1])[0]);
    if (regexTest(/^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/, stem)) {
      throw new TypeError("Canvas download filename is reserved");
    }
    const allowed = makeRecord();
    allowed["text/plain"] = [".txt"]; allowed["text/csv"] = [".csv"]; allowed["application/json"] = [".json"];
    allowed["image/png"] = [".png"]; allowed["image/jpeg"] = [".jpg", ".jpeg"];
    allowed["image/gif"] = [".gif"]; allowed["image/webp"] = [".webp"];
    const extensions = allowed[value.mime_type];
    let extensionMatches = false;
    if (extensions) {
      const lowerFilename = stringToLowerCase(filename);
      for (let index = 0; index < extensions.length; index += 1) {
        if (stringEndsWith(lowerFilename, extensions[index])) extensionMatches = true;
      }
    }
    if (!extensionMatches) {
      throw new TypeError("Canvas download type is not passive or does not match its filename");
    }
    if (stringStartsWith(value.mime_type, "image/")) {
      const prefix = "data:" + value.mime_type + ";base64,";
      const encoded = stringSlice(value.data, prefix.length);
      if (!stringStartsWith(value.data, prefix) || encoded.length % 4 !== 0 || !regexTest(/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/, encoded)) {
        throw new TypeError("Canvas image download must use matching base64 data URL encoding");
      }
      const padding = stringEndsWith(encoded, "==") ? 2 : (stringEndsWith(encoded, "=") ? 1 : 0);
      if ((encoded.length / 4) * 3 - padding > MAX.downloadBytes) {
        throw new RangeError("Canvas download exceeds its byte limit");
      }
    } else {
      if (stringStartsWith(value.data, "data:") || utf8Length(value.data) > MAX.downloadBytes) {
        throw new RangeError("Canvas download exceeds its byte limit");
      }
      if (value.mime_type === "application/json") {
        try { safeJsonParse(value.data); } catch (_) { throw new TypeError("Canvas JSON download data is invalid"); }
      }
    }
  }
  function log(...values) {
    if (logs.length >= MAX.consoleEntries) return;
    const text = values.map((value) => { try { return typeof value === "string" ? value : JSON.stringify(value); } catch (_) { return "[unprintable]"; } }).join(" ");
    const bytes = utf8Length(text); if (logBytes + bytes > MAX.consoleBytes) return;
    logBytes += bytes; logs.push(text);
  }

  Object.defineProperties(globalThis, {
    document: {value: new VirtualDocument(), writable: false, configurable: false},
    Event: {value: VirtualEvent, writable: false, configurable: false},
    setTimeout: {value: (callback, delay, ...args) => schedule(callback, delay, false, args), writable: false, configurable: false},
    clearTimeout: {value: cancelTimer, writable: false, configurable: false},
    setInterval: {value: (callback, delay, ...args) => schedule(callback, delay, true, args), writable: false, configurable: false},
    clearInterval: {value: cancelTimer, writable: false, configurable: false},
    console: {value: Object.freeze({log, info: log, warn: log, error: log}), writable: false, configurable: false},
    canvas: {value: Object.freeze({submit: (value) => bridge("submit", value), download: (value) => bridge("download", value)}), writable: false, configurable: false},
  });
  for (const name of ["window", "parent", "top", "self", "location", "navigator", "fetch", "XMLHttpRequest", "WebSocket", "EventSource", "Worker", "SharedWorker", "importScripts", "localStorage", "sessionStorage", "indexedDB", "caches", "cookieStore", "FileSystemHandle", "showOpenFilePicker", "SharedArrayBuffer", "Atomics", "WebAssembly", "open", "postMessage", "alert", "confirm", "prompt", "print", "Deno", "Bun", "process", "require"]) {
    Object.defineProperty(globalThis, name, {value: undefined, writable: false, configurable: false});
  }

  const controls = makeRecord();
  controls.install = (encoded) => {
    root = installNode(safeJsonParse(encoded), HTML);
    body = [...nodes.values()].find((node) => node.localName === "body") || root;
  };
  controls.beginOperation = (now) => {
    state.patches = makeList();
    state.bridges = makeList();
    state.poison = null;
    state.now = safeNumber(now);
  };
  controls.drainTransaction = () => {
    const timerRecords = makeList();
    mapForEach(timers, (timer, id) => {
      const record = makeRecord();
      record.id = id;
      record.delay = safeMathMax(0, timer.due - state.now);
      push(timerRecords, record);
    });
    const transaction = makeRecord();
    transaction.patches = cloneJson(state.patches, 0, makeList(), 64);
    transaction.bridges = cloneJson(state.bridges, 0, makeList(), 64);
    transaction.timers = timerRecords;
    transaction.poison = state.poison;
    return safeJsonStringify(transaction);
  };
  controls.dispatch = (encoded) => {
    const record = safeJsonParse(encoded);
    const target = nodes.get(record.target_id); if (!target) throw new Error("Unknown event target");
    if (record.value !== null) target.__value = record.value;
    if (record.checked !== null) target.__checked = record.checked;
    target.dispatchEvent(new VirtualEvent(record.type, {key: record.key}));
  };
  controls.fireTimer = (id, now) => {
    state.now = safeNumber(now);
    const timer = mapGet(timers, safeNumber(id));
    if (!timer) return;
    if (timer.repeat) timer.due = state.now + timer.delay;
    else mapDelete(timers, safeNumber(id));
    timer.callback(...timer.args);
  };
  return Object.freeze(controls);
})();
`;

async function prepare(value) {
  validatePlan(value);
  plan = value;
  quickJS = await newQuickJSWASMModule();
  runtime = quickJS.newRuntime();
  runtime.setMemoryLimit(LIMITS.runtimeMemoryBytes);
  runtime.setMaxStackSize(LIMITS.stackBytes);
  runtime.setInterruptHandler(() => {
    if (performance.now() <= deadline) return false;
    interrupted = true;
    return true;
  });
  runtime.removeModuleLoader();
  context = runtime.newContext();
  deadline = performance.now() + LIMITS.startupMilliseconds;
  virtualControls = handleEval(VIRTUAL_RUNTIME_SOURCE, "canvas-trusted-bootstrap.js");
  if (context.typeof(virtualControls) !== "object") {
    virtualControls.dispose();
    virtualControls = null;
    throw new Error("QuickJS virtual controls were invalid");
  }
  callVirtualControl("install", [JSON.stringify(plan.root)]);
  prepared = true;
  postMessage({type: "prepared", native_worker_sentinel: self.__canvasNativeWorkerSentinel});
}

self.onmessage = async (event) => {
  if (terminal) return;
  const message = event.data;
  try {
    if (ownRecord(message, ["type", "plan"]) && message.type === "prepare" && !prepared) {
      await prepare(message.plan);
      return;
    }
    if (ownRecord(message, ["type", "operation_id"]) && message.type === "execute" && prepared && !executed) {
      if (!Number.isSafeInteger(message.operation_id)) throw new Error("operation identity");
      executed = true;
      runStartup(message.operation_id);
      return;
    }
    if (ownRecord(message, ["type", "operation_id", "event"]) && message.type === "event" && executed) {
      if (!Number.isSafeInteger(message.operation_id) || !validateEvent(message.event)) throw new Error("event schema");
      runEvent(message.operation_id, message.event);
      return;
    }
    throw new Error("worker protocol");
  } catch (error) {
    postFailure(failureCode(error), "Canvas script failed inside the bounded virtual runtime.");
  }
};
}
