/*
 * Chatbook Canvas V1 trusted renderer.
 *
 * Model-authored source never enters this realm. The renderer accepts only the
 * compiler's typed plan and the worker's typed patches, validates both again,
 * and uses DOM/CSSOM construction APIs. Keep this file external: the renderer
 * response's CSP deliberately permits no generated/native script text.
 */

"use strict";

const HTML_NAMESPACE = "http://www.w3.org/1999/xhtml";
const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const MAX = Object.freeze({
  htmlBytes: 512 * 1024,
  assets: 64,
  assetBytes: 1024 * 1024,
  aggregateAssetBytes: 4 * 1024 * 1024,
  imageDimension: 4096,
  imagePixels: 4 * 1024 * 1024,
  aggregateImagePixels: 16 * 1024 * 1024,
  imageFrames: 1,
  imageDecodeMilliseconds: 1000,
  assetDecodeTotalMilliseconds: 3000,
  domNodes: 5000,
  cssRules: 2000,
  scriptBytes: 256 * 1024,
  patchesPerEvent: 1000,
  bridgeRequestsPerOperation: 16,
  eventQueue: 100,
  bridgeBytes: 10 * 1024 * 1024,
  bridgeEncodedBytes: Math.ceil((10 * 1024 * 1024) / 3) * 4 + 4096,
  jsonDepth: 16,
  rendererMessageBytes: 15 * 1024 * 1024,
  trustedPrepareMilliseconds: 10000,
  workerStartupBackstopMilliseconds: 750,
  workerEventBackstopMilliseconds: 250,
});

const HTML_TAGS = new Set([
  "a", "abbr", "address", "article", "aside", "b", "bdi", "bdo",
  "blockquote", "body", "br", "button", "caption", "cite", "code", "col",
  "colgroup", "data", "datalist", "dd", "del", "details", "dfn", "div",
  "dl", "dt", "em", "fieldset", "figcaption", "figure", "footer", "form",
  "h1", "h2", "h3", "h4", "h5", "h6", "head", "header", "hr", "html",
  "i", "img", "input", "ins", "kbd", "label", "legend", "li", "main",
  "mark", "menu", "meta", "meter", "nav", "ol", "optgroup", "option",
  "output", "p", "pre", "progress", "q", "s", "samp", "section", "select",
  "small", "span", "strong", "sub", "summary", "sup", "table", "tbody",
  "td", "textarea", "tfoot", "th", "thead", "time", "title", "tr", "u",
  "ul", "var", "wbr",
]);
const SVG_TAGS = new Set([
  "svg", "g", "circle", "ellipse", "line", "path", "polygon", "polyline",
  "rect", "text", "tspan",
]);
const GLOBAL_ATTRIBUTES = new Set([
  "class", "dir", "hidden", "id", "lang", "role", "style", "tabindex",
  "title",
]);
const HTML_ATTRIBUTES = Object.freeze({
  a: new Set(["href"]),
  button: new Set(["disabled", "name", "type", "value"]),
  col: new Set(["span"]),
  colgroup: new Set(["span"]),
  data: new Set(["value"]),
  del: new Set(["datetime"]),
  fieldset: new Set(["disabled", "name"]),
  form: new Set(["name", "novalidate"]),
  img: new Set(["alt", "decoding", "height", "loading", "width"]),
  input: new Set([
    "checked", "disabled", "form", "list", "max", "maxlength", "min",
    "minlength", "multiple", "name", "pattern", "placeholder", "readonly",
    "required", "size", "step", "type", "value",
  ]),
  ins: new Set(["datetime"]),
  label: new Set(["for"]),
  li: new Set(["value"]),
  meta: new Set(["charset"]),
  meter: new Set(["high", "low", "max", "min", "optimum", "value"]),
  ol: new Set(["reversed", "start", "type"]),
  optgroup: new Set(["disabled", "label"]),
  option: new Set(["disabled", "label", "selected", "value"]),
  output: new Set(["for", "form", "name"]),
  progress: new Set(["max", "value"]),
  select: new Set(["disabled", "form", "multiple", "name", "required", "size"]),
  td: new Set(["colspan", "headers", "rowspan"]),
  textarea: new Set([
    "cols", "disabled", "form", "maxlength", "minlength", "name",
    "placeholder", "readonly", "required", "rows", "wrap",
  ]),
  th: new Set(["abbr", "colspan", "headers", "rowspan", "scope"]),
  time: new Set(["datetime"]),
});
const SVG_ATTRIBUTES = new Set([
  "cx", "cy", "d", "fill", "fill-opacity", "height", "points",
  "preserveAspectRatio", "r", "rx", "ry", "stroke", "stroke-dasharray",
  "stroke-dashoffset", "stroke-linecap", "stroke-linejoin", "stroke-opacity",
  "stroke-width", "text-anchor", "transform", "vector-effect", "viewBox",
  "width", "x", "x1", "x2", "y", "y1", "y2",
]);
const BOOLEAN_ATTRIBUTES = new Set([
  "checked", "disabled", "hidden", "multiple", "novalidate", "readonly",
  "required", "reversed", "selected",
]);
const INPUT_TYPES = new Set([
  "button", "checkbox", "color", "date", "datetime-local", "email", "hidden",
  "month", "number", "password", "radio", "range", "reset", "search",
  "submit", "tel", "text", "time", "url", "week",
]);
const BUTTON_TYPES = new Set(["button", "reset", "submit"]);
const EVENTS = new Set([
  "change", "click", "focus", "input", "keydown", "keyup", "reset", "submit",
]);
const STYLE_PROPERTIES = new Set([
  "align-content", "align-items", "align-self", "animation", "animation-delay",
  "animation-direction", "animation-duration", "animation-fill-mode",
  "animation-iteration-count", "animation-name", "animation-play-state",
  "animation-timing-function", "appearance", "aspect-ratio", "backdrop-filter",
  "backface-visibility", "background", "background-attachment",
  "background-blend-mode", "background-clip", "background-color",
  "background-image", "background-origin", "background-position",
  "background-repeat", "background-size", "block-size", "border", "border-block",
  "border-block-color", "border-block-end", "border-block-start",
  "border-block-style", "border-block-width", "border-bottom",
  "border-bottom-color", "border-bottom-left-radius", "border-bottom-right-radius",
  "border-bottom-style", "border-bottom-width", "border-collapse", "border-color",
  "border-inline", "border-inline-color", "border-inline-end", "border-inline-start",
  "border-inline-style", "border-inline-width", "border-left",
  "border-left-color", "border-left-style", "border-left-width", "border-radius",
  "border-right", "border-right-color", "border-right-style", "border-right-width",
  "border-spacing", "border-style", "border-top", "border-top-color",
  "border-top-left-radius", "border-top-right-radius", "border-top-style",
  "border-top-width", "border-width", "bottom", "box-shadow", "box-sizing",
  "break-after", "break-before", "break-inside", "caption-side", "caret-color",
  "clear", "color", "column-gap", "column-width", "columns", "display",
  "empty-cells", "filter", "flex", "flex-basis", "flex-direction", "flex-flow",
  "flex-grow", "flex-shrink", "flex-wrap", "float", "font", "font-family",
  "font-feature-settings", "font-kerning", "font-size", "font-stretch",
  "font-style", "font-variant", "font-weight", "gap", "grid", "grid-area",
  "grid-auto-columns", "grid-auto-flow", "grid-auto-rows", "grid-column",
  "grid-column-end", "grid-column-gap", "grid-column-start", "grid-gap",
  "grid-row", "grid-row-end", "grid-row-gap", "grid-row-start", "grid-template",
  "grid-template-areas", "grid-template-columns", "grid-template-rows", "height",
  "hyphens", "inline-size", "inset", "inset-block", "inset-block-end",
  "inset-block-start", "inset-inline", "inset-inline-end", "inset-inline-start",
  "isolation", "justify-content", "justify-items", "justify-self", "left",
  "letter-spacing", "line-height", "list-style", "list-style-position",
  "list-style-type", "margin", "margin-block", "margin-block-end",
  "margin-block-start", "margin-bottom", "margin-inline", "margin-inline-end",
  "margin-inline-start", "margin-left", "margin-right", "margin-top",
  "max-block-size", "max-height", "max-inline-size", "max-width", "min-block-size",
  "min-height", "min-inline-size", "min-width", "mix-blend-mode", "object-fit",
  "object-position", "opacity", "order", "outline", "outline-color",
  "outline-offset", "outline-style", "outline-width", "overflow", "overflow-wrap",
  "overflow-x", "overflow-y", "padding", "padding-block", "padding-block-end",
  "padding-block-start", "padding-bottom", "padding-inline", "padding-inline-end",
  "padding-inline-start", "padding-left", "padding-right", "padding-top",
  "perspective", "perspective-origin", "place-content", "place-items", "place-self",
  "pointer-events", "position", "resize", "right", "rotate", "row-gap", "scale",
  "scroll-behavior", "shape-rendering", "stroke", "stroke-dasharray",
  "stroke-dashoffset", "stroke-linecap", "stroke-linejoin", "stroke-opacity",
  "stroke-width", "table-layout", "text-align", "text-decoration",
  "text-decoration-color", "text-decoration-line", "text-decoration-style",
  "text-indent", "text-overflow", "text-shadow", "text-transform", "top",
  "transform", "transform-origin", "transform-style", "transition",
  "transition-delay", "transition-duration", "transition-property",
  "transition-timing-function", "translate", "unicode-bidi", "user-select",
  "vertical-align", "visibility", "white-space", "width", "word-break",
  "word-spacing", "writing-mode", "z-index", "fill", "fill-opacity",
]);
const PASSIVE_IMAGE_TYPES = new Set([
  "image/gif", "image/jpeg", "image/png", "image/webp",
]);

const root = document.getElementById("canvas-root");
let nativeById = new Map();
let idByNative = new WeakMap();
let assetUrls = new Map();
const decoder = new TextDecoder("utf-8", {fatal: true});
const encoder = new TextEncoder();
let initialized = false;
let failed = false;
let nonce = null;
let channel = null;
let worker = null;
let prepared = false;
let scriptsStarted = false;
let watchdog = null;
let operationCounter = 0;
let workerBootstrapUrl = null;
let pendingPlan = null;
let runtimeReady = false;
let activeEventOperation = null;
const eventQueue = [];

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

function truncateUtf8(value, byteLimit) {
  const candidate = value.slice(0, byteLimit);
  const encoded = encoder.encode(candidate);
  if (encoded.byteLength <= byteLimit) return candidate;
  const minimum = Math.max(0, byteLimit - 3);
  for (let end = byteLimit; end >= minimum; end -= 1) {
    try {
      return decoder.decode(encoded.subarray(0, end));
    } catch (_) {
      // A UTF-8 code point occupies at most four bytes.
    }
  }
  return "";
}

function jsonDepthAndShape(value, byteLimit = MAX.bridgeBytes) {
  let encoded;
  try {
    encoded = JSON.stringify(value);
  } catch (_) {
    throw new Error("value is not JSON serializable");
  }
  if (typeof encoded !== "string" || encoder.encode(encoded).byteLength > byteLimit) {
    throw new Error("JSON value exceeds byte limit");
  }
  const stack = [{value, depth: 0}];
  while (stack.length) {
    const item = stack.pop();
    if (item.depth > MAX.jsonDepth) throw new Error("JSON value exceeds depth limit");
    if (item.value === null || ["string", "boolean"].includes(typeof item.value)) continue;
    if (typeof item.value === "number" && Number.isFinite(item.value)) continue;
    if (Array.isArray(item.value)) {
      for (const child of item.value) stack.push({value: child, depth: item.depth + 1});
      continue;
    }
    if (item.value && typeof item.value === "object") {
      for (const [key, child] of Object.entries(item.value)) {
        if (typeof key !== "string") throw new Error("JSON key is not a string");
        stack.push({value: child, depth: item.depth + 1});
      }
      continue;
    }
    throw new Error("value is not JSON compatible");
  }
  return JSON.parse(encoded);
}

function isAllowedAttribute(tag, namespace, name) {
  if (name.startsWith("aria-") || name.startsWith("data-")) {
    return name !== "data-canvas-asset";
  }
  if (GLOBAL_ATTRIBUTES.has(name)) return true;
  if (namespace === SVG_NAMESPACE) return SVG_ATTRIBUTES.has(name);
  return Boolean(HTML_ATTRIBUTES[tag]?.has(name));
}

function validateAttributeValue(tag, name, value) {
  if (!boundedString(value, 16 * 1024)) throw new Error("attribute value limit");
  const normalized = value.trim().toLowerCase();
  if (name.startsWith("on")) throw new Error("native event attributes are forbidden");
  if (["href", "src", "action", "formaction", "target", "download"].includes(name)) {
    if (!(tag === "a" && name === "href" && /^#[^\s]+$/.test(value))) {
      throw new Error("URL/navigation attribute is forbidden");
    }
  }
  if (tag === "input" && name === "type" && !INPUT_TYPES.has(normalized)) {
    throw new Error("input type is forbidden");
  }
  if (tag === "button" && name === "type" && !BUTTON_TYPES.has(normalized)) {
    throw new Error("button type is forbidden");
  }
  if (tag === "meta" && name === "charset" && normalized !== "utf-8") {
    throw new Error("metadata charset is forbidden");
  }
  if (["fill", "stroke"].includes(name)) canonicalStyle(name, value);
}

function canonicalStyle(property, value) {
  if (!STYLE_PROPERTIES.has(property) || property.startsWith("--")) {
    throw new Error("style property is not allowlisted");
  }
  if (!boundedString(value, 16 * 1024)) throw new Error("style value limit");
  const scratch = document.createElement("span");
  scratch.style.setProperty(property, value);
  const canonical = scratch.style.getPropertyValue(property);
  if (!canonical && value.trim()) throw new Error("style value is invalid");
  const normalized = canonical.replace(/\s+/g, "").toLowerCase();
  if (
    normalized.includes("url(") || normalized.includes("image-set(") ||
    normalized.includes("cross-fade(") || normalized.includes("element(") ||
    normalized.includes("paint(") || normalized.includes("var(") ||
    normalized.includes("env(") || normalized.includes("attr(")
  ) {
    throw new Error("style resource/computed value is forbidden");
  }
  return canonical;
}

function applyStyleText(target, cssText) {
  if (!boundedString(cssText, 16 * 1024)) throw new Error("style text limit");
  const scratch = document.createElement("span");
  scratch.style.cssText = cssText;
  if (cssText.trim() && scratch.style.length === 0) throw new Error("style text is invalid");
  for (let index = 0; index < scratch.style.length; index += 1) {
    const property = scratch.style.item(index);
    const canonical = canonicalStyle(property, scratch.style.getPropertyValue(property));
    target.style.setProperty(property, canonical, scratch.style.getPropertyPriority(property));
  }
}

function validateStyleSheetRule(rule) {
  if (rule.type === CSSRule.STYLE_RULE) {
    if (rule.cssRules && rule.cssRules.length !== 0) {
      throw new Error("nested style rules are forbidden");
    }
    if (/(^|[^\\]):visited(?:\W|$)/i.test(rule.selectorText)) {
      throw new Error("visited selectors are forbidden");
    }
    for (let index = 0; index < rule.style.length; index += 1) {
      const property = rule.style.item(index);
      canonicalStyle(property, rule.style.getPropertyValue(property));
    }
    return 1;
  }
  if (rule.type === CSSRule.MEDIA_RULE) {
    let count = 1;
    for (const child of rule.cssRules) count += validateStyleSheetRule(child);
    return count;
  }
  throw new Error("CSS rule type is forbidden");
}

function prepareStyleRules(rules) {
  if (!Array.isArray(rules) || rules.length > MAX.cssRules) throw new Error("CSS rule limit");
  if (typeof CSSStyleSheet !== "function") throw new Error("trusted CSSOM is unavailable");
  const sheet = new CSSStyleSheet();
  let count = 0;
  for (const ruleText of rules) {
    if (!boundedString(ruleText, 64 * 1024)) throw new Error("CSS text limit");
    const index = sheet.insertRule(ruleText, sheet.cssRules.length);
    count += validateStyleSheetRule(sheet.cssRules[index]);
    if (count > MAX.cssRules) throw new Error("CSS rule limit");
  }
  return sheet;
}

function readU16BigEndian(bytes, offset) {
  if (offset < 0 || offset + 2 > bytes.length) throw new Error("truncated image metadata");
  return (bytes[offset] << 8) | bytes[offset + 1];
}

function readU16LittleEndian(bytes, offset) {
  if (offset < 0 || offset + 2 > bytes.length) throw new Error("truncated image metadata");
  return bytes[offset] | (bytes[offset + 1] << 8);
}

function readU24LittleEndian(bytes, offset) {
  if (offset < 0 || offset + 3 > bytes.length) throw new Error("truncated image metadata");
  return bytes[offset] | (bytes[offset + 1] << 8) | (bytes[offset + 2] << 16);
}

function readU32BigEndian(bytes, offset) {
  if (offset < 0 || offset + 4 > bytes.length) throw new Error("truncated image metadata");
  return (
    bytes[offset] * 0x1000000 +
    (bytes[offset + 1] << 16) +
    (bytes[offset + 2] << 8) +
    bytes[offset + 3]
  );
}

function readU32LittleEndian(bytes, offset) {
  if (offset < 0 || offset + 4 > bytes.length) throw new Error("truncated image metadata");
  return (
    bytes[offset] +
    (bytes[offset + 1] << 8) +
    (bytes[offset + 2] << 16) +
    bytes[offset + 3] * 0x1000000
  );
}

function imageType(bytes, offset) {
  if (offset < 0 || offset + 4 > bytes.length) throw new Error("truncated image metadata");
  return String.fromCharCode(bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]);
}

function validateRaster(width, height, frames = 1) {
  if (
    !Number.isSafeInteger(width) || !Number.isSafeInteger(height) ||
    width < 1 || height < 1 ||
    width > MAX.imageDimension || height > MAX.imageDimension ||
    width * height > MAX.imagePixels
  ) throw new Error("asset pixel boundary");
  if (!Number.isSafeInteger(frames) || frames < 1 || frames > MAX.imageFrames) {
    throw new Error("animated images are forbidden");
  }
  return {width, height, pixels: width * height, frames};
}

function parsePng(bytes) {
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  if (bytes.length < 8 || !signature.every((value, index) => bytes[index] === value)) {
    throw new Error("asset signature");
  }
  let offset = 8;
  let metadata = null;
  let ended = false;
  while (offset + 12 <= bytes.length) {
    const length = readU32BigEndian(bytes, offset);
    const end = offset + 12 + length;
    if (!Number.isSafeInteger(end) || end > bytes.length) throw new Error("truncated PNG chunk");
    const type = imageType(bytes, offset + 4);
    if (metadata === null && type !== "IHDR") throw new Error("PNG IHDR order");
    if (type === "IHDR") {
      if (metadata !== null || length !== 13) throw new Error("PNG IHDR schema");
      metadata = validateRaster(
        readU32BigEndian(bytes, offset + 8),
        readU32BigEndian(bytes, offset + 12),
      );
    }
    if (type === "acTL") throw new Error("animated PNG is forbidden");
    offset = end;
    if (type === "IEND") {
      if (length !== 0 || metadata === null) throw new Error("PNG IEND schema");
      ended = true;
      break;
    }
  }
  if (!ended || offset !== bytes.length || metadata === null) throw new Error("incomplete PNG");
  return metadata;
}

function parseJpeg(bytes) {
  if (bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) {
    throw new Error("asset signature");
  }
  const startOfFrame = new Set([
    0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7,
    0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf,
  ]);
  let offset = 2;
  while (offset < bytes.length) {
    if (bytes[offset] !== 0xff) throw new Error("JPEG marker boundary");
    while (offset < bytes.length && bytes[offset] === 0xff) offset += 1;
    if (offset >= bytes.length) throw new Error("truncated JPEG marker");
    const marker = bytes[offset];
    offset += 1;
    if (marker === 0xd9 || marker === 0xda) break;
    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd8)) continue;
    const length = readU16BigEndian(bytes, offset);
    if (length < 2 || offset + length > bytes.length) throw new Error("truncated JPEG segment");
    if (startOfFrame.has(marker)) {
      if (length < 7) throw new Error("JPEG frame schema");
      return validateRaster(
        readU16BigEndian(bytes, offset + 5),
        readU16BigEndian(bytes, offset + 3),
      );
    }
    offset += length;
  }
  throw new Error("JPEG frame is missing");
}

function skipGifSubBlocks(bytes, offset) {
  while (true) {
    if (offset >= bytes.length) throw new Error("truncated GIF sub-block");
    const length = bytes[offset];
    offset += 1;
    if (length === 0) return offset;
    if (offset + length > bytes.length) throw new Error("truncated GIF sub-block");
    offset += length;
  }
}

function parseGif(bytes) {
  const header = bytes.length >= 6
    ? String.fromCharCode(...bytes.subarray(0, 6))
    : "";
  if (header !== "GIF87a" && header !== "GIF89a") throw new Error("asset signature");
  if (bytes.length < 13) throw new Error("truncated GIF header");
  const width = readU16LittleEndian(bytes, 6);
  const height = readU16LittleEndian(bytes, 8);
  const packed = bytes[10];
  let offset = 13;
  if (packed & 0x80) offset += 3 * (2 ** ((packed & 0x07) + 1));
  if (offset > bytes.length) throw new Error("truncated GIF color table");
  let frames = 0;
  while (offset < bytes.length) {
    const block = bytes[offset];
    offset += 1;
    if (block === 0x3b) {
      if (offset !== bytes.length) throw new Error("GIF trailer boundary");
      return validateRaster(width, height, frames);
    }
    if (block === 0x21) {
      if (offset >= bytes.length) throw new Error("truncated GIF extension");
      offset += 1;
      offset = skipGifSubBlocks(bytes, offset);
      continue;
    }
    if (block === 0x2c) {
      if (offset + 9 > bytes.length) throw new Error("truncated GIF frame");
      const left = readU16LittleEndian(bytes, offset);
      const top = readU16LittleEndian(bytes, offset + 2);
      const frameWidth = readU16LittleEndian(bytes, offset + 4);
      const frameHeight = readU16LittleEndian(bytes, offset + 6);
      const framePacked = bytes[offset + 8];
      validateRaster(frameWidth, frameHeight);
      if (left + frameWidth > width || top + frameHeight > height) {
        throw new Error("GIF frame exceeds logical screen");
      }
      frames += 1;
      if (frames > MAX.imageFrames) throw new Error("animated GIF is forbidden");
      offset += 9;
      if (framePacked & 0x80) offset += 3 * (2 ** ((framePacked & 0x07) + 1));
      if (offset >= bytes.length) throw new Error("truncated GIF image data");
      offset += 1;
      offset = skipGifSubBlocks(bytes, offset);
      continue;
    }
    if (block !== 0x00) throw new Error("unknown GIF block");
  }
  throw new Error("GIF trailer is missing");
}

function parseWebp(bytes) {
  if (
    bytes.length < 12 || imageType(bytes, 0) !== "RIFF" ||
    imageType(bytes, 8) !== "WEBP" || readU32LittleEndian(bytes, 4) + 8 !== bytes.length
  ) throw new Error("asset signature");
  let offset = 12;
  let containerMetadata = null;
  let frameMetadata = null;
  while (offset + 8 <= bytes.length) {
    const type = imageType(bytes, offset);
    const length = readU32LittleEndian(bytes, offset + 4);
    const dataOffset = offset + 8;
    const end = dataOffset + length;
    if (!Number.isSafeInteger(end) || end > bytes.length) throw new Error("truncated WebP chunk");
    if (type === "ANIM" || type === "ANMF") throw new Error("animated WebP is forbidden");
    if (type === "VP8X") {
      if (containerMetadata !== null || length < 10 || (bytes[dataOffset] & 0x02)) {
        throw new Error("WebP extended header schema");
      }
      containerMetadata = validateRaster(
        readU24LittleEndian(bytes, dataOffset + 4) + 1,
        readU24LittleEndian(bytes, dataOffset + 7) + 1,
      );
    } else if (type === "VP8 ") {
      if (
        frameMetadata !== null || length < 10 || bytes[dataOffset + 3] !== 0x9d ||
        bytes[dataOffset + 4] !== 0x01 || bytes[dataOffset + 5] !== 0x2a
      ) throw new Error("WebP lossy frame schema");
      frameMetadata = validateRaster(
        readU16LittleEndian(bytes, dataOffset + 6) & 0x3fff,
        readU16LittleEndian(bytes, dataOffset + 8) & 0x3fff,
      );
    } else if (type === "VP8L") {
      if (frameMetadata !== null || length < 5 || bytes[dataOffset] !== 0x2f) {
        throw new Error("WebP lossless frame schema");
      }
      const bits = readU32LittleEndian(bytes, dataOffset + 1) >>> 0;
      frameMetadata = validateRaster((bits & 0x3fff) + 1, ((bits >>> 14) & 0x3fff) + 1);
    }
    offset = end + (length & 1);
  }
  if (offset !== bytes.length || frameMetadata === null) throw new Error("WebP frame is missing");
  if (
    containerMetadata !== null &&
    (containerMetadata.width !== frameMetadata.width ||
      containerMetadata.height !== frameMetadata.height)
  ) throw new Error("WebP canvas/frame dimensions differ");
  return containerMetadata ?? frameMetadata;
}

function parseRasterMetadata(mimeType, bytes) {
  if (mimeType === "image/png") return parsePng(bytes);
  if (mimeType === "image/jpeg") return parseJpeg(bytes);
  if (mimeType === "image/gif") return parseGif(bytes);
  if (mimeType === "image/webp") return parseWebp(bytes);
  throw new Error("asset type");
}

function decodeAsset(asset, aggregateBytes) {
  if (!ownRecord(asset, ["asset_id", "mime_type", "data_base64"])) throw new Error("asset schema");
  if (!boundedIdentifier(asset.asset_id) || !PASSIVE_IMAGE_TYPES.has(asset.mime_type)) {
    throw new Error("asset identity/type");
  }
  if (
    !boundedString(asset.data_base64, Math.ceil(MAX.assetBytes / 3) * 4) ||
    !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(asset.data_base64)
  ) {
    throw new Error("asset base64");
  }
  let binary;
  try {
    binary = atob(asset.data_base64);
  } catch (_) {
    throw new Error("asset base64");
  }
  if (binary.length > MAX.assetBytes || aggregateBytes + binary.length > MAX.aggregateAssetBytes) {
    throw new Error("asset byte limit");
  }
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  return {bytes, size: binary.length, ...parseRasterMetadata(asset.mime_type, bytes)};
}

async function verifyImageDecode(blob, metadata, milliseconds) {
  if (typeof createImageBitmap !== "function" || milliseconds < 1) {
    throw new Error("bounded image decode is unavailable");
  }
  let bitmap = null;
  let timer = null;
  let expired = false;
  const decoding = createImageBitmap(blob).then((candidate) => {
    if (expired) {
      candidate.close();
      throw new Error("asset decode timeout");
    }
    return candidate;
  });
  try {
    bitmap = await Promise.race([
      decoding,
      new Promise((_, reject) => {
        timer = setTimeout(() => {
          expired = true;
          reject(new Error("asset decode timeout"));
        }, milliseconds);
      }),
    ]);
    if (bitmap.width !== metadata.width || bitmap.height !== metadata.height) {
      throw new Error("decoded image dimensions differ from metadata");
    }
  } finally {
    if (timer !== null) clearTimeout(timer);
    if (bitmap !== null) bitmap.close();
  }
}

async function prepareAssets(assets) {
  if (!Array.isArray(assets) || assets.length > MAX.assets) throw new Error("asset list schema");
  const prepared = new Map();
  let aggregateBytes = 0;
  let aggregatePixels = 0;
  const deadline = performance.now() + MAX.assetDecodeTotalMilliseconds;
  for (const asset of assets) {
    if (prepared.has(asset.asset_id)) throw new Error("duplicate asset ID");
    const decoded = decodeAsset(asset, aggregateBytes);
    aggregateBytes += decoded.size;
    aggregatePixels += decoded.pixels;
    if (aggregatePixels > MAX.aggregateImagePixels) throw new Error("aggregate image pixel boundary");
    const blob = new Blob([decoded.bytes], {type: asset.mime_type});
    const remaining = Math.floor(deadline - performance.now());
    await verifyImageDecode(blob, decoded, Math.min(MAX.imageDecodeMilliseconds, remaining));
    prepared.set(asset.asset_id, blob);
  }
  return prepared;
}

function validateNodeRecord(node) {
  if (!ownRecord(node, ["node_id", "tag", "attributes", "text", "children"])) {
    throw new Error("node schema");
  }
  if (!boundedIdentifier(node.node_id) || !boundedString(node.tag, 128)) {
    throw new Error("node identity/tag");
  }
  if (!Array.isArray(node.attributes) || !Array.isArray(node.children)) {
    throw new Error("node collection schema");
  }
  if (node.text !== null && !boundedString(node.text, MAX.htmlBytes)) {
    throw new Error("node text limit");
  }
}

function nativeNode(node, inheritedNamespace, assets, assetBindings) {
  if (node.tag === "#text") {
    if (node.attributes.length || node.children.length || node.text === null) {
      throw new Error("text node schema");
    }
    return {native: document.createTextNode(node.text), namespace: inheritedNamespace};
  }
  const namespace = inheritedNamespace === SVG_NAMESPACE || node.tag === "svg"
    ? SVG_NAMESPACE
    : HTML_NAMESPACE;
  const allowed = namespace === SVG_NAMESPACE ? SVG_TAGS : HTML_TAGS;
  if (!allowed.has(node.tag)) throw new Error("element tag is not allowlisted");
  const native = namespace === SVG_NAMESPACE
    ? document.createElementNS(SVG_NAMESPACE, node.tag)
    : document.createElement(node.tag);
  const seen = new Set();
  for (const attribute of node.attributes) {
    if (!Array.isArray(attribute) || attribute.length !== 2) throw new Error("attribute schema");
    const [name, value] = attribute;
    if (!boundedIdentifier(name) || seen.has(name)) throw new Error("attribute identity");
    seen.add(name);
    if (name === "data-canvas-asset") {
      if (node.tag !== "img" || !assets.has(value)) throw new Error("asset handle");
      assetBindings.push({native, assetId: value});
      continue;
    }
    if (!isAllowedAttribute(node.tag, namespace, name)) throw new Error("attribute is not allowlisted");
    validateAttributeValue(node.tag, name, value);
    if (name === "style") applyStyleText(native, value);
    else native.setAttribute(name, value);
  }
  if (node.text !== null) throw new Error("element text slot must be null");
  return {native, namespace};
}

function prepareTree(planRoot, assets) {
  const fragment = document.createDocumentFragment();
  const preparedById = new Map();
  const preparedIdByNative = new WeakMap();
  const assetBindings = [];
  const work = [{node: planRoot, parent: fragment, namespace: HTML_NAMESPACE}];
  let count = 0;
  while (work.length) {
    const item = work.pop();
    validateNodeRecord(item.node);
    count += 1;
    if (count > MAX.domNodes || preparedById.has(item.node.node_id)) {
      throw new Error("DOM node boundary");
    }
    const created = nativeNode(
      item.node,
      item.namespace,
      assets,
      assetBindings,
    );
    preparedById.set(item.node.node_id, created.native);
    preparedIdByNative.set(created.native, item.node.node_id);
    item.parent.appendChild(created.native);
    for (let index = item.node.children.length - 1; index >= 0; index -= 1) {
      work.push({node: item.node.children[index], parent: created.native, namespace: created.namespace});
    }
  }
  return {
    fragment,
    nativeById: preparedById,
    idByNative: preparedIdByNative,
    assetBindings,
  };
}

function commitPlan(preparedAssets, sheet, tree) {
  const urls = new Map();
  const previousSheets = document.adoptedStyleSheets;
  try {
    for (const [assetId, blob] of preparedAssets) {
      urls.set(assetId, URL.createObjectURL(blob));
    }
    for (const binding of tree.assetBindings) {
      binding.native.src = urls.get(binding.assetId);
    }
    document.adoptedStyleSheets = [...previousSheets, sheet];
    root.replaceChildren(tree.fragment);
    nativeById = tree.nativeById;
    idByNative = tree.idByNative;
    assetUrls = urls;
  } catch (error) {
    root.replaceChildren();
    document.adoptedStyleSheets = previousSheets;
    for (const url of urls.values()) URL.revokeObjectURL(url);
    throw error;
  }
}

function validatePlan(plan) {
  if (!ownRecord(plan, ["runtime_profile", "source_identity", "root", "assets", "css_rules", "scripts"])) {
    throw new Error("render plan schema");
  }
  if (plan.runtime_profile !== "canvas-v1") throw new Error("runtime profile");
  if (!ownRecord(plan.source_identity, ["source_bytes", "sha256"])) throw new Error("source identity schema");
  if (!Number.isSafeInteger(plan.source_identity.source_bytes) || plan.source_identity.source_bytes < 0 || plan.source_identity.source_bytes > MAX.htmlBytes) {
    throw new Error("source byte identity");
  }
  if (typeof plan.source_identity.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(plan.source_identity.sha256)) {
    throw new Error("source digest identity");
  }
  if (!Array.isArray(plan.scripts) || !Array.isArray(plan.css_rules)) throw new Error("plan lists");
  const scriptBytes = plan.scripts.reduce((total, script) => {
    if (typeof script !== "string") throw new Error("script type");
    return total + encoder.encode(script).byteLength;
  }, 0);
  if (scriptBytes > MAX.scriptBytes) throw new Error("script byte limit");
  const encodedPlan = JSON.stringify(plan);
  if (!boundedString(encodedPlan, MAX.rendererMessageBytes)) throw new Error("plan message byte limit");
  JSON.parse(decoder.decode(encoder.encode(encodedPlan)));
}

function postTrusted(message) {
  if (!channel) return;
  channel.postMessage({...message, nonce});
}

function clearWatchdog() {
  if (watchdog !== null) clearTimeout(watchdog);
  watchdog = null;
}

function armWatchdog(milliseconds, code) {
  clearWatchdog();
  watchdog = setTimeout(() => fail(code, "Canvas worker exceeded its termination backstop."), milliseconds);
}

function fail(code, message) {
  if (failed) return;
  failed = true;
  clearWatchdog();
  if (worker) {
    worker.terminate();
    worker = null;
  }
  eventQueue.length = 0;
  activeEventOperation = null;
  if (workerBootstrapUrl) {
    URL.revokeObjectURL(workerBootstrapUrl);
    workerBootstrapUrl = null;
  }
  postTrusted({
    type: "canvas:status",
    state: "failed",
    code,
    message,
    scripts_disabled: true,
    engine: "quickjs-wasm",
    native_worker_sentinel: "native-worker-clean",
  });
}

function validatePatch(patch) {
  if (!patch || typeof patch !== "object" || Array.isArray(patch)) throw new Error("patch schema");
  if (!boundedIdentifier(patch.node_id)) throw new Error("patch node ID");
  if (patch.op === "set-text" && ownRecord(patch, ["op", "node_id", "value"])) {
    if (!boundedString(patch.value, MAX.htmlBytes)) throw new Error("patch text");
    return;
  }
  if (patch.op === "set-attribute" && ownRecord(patch, ["op", "node_id", "name", "value"])) {
    if (!boundedIdentifier(patch.name) || !boundedString(patch.value, 16 * 1024)) throw new Error("patch attribute");
    return;
  }
  if (patch.op === "remove-attribute" && ownRecord(patch, ["op", "node_id", "name"])) {
    if (!boundedIdentifier(patch.name)) throw new Error("patch attribute");
    return;
  }
  if (patch.op === "set-property" && ownRecord(patch, ["op", "node_id", "name", "value"])) {
    if (!["checked", "disabled", "selected", "value"].includes(patch.name)) throw new Error("patch property");
    if (patch.name === "value" ? !boundedString(patch.value, 16 * 1024) : typeof patch.value !== "boolean") {
      throw new Error("patch property value");
    }
    return;
  }
  if (patch.op === "set-style" && ownRecord(patch, ["op", "node_id", "name", "value"])) {
    canonicalStyle(patch.name, patch.value);
    return;
  }
  if (patch.op === "remove-style" && ownRecord(patch, ["op", "node_id", "name"])) {
    if (!STYLE_PROPERTIES.has(patch.name)) throw new Error("patch style");
    return;
  }
  if (patch.op === "create-element" && ownRecord(patch, ["op", "node_id", "tag", "namespace"])) {
    if (![HTML_NAMESPACE, SVG_NAMESPACE].includes(patch.namespace)) throw new Error("patch namespace");
    const tags = patch.namespace === SVG_NAMESPACE ? SVG_TAGS : HTML_TAGS;
    if (!tags.has(patch.tag)) throw new Error("patch tag");
    return;
  }
  if (patch.op === "create-text" && ownRecord(patch, ["op", "node_id", "value"])) {
    if (!boundedString(patch.value, MAX.htmlBytes)) throw new Error("patch text");
    return;
  }
  if (["append-child", "remove-child"].includes(patch.op) && ownRecord(patch, ["op", "node_id", "child_id"])) {
    if (!boundedIdentifier(patch.child_id)) throw new Error("patch child ID");
    return;
  }
  if (patch.op === "insert-before" && ownRecord(patch, ["op", "node_id", "child_id", "reference_id"])) {
    if (!boundedIdentifier(patch.child_id) || (patch.reference_id !== null && !boundedIdentifier(patch.reference_id))) {
      throw new Error("patch child/reference ID");
    }
    return;
  }
  throw new Error("patch operation/schema");
}

function removeMappings(native, state) {
  const stack = [native];
  while (stack.length) {
    const current = stack.pop();
    const identifier = state.idByNative.get(current);
    if (identifier) state.nativeById.delete(identifier);
    for (const child of current.childNodes) stack.push(child);
  }
}

function applyPatch(patch, state) {
  validatePatch(patch);
  if (patch.op === "create-element") {
    if (state.nativeById.has(patch.node_id) || state.nativeById.size >= MAX.domNodes) throw new Error("created node boundary");
    const native = patch.namespace === SVG_NAMESPACE
      ? document.createElementNS(SVG_NAMESPACE, patch.tag)
      : document.createElement(patch.tag);
    state.nativeById.set(patch.node_id, native);
    state.idByNative.set(native, patch.node_id);
    return;
  }
  if (patch.op === "create-text") {
    if (state.nativeById.has(patch.node_id) || state.nativeById.size >= MAX.domNodes) throw new Error("created node boundary");
    const native = document.createTextNode(patch.value);
    state.nativeById.set(patch.node_id, native);
    state.idByNative.set(native, patch.node_id);
    return;
  }
  const target = state.nativeById.get(patch.node_id);
  if (!target) throw new Error("unknown patch target");
  if (patch.op === "set-text") {
    for (const child of target.childNodes) removeMappings(child, state);
    target.textContent = patch.value;
  } else if (patch.op === "set-attribute") {
    if (!(target instanceof Element)) throw new Error("attribute target");
    const namespace = target.namespaceURI === SVG_NAMESPACE ? SVG_NAMESPACE : HTML_NAMESPACE;
    const tag = target.localName;
    if (!isAllowedAttribute(tag, namespace, patch.name)) throw new Error("attribute is not allowlisted");
    validateAttributeValue(tag, patch.name, patch.value);
    if (patch.name === "style") applyStyleText(target, patch.value);
    else target.setAttribute(patch.name, patch.value);
  } else if (patch.op === "remove-attribute") {
    if (!(target instanceof Element)) throw new Error("attribute target");
    const namespace = target.namespaceURI === SVG_NAMESPACE ? SVG_NAMESPACE : HTML_NAMESPACE;
    if (!isAllowedAttribute(target.localName, namespace, patch.name)) throw new Error("attribute is not allowlisted");
    target.removeAttribute(patch.name);
  } else if (patch.op === "set-property") {
    target[patch.name] = patch.value;
  } else if (patch.op === "set-style") {
    if (!(target instanceof Element)) throw new Error("style target");
    target.style.setProperty(patch.name, canonicalStyle(patch.name, patch.value));
  } else if (patch.op === "remove-style") {
    if (!(target instanceof Element)) throw new Error("style target");
    target.style.removeProperty(patch.name);
  } else {
    const child = state.nativeById.get(patch.child_id);
    if (!child) throw new Error("unknown patch child");
    if (patch.op === "append-child") target.appendChild(child);
    else if (patch.op === "remove-child") {
      if (child.parentNode !== target) throw new Error("patch parent mismatch");
      target.removeChild(child);
      removeMappings(child, state);
    } else {
      const reference = patch.reference_id === null ? null : state.nativeById.get(patch.reference_id);
      if (patch.reference_id !== null && (!reference || reference.parentNode !== target)) {
        throw new Error("patch reference mismatch");
      }
      target.insertBefore(child, reference);
    }
  }
}

function cloneRendererState() {
  const state = {
    fragment: document.createDocumentFragment(),
    nativeById: new Map(),
    idByNative: new WeakMap(),
  };
  const cloned = new Map();

  function indexClone(original, copy) {
    cloned.set(original, copy);
    const identifier = idByNative.get(original);
    if (identifier) {
      state.nativeById.set(identifier, copy);
      state.idByNative.set(copy, identifier);
    }
    for (let index = 0; index < original.childNodes.length; index += 1) {
      indexClone(original.childNodes[index], copy.childNodes[index]);
    }
  }

  for (const child of root.childNodes) {
    const copy = child.cloneNode(true);
    state.fragment.appendChild(copy);
    indexClone(child, copy);
  }
  for (const original of nativeById.values()) {
    if (cloned.has(original)) continue;
    let top = original;
    while (
      top.parentNode && top.parentNode !== root &&
      idByNative.has(top.parentNode) && !cloned.has(top.parentNode)
    ) top = top.parentNode;
    if (!cloned.has(top)) indexClone(top, top.cloneNode(true));
  }
  return state;
}

function prepareBridge(bridge) {
  if (!ownRecord(bridge, ["request_id", "kind", "value"])) throw new Error("bridge schema");
  if (!boundedIdentifier(bridge.request_id) || !["submit", "download"].includes(bridge.kind)) {
    throw new Error("bridge identity");
  }
  let value;
  if (bridge.kind === "submit" && typeof bridge.value === "string") {
    if (!boundedString(bridge.value, 16 * 1024)) throw new Error("bridge value");
    value = bridge.value;
  } else {
    value = jsonDepthAndShape(
      bridge.value,
      bridge.kind === "submit" ? 16 * 1024 : MAX.bridgeEncodedBytes,
    );
  }
  if (bridge.kind === "download") validateDownloadRequest(value);
  return {
    request_id: bridge.request_id,
    kind: bridge.kind,
    value,
  };
}

function validateDownloadRequest(value) {
  if (!ownRecord(value, ["filename", "mime_type", "data"]) ||
      typeof value.filename !== "string" || typeof value.mime_type !== "string" || typeof value.data !== "string") {
    throw new Error("download schema");
  }
  const filename = value.filename.trim();
  if (!filename || new TextEncoder().encode(filename).length > 255 || /[\\/\x00-\x1f\x7f<>:"|?*]/.test(filename) ||
      filename.startsWith(".") || filename.endsWith(".") || filename.endsWith(" ")) throw new Error("download filename");
  if (/^(con|prn|aux|nul|com[1-9]|lpt[1-9])$/i.test(filename.split(".", 1)[0])) throw new Error("download filename");
  const allowed = {
    "text/plain": [".txt"], "text/csv": [".csv"], "application/json": [".json"],
    "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"], "image/gif": [".gif"], "image/webp": [".webp"],
  };
  const extensions = allowed[value.mime_type];
  if (!extensions || !extensions.some((extension) => filename.toLowerCase().endsWith(extension))) throw new Error("download type");
  if (value.mime_type.startsWith("image/")) {
    const prefix = `data:${value.mime_type};base64,`;
    if (!value.data.startsWith(prefix)) throw new Error("download encoding");
    let decoded;
    try { decoded = atob(value.data.slice(prefix.length)); } catch (_) { throw new Error("download encoding"); }
    if (decoded.length > MAX.bridgeBytes) throw new Error("download bytes");
  } else {
    if (value.data.startsWith("data:") || new TextEncoder().encode(value.data).length > MAX.bridgeBytes) throw new Error("download bytes");
    if (value.mime_type === "application/json") {
      try { JSON.parse(value.data); } catch (_) { throw new Error("download json"); }
    }
  }
}

function prepareTransaction(patches, bridges) {
  const state = cloneRendererState();
  const journal = [];
  for (const patch of patches) {
    applyPatch(patch, state);
    journal.push(Object.freeze({...patch}));
  }
  const preparedBridges = bridges.map((bridge) => Object.freeze(prepareBridge(bridge)));
  return Object.freeze({
    journal: Object.freeze(journal),
    bridges: Object.freeze(preparedBridges),
  });
}

function commitTransaction(transaction) {
  const liveState = {nativeById, idByNative};
  for (const patch of transaction.journal) applyPatch(patch, liveState);
  nativeById = liveState.nativeById;
  idByNative = liveState.idByNative;
  for (const bridge of transaction.bridges) {
    postTrusted({type: "canvas:bridge-request", ...bridge});
  }
}

function validWorkerMessage(message) {
  try {
    return (
      message && typeof message === "object" && !Array.isArray(message) &&
      boundedString(JSON.stringify(message), MAX.rendererMessageBytes)
    );
  } catch (_) {
    return false;
  }
}

function handleWorkerMessage(event) {
  const message = event.data;
  if (!validWorkerMessage(message) || failed) return fail("worker-protocol", "Canvas worker emitted an invalid message.");
  if (message.type === "prepared" && ownRecord(message, ["type", "native_worker_sentinel"])) {
    if (message.native_worker_sentinel !== "native-worker-clean" || prepared) {
      return fail("worker-protocol", "Canvas worker preparation was invalid.");
    }
    prepared = true;
    clearWatchdog();
    postTrusted({type: "canvas:execution-started"});
    return;
  }
  if (message.type === "bootstrap-ready" && ownRecord(message, ["type"])) {
    if (pendingPlan === null || prepared) {
      return fail("worker-protocol", "Canvas worker bootstrap was invalid.");
    }
    armWatchdog(MAX.trustedPrepareMilliseconds, "worker-prepare-timeout");
    worker.postMessage({type: "prepare", plan: pendingPlan});
    pendingPlan = null;
    return;
  }
  if (message.type === "failure" && ownRecord(message, ["type", "code", "message", "native_worker_sentinel"])) {
    return fail(
      boundedIdentifier(message.code) ? message.code : "runtime-error",
      boundedString(message.message, 4096) ? message.message : "Canvas execution failed.",
    );
  }
  if (message.type === "bootstrap-failure" && ownRecord(message, ["type", "name"])) {
    console.error("Canvas trusted worker module bootstrap failure", message.name);
    return fail("worker-load-failed", "Canvas worker module could not start.");
  }
  if (message.type === "transaction" && ownRecord(message, ["type", "operation_id", "operation_kind", "patches", "bridges", "native_worker_sentinel"])) {
    if (
      message.native_worker_sentinel !== "native-worker-clean" ||
      !Number.isSafeInteger(message.operation_id) ||
      !["startup", "event", "timer"].includes(message.operation_kind) ||
      !Array.isArray(message.patches) || message.patches.length > MAX.patchesPerEvent ||
      !Array.isArray(message.bridges) ||
      message.bridges.length > MAX.bridgeRequestsPerOperation
    ) return fail("worker-protocol", "Canvas worker transaction was invalid.");
    if (message.operation_kind === "startup" && runtimeReady) {
      return fail("worker-protocol", "Canvas worker repeated its startup transaction.");
    }
    if (
      message.operation_kind === "event" &&
      message.operation_id !== activeEventOperation
    ) {
      return fail("worker-protocol", "Canvas worker event ordering was invalid.");
    }
    try {
      commitTransaction(prepareTransaction(message.patches, message.bridges));
    } catch (_) {
      return fail("invalid-patch", "Canvas emitted a patch outside the V1 renderer vocabulary.");
    }
    if (message.operation_kind === "startup") {
      clearWatchdog();
      runtimeReady = true;
      postTrusted({
        type: "canvas:status",
        state: "ready",
        code: null,
        message: null,
        scripts_disabled: false,
        engine: "quickjs-wasm",
        native_worker_sentinel: "native-worker-clean",
      });
      pumpEventQueue();
    } else if (message.operation_kind === "event") {
      clearWatchdog();
      activeEventOperation = null;
      pumpEventQueue();
    }
    return;
  }
  fail("worker-protocol", "Canvas worker emitted an unknown message.");
}

function executeScripts() {
  if (failed || scriptsStarted || !prepared) return;
  scriptsStarted = true;
  operationCounter += 1;
  armWatchdog(MAX.workerStartupBackstopMilliseconds, "worker-unresponsive");
  worker.postMessage({type: "execute", operation_id: operationCounter});
}

function pumpEventQueue() {
  if (failed || !runtimeReady || activeEventOperation !== null || !eventQueue.length) return;
  const record = eventQueue.shift();
  operationCounter += 1;
  activeEventOperation = operationCounter;
  armWatchdog(MAX.workerEventBackstopMilliseconds, "worker-unresponsive");
  worker.postMessage({type: "event", operation_id: activeEventOperation, event: record});
}

function sendEvent(type, nativeEvent, explicitTarget = null) {
  if (failed || !scriptsStarted || !EVENTS.has(type)) return;
  let target = explicitTarget || nativeEvent.target;
  while (target && !idByNative.has(target)) target = target.parentNode;
  if (!target) return;
  const record = {
    type,
    target_id: idByNative.get(target),
    value: typeof target.value === "string" ? truncateUtf8(target.value, 16 * 1024) : null,
    checked: typeof target.checked === "boolean" ? target.checked : null,
    key: typeof nativeEvent.key === "string" ? truncateUtf8(nativeEvent.key, 64) : null,
  };
  if (eventQueue.length >= MAX.eventQueue) {
    fail("event-queue-limit", "Canvas event queue exceeded its bounded capacity.");
    return;
  }
  eventQueue.push(record);
  pumpEventQueue();
}

function submitFormForControl(target) {
  if (!(target instanceof Element)) return null;
  const control = target.closest("button, input");
  if (!control || !control.form) return null;
  const defaultType = control.localName === "button" ? "submit" : "text";
  return (control.getAttribute("type") || defaultType).toLowerCase() === "submit"
    ? control.form
    : null;
}

for (const type of EVENTS) {
  document.addEventListener(type, (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const syntheticSubmit = type === "click" ? submitFormForControl(target) : null;
    if (
      type === "submit" || type === "reset" ||
      (type === "click" && (target?.closest("a") || syntheticSubmit))
    ) event.preventDefault();
    sendEvent(type, event);
    if (syntheticSubmit) sendEvent("submit", event, syntheticSubmit);
  }, true);
}

async function start(plan) {
  try {
    validatePlan(plan);
    const preparedAssets = await prepareAssets(plan.assets);
    const sheet = prepareStyleRules(plan.css_rules);
    const tree = prepareTree(plan.root, preparedAssets);
    commitPlan(preparedAssets, sheet, tree);
  } catch (_) {
    fail("invalid-plan", "Canvas render plan failed trusted renderer validation.");
    return;
  }
  const workerModuleUrl = new URL("./canvas_runtime_worker.js", import.meta.url).href;
  const bootstrap = `import(${JSON.stringify(workerModuleUrl)}).then((module) => { module.startCanvasRuntimeWorker(globalThis); postMessage({type: "bootstrap-ready"}); }).catch((error) => postMessage({type: "bootstrap-failure", name: String(error && error.name || "Error")}));`;
  workerBootstrapUrl = `data:text/javascript;base64,${btoa(bootstrap)}`;
  worker = new Worker(workerBootstrapUrl, {
    type: "module",
    name: "chatbook-canvas-v1",
  });
  worker.onmessage = handleWorkerMessage;
  worker.onerror = (event) => {
    console.error("Canvas trusted worker load failure", event.message, event.filename, event.lineno);
    fail("worker-load-failed", "Canvas worker could not start.");
  };
  pendingPlan = plan;
  armWatchdog(MAX.trustedPrepareMilliseconds, "worker-bootstrap-timeout");
}

window.addEventListener("message", (event) => {
  const expectedOrigin = new URL(location.href).origin;
  if (
    initialized || event.source !== parent || event.origin !== expectedOrigin ||
    !ownRecord(event.data, ["type", "nonce", "plan"]) ||
    event.data.type !== "canvas:init" || !boundedIdentifier(event.data.nonce) ||
    event.ports.length !== 1
  ) return;
  initialized = true;
  nonce = event.data.nonce;
  channel = event.ports[0];
  channel.onmessage = (portEvent) => {
    const message = portEvent.data;
    if (
      ownRecord(message, ["type", "nonce"]) &&
      message.type === "canvas:execution-ack" && message.nonce === nonce
    ) executeScripts();
  };
  channel.start();
  void start(event.data.plan);
});

window.addEventListener("pagehide", () => {
  for (const url of assetUrls.values()) URL.revokeObjectURL(url);
  assetUrls.clear();
});

parent.postMessage({type: "canvas:renderer-ready"}, new URL(location.href).origin);
