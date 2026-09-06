// One owner per document; beginDiagram never resets document accounting.
class DiagramError extends Error {
  constructor(code, ordinal, line = null, column = null) {
    super(code);
    this.code = code;
    this.ordinal = ordinal;
    this.line = Number.isInteger(line) ? Math.min(8192, Math.max(1, line)) : null;
    this.column = Number.isInteger(column) ? Math.min(8192, Math.max(1, column)) : null;
  }
}
function utf8Bytes(text) {
  let count = 0;
  for (const point of text) {
    const cp = point.codePointAt(0);
    if (cp >= 0xd800 && cp <= 0xdfff) throw new DiagramError("invalid-text", 1);
    count += cp <= 0x7f ? 1 : cp <= 0x7ff ? 2 : cp <= 0xffff ? 3 : 4;
  }
  return count;
}
const semanticLimits = Object.freeze({input: [8192, 16384], labels: [4096, 8192],
  nodes: [16, 24], edges: [24, 32], participants: [6, 8], messages: [16, 24], notes: [8, 12]});
class DiagramBudget {
  constructor() { this.ordinal = 0; this.document = new Map(); this.diagram = new Map(); }
  fail(code, line = null, column = null) { throw new DiagramError(code, this.ordinal || 1, line, column); }
  beginDiagram(source) {
    this.ordinal += 1;
    if (this.ordinal > 4) this.fail("declaration-limit");
    this.diagram = new Map();
    if (typeof source !== "string") this.fail("invalid-text");
    this.charge("input", utf8Bytes(source));
  }
  charge(kind, amount = 1) {
    const limits = semanticLimits[kind];
    if (!limits || !Number.isSafeInteger(amount) || amount < 0) throw new Error("invalid budget charge");
    const local = (this.diagram.get(kind) || 0) + amount;
    const total = (this.document.get(kind) || 0) + amount;
    if (local > limits[0] || total > limits[1]) this.fail(kind + "-limit");
    this.diagram.set(kind, local); this.document.set(kind, total);
  }
  label(text) {
    const size = utf8Bytes(text);
    if (size > 512) this.fail("label-limit");
    this.charge("labels", size);
    return text;
  }
}
