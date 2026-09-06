// Closed adapter over the two pinned Jison parsers. No upstream renderer/DB.
function cleanComments(source, budget) {
  // Discard only comment suffixes; retain newlines and token line/columns.
  let out = "", quoted = false, pipeLabel = false, depth = 0;
  for (let i = 0; i < source.length; i += 1) {
    const c = source[i];
    if (!quoted && !pipeLabel && depth === 0 && c === "%" && source[i + 1] === "%") {
      let end = source.indexOf("\n", i);
      if (end < 0) end = source.length;
      if (/^%%\s*\{/.test(source.slice(i, end))) budget.fail("unsupported-syntax");
      i = end - 1; continue;
    }
    if (c === '"') quoted = !quoted;
    if (!quoted) {
      if (c === "|" && depth === 0) pipeLabel = !pipeLabel;
      if (!pipeLabel) {
        if ("[({".includes(c)) depth += 1;
        if ("])}".includes(c)) depth = Math.max(0, depth - 1);
      }
    }
    out += c;
  }
  return out;
}
function plainLabel(value, budget) {
  if (typeof value !== "string") budget.fail("unsupported-label");
  if (/<\/?[A-Za-z!][^>]*>|`|\*\*|__|~~|\[[^\]]*\]\s*\(/u.test(value)) budget.fail("unsupported-label");
  if (/\*(?=\S)(?:[^*\n]*\S)?\*|(^|[\s\p{P}])_(?=\S)(?:[^_\n]*\S)?_(?=$|[\s\p{P}])/u.test(value)) budget.fail("unsupported-label");
  if (/[\u0000-\u0008\u000b-\u001f\u007f]/u.test(value)) budget.fail("invalid-text");
  return value;
}
function identifier(value, budget) {
  if (typeof value !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) budget.fail("invalid-id");
  return value;
}
function configureParser(original, budget, allowedTokens, allowedProductions) {
  const parser = new original.Parser();
  parser.lexer = Object.create(original.lexer);
  const refuse = () => budget.fail("unsupported-syntax");
  parser.yy = {parseError: (_message, hash) => budget.fail("parse-error", hash?.loc?.first_line, (hash?.loc?.first_column ?? -1) + 1)};
  const action = original.performAction;
  parser.performAction = function (...args) {
    if (!allowedProductions.has(args[4])) refuse();
    // A -> B -> C is compound shorthand even though each callback has one pair.
    if (original === flowParser && (args[4] === 46 || args[4] === 47)) {
      const offset = args[4] === 46 ? 3 : 4;
      if (args[5][args[5].length - offset].nodes.length !== 1) refuse();
    }
    return action.apply(this, args);
  };
  const lexAction = original.lexer.performAction;
  parser.lexer.performAction = function (...args) {
    const raw = this.yytext;
    const token = lexAction.apply(this, args);
    // Only skipped comment tokens are directives. Percent pairs inside a
    // sequence TXT/restOfLine token are literal label content.
    if (original === sequenceParser && token === undefined && raw.trim()) {
      const skipped = raw.trimStart();
      if (!skipped.startsWith("%%") || /^%%\s*\{/.test(skipped)) refuse();
    }
    const name = typeof token === "number" ? original.terminals_[token] : token;
    if (token !== undefined && !allowedTokens.has(name)) budget.fail("unsupported-syntax", this.yylloc?.first_line, (this.yylloc?.first_column ?? -1) + 1);
    return token;
  };
  return parser;
}
function integerSet(ranges) {
  const values = new Set();
  for (const [lo, hi] of ranges) for (let i = lo; i <= hi; i += 1) values.add(i);
  return values;
}
const flowTokens = new Set(["SEMI", "NEWLINE", "SPACE", "EOF", "GRAPH", "DIR", "SQS", "SQE", "PS", "PE", "DIAMOND_START", "DIAMOND_STOP", "PIPE", "LINK", "STR", "DOWN", "NUM", "COMMA", "NODE_STRING", "UNIT", "BRKT", "PCT", "MINUS", "MULT", "UNICODE_TEXT", "TEXT", "COLON"]);
const flowProductions = integerSet([[1, 10], [12, 27], [40, 42], [46, 48], [50, 51], [54, 54], [56, 56], [64, 65], [72, 76], [83, 83], [85, 88], [90, 100], [142, 156], [181, 184]]);
function parseFlow(source, budget) {
  const model = {kind: "flow", direction: "TD", nodes: [], edges: []};
  const nodes = new Map();
  const parser = configureParser(flowParser, budget, flowTokens, flowProductions);
  const refuse = () => budget.fail("unsupported-syntax");
  let sawHeader = false;
  Object.assign(parser.yy, {
    lex: {firstGraph: () => {
      if (sawHeader) refuse();
      sawHeader = true;
      return true;
    }},
    setDirection(direction) {
      direction = direction.trim();
      if (!["TD", "TB", "LR"].includes(direction)) refuse();
      model.direction = direction === "TB" ? "TD" : direction;
    },
    addVertex(id, label, shape, ...extras) {
      identifier(id, budget);
      if (extras.some(x => x !== undefined)) refuse();
      const shapes = new Map([[undefined, "rect"], ["square", "rect"], ["round", "rounded"], ["diamond", "diamond"]]);
      if (!shapes.has(shape)) refuse();
      if (label !== undefined && (!label || !["text", "string"].includes(label.type))) budget.fail("unsupported-label");
      const text = label === undefined ? id : plainLabel(label.text, budget);
      const old = nodes.get(id);
      if (old) {
        if (label !== undefined && (old.label !== text || old.shape !== shapes.get(shape))) budget.fail("conflicting-node");
        return;
      }
      budget.charge("nodes");
      const node = {id, label: budget.label(text), shape: shapes.get(shape)};
      nodes.set(id, node); model.nodes.push(node);
    },
    destructLink(link, start) {
      if (link.trim() !== "-->" || start !== undefined) refuse();
      return {type: "arrow_point", stroke: "normal", length: 1};
    },
    addLink(from, to, link) {
      if (from.length !== 1 || to.length !== 1 || link.id !== undefined || link.type !== "arrow_point" || link.stroke !== "normal" || link.length !== 1) refuse();
      if (!nodes.has(from[0]) || !nodes.has(to[0])) budget.fail("missing-endpoint");
      if (from[0] === to[0]) budget.fail("cycle");
      let label = "";
      if (link.text !== undefined) {
        if (!["text", "string"].includes(link.text.type)) budget.fail("unsupported-label");
        label = plainLabel(link.text.text, budget);
      }
      budget.charge("edges");
      model.edges.push({from: from[0], to: to[0], label: budget.label(label)});
    },
    addSubGraph: refuse, setAccTitle: refuse, setAccDescription: refuse,
    setClass: refuse, addClass: refuse, setClickEvent: refuse, setTooltip: refuse,
    setLink: refuse, updateLink: refuse, updateLinkInterpolate: refuse,
  });
  parser.parse(source + "\n");
  if (!model.nodes.length) budget.fail("empty-diagram");
  const indegree = new Map(model.nodes.map(n => [n.id, 0]));
  for (const edge of model.edges) indegree.set(edge.to, indegree.get(edge.to) + 1);
  const ready = model.nodes.filter(n => indegree.get(n.id) === 0).map(n => n.id);
  let count = 0;
  while (ready.length) {
    const id = ready.shift(); count += 1;
    for (const edge of model.edges) if (edge.from === id) {
      indegree.set(edge.to, indegree.get(edge.to) - 1);
      if (indegree.get(edge.to) === 0) ready.push(edge.to);
    }
  }
  if (count !== model.nodes.length) budget.fail("cycle");
  return model;
}
const sequenceTokens = new Set(["SPACE", "NEWLINE", "SD", "participant", "AS", "ACTOR", "restOfLine", "note", "over", "left_of", "right_of", ",", "SOLID_ARROW", "DOTTED_ARROW", "TXT"]);
const sequenceProductions = integerSet([[1, 9], [15, 15], [18, 18], [25, 25], [49, 50], [58, 59], [64, 69], [75, 75], [78, 78], [81, 81], [99, 99], [105, 105]]);
function parseSequence(source, budget) {
  const model = {kind: "sequence", participants: [], messages: [], notes: []};
  const participants = new Map();
  const parser = configureParser(sequenceParser, budget, sequenceTokens, sequenceProductions);
  const refuse = () => budget.fail("unsupported-syntax");
  let order = 0;
  function requireActor(id) {
    identifier(id, budget);
    if (!participants.has(id)) budget.fail("missing-endpoint");
    return id;
  }
  function apply(value) {
    if (Array.isArray(value)) { for (const row of value) apply(row); return; }
    if (!value || typeof value !== "object") refuse();
    switch (value.type) {
      case "addParticipant": {
        identifier(value.actor, budget);
        if (value.config !== undefined) refuse();
        if (value.draw === undefined) { requireActor(value.actor); return; }
        if (value.draw !== "participant") refuse();
        if (participants.has(value.actor)) budget.fail("duplicate-participant");
        budget.charge("participants");
        const participant = {id: value.actor, label: budget.label(value.description === undefined ? value.actor : plainLabel(value.description, budget))};
        participants.set(value.actor, participant); model.participants.push(participant); return;
      }
      case "addMessage": {
        const from = requireActor(value.from), to = requireActor(value.to);
        if (from === to) budget.fail("self-message");
        if (!["solid", "dashed"].includes(value.signalType) || value.activate !== undefined || value.centralConnection !== undefined) refuse();
        budget.charge("messages");
        model.messages.push({from, to, label: budget.label(plainLabel(value.msg, budget)), dashed: value.signalType === "dashed", order: order++}); return;
      }
      case "addNote": {
        if (!["left", "right", "over"].includes(value.placement)) refuse();
        let ids = Array.isArray(value.actor) ? value.actor : [value.actor];
        if (ids.length === 2 && ids[0] === ids[1]) ids = [ids[0]];
        if (!ids.length || ids.length > 2 || (value.placement !== "over" && ids.length !== 1)) refuse();
        ids = ids.map(requireActor);
        budget.charge("notes");
        model.notes.push({side: value.placement, participants: ids, label: budget.label(plainLabel(value.text, budget)), order: order++}); return;
      }
      default: refuse();
    }
  }
  Object.assign(parser.yy, {apply, parseMessage: value => {
    const text = value.trim();
    if (/^(?:no)?wrap\s*:/i.test(text)) budget.fail("unsupported-label");
    return plainLabel(text, budget);
  },
    LINETYPE: Object.freeze({SOLID: "solid", DOTTED: "dashed"}),
    PLACEMENT: Object.freeze({LEFTOF: "left", RIGHTOF: "right", OVER: "over"}),
    parseBoxData: refuse, setDiagramTitle: refuse, setAccTitle: refuse, setAccDescription: refuse});
  parser.parse(source + "\n");
  if (!model.participants.length) budget.fail("empty-diagram");
  return model;
}
function parseMermaid(source, budget) {
  budget.beginDiagram(source);
  let first = "";
  for (const line of source.split("\n")) {
    const text = line.trimStart();
    if (text.startsWith("%%")) {
      if (/^%%\s*\{/.test(text)) budget.fail("unsupported-syntax");
      continue;
    }
    if (text.trim()) { first = text; break; }
  }
  if (/^flowchart[ \t]+(?:TD|TB|LR)(?=[ \t\r\n;]|$)/.test(first)) return parseFlow(cleanComments(source, budget), budget);
  // The pinned sequence lexer already distinguishes comments from label tokens.
  // Giving it the original source avoids a second, incompatible label scanner.
  if (/^sequenceDiagram(?=[ \t\r\n;]|$)/.test(first)) return parseSequence(source, budget);
  budget.fail(first ? "unsupported-syntax" : "empty-diagram");
}
