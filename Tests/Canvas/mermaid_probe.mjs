import fs from "node:fs";
import crypto from "node:crypto";

const staticURL = new URL("../../tldw_chatbook/Canvas/static/", import.meta.url);
const sha = bytes => crypto.createHash("sha256").update(bytes).digest("hex");
const request = JSON.parse(fs.readFileSync(0, "utf8"));
const catalog = JSON.parse(fs.readFileSync(new URL("profile-catalog.json", staticURL)));
const candidate = catalog.profiles.find(row => row.profile_id === "canvas-v2-mermaid-1");
if (!candidate) throw new Error("missing real candidate");
const manifestBytes = fs.readFileSync(new URL(candidate.manifest, staticURL));
if (sha(manifestBytes) !== candidate.manifest_sha256) throw new Error("manifest integrity");
const manifest = JSON.parse(manifestBytes);
const libraryBytes = fs.readFileSync(new URL("mermaid-subset.json", staticURL));
const record = candidate.library.files["mermaid-subset.json"];
if (sha(libraryBytes) !== record.sha256 || libraryBytes.length !== record.bytes) throw new Error("library integrity");
const library = JSON.parse(libraryBytes);
if (sha(library.source) !== library.source_sha256 || Buffer.byteLength(library.source) !== library.source_bytes || library.source_bytes > 262144) throw new Error("source integrity");
const engineURL = new URL("quickjs-runtime.js", staticURL);
const engine = fs.readFileSync(engineURL);
if (sha(engine) !== manifest.outputs["quickjs-runtime.js"].sha256) throw new Error("engine integrity");
const {newQuickJSWASMModule} = await import("data:text/javascript;base64," + engine.toString("base64"));
const module = await newQuickJSWASMModule();
const runtime = module.newRuntime();
runtime.setMemoryLimit(32 * 1024 * 1024);
runtime.setMaxStackSize(512 * 1024);
runtime.removeModuleLoader();
const started = performance.now();
const deadline = started + 250;
runtime.setInterruptHandler(() => performance.now() > deadline);
const vm = runtime.newContext();
const handles = [];
function owned(handle) { handles.push(handle); return handle; }
function check(result) {
  if (result.error) {
    const handle = owned(result.error);
    const value = vm.dump(handle);
    throw new Error("unexpected guest/API failure: " + JSON.stringify(value));
  }
  return owned(result.value);
}
try {
  const api = check(vm.evalCode(library.source, "canvas-mermaid-private.js"));
  for (const name of ["parseMermaid", "DiagramBudget", "flowParser", "sequenceParser", "window", "document", "fetch", "require", "process"]) {
    if (vm.dump(owned(vm.getProp(vm.global, name))) !== undefined) throw new Error("private boundary leak");
  }
  const parse = owned(vm.getProp(api, "parseMermaid"));
  const budgetClass = owned(vm.getProp(api, "DiagramBudget"));
  // Reflect.construct is looked up from this guest realm; no generated source.
  const reflect = owned(vm.getProp(vm.global, "Reflect"));
  const construct = owned(vm.getProp(reflect, "construct"));
  const empty = owned(vm.newArray());
  const budget = check(vm.callFunction(construct, reflect, budgetClass, empty));
  const models = [];
  const scenes = [];
  const layout = owned(vm.getProp(api, "layoutDiagram"));
  let failure = null;
  if (request.operation === "text") {
    const segment = owned(vm.getProp(api, "segmentGraphemes"));
    const width = owned(vm.getProp(api, "graphemeWidth"));
    for (const source of request.sources) {
      const arg = owned(vm.newString(source));
      const clusters = vm.dump(check(vm.callFunction(segment, vm.undefined, arg)));
      const widths = clusters.map(cluster => vm.dump(check(vm.callFunction(width, vm.undefined, owned(vm.newString(cluster))))));
      models.push({clusters, widths});
    }
  } else {
    for (const source of request.operation === "render" ? [null] : request.sources ?? [request.source]) {
      let result;
      if (request.operation === "render") {
        const render = owned(vm.getProp(api, "renderDiagrams"));
        const records = owned(vm.newArray());
        for (const [index, text] of (request.sources ?? [request.source]).entries()) {
          const record = owned(vm.newObject());
          vm.setProp(record, "source", owned(vm.newString(text)));
          vm.setProp(records, index, record);
        }
        result = vm.callFunction(render, vm.undefined, records);
      } else {
        const arg = owned(vm.newString(source));
        result = vm.callFunction(parse, vm.undefined, arg, budget);
      }
      if (!result.error && request.operation === "layout") {
        const model = owned(result.value);
        models.push(vm.dump(model));
        // Component tests exhaust one real budget scope without raising its caps.
        for (const scope of ["diagram", "document"]) {
          const counts = owned(vm.getProp(budget, scope));
          const set = owned(vm.getProp(counts, "set"));
          for (const [kind, amount] of Object.entries(request.seed?.[scope] ?? {})) {
            check(vm.callFunction(set, counts, owned(vm.newString(kind)), owned(vm.newNumber(amount))));
          }
        }
        result = vm.callFunction(layout, vm.undefined, model, budget);
      }
      if (result.error) {
        const error = owned(result.error);
        const code = owned(vm.getProp(error, "code"));
        const value = vm.dump(code);
        const codes = new Set(["parse-error", "unsupported-syntax", "unsupported-label", "invalid-id", "invalid-text", "conflicting-node", "missing-endpoint", "cycle", "self-message", "duplicate-participant", "empty-diagram", "declaration-limit", "input-limit", "labels-limit", "label-limit", "nodes-limit", "edges-limit", "participants-limit", "messages-limit", "notes-limit", "work-limit", "elements-limit", "output-limit", "area-limit", "geometry-limit"]);
        if (!codes.has(value)) throw new Error("unexpected guest failure: " + JSON.stringify(vm.dump(error)));
        failure = {code: value};
        for (const key of ["ordinal", "line", "column"]) failure[key] = vm.dump(owned(vm.getProp(error, key)));
        break;
      }
      const value = vm.dump(owned(result.value));
      if (request.operation === "render") scenes.push(...value);
      else (request.operation === "layout" ? scenes : models).push(value);
    }
  }
  const output = {ok: failure === null, model: request.sources ? models : models[0] ?? null, error: failure};
  if (["layout", "render"].includes(request.operation)) output.scene = request.sources ? scenes : scenes[0] ?? null;
  if (request.measure === true) {
    const elapsed = performance.now() - started;
    const usage = runtime.computeMemoryUsage();
    try {
      output.resources = {library_bytes: library.source_bytes, guest_heap_bytes: runtime.getSystemContext().dump(usage).memory_used_size, library_and_layout_ms: elapsed};
    } finally { usage.dispose(); }
  }
  process.stdout.write(JSON.stringify(output));
} finally {
  for (const handle of handles.reverse()) handle.dispose();
  vm.dispose(); runtime.dispose();
}
