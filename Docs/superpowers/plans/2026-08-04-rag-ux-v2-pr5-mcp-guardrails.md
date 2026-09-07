# RAG UX v2 — PR-5: MCP + Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the final six items of the RAG UX v2 program (RAG-48, 49, 50, 51, 53, 55): built-in MCP tools render as typed forms, tool-run results get an honest summary line with collapsible raw JSON, the inspector's stale empty-state badge disappears when detail is shown, the resolved permission decision is named in words (and logged honestly), concurrent app instances get a visible non-blocking warning, and a CSS-coverage contract test makes composed-but-unstyled widgets unshippable in the RAG/Console/Library/MCP surface.

**Architecture:** RAG-48 is fixed upstream of the renderer: the AST-based manifest in `MCP/server.py` learns to synthesize `inputSchema` from function signatures (preserving the MCP_AVAILABLE=False contract), the catalog stops hardcoding `input_schema=None`, and the schema form gains array-of-simple support — with `builtin:tldw_chatbook` added to `HASH_FREE_SERVER_KEYS` first so populating schemas does not rug-pull every stored "allow" to "ask". RAG-49/51 share one seam (`_run_tool_test` → `show_tool_result`): the structured envelope survives to the inspector, which renders `OK · local · 981ms · 3 results` + a quiet interpretation + a Collapsible raw body, plus a one-sentence permission-decision note; the control plane stops hardcoding `decision="allowed"` in the execution log. RAG-53 is an advisory non-blocking portalocker lock beside the profile's data dir, surfaced as a toast sibling to the first-run-wizard notice. RAG-55 is a frozen-registry contract test over literal `classes=` tokens in the four program surfaces.

**Tech Stack:** Python ≥3.11, Textual 8.x, `ast` stdlib (schema synthesis), portalocker 3.2.0 (hard dep), pytest.

**Branch/worktree:** `feat/rag-v2-mcp-guardrails` in `.worktrees/rag-v2-pr5`, base `b5a8fcec5` (PR-4 merge). All paths below are relative to the worktree root.

## Global Constraints

- **Escaping is terminal / markup discipline:** in the MCP Hub the convention is `markup=False` on Statics (`mcp_inspector.py:758,759,1541`, rationale `:387-390`) and `escape_markup` only at `notify()` via `_toast()` (`mcp_workbench.py:119-130`). Every new Static rendering tool output MUST set `markup=False`. Tool output is untrusted (the `builtin:` branch executes in-process code — `mcp_workbench.py:3241-3246` says so).
- **Quiet register:** copy is verb-first plain language. Exemplars: `_ORIGIN_SENTENCES` (`mcp_inspector.py:130-135`), `_EMPTY_STATE_COPY` (`:157-161`), `_toast()` calls (`mcp_workbench.py:3200-3206`). Never re-add the "Owner:/Unavailable:" dump register PR-2 retired.
- **Targeted tests only** (owner ruling 2026-08-02): each task's gate = the test files it touched + `pytest Tests/ --collect-only -q` sweep tail. Never run full `Tests/UI`.
- **venv-only pytest:** `source /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/activate` first. pytest under `Tests/` is the ONLY python entry point — a bare `python3 -c` importing app modules writes to the LIVE config. Scratch probes = pytest files under `Tests/`, deleted after.
- **CSS bundle:** never hand-edit `tldw_chatbook/css/tldw_cli_modular.tcss`; edit source `.tcss` under `css/components|features/` then run `python3 -m tldw_chatbook.css.build_css` (as a pytest-free build step this is safe — it does not import the app config; if unsure, check `build_css.py` imports first).
- **Protected oracles:** existing pinned tests may be updated ONLY where this plan deliberately changes the contract, and the change must be named in the task report. Never edit a test and report it unmodified.
- **git stash is forbidden** (repo-wide stash stack shared across worktrees).
- **Backlog IDs:** any follow-up task filed at ship time needs a fresh cross-worktree `os.listdir`+regex scan (last known max: 2223).
- **Line numbers** in this plan were verified at `b5a8fcec5` by the scout; re-verify against the real file before editing — never edit blind.

## File Structure

| File | Responsibility in this PR |
|---|---|
| `tldw_chatbook/MCP/server.py` | Task 1: `_extract_registered_entries` synthesizes `inputSchema` from AST |
| `tldw_chatbook/MCP/hub_tool_catalog.py` | Task 2: builtins read `inputSchema` like their two siblings |
| `tldw_chatbook/MCP/permission_store.py` | Task 2: `builtin:tldw_chatbook` → `HASH_FREE_SERVER_KEYS` |
| `tldw_chatbook/UI/MCP_Modules/mcp_schema_form.py` | Task 3: array-of-simple support |
| `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py` | Task 4: structured result to inspector; Task 5: gate plumb-through |
| `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py` | Task 4: summary + Collapsible; Task 5: decision note; Task 6: badge fix |
| `tldw_chatbook/MCP/unified_control_plane_service.py` | Task 5: real decision into execution log |
| `tldw_chatbook/UI/MCP_Modules/mcp_audit_mode.py` | Task 5: render the new decision value |
| `tldw_chatbook/Utils/instance_lock.py` (new) | Task 7: advisory profile lock |
| `tldw_chatbook/app.py` | Task 7: acquire at boot + toast in `_push_initial_screen` tail |
| `Tests/UI/test_css_class_coverage_contract.py` (new) | Task 8: the RAG-55 guardrail |
| `Tests/MCP/test_local_control_service.py`, `Tests/MCP/test_hub_tool_catalog.py`, `Tests/UI/test_mcp_schema_form.py`, `Tests/UI/test_mcp_tools_mode.py`, `Tests/UI/test_mcp_inspector.py`, `Tests/UI/test_mcp_workbench.py`, `Tests/MCP/test_permission_store.py` | updated in lockstep per task |

---

### Task 1: Synthesize `inputSchema` in the AST manifest (RAG-48 part 1)

**Files:**
- Modify: `tldw_chatbook/MCP/server.py:66-104` (`_extract_registered_entries`, `_describe_local_tools`)
- Test: `Tests/MCP/test_local_control_service.py:549-616` (mirror test), plus new cases in the same file or `Tests/MCP/test_server_manifest.py` if that file exists

**Interfaces:**
- Produces: every entry in `describe_local_mcp_capabilities()["tools"]` gains `"inputSchema": {"type": "object", "properties": {...}, "required": [...]}` synthesized from the tool function's signature. Simple params map to `{"type": "string"|"integer"|"number"|"boolean"}`; `Optional[T]` maps to `{"type": [T_json, "null"]}` (the type-list idiom `parse_schema` already understands); `List[str]`/`list[str]` maps to `{"type": "array", "items": {"type": "string"}}`; `Optional[List[str]]` maps to `{"type": ["array", "null"], "items": {"type": "string"}}`. Params with defaults get `"default": <literal>` (via `ast.literal_eval`, only when it succeeds) and are omitted from `required`. An annotation the mapper does not understand yields `{}` for that property (the form layer will then honestly fall back to raw for that tool).
- Consumes: nothing from other tasks.

**Context:** The Hub never instantiates FastMCP. Its manifest comes from `describe_local_mcp_capabilities()` (`server.py:122-130`) → `_describe_local_tools()` (`:118`) → `_extract_registered_entries("_register_tools", "tool")` (`:70-104`), which AST-parses server.py's own source and emits only `{"name", "description"}` (`:88-92`). This is deliberate — it must work when `MCP_AVAILABLE = False` (`:17-25`). Extend the AST walk; do NOT import `mcp`/FastMCP on this path.

- [ ] **Step 1: Read the real code.** Read `server.py:60-130` and the ten tool registrations at `server.py:286-448`. Enumerate every parameter annotation that actually appears (expect: `str`, `int`, `bool`, `Optional[str]`, `Optional[int]`, `Optional[List[str]]`, possibly `float`). The mapper must cover exactly these plus `list[...]` lowercase.

- [ ] **Step 2: Write failing tests.** In `Tests/MCP/test_local_control_service.py` (beside the mirror test at `:568-616`), add:

```python
def test_manifest_tools_carry_input_schema():
    manifest = describe_local_mcp_capabilities()
    tools = {t["name"]: t for t in manifest["tools"]}
    # search_rag: query is required str; media_types is Optional[List[str]]
    schema = tools["search_rag"]["inputSchema"]
    assert schema["type"] == "object"
    assert schema["properties"]["query"] == {"type": "string"}
    assert "query" in schema["required"]
    mt = schema["properties"]["media_types"]
    assert mt["type"] == ["array", "null"] and mt["items"] == {"type": "string"}
    assert "media_types" not in schema["required"]

def test_every_builtin_tool_has_object_schema():
    for tool in describe_local_mcp_capabilities()["tools"]:
        assert tool["inputSchema"]["type"] == "object", tool["name"]

def test_schema_defaults_recorded():
    tools = {t["name"]: t for t in describe_local_mcp_capabilities()["tools"]}
    # pick one real param with a literal default after reading server.py; e.g. search_rag limit
    prop = tools["search_rag"]["inputSchema"]["properties"]["limit"]
    assert prop.get("default") is not None
```

Adjust the exact param names to what Step 1 found — the assertions must pin REAL signatures, not invented ones.

- [ ] **Step 3: Run to verify they fail** (`KeyError: 'inputSchema'`).

Run: `pytest Tests/MCP/test_local_control_service.py -q`

- [ ] **Step 4: Implement.** In `server.py`, add a module-level helper and wire it into `_extract_registered_entries`:

```python
_AST_SIMPLE_TYPES = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}

def _annotation_to_property(node: ast.expr | None) -> dict:
    """Best-effort JSON-schema fragment for one annotation AST node.

    Returns {} for anything unrecognised so the form layer falls back to
    raw JSON for that tool instead of rendering a wrong field.
    """
    if isinstance(node, ast.Name) and node.id in _AST_SIMPLE_TYPES:
        return {"type": _AST_SIMPLE_TYPES[node.id]}
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        inner = node.slice
        if node.value.id == "Optional":
            base = _annotation_to_property(inner)
            if isinstance(base.get("type"), str):
                return {**base, "type": [base["type"], "null"]}
            return {}
        if node.value.id in ("List", "list"):
            items = _annotation_to_property(inner)
            if items:
                return {"type": "array", "items": items}
    return {}

def _signature_to_input_schema(fn: ast.AsyncFunctionDef | ast.FunctionDef) -> dict:
    properties: dict = {}
    required: list[str] = []
    args = fn.args.args
    defaults: list = fn.args.defaults
    first_default_index = len(args) - len(defaults)
    for index, arg in enumerate(args):
        if arg.arg in ("self", "cls"):
            continue
        prop = _annotation_to_property(arg.annotation)
        if index >= first_default_index:
            default_node = defaults[index - first_default_index]
            try:
                default_value = ast.literal_eval(default_node)
            except (ValueError, SyntaxError):
                default_value = None
            if default_value is not None and prop:
                prop = {**prop, "default": default_value}
        else:
            required.append(arg.arg)
        properties[arg.arg] = prop
    return {"type": "object", "properties": properties, "required": required}
```

In `_extract_registered_entries` (`:82-92`), where the entry dict is built, add `entry["inputSchema"] = _signature_to_input_schema(fn_node)`. Match the actual AST node variable name in the real code. Keyword-only args (`fn.args.kwonlyargs`/`kw_defaults`): if any tool uses them (check in Step 1), handle them with the same logic; if none do, add a comment stating they're intentionally unhandled.

- [ ] **Step 5: Update the mirror test.** `Tests/MCP/test_local_control_service.py:568-616` re-implements the extraction and asserts `helper_manifest["tools"] == expected_tools`. Update its expectation to include `inputSchema` (import and reuse `_signature_to_input_schema` in the mirror rather than duplicating the mapper — the mirror's job is pinning the *walk*, not the mapper). This IS a deliberate contract change; say so in the task report.

- [ ] **Step 6: Run the touched files' tests + collect sweep.**

Run: `pytest Tests/MCP/test_local_control_service.py -q && pytest Tests/ --collect-only -q | tail -3`

- [ ] **Step 7: Commit** — `feat(mcp): synthesize inputSchema in AST tool manifest`

---

### Task 2: Plumb schemas into the catalog without the permission rug-pull (RAG-48 part 2)

**Files:**
- Modify: `tldw_chatbook/MCP/permission_store.py:507-519` (`HASH_FREE_SERVER_KEYS`)
- Modify: `tldw_chatbook/MCP/hub_tool_catalog.py:121-122,149`
- Test: `Tests/MCP/test_permission_store.py` (the `HASH_FREE_SERVER_KEYS` pin at `:517-518`'s companion test), `Tests/MCP/test_hub_tool_catalog.py:50-55`

**Interfaces:**
- Consumes: Task 1's `inputSchema` key in the inventory's builtin tool dicts (flows through `local_control_service.get_inventory()` untouched — verify, don't assume).
- Produces: `builtin_tools_from_inventory()` returns `HubTool`s with real `input_schema`; permission resolution for `builtin:tldw_chatbook` no longer compares definition hashes, so pre-existing stored "allow" decisions survive the schema arrival.

**ORDER MATTERS:** commit the hash-free change FIRST (or in the same commit as the catalog flip, never after). If the catalog flip lands alone, every stored `allow` for a builtin downgrades to `ask` with "Definition changed since you allowed it." (`mcp_inspector.py:142`), because `resolve_effective_state()` recomputes `definition_hash(tool.description, tool.input_schema)` at `permission_store.py:643`.

- [ ] **Step 1: Write the failing permission test.** Beside the existing `HASH_FREE_SERVER_KEYS` pin (find it near `permission_store.py:517-518`'s referenced test — grep `HASH_FREE` in `Tests/MCP/`), add:

```python
def test_builtin_server_is_hash_free():
    assert "builtin:tldw_chatbook" in HASH_FREE_SERVER_KEYS

def test_builtin_allow_survives_schema_change(tmp_path):
    # Store an allow for a builtin tool with input_schema=None, then resolve
    # with a populated schema; effective state must remain allow with no
    # config_changed flag. Mirror the shape of the existing hash-mismatch
    # test in this file (find it via grep "config_changed") with
    # server_key="builtin:tldw_chatbook".
    ...
```

Write the second test fully by copying the existing hash-mismatch test's arrange/act and flipping the expectation — its exact helper names live in that file; do not invent new fixtures.

- [ ] **Step 2: Run to verify failure**, then add `"builtin:tldw_chatbook"` to `HASH_FREE_SERVER_KEYS` (`permission_store.py:519`) with a rationale comment mirroring the `agent:builtin` one at `:507-518`: built-in definitions live in this codebase and change only via app updates; the hash guard exists to catch remote servers silently redefining a tool, which cannot happen here.

- [ ] **Step 3: Flip the catalog.** `hub_tool_catalog.py:149`: replace `input_schema=None` with `input_schema=_normalized_schema(raw_tool.get("inputSchema"))` — the exact pattern of its two siblings (`local_tools_from_record` `:108`, `server_tools_from_inventory` `:211`). Update the stale comment at `:121-122`.

- [ ] **Step 4: Flip the catalog pin.** `Tests/MCP/test_hub_tool_catalog.py:50-55` pins `input_schema is None` for builtins by name — change it to assert the schema is a dict with `type == "object"`. Deliberate contract change; name it in the report.

- [ ] **Step 5: Run gates.**

Run: `pytest Tests/MCP/test_permission_store.py Tests/MCP/test_hub_tool_catalog.py Tests/MCP/test_permission_resolution.py -q && pytest Tests/ --collect-only -q | tail -3`

- [ ] **Step 6: Commit** — `feat(mcp): builtin tool schemas reach the catalog; builtin server key is hash-free`

---

### Task 3: Array support in the schema form (RAG-48 part 3)

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_schema_form.py:27,45-95` (`_SIMPLE_KINDS`, `_resolve_property`), plus the value-collection method (find where the form reads widget values back into an arguments dict)
- Test: `Tests/UI/test_mcp_schema_form.py`, `Tests/UI/test_mcp_tools_mode.py:67-124`

**Interfaces:**
- Consumes: Task 1's schema shapes: `{"type": "array", "items": {"type": "string"}}` and `{"type": ["array", "null"], "items": {"type": "string"}}` (also accept the `anyOf` Optional idiom since remote servers emit it).
- Produces: `parse_schema()` returns a renderable form spec for arrays of simple items; the form renders a single `Input` with placeholder `comma-separated` and parses `"a, b"` → `["a", "b"]`, `""` → omitted for optional / `[]` for required. With this, all 10 builtin tools render `form` in the Schema column (`mcp_tools_mode.py:383`); the two stragglers were `search_rag.media_types` and `ingest_media.tags`.

- [ ] **Step 1: Read `mcp_schema_form.py` fully** (~200 lines). Understand the property-spec dataclass/tuple `_resolve_property` returns, how `compose()` maps a spec to a widget, and how values are read back (there is a collect/values method — the exact name matters for the parse-back edit).

- [ ] **Step 2: Write failing tests** in `Tests/UI/test_mcp_schema_form.py` (module header says no stylesheet — keep it that way):

```python
ARRAY_SCHEMA = {
    "type": "object",
    "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    "required": ["tags"],
}
OPTIONAL_ARRAY_SCHEMA = {
    "type": "object",
    "properties": {"media_types": {"type": ["array", "null"], "items": {"type": "string"}}},
    "required": [],
}
ANYOF_ARRAY_SCHEMA = {
    "type": "object",
    "properties": {"media_types": {"anyOf": [
        {"type": "array", "items": {"type": "string"}}, {"type": "null"}]}},
    "required": [],
}

def test_parse_schema_accepts_array_of_strings():
    assert parse_schema(ARRAY_SCHEMA) is not None

def test_parse_schema_accepts_optional_array_both_idioms():
    assert parse_schema(OPTIONAL_ARRAY_SCHEMA) is not None
    assert parse_schema(ANYOF_ARRAY_SCHEMA) is not None

def test_parse_schema_still_rejects_array_of_objects():
    nested = {"type": "object", "properties": {
        "rows": {"type": "array", "items": {"type": "object"}}}, "required": []}
    assert parse_schema(nested) is None
```

Plus value-parse-back tests using the form's real collect method (write them after Step 1 reveals its name): `"a, b"` → `["a", "b"]`; `" a ,, b "` → `["a", "b"]` (strip empties); empty input on optional → key omitted; empty input on required array → `[]`.

- [ ] **Step 3: Run to verify failures.**

Run: `pytest Tests/UI/test_mcp_schema_form.py -q`

- [ ] **Step 4: Implement.** In `_resolve_property`: where the simple-kind resolution happens, add an `array` branch — renderable iff `items` resolves to a member of `_SIMPLE_KINDS` (reuse `_resolve_property` recursively on `items`, reject if the item spec is itself enum/array). Thread the Optional wrappers exactly as the existing simple-type code does (both `anyOf` and type-list forms). In the widget mapping, arrays render as `Input` with `placeholder="comma-separated"`. In value collection, split on `,`, strip, drop empties; cast items per item kind (`int()`/`float()`/bool parsing consistent with how the form already parses scalars). Follow the file's existing spec idiom — do not invent a parallel structure.

- [ ] **Step 5: Prove the end-to-end column.** In `Tests/UI/test_mcp_tools_mode.py` (`:67-124` prior art), add a case that builds the builtin catalog from the real manifest and asserts every builtin tool's schema cell is `"form"`:

```python
def test_all_builtin_tools_render_form_column():
    from tldw_chatbook.MCP.server import describe_local_mcp_capabilities
    from tldw_chatbook.UI.MCP_Modules.mcp_schema_form import parse_schema
    for tool in describe_local_mcp_capabilities()["tools"]:
        assert parse_schema(tool["inputSchema"]) is not None, tool["name"]
```

(If any tool honestly cannot render — Step 1 of Task 1 found an exotic annotation — assert the true count instead and name the exception; do not force it.)

- [ ] **Step 6: Run gates.**

Run: `pytest Tests/UI/test_mcp_schema_form.py Tests/UI/test_mcp_tools_mode.py -q && pytest Tests/ --collect-only -q | tail -3`

- [ ] **Step 7: Commit** — `feat(mcp): schema form renders arrays of simple items — all builtin tools get typed forms`

---

### Task 4: Tool-run result summary + collapsible raw JSON (RAG-49)

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py:3215-3280` (`_run_tool_test`, `_show_tool_test_result`)
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py:1541,1774-1834` (`show_tool_result` + the result region)
- Modify: CSS source for any new classes (likely `tldw_chatbook/css/components/_mcp_hub.tcss` or wherever `mcp-inspector-*` rules live — grep the bundle for `mcp-inspector-test-result` to find the source file), then regenerate the bundle
- Test: `Tests/UI/test_mcp_inspector.py:1575-1676`, `Tests/UI/test_mcp_workbench.py:3291-3320`

**Interfaces:**
- Consumes: the builtin envelope `{"source", "tool_name", "result", "governance"}` built at `local_control_service.py:598-603`; `format_duration_ms()` (`mcp_inspector.py:262-282`); `redact_mapping` (already used at `mcp_workbench.py:3253`).
- Produces: `show_tool_result(...)` accepts a structured result and renders: **line 1** status summary `OK · local · 981ms · 3 results` (segments joined with ` · `; `source` segment only when known; count segment only when the tool result is a list); **line 2 (conditional)** a quiet interpretation; **collapsible** "Raw response" (collapsed) holding pretty-printed redacted JSON, `markup=False`. Task 5 will add one more optional parameter (`decision_note`) to this same method — keep the signature keyword-only so that lands cleanly.

**Behavior contract (write tests to exactly this):**
- Result is a list of N ≥ 1 (not error-shaped): `OK · local · 981ms · 3 results` (singular `1 result`).
- Result is `[]`: `OK · local · 981ms · 0 results` + quiet line `The tool ran and returned no results.`
- Result is the error shape `[{"error": "..."}]` (exactly one element, a mapping whose only key is `"error"` — the `MCP/tools.py:326` contract): `OK · local · 981ms · tool returned an error` + the error string as the interpretation line.
- Result is a non-list (dict/str/number): no count segment — `OK · local · 981ms`.
- Failure/blocked paths keep their existing rendering (`Failed · …`, `Blocked · not run`) and the blocked jump-button behavior (`test_mcp_inspector.py:2382-2421` pins it).
- Raw body: `json.dumps(redact_mapping(envelope), indent=2, default=str)`, capped at 20 000 chars with a trailing `… truncated (showing 20000 of N chars)` when over; the 500-char cap at `mcp_workbench.py:3256` is retired for this path. Deliberate contract change to the pins at `test_mcp_inspector.py:1575-1595` and `test_mcp_workbench.py:3291-3320` — update them and say so.
- Stale-result drop by `(server_key, tool_name)` (`test_mcp_inspector.py:1599-1676`) must keep passing unchanged.

- [ ] **Step 1: Read the real seam.** `mcp_workbench.py:3215-3280` and `mcp_inspector.py:1500-1560,1774-1834`. Note the current `show_tool_result` signature and the exact widget ids. The plan's target shape is:

```python
def show_tool_result(
    self, *, server_key: str, tool_name: str, ok: bool,
    duration_ms: float | None = None, result: object = None,
    source: str | None = None, error: str | None = None,
    blocked: bool = False,
) -> None:
```

Adapt the parameter set to preserve whatever the current signature carries for the blocked/failed paths; keep it keyword-only.

- [ ] **Step 2: Write failing tests** in `Tests/UI/test_mcp_inspector.py` beside `:1575` — one per bullet of the behavior contract above, using the panel harness the existing tests at `:1500-1676` use. Assert the summary line via the result widget's content and the Collapsible's presence + `collapsed=True`. Also a summary-unit test for the error-shape detector:

```python
def test_error_shape_detection():
    assert _is_tool_error_shape([{"error": "boom"}])
    assert not _is_tool_error_shape([])
    assert not _is_tool_error_shape([{"error": "x", "id": 1}])
    assert not _is_tool_error_shape([{"id": 1}, {"error": "x"}])
    assert not _is_tool_error_shape({"error": "x"})
```

- [ ] **Step 3: Run to verify failures.**

Run: `pytest Tests/UI/test_mcp_inspector.py -q -k "tool_result or error_shape"`

- [ ] **Step 4: Implement.**
  - `mcp_workbench.py::_run_tool_test`: stop flattening to a 500-char string; carry the envelope (already redact-safe via `redact_mapping` at render time) through `_show_tool_test_result` to `show_tool_result(result=envelope.get("result"), source=envelope.get("source"), ...)` for mappings, falling back to the current stringification for non-mapping results. Keep the formatting-failure containment the `test_mcp_workbench.py:3291` pin exercises — a result that cannot be dumped must still produce a rendered failure line, not an exception.
  - `mcp_inspector.py::show_tool_result`: build the summary via a small pure helper (unit-testable):

```python
def _summarize_tool_result(*, ok: bool, duration_ms, source, result) -> tuple[str, str | None]:
    """Returns (status_line, interpretation_or_None)."""
    segments = ["OK" if ok else "Failed"]
    if source:
        segments.append(str(source))
    if duration_ms is not None:
        segments.append(format_duration_ms(duration_ms))
    interpretation = None
    if isinstance(result, list):
        if _is_tool_error_shape(result):
            segments.append("tool returned an error")
            interpretation = str(result[0]["error"])
        elif not result:
            segments.append("0 results")
            interpretation = "The tool ran and returned no results."
        else:
            count = len(result)
            segments.append(f"{count} result" + ("s" if count != 1 else ""))
    return " · ".join(segments), interpretation
```

  - Render: summary into the existing `#mcp-inspector-test-result` Static (`markup=False`); interpretation into a sibling Static with a quiet class (reuse an existing quiet class if the Hub has one — grep for how `_EMPTY_STATE_COPY`'s badge is styled — else add `.mcp-inspector-result-note` with a dim color in the CSS source); raw body inside a `Collapsible(title="Raw response", collapsed=True)` mounted in the result region. `Collapsible` is already imported (`mcp_inspector.py:18`); respect the height caveat noted at `:492-497`. Mount the Collapsible once in `compose` with `display=False` and fill/show it in `show_tool_result` (always-mounted + display-toggle — the program's PR-2 lesson; never conditionally compose it).
  - Style every new class in CSS source and regenerate the bundle (`python3 -m tldw_chatbook.css.build_css`), or use `DEFAULT_CSS` — Task 8's guardrail will fail the branch otherwise.

- [ ] **Step 5: Update the deliberately-changed pins** (`test_mcp_inspector.py:1575-1595` `OK · 123ms` prefix + `'{"ok": true}'` containment; `test_mcp_workbench.py:3291-3320` formatting-failure path). Name each in the task report.

- [ ] **Step 6: Run gates.**

Run: `pytest Tests/UI/test_mcp_inspector.py Tests/UI/test_mcp_workbench.py -q && pytest Tests/ --collect-only -q | tail -3`

- [ ] **Step 7: Commit** — `feat(mcp): tool-run summary line + collapsible raw response`

---

### Task 5: Name the permission decision — in the result and in the audit log (RAG-51)

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py:3127-3280` (carry gate + armed fact through the worker)
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py` (`show_tool_result` decision note; `_ORIGIN_SENTENCES` reuse)
- Modify: `tldw_chatbook/MCP/unified_control_plane_service.py:2274-2312` (`test_hub_tool` decision pass-through), `:2132-2268` (`execute_hub_tool`) if the decision funnels there
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_audit_mode.py:251-254` (render the new decision value)
- Test: `Tests/UI/test_mcp_inspector.py`, `Tests/UI/test_mcp_workbench.py`, `Tests/MCP/test_control_plane_tool_execute.py`

**Interfaces:**
- Consumes: Task 4's keyword-only `show_tool_result`; `EffectiveToolState` (`permission_store.py:545-582`, `.ui_label` → `"Allow"|"Ask"|"Off"`, `.origin`, `.config_changed`, `.risk_floored`); the ask-armed fact `inspector.test_run_armed` (`mcp_inspector.py:1643`) read at `mcp_workbench.py:3186` before `disarm_test_run()` (`:3194`).
- Produces: `show_tool_result(..., decision_note: str | None = None)`; the execution log records `"ask-approved"` (or the file's real vocabulary — see Step 3) instead of a hardcoded `"allowed"` for ask-gated test runs.

**Context:** `on_mcp_inspector_tool_test_requested()` (`mcp_workbench.py:3175`) computes `gate = self._resolve_test_gate(...)`, branches on it (`:3177-3193`), then discards it — `_run_tool_test(server_key, tool_name, arguments)` (`:3209-3213`) carries neither the gate nor the just-approved fact. Both are known synchronously at dispatch. Separately, `test_hub_tool()` hardcodes `decision="allowed"` (`unified_control_plane_service.py:2310`), so ask-then-approved runs are logged as plain `allowed` in the execution log (`MCP/execution_log.py:91,122`) and shown that way in Audit mode.

**Decision-note copy (quiet register, one sentence, reuse `_ORIGIN_SENTENCES` for the origin clause):**
- Allow gate: `Ran because this tool is set to Allow. <origin sentence>`
- Ask gate, armed approval: `Ran because you approved this run (the tool is set to Ask).`
- Off gate: unchanged existing `Blocked · not run` path — add note `This tool is set to Off. <origin sentence>`.

- [ ] **Step 1: Write failing tests.**
  - Inspector: `show_tool_result(..., decision_note="Ran because you approved this run (the tool is set to Ask).")` renders the sentence in the result region (`markup=False` — the origin sentence is trusted copy but keep the discipline).
  - Workbench: dispatching a test run with an Ask gate + armed approval reaches `show_tool_result` with the ask-approved note; an Allow gate reaches it with the allow note. Follow the harness the existing dispatch tests use (grep `test_run_armed` in `Tests/UI/test_mcp_workbench.py`).
  - Control plane: in `Tests/MCP/test_control_plane_tool_execute.py`, a `test_hub_tool` call with an ask-approved decision records that decision in the execution log entry, not `"allowed"`.

- [ ] **Step 2: Run to verify failures.**

Run: `pytest Tests/UI/test_mcp_inspector.py Tests/UI/test_mcp_workbench.py Tests/MCP/test_control_plane_tool_execute.py -q -k decision`

- [ ] **Step 3: Implement.**
  - Read `MCP/execution_log.py:80-130` first: learn the decision vocabulary. If decisions are free strings, record `"ask-approved"`; if an enum/validated set, extend it minimally. Then `unified_control_plane_service.py::test_hub_tool` gains a `decision: str = "allowed"` parameter (or threads it to `execute_hub_tool`'s recording — follow where `decision="allowed"` at `:2310` actually lands) and the workbench passes the real value.
  - `mcp_audit_mode.py:251-254`: add the rendering for the new value — plain words, e.g. `asked · you approved`. Match the existing rendering idiom at that site exactly.
  - `mcp_workbench.py`: capture `gate` and `armed = <the :3186 read>` before `disarm_test_run()`, pass through `_run_tool_test(..., gate=gate, ask_approved=armed)` → `_show_tool_test_result` → `show_tool_result(decision_note=...)`, and pass the decision string into the service call. Build the note with a pure helper so it unit-tests without the UI:

```python
def _decision_note(gate, ask_approved: bool) -> str | None:
    if gate is None:
        return None
    origin = _ORIGIN_SENTENCES.get(gate.origin, "")
    if gate.ui_label == "Ask" and ask_approved:
        return "Ran because you approved this run (the tool is set to Ask)."
    if gate.ui_label == "Allow":
        return f"Ran because this tool is set to Allow. {origin}".strip()
    if gate.ui_label == "Off":
        return f"This tool is set to Off. {origin}".strip()
    return None
```

(Confirm `_ORIGIN_SENTENCES` keys match `gate.origin` values — read both before wiring.)

- [ ] **Step 4: Run gates.**

Run: `pytest Tests/UI/test_mcp_inspector.py Tests/UI/test_mcp_workbench.py Tests/MCP/test_control_plane_tool_execute.py -q && pytest Tests/ --collect-only -q | tail -3`

- [ ] **Step 5: Commit** — `feat(mcp): name the permission decision in the run result and audit log`

---

### Task 6: Retire the stale empty-state badge over populated detail (RAG-50)

**Files:**
- Modify: `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py:1023-1055` (`update_readiness`), `:1118-1237` (`show_tool`), and whatever clear/blank method hides the detail containers
- Test: `Tests/UI/test_mcp_inspector.py`, `Tests/UI/test_mcp_workbench.py:258`

**Interfaces:** self-contained; no other task consumes this.

**Context:** `#mcp-inspector-state` is composed once seeded with `_EMPTY_STATE_COPY` (`:757-758`) and written only by `update_readiness()` (`:1023-1055`), whose only caller is `MCPWorkbench._sync_children()` (`mcp_workbench.py:1192`) fed by the selected SERVER. The tool populate path `show_tool()` never touches it — so in Tools mode with no server selected, the empty-state badge sits above fully populated tool detail. (The copy itself was already fixed by F-054; the staleness is structural.)

- [ ] **Step 1: Write failing tests.**

```python
def test_empty_state_badge_hidden_when_tool_shown():
    # harness: mount inspector, call show_tool(tool, effective=...) with no
    # prior update_readiness call; assert query_one("#mcp-inspector-state").display is False

def test_empty_state_badge_returns_when_detail_cleared():
    # show_tool(...) then the clear path (find the method the workbench calls
    # when selection empties — grep show_tool callers and the blank/clear sibling);
    # assert badge display True and content == _EMPTY_STATE_COPY
```

Use the mounting harness the existing inspector tests use; fill in the real clear-method name after reading `:1118-1237` and its callers.

- [ ] **Step 2: Run to verify failure.**

Run: `pytest Tests/UI/test_mcp_inspector.py -q -k empty_state_badge`

- [ ] **Step 3: Implement** at the `show_tool()` container-display seam (`:1152-1157`): when any detail container becomes displayed, set the badge's `display = False`; in the clear path, restore `display = True` (content is still maintained by `update_readiness`). Check `update_readiness()` doesn't force it back visible while a tool is displayed — if it writes unconditionally, guard it with "a detail container is currently displayed". Keep the fix inside the inspector; the workbench should not learn new responsibilities.

- [ ] **Step 4: Run gates** (`test_mcp_workbench.py:258` pins badge content — must keep passing).

Run: `pytest Tests/UI/test_mcp_inspector.py Tests/UI/test_mcp_workbench.py -q && pytest Tests/ --collect-only -q | tail -3`

- [ ] **Step 5: Commit** — `fix(mcp): hide inspector empty-state badge while detail is shown`

---

### Task 7: Multi-instance detection + non-blocking boot warning (RAG-53)

**Files:**
- Create: `tldw_chatbook/Utils/instance_lock.py`
- Modify: `tldw_chatbook/app.py` (acquire near boot; `_maybe_warn_second_instance` called from `_push_initial_screen` tail beside `:7630`)
- Test: `Tests/Utils/test_instance_lock.py` (new), `Tests/UI/test_app_instance_warning.py` (new, or fold into an existing app-boot test file if one collects cheaply)

**Interfaces:**
- Consumes: `config.get_user_data_dir()` (`config.py:5256-5273`) — the one directory every bleeding store lives under; portalocker prior art at `Model_Artifacts/leases.py:216-265`; the boot-notice channel prior art at `app.py:7500-7519` (`_maybe_offer_first_run_wizard`).
- Produces: `acquire_profile_instance_lock(user_data_dir: Path) -> InstanceLockStatus` and an app-lifetime handle. **NEVER blocks, NEVER raises, NEVER prevents boot** — the owner runs concurrent instances on purpose; this is a warning, not a lock-out.

- [ ] **Step 1: Write failing unit tests** in `Tests/Utils/test_instance_lock.py`:

```python
from pathlib import Path
from tldw_chatbook.Utils.instance_lock import acquire_profile_instance_lock

def test_first_acquire_succeeds(tmp_path):
    status = acquire_profile_instance_lock(tmp_path)
    assert status.acquired is True
    assert status.handle is not None
    status.handle.close()

def test_second_acquire_reports_holder(tmp_path):
    first = acquire_profile_instance_lock(tmp_path)
    second = acquire_profile_instance_lock(tmp_path)
    assert second.acquired is False
    assert second.holder_pid == first.written_pid  # our own pid
    assert second.handle is None
    first.handle.close()

def test_reacquire_after_release(tmp_path):
    first = acquire_profile_instance_lock(tmp_path)
    first.handle.close()
    second = acquire_profile_instance_lock(tmp_path)
    assert second.acquired is True
    second.handle.close()

def test_unwritable_dir_never_raises(tmp_path):
    target = tmp_path / "nope"
    target.mkdir()
    target.chmod(0o400)
    try:
        status = acquire_profile_instance_lock(target)
        assert status.acquired is True  # unknown → quiet, never a false warning
    finally:
        target.chmod(0o700)
```

`flock`-style exclusive locks conflict between two file descriptors in the SAME process on macOS/Linux, so `test_second_acquire_reports_holder` works in-process — but verify on this machine; if portalocker's mechanism doesn't conflict intra-process here, switch that test to a `multiprocessing` child.

- [ ] **Step 2: Run to verify failure** (`ModuleNotFoundError`).

Run: `pytest Tests/Utils/test_instance_lock.py -q`

- [ ] **Step 3: Implement `Utils/instance_lock.py`** — copy the leases.py prior-art shape:

```python
"""Advisory per-profile instance lock.

Detection only — a second instance gets a warning toast, never a lock-out
(the owner runs concurrent instances deliberately; permission/settings
stores are last-write-wins by design). The OS lock, not the file's
existence, is the liveness signal: locks vanish with the process, so stale
files never false-positive. The lock file is deliberately never unlinked —
unlinking races a third instance onto a fresh inode and splits the lock.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO
import portalocker
from portalocker import LockFlags
from loguru import logger

_LOCK_FILENAME = ".instance.lock"

@dataclass
class InstanceLockStatus:
    acquired: bool
    handle: BinaryIO | None = None
    written_pid: int | None = None
    holder_pid: int | None = None
    holder_since: str | None = None

def acquire_profile_instance_lock(user_data_dir: Path) -> InstanceLockStatus:
    lock_path = Path(user_data_dir) / _LOCK_FILENAME
    try:
        handle = lock_path.open("a+b")
    except OSError as exc:
        logger.debug("instance lock unavailable ({}): {}", type(exc).__name__, exc)
        return InstanceLockStatus(acquired=True)
    try:
        portalocker.lock(handle, LockFlags.EXCLUSIVE | LockFlags.NON_BLOCKING)
    except portalocker.exceptions.BaseLockException:
        holder_pid, holder_since = _read_holder(lock_path)
        handle.close()
        return InstanceLockStatus(
            acquired=False, holder_pid=holder_pid, holder_since=holder_since)
    except Exception as exc:
        logger.debug("instance lock error ({}): {}", type(exc).__name__, exc)
        handle.close()
        return InstanceLockStatus(acquired=True)
    pid = os.getpid()
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"{pid}\n{datetime.now(timezone.utc).isoformat()}\n".encode("utf-8"))
        handle.flush()
    except OSError:
        pass  # body is informational; the lock itself is the signal
    return InstanceLockStatus(acquired=True, handle=handle, written_pid=pid)

def _read_holder(lock_path: Path) -> tuple[int | None, str | None]:
    try:
        lines = lock_path.read_text(encoding="utf-8", errors="replace").splitlines()
        pid = int(lines[0]) if lines and lines[0].strip().isdigit() else None
        since = lines[1].strip() if len(lines) > 1 else None
        return pid, since
    except OSError:
        return None, None
```

Check `Model_Artifacts/leases.py:216-265` for whether this codebase catches `AlreadyLocked` specifically — mirror its exception choice (`BaseLockException` covers `AlreadyLocked`; confirm the class exists in portalocker 3.2.0 before relying on it).

- [ ] **Step 4: Run unit tests to green.**

Run: `pytest Tests/Utils/test_instance_lock.py -q`

- [ ] **Step 5: Wire into app boot.** In `app.py`:
  - Acquire once where the profile is known and boot work happens (find where other per-profile startup happens near `get_user_data_dir()` usage — likely in `__init__` or early `on_mount`; pick the earliest point AFTER the data dir is final): `self._instance_lock_status = acquire_profile_instance_lock(get_user_data_dir())`, in a `try/except Exception` that defaults to `InstanceLockStatus(acquired=True)`. Keep the status (and thus the open handle) referenced for app lifetime.
  - Add the warning method:

```python
def _maybe_warn_second_instance(self) -> None:
    status = getattr(self, "_instance_lock_status", None)
    if status is None or status.acquired:
        return
    detail = ""
    if status.holder_pid:
        detail = f" (pid {status.holder_pid})"
    self.notify(
        "Another copy of tldw is already using this profile"
        f"{detail}. Everything keeps working, but the last instance to "
        "change settings or permissions wins, and a restart sweep may mark "
        "the other instance's running jobs as interrupted.",
        title="Profile already open",
        severity="warning",
        timeout=10,
    )
```

  - Call it at the tail of `_push_initial_screen()` as a sibling to the `:7630` wizard call, wrapped in `try/except Exception: pass` exactly like `:7518-7519`.

- [ ] **Step 6: Write the wiring tests** (`Tests/UI/test_app_instance_warning.py`): unit-style on a stub object exercising `_maybe_warn_second_instance` via `TldwCli._maybe_warn_second_instance(stub)` — not-acquired → `notify` called once with `severity="warning"`; acquired → not called; attribute absent → not called. Plus a source-level pin that `_push_initial_screen`'s body contains the call (read `app.py` source with `inspect.getsource`, assert `"_maybe_warn_second_instance"` appears) so the wiring can't silently drop — mirror how `Tests/UI/test_first_run_wizard_live_contract.py:131-152` pins its wiring if that pattern is cheaper.

- [ ] **Step 7: Run gates.**

Run: `pytest Tests/Utils/test_instance_lock.py Tests/UI/test_app_instance_warning.py -q && pytest Tests/ --collect-only -q | tail -3`

- [ ] **Step 8: Commit** — `feat(app): warn (never block) when a second instance shares the profile`

---

### Task 8: CSS class-coverage guardrail (RAG-55)

**Files:**
- Create: `Tests/UI/test_css_class_coverage_contract.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:14314,14325` (remove PR-4's two inert tokens — conditions below)
- Test: the new file is the test.

**Interfaces:** consumes bundle helpers from `Tests/UI/test_non_obscuring_focus_contract.py:25,94-138` (`BUNDLE`, `css_selectors`, `css_selectors_contain_class`, `css_blocks`). Runs LAST so it covers the branch's final state — Tasks 4/5's new classes must already be styled.

**Scope (the narrowest honest version, per scout measurement):** `tldw_chatbook/UI/MCP_Modules/**`, `tldw_chatbook/Widgets/Console/**`, `tldw_chatbook/UI/Screens/chat_screen.py`, `tldw_chatbook/UI/Screens/library_screen.py`. Literal `classes="…"` double-quoted tokens only — f-strings and `add_class(var)` are documented out of scope. A token counts as styled if it has a `.token` rule in the built bundle, a `.token` rule in any `DEFAULT_CSS`/`CSS` string of the scoped files, **or** a `#token` id rule in the bundle (e.g. `console-staged-context-empty` is styled via `#console-staged-context-empty` at `tldw_cli_modular.tcss:5328`). Everything else must be on the frozen `KNOWN_UNSTYLED` registry (~26 entries at scout count) with a one-line reason each. **Verified non-facts to state in the docstring:** there is no `is-*` style-free convention in this repo (those classes are styled and pinned by `test_master_shell_design_system_contract.py:21-34`), and none of the currently-unstyled tokens are query-selector markers (cross-checked against every `query*(".x")`/`has_class("x")` literal).

- [ ] **Step 1: Handle PR-4's own inert tokens first.** `chat_screen.py:14314` composes `classes="destination-section console-library-rag-scope"` and `:14325` similar with `console-library-rag-run` — both second tokens are inert (no rule anywhere, not queried). Grep `Tests/` for each literal; if unpinned, delete the token from the compose call; if pinned, keep it and put it on the registry with reason `pinned by <test>` instead. Commit separately: `chore(console): drop inert class tokens from PR-4 compose calls`.

- [ ] **Step 2: Write the contract test.**

```python
"""Every literal classes= token in the RAG/Console/Library/MCP surface must
be styled, or explicitly registered as known-unstyled.

Bug class this guards (RAG-55): a widget composed with classes that no
stylesheet rule ever touches ships invisible or default-styled — PR-2
found a zero-height toggle with green behavioral tests; PR-4's own
chat_screen compose calls carried two inert tokens. Exhibits:
`mcp-optin` (mcp_servers_mode.py), `console-library-rag-scope`.

Scope: literal double-quoted classes="..." only; f-strings and
add_class(variable) are out of scope by design. `#token` id rules count
as styled (console-staged-context-empty). There is no style-free `is-*`
convention here (test_master_shell_design_system_contract pins those),
and no scoped unstyled token is a query-selector marker (verified
2026-08-04).
"""
import re
from pathlib import Path

from Tests.UI.test_non_obscuring_focus_contract import (
    BUNDLE, css_selectors, css_selectors_contain_class,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "tldw_chatbook"
SCOPES = [
    PACKAGE / "UI" / "MCP_Modules",
    PACKAGE / "Widgets" / "Console",
    PACKAGE / "UI" / "Screens" / "chat_screen.py",
    PACKAGE / "UI" / "Screens" / "library_screen.py",
]
CLASSES_ATTR = re.compile(r'classes="([^"{}]+)"')
DEFAULT_CSS_BLOCK = re.compile(
    r'(?:DEFAULT_CSS|CSS)\s*(?::\s*\w+\s*)?=\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', re.DOTALL)

KNOWN_UNSTYLED: dict[str, str] = {
    # token: one-line reason it is allowed to have no rule.
    # Frozen registry — additions require an explicit edit here, which is
    # the point. Populate from the initial run's failure output, one
    # reviewed line per token, e.g.:
    # "mcp-optin": "legacy marker on servers-mode opt-in row; styling TBD (task-XXXX)",
}

def _scoped_files():
    for scope in SCOPES:
        if scope.is_file():
            yield scope
        else:
            yield from sorted(scope.rglob("*.py"))

def _composed_tokens():
    tokens = {}
    for path in _scoped_files():
        text = path.read_text(encoding="utf-8")
        for match in CLASSES_ATTR.finditer(text):
            for token in match.group(1).split():
                tokens.setdefault(token, path.relative_to(ROOT))
    return tokens

def _styled_tokens():
    bundle_text = BUNDLE.read_text(encoding="utf-8")
    selectors = css_selectors(bundle_text)
    for path in _scoped_files():
        for block in DEFAULT_CSS_BLOCK.finditer(path.read_text(encoding="utf-8")):
            selectors.extend(css_selectors(block.group(1)))
    return selectors

def test_every_composed_class_is_styled_or_registered():
    selectors = _styled_tokens()
    missing = []
    for token, path in sorted(_composed_tokens().items()):
        if token in KNOWN_UNSTYLED:
            continue
        if css_selectors_contain_class(selectors, f".{token}"):
            continue
        if any(re.search(rf"#{re.escape(token)}(?![\w-])", s) for s in selectors):
            continue
        missing.append(f"{token}  (first composed in {path})")
    assert not missing, (
        "Composed class tokens with no .class or #id rule in the bundle or "
        "any DEFAULT_CSS, and not on KNOWN_UNSTYLED:\n  " + "\n  ".join(missing))

def test_registry_entries_are_still_unstyled():
    """A registry entry whose token gained a rule is stale — remove it."""
    selectors = _styled_tokens()
    stale = [t for t in KNOWN_UNSTYLED
             if css_selectors_contain_class(selectors, f".{t}")]
    assert not stale, f"KNOWN_UNSTYLED entries now styled — delete them: {stale}"

def test_registry_entries_are_still_composed():
    """A registry entry no one composes anymore is dead weight — remove it."""
    composed = _composed_tokens()
    dead = [t for t in KNOWN_UNSTYLED if t not in composed]
    assert not dead, f"KNOWN_UNSTYLED entries no longer composed — delete them: {dead}"
```

First verify the cross-test import works (`Tests` packaging varies): `pytest Tests/UI/test_css_class_coverage_contract.py --collect-only`. If the import fails, copy the three helpers (~35 lines) with a header comment naming their source of truth.

- [ ] **Step 3: Run it, populate the registry.** The first run fails listing every unstyled token (~26 expected). For each: (a) if it's obviously a bug in code this program owns, style or delete it; (b) otherwise add it to `KNOWN_UNSTYLED` with an honest one-line reason. Do NOT bulk-style pre-existing tokens — that's scope creep; the registry documents them.

- [ ] **Step 4: Run gates.**

Run: `pytest Tests/UI/test_css_class_coverage_contract.py Tests/UI/test_non_obscuring_focus_contract.py -q && pytest Tests/ --collect-only -q | tail -3`

- [ ] **Step 5: Commit** — `test(css): class-coverage contract for RAG/Console/Library/MCP surfaces (RAG-55)`

---

### Task 9: Live verification, docs, ship

**Files:**
- Modify: `Docs/User_Guide/` MCP page(s) — grep for the hub/tools doc; stamp with the live-check commit per program practice
- No new code except fixes the live check or whole-branch review demands.

- [ ] **Step 1: Whole-branch review** (strongest model, per SDD): full `git diff b5a8fcec5..HEAD`, hunting between-task composition defects. Known watch-items: Task 4/5 both reshape `show_tool_result` — verify the final signature is coherent and every caller updated; Task 2's hash-free change actually covers the key the builtin tools resolve under (verify the literal server key string end-to-end); Task 8's guardrail passes on the FINAL branch state.
- [ ] **Step 2: Single fix wave** for review findings; scoped re-review of fixed files only.
- [ ] **Step 3: Live check** (scratch profile, proven recipe: copy `tldw_chatbook_ChaChaNotes.db` + `tldw_chatbook_media_v2.db` + `chromadb/` BEFORE first launch; scratch config with `[general] users_name` + `[first_run] setup_started/setup_completed = true`; session-suffixed tmux socket; click columns via python char-index, never byte offsets):
  1. MCP Hub → Tools mode: builtin tools show `form` in the Schema column; open Test Tool on `search_rag` → typed fields incl. comma-separated media_types; no stale empty-state badge above the detail (RAG-48, 50).
  2. Run `search_rag` with a query that matches nothing → `OK · local · <duration> · 0 results` + quiet interpretation + collapsed Raw response that expands (RAG-49).
  3. With the tool set to Ask: approve → result shows "Ran because you approved this run…"; Audit mode shows the ask-approved decision (RAG-51).
  4. Second app instance on the SAME scratch profile in another tmux pane → warning toast appears on the second instance; both instances stay fully functional (RAG-53).
  5. Permission continuity: a stored Allow from BEFORE this branch (set one on dev first, or seed the store) still shows Allow after boot on this branch — the rug-pull guard held (Task 2).
- [ ] **Step 4: Docs** — update the MCP user-guide page for typed forms + result summary + decision note, and wherever multi-instance behavior is documented (the AgentRuns/library restart-sweep docs mention the accepted race; link the new warning); stamp with the verification commit hash + date.
- [ ] **Step 5: Ship** — regenerate CSS bundle if dev moved it (never hand-merge); merge latest `origin/dev` into the branch; targeted gates once more; fresh cross-worktree backlog ID scan for any follow-ups filed; push; PR titled `RAG UX v2 PR-5: MCP typed forms, honest run results, instance warning, CSS guardrail (RAG-48…51, 53, 55)`; merge on green per the standing "merge when verified" authorization; confirm merge landed (`gh pr view --json merged` — exit 1 on `--delete-branch` cleanup is NOT a failure signal).

---

## Self-Review Notes

- **Spec coverage:** RAG-48 → Tasks 1-3; RAG-49 → Task 4; RAG-50 → Task 6; RAG-51 → Task 5; RAG-53 → Task 7; RAG-55 → Task 8. RAG-52/54 were closed in earlier PRs (see the program critique file); no other open items remain.
- **Type consistency:** `show_tool_result` keyword-only signature defined in Task 4 and extended (one optional param) in Task 5; `InstanceLockStatus` fields used by app.py match the Task 7 dataclass; `_signature_to_input_schema` name consistent between Task 1 impl and its mirror-test reuse.
- **Known unknowns delegated with instructions, not placeholders:** the form value-collection method name (Task 3 Step 1), execution-log decision vocabulary (Task 5 Step 3), the inspector clear-path method (Task 6 Step 1), portalocker intra-process conflict semantics (Task 7 Step 1) — each has a concrete read-first instruction and a fallback.
