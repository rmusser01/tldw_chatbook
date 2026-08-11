# MCP-Unified Standalone Server Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Chatbook's broken FastMCP standalone server with the exact public `mcp-unified==0.2.1` stdio runtime while preserving its intended tools, resources, prompts, client compatibility, permission boundary, and private in-process Library surface.

**Architecture:** Add one focused `ChatbookGatewayRuntime` adapter implementing the public `GatewayCoreRuntime` contract and retaining the existing decorator-shaped registrations in `MCP/server.py`. The adapter owns strict registration, canonical application-return mapping, bounded resource continuation, prompt validation, and local-tool error projection; `mcp-unified` continues to own revision negotiation, schema/result validation, JSON-RPC projection, pagination, and stdio lifecycle. The existing in-process Library runtime remains separate and is never used to assemble the standalone catalog.

**Tech Stack:** Python 3.11+, `mcp-unified==0.2.1`, asyncio binary stdio, SQLite/FTS-backed Chatbook services, pytest/pytest-asyncio/Hypothesis, Ruff, mypy, Bandit, `build`, and isolated wheel/sdist virtual environments.

**Post-rebase inventory correction (approved 2026-08-10):** TASK-4000 retired
the dishonest `ingest_media` placeholder. Preserve exactly nine implemented
standalone built-ins, keep `ingest_media` absent from discovery and refused by
direct dispatch, and direct persistent ingestion to Library Import. Do not
restore the placeholder or expand TASK-2512 into real ingestion.

---

## Governing documents and working rules

- Specification: `Docs/superpowers/specs/2026-08-09-mcp-unified-standalone-server-migration-design.md`
- ADR: `backlog/decisions/053-mcp-unified-standalone-runtime-boundary.md`
- Task: `backlog/tasks/task-2512 - Migrate-MCP-server-from-FastMCP-to-tldw_servers-mcp-unified-package.md`
- Existing Library boundary: `backlog/decisions/030-local-library-agent-tool-boundary.md`
- Existing local-tool permission/process boundaries: `backlog/decisions/032-local-agent-tool-permission-boundary.md` and `backlog/decisions/033-local-agent-process-execution-boundary.md`
- Required implementation method: `@superpowers:test-driven-development` for every behavior change and `@superpowers:verification-before-completion` before every completion claim.
- Use the absolute repository interpreter, `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`, for source tests. Do not use bare `python3`.
- For every ad-hoc app/process probe, set `TLDW_TEST_MODE=1`, a temporary `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TMPDIR`, and `TLDW_CONFIG_PATH` before importing Chatbook.
- Never point a test or probe at the user's configured databases, permission store, config, workspace, or provider credentials.
- Before implementation and again before integration, repeat the TASK-2512 branch/PR search documented in `backlog/docs/lessons-backlog-hygiene.md`.
- If a broad test command fails, run the identical command on a clean `origin/dev` worktree and compare exact failing node IDs, not counts.

## File responsibility map

### New files

- `tldw_chatbook/MCP/gateway_runtime.py` — the only Chatbook-owned `GatewayCoreRuntime` adapter: registrations, dispatch, local error mapping, URI-template routing, resource continuation, and prompt mapping.
- `Tests/MCP/test_mcp_unified_public_contract.py` — exact 0.2.1 public imports/signatures and optional-dependency metadata.
- `Tests/MCP/test_gateway_runtime_tools.py` — built-in and local tool registration/dispatch/error behavior.
- `Tests/MCP/test_gateway_runtime_resources.py` — resource catalog, URI routing, canonical result, metadata, and continuation behavior.
- `Tests/MCP/test_gateway_runtime_prompts.py` — prompt descriptors, argument coercion, role folding, and invalid result behavior.
- `Tests/MCP/test_mcp_unified_stdio.py` — real multi-revision in-memory/process stdio coverage, entrypoint behavior, and private-Library exclusion.
- `Tests/MCP/test_client_catalog_pagination.py` — bounded catalog aggregation and resource metadata preservation.
- `Tests/MCP/test_mcp_documentation_contract.py` — install, protocol, inventory, continuation, privacy, and retired-FastMCP documentation assertions.
- `Tests/Packaging/test_mcp_unified_distribution.py` — independent wheel/sdist `[mcp]` installation and site-packages-only protocol smoke.

### Existing production files

- `tldw_chatbook/MCP/server.py` — retain the nine implemented nested handlers and AST authority; replace FastMCP construction/binding with `ChatbookGatewayRuntime`, strengthen schemas, compose optional local tools, and call `serve_stdio`.
- `tldw_chatbook/MCP/local_server_tools.py` — retain external-client provider composition, return canonical `ToolResult` objects to the adapter, remove FastMCP-only schema/copy workarounds, and stop interpolating raw exceptions.
- `tldw_chatbook/MCP/prompts.py` — add the missing `await` in `search_and_synthesize_prompt`; no external role folding here.
- `tldw_chatbook/MCP/client.py` — bounded cursor aggregation and exact resource `_meta` preservation.
- `tldw_chatbook/MCP/__init__.py` — availability checks target `mcp_unified`.
- `tldw_chatbook/MCP/__main__.py` — remove checkout `sys.path` mutation, keep stdout protocol-clean, and propagate the server status.
- `tldw_chatbook/Utils/optional_deps.py` — use the `mcp-unified` distribution/import boundary.
- `pyproject.toml` — exact pin in both `mcp` and `all-tools` extras.

### Existing tests/docs/task records

- `Tests/MCP/test_local_server_tools.py` — replace generic FastMCP binding expectations with provider-schema and typed-error adapter expectations.
- `Tests/MCP/test_server_notes_service.py` — replace the `_RecordingFastMCP` harness with the real decorator-compatible adapter seam.
- `Tests/MCP/test_tools_resources_prompts_real_methods.py` — extend real SQLite handler coverage through the adapter.
- `Tests/MCP/test_library_tools.py` and `Tests/MCP/test_local_runtime_delegate.py` — preserve and rerun the private Library manifest/execution/refusal contract; modify only if an assertion must name the new standalone boundary.
- `Tests/Utils/test_optional_deps.py` and `Tests/Utils/test_subscriptions_dependency_gate.py` — new distribution-to-import mapping and availability behavior.
- `Tests/Packaging/test_installed_distribution.py` — retain the general distribution contract; add only shared assertions that genuinely belong to every installed artifact.
- `Docs/Design/MCP.md` — authoritative developer/user architecture, catalog, protocol, continuation, installation, and privacy update.
- `Docs/User_Guide/mcp.md` — external-server privacy warning and standalone boundary clarification.
- `Docs/Development/release-recovery-setup.md` — recovery/install copy for the new dependency.
- `backlog/tasks/task-2511 - Smoke-test-FastMCP-local-tool-binding-with-the-mcp-extra.md` — record supersession only after artifact smoke passes.
- `backlog/tasks/task-2512 - Migrate-MCP-server-from-FastMCP-to-tldw_servers-mcp-unified-package.md` — plan link, final evidence, checked acceptance criteria, notes, and Done status only after all gates pass.

Do not add an adapter hierarchy, plugin registry, transport abstraction, HTTP surface, authenticated continuation format, or replacement MCP client. Those are outside the accepted design.

### Task 1: Pin and verify the released public contract

**Files:**
- Create: `Tests/MCP/test_mcp_unified_public_contract.py`
- Modify: `pyproject.toml:130-134`
- Modify: `pyproject.toml:413-420`
- Modify: `tldw_chatbook/MCP/__init__.py:1-25`
- Modify: `tldw_chatbook/Utils/optional_deps.py:350-370`
- Modify: `tldw_chatbook/Utils/optional_deps.py:1298-1317`
- Modify: `Tests/Utils/test_optional_deps.py:425-475`
- Modify: `Tests/Utils/test_subscriptions_dependency_gate.py:1-40`

- [ ] **Step 1: Write the exact dependency and public-surface tests**

Add tests which assert:

```python
from importlib.metadata import version

from mcp_unified.gateway import (
    GatewayApplicationError,
    GatewayCoreRuntime,
    GatewayLimits,
    GatewayRequestContext,
    GatewayResourceTemplateRuntime,
    GatewayToolExecutionError,
    serve_stdio,
)


def test_mcp_unified_public_contract_is_exact_release():
    assert version("mcp-unified") == "0.2.1"
    assert GatewayCoreRuntime
    assert GatewayResourceTemplateRuntime
    assert GatewayRequestContext
    assert GatewayApplicationError
    assert GatewayToolExecutionError
    assert GatewayLimits
    assert callable(serve_stdio)
```

Also assert the six `GatewayCoreRuntime` methods (`list_tools`, `call_tool`, `list_resources`, `read_resource`, `list_prompts`, `get_prompt`) and the separate `GatewayResourceTemplateRuntime.list_resource_templates` method are callable. Pin the released signatures with `inspect.signature`: `serve_stdio` has positional `runtime` followed by keyword-only `input_stream=None`, `output_stream=None`, `limits=GatewayLimits()`, and `metadata=None`; `GatewayApplicationError` has positional `public_message` followed by keyword-only `reason_code` and `kind="application"`; `GatewayToolExecutionError` has positional `public_message` followed by required keyword-only `reason_code`. Parse `pyproject.toml` and assert both optional-extra occurrences equal `mcp-unified==0.2.1`, assert `is_mcp_available()` probes `mcp_unified`, and update the package/import map assertion to `"mcp-unified": "mcp_unified"`.

- [ ] **Step 2: Run the focused tests and capture RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP/test_mcp_unified_public_contract.py \
  Tests/Utils/test_optional_deps.py \
  Tests/Utils/test_subscriptions_dependency_gate.py -q
```

Expected: collection/import and metadata assertions fail because the venv and extras still use the old official SDK/FastMCP dependency.

- [ ] **Step 3: Change only the declared and live dependency boundary**

Use exactly:

```toml
mcp = ["mcp-unified==0.2.1"]
```

Replace the `all-tools` entry with the same exact pin. Change availability checks to `import mcp_unified`, optional-feature distribution text to `mcp-unified`, and dependency checks to the import name `mcp_unified`. Do not yet change server registration or protocol behavior.

- [ ] **Step 4: Install the changed optional extra into the repository venv**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pip install -e ".[mcp]"
```

Expected: install succeeds with `mcp-unified==0.2.1`; do not install `mcp==2.0.0` as a runtime dependency.

- [ ] **Step 5: Verify GREEN and the absence case**

Run the Step 2 command again. Add a subprocess test which blocks `mcp_unified` import and proves `is_mcp_available()` returns `False` without importing `MCP/server.py`.

Expected: all selected tests pass.

- [ ] **Step 6: Commit the public boundary**

```bash
git add pyproject.toml tldw_chatbook/MCP/__init__.py tldw_chatbook/Utils/optional_deps.py Tests/MCP/test_mcp_unified_public_contract.py Tests/Utils/test_optional_deps.py Tests/Utils/test_subscriptions_dependency_gate.py
git commit -m "build(mcp): pin mcp-unified public runtime"
```

### Task 2: Implement strict built-in tool registration and dispatch

**Files:**
- Create: `tldw_chatbook/MCP/gateway_runtime.py`
- Create: `Tests/MCP/test_gateway_runtime_tools.py`
- Modify: `tldw_chatbook/MCP/server.py:100-253`
- Modify: `Tests/MCP/test_server_notes_service.py:1-110`

- [ ] **Step 1: Write adapter construction and schema RED tests**

Pin these behaviors by name:

- `test_runtime_requires_one_handler_for_every_expected_builtin`
- `test_runtime_rejects_duplicate_tool_names`
- `test_runtime_rejects_handler_without_descriptor`
- `test_all_nine_builtin_schemas_reject_additional_properties`
- `test_all_nine_builtin_handlers_register_with_exact_names`
- `test_standalone_tool_descriptors_exclude_library_tools`

Use `_describe_local_tools()` as the expected descriptor source and assert the exact nine names from the specification. Assert `ingest_media` is absent and preserve the direct-runtime refusal test. Mutation-test the guard by deleting one handler, duplicating one name, and setting one schema's `additionalProperties` back to `True`.

- [ ] **Step 2: Run focused tests and capture RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP/test_gateway_runtime_tools.py \
  Tests/MCP/test_server_notes_service.py -q
```

Expected: import failure for `tldw_chatbook.MCP.gateway_runtime` and schema assertion failures.

- [ ] **Step 3: Add the minimal decorator-compatible runtime**

Implement one concrete class with only the accepted surface:

```python
class ChatbookGatewayRuntime:
    def __init__(
        self,
        *,
        name: str,
        version: str,
        tool_descriptors: list[dict[str, Any]],
    ) -> None: ...
    def tool(self, *, name: str | None = None): ...
    def resource(self, uri_template: str): ...
    def list_resources(self): ...
    def prompt(self, *, name: str | None = None): ...
    def finalize(self) -> None: ...
    async def list_tools(self, context: GatewayRequestContext) -> list[dict[str, Any]]: ...
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> Any: ...
```

Validate and retain bounded non-empty `name` and `version` public attributes because the strict connection reads them for `serverInfo`. Use private dictionaries keyed by stable names. Decorators record handlers only; `finalize()` validates the complete descriptor/handler bijection and freezes publication. `call_tool` invokes the matching async handler and returns its JSON value unchanged so upstream projection remains authoritative. Do not infer failures from an `{"error": ...}` result.

- [ ] **Step 4: Strengthen the AST schema authority**

In `_signature_to_input_schema`, emit:

```python
return {
    "type": "object",
    "properties": properties,
    "required": required,
    "additionalProperties": False,
}
```

Keep `_describe_local_tools()` separate from `describe_local_mcp_capabilities()` and do not register the eighteen Library descriptors.

- [ ] **Step 5: Replace the FastMCP-shaped notes test harness**

Construct a bare `TldwMCPServer`, attach a real `ChatbookGatewayRuntime`, call the existing registration method, finalize it, and dispatch `create_note`/`search_notes` through the adapter. This proves the actual seam instead of another `_RecordingFastMCP` fake.

- [ ] **Step 6: Run tool and existing manifest regressions**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP/test_gateway_runtime_tools.py \
  Tests/MCP/test_server_notes_service.py \
  Tests/MCP/test_library_tools.py \
  Tests/MCP/test_local_runtime_delegate.py -q
```

Expected: all pass; the private Library manifest and raw-call refusal tests remain unchanged.

- [ ] **Step 7: Commit built-in registration**

```bash
git add tldw_chatbook/MCP/gateway_runtime.py tldw_chatbook/MCP/server.py Tests/MCP/test_gateway_runtime_tools.py Tests/MCP/test_server_notes_service.py
git commit -m "feat(mcp): add strict chatbook gateway runtime"
```

### Task 3: Preserve local-agent schemas, permission errors, and atomic publication

**Files:**
- Modify: `tldw_chatbook/MCP/gateway_runtime.py`
- Modify: `tldw_chatbook/MCP/local_server_tools.py`
- Modify: `tldw_chatbook/MCP/server.py:800-875`
- Modify: `Tests/MCP/test_gateway_runtime_tools.py`
- Modify: `Tests/MCP/test_local_server_tools.py`

- [ ] **Step 1: Write the local-tool schema and failure RED matrix**

Add parameterized tests for every exact mapping:

```python
LOCAL_FAILURES = [
    (EXTERNAL_NO_CALLBACK_REFUSAL, "operator_approval_required", "Operator approval is required for this local tool."),
    (LOCAL_TIMEOUT_REFUSAL, "operator_approval_required", "Operator approval is required for this local tool."),
    (LOCAL_DENY_REFUSAL, "tool_permission_denied", "This local tool is disabled by operator policy."),
    (LOCAL_KILL_SWITCH_REFUSAL, "local_tools_disabled", "Local tools are disabled."),
    (LOCAL_GATE_ERROR_REFUSAL, "permission_state_unavailable", "Local tool permission state is unavailable."),
    ("SENTINEL /private/path API_KEY=secret", "local_tool_failed", "Local tool execution failed."),
]
```

Assert the real provider schema is byte-for-byte/equality-preserved in `list_tools`, successful content is returned raw, failures become `GatewayToolExecutionError`, and the sentinel appears in neither the exception's public fields nor captured stdout/stderr/logs.

Add a provider handler which raises `RuntimeError("SENTINEL /private/path API_KEY=secret")`; it must map to the same fixed `local_tool_failed` public error without exposing the exception.

- [ ] **Step 2: Write atomic-publication and event-loop RED tests**

Cover a duplicate local name, collision with a built-in, invalid non-object schema, non-callable handler, and a mid-list invalid registration. After every failure, assert the tool catalog is exactly the original nine built-ins. Add a heartbeat task around a blocking fake provider invocation and assert the heartbeat advances while the handler runs in `asyncio.to_thread`.

Using one already-running adapter/provider instance and a real temporary `MCPPermissionStore`, make two calls around each state transition: ask → grant (second call succeeds), grant → revoke/deny (second call refuses), kill switch off → on (second call refuses), and on → off with a still-valid grant (second call succeeds). These tests must fail if either effective permission state or the kill switch is cached at registration/server startup.

- [ ] **Step 3: Run focused RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP/test_gateway_runtime_tools.py \
  Tests/MCP/test_local_server_tools.py -q
```

Expected: old handlers return error dictionaries, schemas are not directly published, and staging/off-loop assertions fail.

- [ ] **Step 4: Return `ToolResult` from local registrations**

Remove `_parameter_summary` and the FastMCP generic-arguments workaround. Make each `LocalToolRegistration` carry `name`, `description`, the real object-root `parameters`, and a callable returning the provider's `ToolResult`. Do not log raw `ToolResult.error`; replace the kill-switch read warning with fixed payload-free copy.

- [ ] **Step 5: Add one all-or-none runtime publication method**

Implement `register_local_tools(registrations)` by validating into temporary descriptor/handler dictionaries, checking all collisions and shapes, then publishing with one update only after the complete list is valid. Mark the staged handlers as local so `call_tool` performs `await asyncio.to_thread(handler, arguments)` and maps `ToolResult.ok`/stable constants to `GatewayToolExecutionError`.

- [ ] **Step 6: Keep optional composition recoverable and private**

When `[mcp].expose_local_tools` is false, publish nothing. Stage optional locals before the runtime's single finalization step. When provider construction or staging fails, keep the complete built-in catalog, discard every staged local, then finalize the built-ins and write exactly:

```text
Local MCP tools unavailable; continuing with built-in tools.
```

to stderr once. Do not interpolate the exception, traceback, paths, arguments, or provider data. `todo_write` remains absent.

- [ ] **Step 7: Run GREEN plus permission mutations**

Run the Step 3 command, then mutate each refusal constant comparison, cache one permission decision/kill-switch value, and publish one local map before validation completes. Prove the corresponding tests fail before restoring each mutation.

Expected: all selected tests pass and the mutations are detected.

- [ ] **Step 8: Commit local-tool migration**

```bash
git add tldw_chatbook/MCP/gateway_runtime.py tldw_chatbook/MCP/local_server_tools.py tldw_chatbook/MCP/server.py Tests/MCP/test_gateway_runtime_tools.py Tests/MCP/test_local_server_tools.py
git commit -m "feat(mcp): preserve gated local tool contracts"
```

### Task 4: Map and continue bounded resources

**Files:**
- Modify: `tldw_chatbook/MCP/gateway_runtime.py`
- Modify: `tldw_chatbook/MCP/server.py:706-753`
- Create: `Tests/MCP/test_gateway_runtime_resources.py`
- Modify: `Tests/MCP/test_tools_resources_prompts_real_methods.py`

- [ ] **Step 1: Write resource registration/routing RED tests**

Assert exact registration and routing for:

- `conversation://{conversation_id}`
- `note://{note_id}`
- `character://{character_id}`
- `media://{media_id}`
- `rag-chunk://{chunk_uuid}`

Reject duplicate templates, unknown schemes, fragments, malformed percent encoding, extra path segments, duplicate/unknown query parameters, empty identifiers, and identifier/template mismatches before handler invocation. Assert dynamic resources preserve `uri`, `name`, optional `description`, and `mimeType` in order.

- [ ] **Step 2: Write canonical result and continuation RED tests**

Use ASCII, multibyte UTF-8, exact-limit, and over-limit content. Assert one text block, at most 256 KiB UTF-8, no split code point, exact counts, and this metadata shape:

```python
result["_meta"] == {
    "tldw.chatbook/continuation": {
        "startChar": 0,
        "endChar": expected_end,
        "totalChars": len(text),
        "totalBytes": len(text.encode("utf-8")),
        "returnedBytes": len(chunk.encode("utf-8")),
        "hasMore": expected_more,
        "nextUri": expected_uri,
    },
    "tldw.chatbook/resource": handler_metadata,
}
```

Test malformed, wrong-base, out-of-range, and changed-content continuations. Demonstrate explicitly that the token is not an authorization test: acceptance depends on its bounded version/offset/base/content fields, not a secret/HMAC.

For handler metadata, test all three cases: a non-empty valid mapping appears only at `_meta["tldw.chatbook/resource"]`; absent or empty metadata omits that key entirely; invalid/non-JSON/deep metadata fails closed rather than being flattened or dropped.

- [ ] **Step 3: Run focused RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP/test_gateway_runtime_resources.py \
  Tests/MCP/test_tools_resources_prompts_real_methods.py -q
```

Expected: resource methods/continuation helpers are missing.

- [ ] **Step 4: Implement the five-template router and catalog**

Compile only the five accepted one-variable custom-scheme templates into anchored matchers. Parse/validate query and fragment first, normalize the query-free base URI, match it, and call the corresponding async handler with the decoded identifier. Implement `list_resources` and `list_resource_templates` using immutable copies so callers cannot mutate the registered catalog.

- [ ] **Step 5: Implement bounded UTF-8 continuation**

Use a single constant:

```python
MAX_RESOURCE_CHUNK_BYTES = 256 * 1024
CONTINUATION_QUERY_KEY = "tldw_continue"
```

Encode a bounded URL-safe token containing version, character offset, SHA-256 normalized-base digest, and SHA-256 content digest. Re-materialize the resource on continuation, verify the base/content/offset, and raise a bounded `GatewayApplicationError` with reason `resource_changed` if the content digest differs. Store only valid JSON handler metadata under `_meta["tldw.chatbook/resource"]`.

- [ ] **Step 6: Exercise all five real resource handlers**

Seed temporary SQLite databases and dispatch each registered handler through `ChatbookGatewayRuntime.read_resource`. Add a large conversation/media fixture which requires at least two reads and reconstruct the exact original text by following `nextUri`.

- [ ] **Step 7: Run GREEN and mutate each validation guard**

Run the Step 3 command. Temporarily allow an unknown query key, remove the content digest check, and flatten handler metadata; confirm one test fails for each mutation before restoring.

- [ ] **Step 8: Commit resources**

```bash
git add tldw_chatbook/MCP/gateway_runtime.py tldw_chatbook/MCP/server.py Tests/MCP/test_gateway_runtime_resources.py Tests/MCP/test_tools_resources_prompts_real_methods.py
git commit -m "feat(mcp): add bounded resource continuation"
```

### Task 5: Validate and map prompt descriptors and results

**Files:**
- Modify: `tldw_chatbook/MCP/gateway_runtime.py`
- Modify: `tldw_chatbook/MCP/server.py:173-212`
- Modify: `tldw_chatbook/MCP/server.py:755-809`
- Modify: `tldw_chatbook/MCP/prompts.py:235-292`
- Create: `Tests/MCP/test_gateway_runtime_prompts.py`
- Modify: `Tests/MCP/test_tools_resources_prompts_real_methods.py`

- [ ] **Step 1: Write exact prompt descriptor RED tests**

Assert the five exact names and derive every public parameter as `{name, description?, required}` from the AST/signature source. Pin `str`, `int`, and `Optional[str]` primitive expectations without exposing Python annotations on the wire. Reject duplicate names and unsupported parameter types during finalization.

- [ ] **Step 2: Write coercion and result-shape RED tests**

Cover correct JSON values, string-to-int coercion, string preservation, omitted defaults, missing required values, unknown arguments, booleans supplied as integers, invalid integers, and non-object arguments. Cover user/assistant pass-through and exact fold:

```text
System instructions:
{system one}\n\n{system two}

User request:
{original user text}
```

Reject empty lists, trailing/mid-stream system roles, unknown roles, non-string content, and non-list/non-dict shapes.

- [ ] **Step 3: Reproduce the missing-await failure**

Add a real async fake/temporary DB search whose `keyword_search` result is awaited. Run only:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP/test_gateway_runtime_prompts.py::test_search_and_synthesize_awaits_keyword_search -q
```

Expected: FAIL with the current coroutine iteration/error-fallback behavior.

- [ ] **Step 4: Implement prompt registration, coercion, and mapping**

Store descriptor arguments plus private primitive kinds, validate the argument object before dispatch, allow Python defaults by omitting optional absent keys, and map the non-empty handler list to MCP text blocks. Fold only a contiguous leading system block immediately followed by the first user message; never change `MCPPrompts.character_writing_prompt`'s internal return shape.

- [ ] **Step 5: Add the missing `await`**

Change only the real call in `search_and_synthesize_prompt` from the coroutine value to `await self.media_db.keyword_search(...)` (using its actual named arguments/signature already exercised in the test).

- [ ] **Step 6: Exercise all five real prompt handlers**

Use temporary databases and deterministic provider/search fakes to get each prompt through `get_prompt`. Assert every result is non-empty and canonical; for `character_writing`, assert the external adapter folds the system message while direct in-process use still returns system-plus-user.

- [ ] **Step 7: Run GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP/test_gateway_runtime_prompts.py \
  Tests/MCP/test_tools_resources_prompts_real_methods.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit prompts**

```bash
git add tldw_chatbook/MCP/gateway_runtime.py tldw_chatbook/MCP/server.py tldw_chatbook/MCP/prompts.py Tests/MCP/test_gateway_runtime_prompts.py Tests/MCP/test_tools_resources_prompts_real_methods.py
git commit -m "feat(mcp): map bounded prompt results"
```

### Task 6: Compose the real server and prove multi-revision strict stdio

**Files:**
- Modify: `tldw_chatbook/MCP/server.py:44-80`
- Modify: `tldw_chatbook/MCP/server.py:379-412`
- Modify: `tldw_chatbook/MCP/server.py:875-905`
- Modify: `tldw_chatbook/MCP/__main__.py`
- Create: `Tests/MCP/test_mcp_unified_stdio.py`
- Modify: `Tests/MCP/test_mcp_import.py`

- [ ] **Step 1: Write construction and entrypoint RED tests**

Assert `TldwMCPServer.mcp` is a finalized `ChatbookGatewayRuntime`, exact counts are 9 built-ins/5 templates/5 prompts with local exposure off, `ingest_media` is absent, HTTP raises `NotImplementedError`, `run("stdio")` returns the injected `serve_stdio` integer, `main()` returns it, and `MCP/__main__.py` exits with it without mutating `sys.path` or writing human text to stdout.

- [ ] **Step 2: Write real protocol RED tests**

Using in-memory binary reader/writer adapters or a subprocess fixture, cover:

- initialize/list/call/read/get at `2025-03-26`, `2025-11-25`, and `2026-07-28`;
- dict/list/string tool projection and `2025-11-25` object-only `structuredContent` behavior;
- a `2026-07-28` request carrying bounded `_meta` reaches `GatewayRequestContext.metadata` with gateway-reserved context keys authoritative and no client overwrite;
- deterministic unsupported-version error;
- a JSON-RPC batch accepted only at `2025-03-26` and rejected for the other two revisions;
- notification/cancellation, including cancellation of a blocked `asyncio.to_thread` local call: no late response is emitted after the worker is released, while the test records that already-started side effects are not claimed rolled back;
- clean EOF, broken output, and bounded shutdown;
- no `library_*` descriptor on the wire even though all eighteen remain in `describe_local_mcp_capabilities()`.

- [ ] **Step 3: Run focused RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP/test_mcp_unified_stdio.py \
  Tests/MCP/test_mcp_import.py -q
```

Expected: FastMCP import/construction and old `run()` behavior fail the new contract.

- [ ] **Step 4: Replace FastMCP construction and serving**

Conditionally import only the public `mcp_unified.gateway` names. Construct `ChatbookGatewayRuntime(name=name, version=version, tool_descriptors=_describe_local_tools())`, register the existing tools/resources/prompts, optionally stage local tools, call `finalize()`, and retain `self.mcp` as the compatibility attribute. Implement:

```python
async def run(self, transport: str = "stdio") -> int:
    if transport != "stdio":
        raise NotImplementedError("Only stdio transport is supported")
    return await serve_stdio(self.mcp)


async def main() -> int:
    return await TldwMCPServer().run("stdio")
```

The package owns limits, negotiation, projection, and lifecycle defaults; do not fork or copy its protocol implementation.

- [ ] **Step 5: Make the module entrypoint protocol-clean**

Remove `sys.path.insert`. Use `raise SystemExit(asyncio.run(main()))`; print only fixed fatal/Ctrl-C diagnostics to stderr. Never print a startup banner or result payload to stdout.

- [ ] **Step 6: Run protocol GREEN and a real subprocess smoke**

Run the Step 3 command. Then run the subprocess test which launches the absolute venv interpreter with `-m tldw_chatbook.MCP` under an isolated temporary profile, performs initialize/catalog/call/read/get, closes stdin, and asserts exit 0 plus JSON-only stdout and payload-free stderr.

Expected: all pass; no server child remains.

- [ ] **Step 7: Commit server migration**

```bash
git add tldw_chatbook/MCP/server.py tldw_chatbook/MCP/__main__.py Tests/MCP/test_mcp_unified_stdio.py Tests/MCP/test_mcp_import.py
git commit -m "feat(mcp): serve chatbook through strict stdio"
```

### Task 7: Make the hand-written client cursor-safe and metadata-complete

**Files:**
- Modify: `tldw_chatbook/MCP/client.py:25-83`
- Modify: `tldw_chatbook/MCP/client.py:121-201`
- Modify: `tldw_chatbook/MCP/client.py:565-699`
- Create: `Tests/MCP/test_client_catalog_pagination.py`
- Modify: `Tests/MCP/test_mcp_unified_stdio.py`

- [ ] **Step 1: Write bounded pagination RED tests**

Use a scripted connection to assert first-page-without-cursor, ordered aggregation, absent/null termination, and cursor forwarding for tools/resources/prompts. Parameterize rejection of empty, non-string, and repeated cursors; non-list item arrays; page 101; and item 10,001. In every overbound case assert no partial list is returned.

- [ ] **Step 2: Write resource metadata RED tests**

Assert low-level `read_resource` exposes exact result `_meta`, missing metadata becomes `{}`, and high-level `MCPClient.read_resource` returns:

```python
{
    "uri": resource_uri,
    "content": expected_text,
    "mimeType": expected_mime,
    "_meta": original_meta,
}
```

Do not rename, flatten, or merge continuation metadata.

- [ ] **Step 3: Run focused RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP/test_client_catalog_pagination.py -q
```

Expected: first-page-only behavior and dropped `_meta` fail.

- [ ] **Step 4: Implement one shared bounded collector**

Add private constants `MAX_CATALOG_PAGES = 100` and `MAX_CATALOG_ITEMS = 10_000`, plus one helper used by all three list methods. Only absent/`None` ends; validate the item list and cursor before appending/continuing; detect the limit before returning any aggregate. Preserve source order.

- [ ] **Step 5: Preserve exact `_meta` at both client levels**

Keep result-level metadata beside `contents` in the low-level namespace, copy it as a mapping, and return it under the exact `"_meta"` key from `MCPClient.read_resource`.

- [ ] **Step 6: Run GREEN and the real 2025-03-26 client flow**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP/test_client_catalog_pagination.py \
  Tests/MCP/test_mcp_unified_stdio.py -q
```

Add/retain a real subprocess case using `MCPClient.connect_to_server` against `python -m tldw_chatbook.MCP`, asserting catalog aggregation, one tool call, continued resource reads, and one prompt.

Expected: all pass.

- [ ] **Step 7: Commit client compatibility**

```bash
git add tldw_chatbook/MCP/client.py Tests/MCP/test_client_catalog_pagination.py Tests/MCP/test_mcp_unified_stdio.py
git commit -m "fix(mcp): bound catalog pagination and preserve metadata"
```

### Task 8: Prove wheel/sdist isolation and update public documentation

**Files:**
- Create: `Tests/Packaging/test_mcp_unified_distribution.py`
- Create: `Tests/MCP/test_mcp_documentation_contract.py`
- Modify: `Tests/Packaging/test_installed_distribution.py` only if a shared artifact assertion belongs there
- Modify: `Docs/Design/MCP.md`
- Modify: `Docs/User_Guide/mcp.md`
- Modify: `Docs/Development/release-recovery-setup.md`
- Modify: `backlog/tasks/task-2511 - Smoke-test-FastMCP-local-tool-binding-with-the-mcp-extra.md`

- [ ] **Step 1: Write the artifact-isolation RED test before changing docs**

Build once, then parameterize `wheel` and `sdist`. For each artifact create an independent temporary root containing its venv, `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TMPDIR`, config, working directory, databases, permission store, and workspace. Sanitize inherited environment by removing `PYTHONPATH`, Chatbook path/config overrides, proxy variables not needed by the test, and every provider credential/token/secret variable.

Install the artifact with `[mcp]`, then assert both `tldw_chatbook.__file__` and `mcp_unified.__file__` are inside that venv's site-packages and outside both checkout/build roots. Before server launch, assert resolved config/data/DB/permission/workspace paths are descendants of the temporary root.

Inspect wheel `METADATA` and sdist `PKG-INFO` and assert the Chatbook license remains `AGPL-3.0-or-later` with its license file present, and that `Requires-Dist: mcp-unified==0.2.1` is attached to the `mcp` extra. This is the explicit dependency/license-metadata gate.

- [ ] **Step 2: Run the new artifact test and capture RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Packaging/test_mcp_unified_distribution.py -q
```

Expected: FAIL until the complete migration is packaged and the isolated server smoke is implemented.

- [ ] **Step 3: Add a site-packages-only protocol consumer**

From the temporary working directory, launch that artifact venv's interpreter with `-I -m tldw_chatbook.MCP`; run initialize/catalog/call/read/get using deterministic temporary data; assert nine implemented built-ins, no `ingest_media`, five templates, five prompts, zero `library_*`, expected continuation metadata, fixed local refusal behavior when enabled, clean EOF, exit 0, JSON-only stdout, and no sentinel path/secret in stderr.

Run wheel and sdist as separate parameter cases so one cannot mask the other. Bound install/build/server waits and retain no process or venv after the test.

- [ ] **Step 4: Remove every live FastMCP/official-SDK dependency reference**

Run:

```bash
rg -n "mcp[.]server[.]fastmcp|from[[:space:]]+mcp(?:[.]|[[:space:]]|$)|import[[:space:]]+mcp(?:[.]|[[:space:]]|$)|mcp\[cli\]|FastMCP" \
  tldw_chatbook pyproject.toml Tests Docs/Design/MCP.md Docs/User_Guide/mcp.md Docs/Development/release-recovery-setup.md
```

Expected after edits: no production/dependency occurrence; only deliberately historical test/task prose may remain. Replace outdated `mcp install`/`mcp dev` instructions with `pip install "tldw_chatbook[mcp]"` and `python -m tldw_chatbook.MCP`.

- [ ] **Step 5: Document the exact external boundary and privacy risk**

Update the developer and user docs with:

- supported `2025-03-26`, `2025-11-25`, and current `2026-07-28` behavior and batching only at `2025-03-26`;
- exact standalone inventory and explicit exclusion of all eighteen `library_*` tools;
- 256 KiB continuation and `_meta["tldw.chatbook/continuation"]`/`_meta["tldw.chatbook/resource"]` keys;
- default-off local tools, permission-store/kill-switch/workspace behavior, and external ask-state refusal;
- a prominent warning that an external MCP client runs with the user's OS access, can read private local Library content through exposed tools/resources/prompts, and may send that content off-device to a cloud model.

Encode these as `Tests/MCP/test_mcp_documentation_contract.py` assertions over the three live documents: exact install command, three required revisions, batch restriction, standalone/private-Library split, continuation keys, default-off local tools, and both local-data/cloud-model privacy statements. The test must also reject live FastMCP/official-SDK commands in those documents.

- [ ] **Step 6: Supersede TASK-2511 only after artifact GREEN**

Record that its FastMCP smoke is obsolete and replaced by TASK-2512's independent wheel/sdist `mcp-unified` protocol smoke. Do not mark TASK-2512 Done yet.

- [ ] **Step 7: Run artifact and documentation GREEN**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Packaging/test_mcp_unified_distribution.py \
  Tests/Packaging/test_installed_distribution.py \
  Tests/MCP/test_mcp_documentation_contract.py \
  Tests/Utils/test_optional_deps.py \
  Tests/Utils/test_subscriptions_dependency_gate.py -q
```

Expected: wheel and sdist each pass independently from site-packages with isolated local data.

- [ ] **Step 8: Commit artifacts and docs**

```bash
git add Tests/Packaging/test_mcp_unified_distribution.py Tests/Packaging/test_installed_distribution.py Tests/MCP/test_mcp_documentation_contract.py Docs/Design/MCP.md Docs/User_Guide/mcp.md Docs/Development/release-recovery-setup.md 'backlog/tasks/task-2511 - Smoke-test-FastMCP-local-tool-binding-with-the-mcp-extra.md'
git commit -m "test(mcp): prove isolated standalone distributions"
```

### Task 9: Run final regression/security gates and close TASK-2512

**Files:**
- Modify: `Docs/superpowers/plans/2026-08-09-mcp-unified-standalone-server-migration.md`
- Modify: `Docs/superpowers/specs/2026-08-09-mcp-unified-standalone-server-migration-design.md`
- Modify: `backlog/decisions/053-mcp-unified-standalone-runtime-boundary.md`
- Modify: `backlog/tasks/task-2512 - Migrate-MCP-server-from-FastMCP-to-tldw_servers-mcp-unified-package.md`
- Modify: `Docs/Design/MCP.md`
- Modify: `Docs/User_Guide/mcp.md`
- Modify: `Docs/Development/release-recovery-setup.md`
- Modify: `Docs/Development/Agent-Tools/local-library-tools.md`
- Modify: `Tests/MCP/test_gateway_runtime_tools.py`
- Modify: `Tests/MCP/test_mcp_unified_stdio.py`
- Modify: `Tests/MCP/test_mcp_documentation_contract.py`
- Modify: `Tests/Packaging/test_mcp_unified_distribution.py`
- Modify: `Tests/UI/test_mcp_workbench.py`
- Modify: `backlog/docs/lessons-testing-evidence.md`, `backlog/docs/lessons-live-verification.md`, or `backlog/docs/lessons-backlog-hygiene.md` only if this implementation produced a new evidence-backed reusable lesson

- [ ] **Step 1: Rebase or merge the latest `origin/dev` before final evidence**

Fetch, inspect the semantic delta, integrate without destructive reset, and rerun the in-flight TASK-2512 PR/branch search. If a conflict touches generated CSS or backlog filenames, follow the repository lessons rather than hand-merging or staging broadly.

- [ ] **Step 1a: Reconcile the post-rebase inventory contract before behavior edits**

Update TASK-2512, ADR-053, this specification, and this plan first to record
TASK-4000's approved nine-tool contract, explicit `ingest_media`
absence/refusal, and Library Import replacement path. Run the required
plan/spec review gate and do not edit inventory tests or live docs until that
review passes.

After the review passes, commit only the governing correction:

```bash
git add \
  'backlog/tasks/task-2512 - Migrate-MCP-server-from-FastMCP-to-tldw_servers-mcp-unified-package.md' \
  backlog/decisions/053-mcp-unified-standalone-runtime-boundary.md \
  Docs/superpowers/specs/2026-08-09-mcp-unified-standalone-server-migration-design.md \
  Docs/superpowers/plans/2026-08-09-mcp-unified-standalone-server-migration.md
git commit -m "docs(mcp): reconcile standalone inventory contract"
```

- [ ] **Step 1b: Capture the stale ten-tool expectations as RED**

Run the focused failing inventory nodes before changing their expectations:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP/test_gateway_runtime_tools.py::test_all_ten_builtin_handlers_register_with_exact_names \
  Tests/MCP/test_gateway_runtime_tools.py::test_all_ten_builtin_schemas_reject_additional_properties \
  Tests/MCP/test_mcp_unified_stdio.py::test_constructor_finalizes_the_exact_default_standalone_surface \
  Tests/Packaging/test_mcp_unified_distribution.py -q
```

Expected RED: every failure reports the same 10-expected/9-actual inventory
mismatch; wheel and sdist must reach their isolated server inventories under
normal PyPI access rather than failing on dependency download.

- [ ] **Step 1c: Write the corrected documentation contract and capture RED**

Before editing live documentation, change/add assertions in
`Tests/MCP/test_mcp_documentation_contract.py` for exactly nine implemented
built-ins, explicit `ingest_media` absence, and the Library Import replacement
path across its existing standalone-document set, including
`Docs/Development/release-recovery-setup.md`.

Add a separate targeted
`test_local_library_tools_documentation_uses_current_standalone_inventory`
which reads only `Docs/Development/Agent-Tools/local-library-tools.md` and
requires its exact nine-tool, `ingest_media`-absent, Library Import sentence.
Do not add this focused Library-boundary document to the global `DOCUMENTS`
fixture or require it to carry the full standalone guide contract. First run
the targeted test alone, then the complete documentation contract:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP/test_mcp_documentation_contract.py::test_local_library_tools_documentation_uses_current_standalone_inventory -q
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP/test_mcp_documentation_contract.py -q
```

Expected RED: the targeted test fails on the unchanged Agent Tools ten-tool
claim, and the complete contract reports the unchanged standalone docs which
still advertise ten tools and/or omit the explicit Library Import
replacement. A collection error or unrelated failure does not count as RED.

- [ ] **Step 1d: Make the minimum inventory/docs correction and prove GREEN**

Change only stale standalone inventory constants, counts, protocol fixtures,
and public documentation to nine implemented built-ins. Remove
`ingest_media` call fixtures and placeholder-success copy, add/retain explicit
absence assertions, preserve `Tests/MCP/test_library_tools.py`'s upstream
absence/refusal coverage, and direct ingestion to Library Import. Do not
change production dispatch. Correct the stale ten-tool claim in
`Docs/Development/Agent-Tools/local-library-tools.md` and the obsolete
"`ingest_media` is currently a stub" claim in
`Tests/UI/test_mcp_workbench.py` without weakening that UI test's actual
permission-classification assertion. Verify/update
`Docs/Development/release-recovery-setup.md` within the same nine-tool,
`ingest_media`-absent, Library Import contract. Run the targeted Agent Tools
documentation test GREEN, then the complete focused suite:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP/test_mcp_documentation_contract.py::test_local_library_tools_documentation_uses_current_standalone_inventory -q
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP/test_gateway_runtime_tools.py \
  Tests/MCP/test_mcp_unified_stdio.py \
  Tests/MCP/test_mcp_documentation_contract.py \
  Tests/MCP/test_library_tools.py \
  Tests/Packaging/test_mcp_unified_distribution.py \
  Tests/UI/test_mcp_workbench.py::test_is_permission_refusal_bare_permission_error_from_tool_body_is_not_a_refusal -q
```

Also run the stale-claim scan:

```bash
rg -n "ten legacy MCP tools|one file-shaped built-in.*ingest_media|ingest_media.*currently a stub|Built-in tools \(10\)" \
  Docs/Design/MCP.md \
  Docs/User_Guide/mcp.md \
  Docs/Development/release-recovery-setup.md \
  Docs/Development/Agent-Tools/local-library-tools.md \
  Tests/UI/test_mcp_workbench.py
```

Expected: no matches.

Expected GREEN: exact nine-tool source, wire, subprocess, wheel, and sdist
inventories pass; `ingest_media` remains absent/refused and Library Import is
the only documented persistent-ingestion path. Then rerun Steps 2-5 in full
on the corrected tree.

Commit only the reconciliation tests/docs after GREEN:

```bash
git add \
  Docs/Design/MCP.md \
  Docs/User_Guide/mcp.md \
  Docs/Development/release-recovery-setup.md \
  Docs/Development/Agent-Tools/local-library-tools.md \
  Tests/MCP/test_gateway_runtime_tools.py \
  Tests/MCP/test_mcp_unified_stdio.py \
  Tests/MCP/test_mcp_documentation_contract.py \
  Tests/Packaging/test_mcp_unified_distribution.py \
  Tests/UI/test_mcp_workbench.py
git commit -m "test(mcp): reconcile nine-tool standalone inventory"
```

- [ ] **Step 2: Run the complete scoped behavior suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP \
  Tests/RuntimePolicy/test_boundary_guards.py \
  Tests/Utils/test_optional_deps.py \
  Tests/Utils/test_subscriptions_dependency_gate.py \
  Tests/Packaging/test_mcp_unified_distribution.py \
  Tests/Packaging/test_installed_distribution.py -q
```

This command explicitly includes the direct Library contract/refusal tests, the new documentation contract, the architecture boundary guard, and both artifact/license-metadata tests. Expected: all new/scoped tests pass.

- [ ] **Step 3: Run the full repository suite as a mandatory gate**

Run the exact command below and record it verbatim in the task evidence:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q
```

This gate is not optional. If the environment prevents it from starting or completing, TASK-2512 remains In Progress. If it fails, run the identical command on a clean `origin/dev` worktree and record exact shared versus branch-only failing node IDs. Do not relabel a branch regression as baseline based only on a similar count, and do not mark Done with a branch-only failure.

- [ ] **Step 4: Run static, type, security, syntax, and diff gates**

First run `git diff --name-only origin/dev...HEAD -- '*.py'` and verify every
reported path is covered by the Ruff scopes below; verify every reported
non-test path is also present in the mypy, Bandit, and compile scopes. The
explicit scopes include all TASK-2512 Python changes plus the post-rebase UI
test correction.

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  tldw_chatbook/MCP \
  tldw_chatbook/RAG_Search/simplified/search_service.py \
  tldw_chatbook/Utils/optional_deps.py \
  Tests/MCP \
  Tests/Packaging/test_mcp_unified_distribution.py \
  Tests/Utils/test_optional_deps.py \
  Tests/Utils/test_subscriptions_dependency_gate.py \
  Tests/UI/test_mcp_workbench.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/MCP \
  tldw_chatbook/RAG_Search/simplified/search_service.py \
  tldw_chatbook/Utils/optional_deps.py \
  Tests/MCP \
  Tests/Packaging/test_mcp_unified_distribution.py \
  Tests/Utils/test_optional_deps.py \
  Tests/Utils/test_subscriptions_dependency_gate.py \
  Tests/UI/test_mcp_workbench.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy \
  tldw_chatbook/MCP/__init__.py \
  tldw_chatbook/MCP/__main__.py \
  tldw_chatbook/MCP/gateway_runtime.py \
  tldw_chatbook/MCP/server.py \
  tldw_chatbook/MCP/client.py \
  tldw_chatbook/MCP/local_runtime_delegate.py \
  tldw_chatbook/MCP/local_server_tools.py \
  tldw_chatbook/MCP/prompts.py \
  tldw_chatbook/RAG_Search/simplified/search_service.py \
  tldw_chatbook/Utils/optional_deps.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m bandit -q \
  tldw_chatbook/MCP/__init__.py \
  tldw_chatbook/MCP/__main__.py \
  tldw_chatbook/MCP/gateway_runtime.py \
  tldw_chatbook/MCP/server.py \
  tldw_chatbook/MCP/client.py \
  tldw_chatbook/MCP/local_runtime_delegate.py \
  tldw_chatbook/MCP/local_server_tools.py \
  tldw_chatbook/MCP/prompts.py \
  tldw_chatbook/RAG_Search/simplified/search_service.py \
  tldw_chatbook/Utils/optional_deps.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q \
  tldw_chatbook/MCP \
  tldw_chatbook/RAG_Search/simplified/search_service.py \
  tldw_chatbook/Utils/optional_deps.py
git diff --check origin/dev...HEAD
git diff --check
```

Also run the explicit documentation/architecture/license slice:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/MCP/test_mcp_documentation_contract.py \
  Tests/MCP/test_library_tools.py \
  Tests/MCP/test_local_runtime_delegate.py \
  Tests/RuntimePolicy/test_boundary_guards.py \
  Tests/Packaging/test_installed_distribution.py::test_built_artifacts_match_distribution_contract \
  Tests/Packaging/test_mcp_unified_distribution.py -q
```

Expected: all pass, the built metadata proves `AGPL-3.0-or-later` plus the exact conditional `mcp-unified==0.2.1` dependency, and live docs contain the required architecture/privacy contract. Only exact pre-existing mypy/full-suite baselines with identical clean-dev evidence may be documented as baseline.

- [ ] **Step 5: Perform a file-by-file self-review and mutation audit**

Review every changed production line against ADR-053 and the spec. Re-run the deliberate mutations for registration bijection, Library exclusion, local refusal classification/public copy, atomic local publication, resource cursor validation/content-change detection, prompt role order/non-empty results, catalog cursor bounds, output purity, and artifact source isolation. Restore each mutation and rerun its focused test.

- [ ] **Step 6: Request final code review while TASK-2512 remains In Progress**

Use `@superpowers:requesting-code-review` against the complete branch diff. The review must cover ADR/spec alignment, security/privacy, protocol revisions, cancellation/lifecycle, package isolation, and regression tests. Do not check acceptance criteria or mark the task Done before this review returns.
Record the exact tested commit and require the review to target that commit
with a clean worktree.

- [ ] **Step 7: Address review findings and refresh evidence**

For each verified Critical/Important/Minor finding, add a focused failing regression test before production changes, implement the minimal fix, and commit it separately. If review changes any tracked file, rerun Steps 2–5 in full, commit the refreshed tree, and repeat Step 6 against that exact tested commit. Continue this loop until the final review returns Ready with no tracked-file change; only then record the review pass for closeout.

- [ ] **Step 8: Update TASK-2512 truthfully after review**

Check each acceptance criterion only when its post-review evidence exists. Add concise Implementation Notes covering architecture, security/privacy boundaries, exact modified files, test/static/security/artifact evidence, review outcome, deviations, and accepted worker-thread cancellation limitation. Add a lesson only if an incident produced genuinely new reusable knowledge.

- [ ] **Step 9: Mark Done and commit closeout only after the repository Definition of Done is satisfied**

Run:

```bash
backlog task edit 2512 -s Done
backlog task 2512 --plain
```

Read the rendered task back and verify all acceptance criteria are checked, the ADR/spec/plan are linked, Implementation Notes survived the CLI edit, and status is Done.

```bash
git add 'backlog/tasks/task-2512 - Migrate-MCP-server-from-FastMCP-to-tldw_servers-mcp-unified-package.md'
# If and only if Step 8 added a new evidence-backed lesson, add that exact lesson file by name.
git commit -m "docs(mcp): close mcp-unified migration"
```

### Task 9 closeout evidence (2026-08-10)

- Integrated `origin/dev` `ced98b9a42da8fa834e7851b1e7e357bb9a7dfd2`;
  independent final review returned **Ready** with no Critical, Important, or
  Minor findings on clean commit
  `0f7200aced210038c2868d132c6ccdf630f43866`.
- The exact final MCP suite passed 1,007 tests with one known dependency
  warning. The focused lifecycle/race subset passed 17 tests, and the
  documentation contract passed 59 tests. Changed-file Ruff format/check,
  mypy over the four MCP production modules, Bandit over the five MCP security
  targets, compileall, and both diff-quality checks passed.
- Wheel and sdist isolation passed earlier in Task 8. A final normal-network
  refresh installed both artifacts successfully, then both hit the newly
  shared upstream omission of
  `chachanotes_v32_to_v33_console_context_memory.sql`; the identical installed-
  distribution node failed for the same missing migration on clean
  `origin/dev`. The complete final scoped command therefore reported 1,083
  passed and four non-MCP failures: that shared artifact omission, the shared
  `frontmatter` optional-feature metadata mismatch, and the two artifact cases
  blocked by the same omission.
- The repository owner explicitly instructed the task to **ignore all CI
  checks**. Repository-full/CI are waived, not green: the earlier full run
  reached about 83% and stopped at the shared Library navigation test hang.
  Exact branch and clean-dev 300-second timeouts isolated the stale test stub;
  TASK-15104 changed it to the typed permitted `NoteFlushOutcome` and made the
  exact node and adjacent transition group pass. No later repository-full or
  CI result is represented as successful.
