# Task 3.2 Report: Register scoped Canvas tools

## Status

Implemented the scoped Canvas tool provider and Console registration seam. The
four reserved tools are advertised only for an enabled provider with a live,
exactly authenticated session/run coordinator binding. Canvas mutations route
complete replacement HTML through the injected coordinator and retain no HTML
in generic display, log, cycle, or continuation records.

ADR required: no

ADR path: `backlog/decisions/115-local-versioned-canvas-artifacts-and-browser-sandbox.md`

Reason: this delivery directly implements ADR-115's established Canvas
authority, storage, staging, and retention boundaries; it makes no new
architectural decision.

## Implementation

- Added `CanvasToolProvider` with closed schemas for `canvas_list`,
  `canvas_read`, `canvas_create(title, html)`, and
  `canvas_update(canvas_id, expected_parent_revision_id, html)`.
- Reused the shared Canvas title, revision-source, canvas-count, and
  revision-count limits at schema and runtime boundaries, including UTF-8 byte
  validation rather than relying on JSON Schema character counts.
- Bound each provider instance to an injected immutable `CanvasScope`; model
  arguments cannot supply session, conversation, branch/reachability, run,
  message-origin, or tool-call authority.
- Required the live run context and server-generated tool-call identity before
  dispatch, and passed those values with the bound scope to the coordinator.
- Serialized bounded list/read/staged-revision/compatibility/conflict results.
  Only successful `canvas_read` model results contain complete HTML.
- Reserved all four Canvas names across builtin/local/skill/MCP composition and
  added authenticated Canvas registration to the per-run catalog.
- Kept authenticated Canvas available for temporary conversations while
  retaining the existing temporary-session block for unrelated providers.

## Authority and approval contracts

- `CanvasToolRegistrationAuthority` is nominal, frozen, issuer-bound, and held
  by weak identity. Copying its values, matching its source/name strings, or
  implementing a structural lookalike does not authenticate a provider.
- Registration rechecks provider type, exact authority identity, session/run
  binding, enabled state, coordinator freshness, source, and the exact reserved
  catalog. Invocation rechecks live authority and scope after registration.
- The narrow `canvas_reversible_conversation_local` classification is returned
  only for authenticated `canvas_create` and `canvas_update`. `canvas_list`,
  `canvas_read`, lookalikes, copied capabilities, and every non-Canvas tool do
  not receive that classification and therefore retain their ordinary policy.
- The provider is a coordinator adapter, not an authority owner. Delivery 3.3
  will supply the Console lifecycle coordinator that selects durable
  `CanvasService` versus temporary `CanvasStagingStore` handling and performs
  commit/finalization.

## Projection and retention contracts

- The model receives complete source only from an explicit successful
  `canvas_read` tool result.
- `canvas_create` and `canvas_update` never echo source in their results.
- Display, log, cycle, and continuation projections strip the `html` argument
  and result field. Mutation arguments retain only IDs/title plus SHA-256 and
  UTF-8 byte count; results retain bounded identity, revision, digest, status,
  compatibility, and conflict metadata.
- Malformed results project to source-free bounded metadata. The registry's
  generic opt-in projection failure fallback is also source-free and is covered
  by the focused Task 3.1 projection regression tests.

## TDD evidence

Initial RED, before `canvas_tool_provider.py` existed:

```text
$ pytest -q Tests/Agents/test_canvas_tool_provider.py
ERROR collecting Tests/Agents/test_canvas_tool_provider.py
ModuleNotFoundError: No module named 'tldw_chatbook.Agents.canvas_tool_provider'
Interrupted: 1 error during collection
```

Focused RED added for temporary-session handling:

```text
FAILED Tests/Agents/test_canvas_tool_provider.py::test_authenticated_canvas_provider_remains_available_in_temporary_session
```

The failure proved that the generic ephemeral third-party gate blocked even an
authenticated conversation-local Canvas provider. The registry now admits only
the exact authenticated Canvas owner through that gate.

Focused RED added for the independent enablement gate:

```text
FAILED Tests/Agents/test_canvas_tool_provider.py::test_disabled_canvas_provider_is_neither_advertised_nor_invoked
TypeError: CanvasToolProvider.__init__() got an unexpected keyword argument 'enabled'
```

Final GREEN:

```text
$ pytest -q Tests/Agents/test_canvas_tool_provider.py
................................                                         [100%]
32 passed, 1 warning in 0.42s
```

Focused catalog/approval/projection GREEN:

```text
$ pytest -q Tests/Agents/test_canvas_tool_provider.py Tests/Agents/test_tool_record_projection.py Tests/Agents/test_tool_catalog.py Tests/Agents/test_tool_catalog_owner_cache.py Tests/Agents/test_tool_catalog_concurrency.py Tests/Agents/test_builtin_tool_gate.py
........................................................................ [ 34%]
........................................................................ [ 69%]
................................................................         [100%]
208 passed, 1 warning in 1.13s
```

Focused Console composition GREEN:

```text
$ pytest -q Tests/Chat/test_console_agent_bridge.py -k 'compose_run_registry or temporary_run or invoke_by_name_refuses'
............................                                             [100%]
28 passed, 246 deselected, 1 warning in 1.13s
```

Static verification:

```text
$ ruff check tldw_chatbook/Agents/canvas_tool_provider.py Tests/Agents/test_canvas_tool_provider.py
All checks passed!
$ ruff format --check tldw_chatbook/Agents/canvas_tool_provider.py Tests/Agents/test_canvas_tool_provider.py
2 files already formatted
$ ruff check --select E9,F63,F7,F82 tldw_chatbook/Agents/tool_catalog.py tldw_chatbook/Chat/console_agent_bridge.py
All checks passed!
$ python -m py_compile tldw_chatbook/Agents/canvas_tool_provider.py tldw_chatbook/Agents/tool_catalog.py tldw_chatbook/Chat/console_agent_bridge.py Tests/Agents/test_canvas_tool_provider.py
$ git diff --check
```

The warning in pytest output is the environment's existing
`RequestsDependencyWarning` for its urllib3/chardet/charset-normalizer versions.

## Files changed

- `tldw_chatbook/Agents/canvas_tool_provider.py`
- `tldw_chatbook/Agents/tool_catalog.py`
- `tldw_chatbook/Chat/console_agent_bridge.py`
- `Tests/Agents/test_canvas_tool_provider.py`
- `.superpowers/sdd/2026-09-03-chatbook-canvas-implementation/task-3.2-report.md`

## Self-review

- Checked every brief acceptance item against implementation and focused tests.
- Verified Canvas schemas contain no model-supplied authority fields and reject
  additional properties before coordinator dispatch.
- Verified stale/disabled/missing/copied/lookalike authority fails closed at
  advertisement, ownership resolution, classification, and invocation.
- Verified temporary Canvas is permitted only for the authenticated Canvas
  provider; existing builtin approval and unrelated provider behavior remains
  covered by the focused suites.
- Verified the change does not instantiate a coordinator or commit revisions;
  those lifecycle responsibilities remain explicitly reserved for Task 3.3.
- Reviewed the complete diff for source-bearing error/log paths and unintended
  changes outside the four delivery files and this report.

## Concerns

- `Tests/Agents/test_trace_approval_capture.py::test_generic_tool_credentials_are_scrubbed_at_durable_agent_step_boundary`
  currently fails with `StopIteration` when run alone because the expected
  `credential_result` step is absent. This test and its AgentService/trace code
  are outside the Canvas diff; the focused builtin gate, catalog, projection,
  and Console composition suites above are green. The issue should be tracked
  separately rather than changing unrelated trace behavior in Task 3.2.
- Task 3.3 still must create and revoke the concrete per-session coordinator,
  connect service/staging finalization, and publish Canvas cards. Until then,
  the new tools are intentionally not supplied by the Console controller.
