---
id: TASK-2740
title: Test Tool panel mount crashes for an all-boolean argument schema
status: Done
assignee:
  - '@claude'
created_date: '2026-08-06 23:45'
updated_date: '2026-08-07 01:27'
labels:
  - mcp
  - ui
  - crash
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during PR-T3 fix round I (the first draft of
`test_toggling_an_argument_checkbox_disarms_pending_confirm` used an
all-boolean schema and crashed the panel mount — the test was re-pointed at a
mixed schema and this defect filed instead of fixed, since it predates the
round and is outside its items).

`MCPInspector._mount_test_tool_panel()`'s focus code
(`UI/MCP_Modules/mcp_inspector.py`, the F-056 block right after
`await container.mount(panel)`) runs:

    first_control = panel.query("Input, Select, TextArea").first()
    if first_control is None:
        first_control = panel.query_one("#mcp-inspector-test-close", Button)

Two stacked defects:

1. The query omits `Checkbox`, so a tool whose schema renders ONLY boolean
   fields (e.g. `{"type": "object", "properties": {"verbose": {"type":
   "boolean"}}}` — `MCPSchemaForm` mounts one `Checkbox` and nothing else)
   matches zero nodes.
2. `DOMQuery.first()` RAISES `NoMatches` on an empty result — it never
   returns `None` — so the `is None` fallback to the Close button is dead
   code, and the comment's promise ("the Close button otherwise") never
   happens.

The exception escapes `_mount_test_tool_panel()`, which runs as a worker
(`run_worker(..., group="mcp-inspector-test-panel", exclusive=True)`) with
default `exit_on_error` — an unhandled worker error, i.e. opening Test Tool
on an all-boolean tool takes down the app, live. Reproduced in a Textual
`run_test` harness 2026-08-06: `NoMatches("No nodes match <DOMQuery
query='Input,Select,TextArea'> on Vertical(id='mcp-inspector-test-panel')")`
out of the worker.

Design decision needed alongside the fix: should the first Checkbox receive
focus (add `Checkbox` to the query), or should an all-boolean form focus the
Close button (make the fallback real via `.first()` guarded by a truthiness
check on the query result)? Either way the dead `is None` branch must become
reachable or be removed — it currently documents behavior that cannot occur.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Opening the Test Tool panel for a tool whose schema renders only Checkbox controls does not raise and does not kill the mount worker
- [x] #2 Focus lands on a deliberate target for the all-boolean case (first Checkbox or Close button — decided, not accidental) and a test pins it
- [x] #3 The dead `is None` fallback is either made reachable or removed; no comment promises a fallback that cannot fire
- [x] #4 A regression test mounts the panel with an all-boolean schema (the exact scenario that crashed) and passes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: regression tests — all-boolean schema and empty-properties schema (the live list_characters shape) both mount the Test Tool panel without crashing; focus pins: first Checkbox for the boolean form, Close button for the zero-control form. 2. GREEN: add Checkbox to the focus query; replace the dead .first()/is-None fallback with a truthiness-guarded real fallback. 3. Mutation: re-drop Checkbox from the query and re-break the fallback, confirm each test reds.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in `_mount_test_tool_panel()` (mcp_inspector.py): `Checkbox` joins
the F-056 focus query, and the `.first()` call is guarded by DOMQuery
truthiness — the old `is None` fallback was dead code because `.first()`
raises `NoMatches` on an empty result, and the raise escaped a
default-`exit_on_error` worker (app crash). Investigation upgraded the
severity: the real built-in `list_characters` ships empty `properties`,
so the crash needed no exotic schema. Focus decisions per AC #2: first
Checkbox for a boolean form (F-056's "first form control" contract);
Close button for a genuinely zero-control form (the fallback, now
reachable and pinned). Tests: `test_test_panel_mounts_for_an_all_boolean_
schema_and_focuses_the_checkbox`, `test_test_panel_mounts_for_a_zero_
control_schema_and_focuses_close`; each mutation-verified independently.
Commit bfd5d38d9 on fix/mcp-inspector-honesty-residue.
<!-- SECTION:NOTES:END -->
