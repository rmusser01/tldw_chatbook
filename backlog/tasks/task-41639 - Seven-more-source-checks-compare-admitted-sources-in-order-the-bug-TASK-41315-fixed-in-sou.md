---
id: TASK-41639
title: "Seven more source checks compare admitted sources in order, the bug TASK-41315 fixed in sources_changed()"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- agents
- console
- mcp
priority: high
references:
- tldw_chatbook/Agents/legacy_hook_ownership.py
- tldw_chatbook/Chat/console_plugin_hook_preparation.py
- tldw_chatbook/Chat/console_reference_preparation.py
- tldw_chatbook/Chat/console_workflow_runs.py
dependencies:
- TASK-41315
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41315 found that every Console agent turn with a context-returning MCP hook was refused before its first model call. `sources_changed()` compared the admitted sources as an ordered sequence, and the same set captured in a different order read as a change. The fix compares them as a set. The same `captured_sources(...) != sources` comparison remains here:
- `Agents/legacy_hook_ownership.py:205`: likely live (same refusal path)
- `Chat/console_plugin_hook_preparation.py:177`: plausibly live
- `Chat/console_reference_preparation.py:312`: plausibly live
- `Chat/console_workflow_runs.py:241`, `:390`, `:947`, `:2263`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each site compares admitted sources without depending on order, or a comment there says why order matters
- [x] #2 A test per live site captures the same sources in a different order and asserts no refusal; red before the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (applies TASK-41315's existing set comparison to seven more call sites; no new contract)

1. Give `Agents/activation.sources_changed()` an optional `service` (the workflow checks observe the agent bridge, not `None`).
2. Route all seven checks through it: `legacy_hook_ownership._EventSource.check`, `console_plugin_hook_preparation._Preparation.check`, `console_reference_preparation._ReferenceRead.check`, and the four `console_workflow_runs` checks (remaining acceptance read and settle, metadata read, native control).
3. Find which sites can see the same sources in another order: instrument `sources_changed` (temporarily) and run each site's suites; read the capture and check contexts.
4. Tests in `Tests/Chat/test_console_agent_storage_sources.py`: a reordered capture is no refusal at each site that can be driven alone, a removed source still refuses, `service` is honoured, and a census that no ordered `captured_sources(...) ==/!=` comparison remains. Show red on the unfixed code.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fix.** `Agents/activation.sources_changed()` takes an optional `service` (the workflow checks observe the agent bridge), and all seven checks now call it, so each compares the admitted (owner, path) set: `legacy_hook_ownership._EventSource.check`, `console_plugin_hook_preparation._Preparation.check`, `console_reference_preparation._ReferenceRead.check`, and in `console_workflow_runs` the remaining-acceptance read (241) and settle (390), the metadata read (947) and the native control (2263). Order carries no meaning at any of them (admission is one lease per path), so no site keeps an ordered check. Set inequality implies tuple inequality, so the change can only remove refusals. This is defensive hardening: no refusal caused by order was reproduced at any of the seven sites (see below).

**Which sites are live.** A temporary probe in `sources_changed` (removed before commit) recorded, per caller, whether the two tuples were equal, a reorder of the same set, or a different set. Over the hook, plugin, reference and storage-source suites, and six workflow suites: legacy 236, plugin 235, reference 252, workflow 241: 24, 947: 3793, 2263: 10 checks, all equal; 390 was not reached. The same probe recorded 42 reorders at the already-fixed `native_commands.py:362`, so it does see reorders where they happen. The structural reason: each of the seven checks runs either in the capture's own admission, or inside `execution(sources=captured)` entered with no inherited admission (pool threads have none; a copied context refuses a foreign admission at entry), so the capture leads the admitted tuple and de-duplication keeps its order. Order alone can still differ in two narrow cases: a raw (unadmitted) capture holding a duplicate (owner, path), which admission collapses (workflow 241/390/947/2263); and a legacy notification prepared under one admission and checked on the notify worker against a fresh observation. No site was shown live in the journeys.

**What is not covered.** The four workflow checks have no behaviour test: they sit in closures inside `_accept_remaining_workflow_phase`, `_metadata` and `_native_control`, whose journeys run in child processes, and a test there would need a journey that forces a duplicate source. Only the census pins them. The census does not follow an indirect form, where a `captured_sources(...)` result is assigned to a name and that name is compared on a later line.

**Tests** (`Tests/Chat/test_console_agent_storage_sources.py`, next to TASK-41315's test):
- `test_source_check_accepts_reordered_admission[...]`: the three sites that can be driven alone (built without `__init__`, their other checks stubbed) accept the same sources captured in reversed order and still refuse a removed source.
- `test_sources_changed_observes_the_named_service`: `service` is honoured.
- `test_no_source_check_compares_captured_sources_in_order`: an AST census (`_ordered_source_checks`) of every `tldw_chatbook/` module that names `captured_sources`. It flags an `==`/`!=` comparison when either side contains a `captured_sources` call (by name, as an attribute, or through a `from ... import captured_sources as` alias), except inside `set(...)`/`frozenset(...)`, the order-free form `sources_changed` itself uses. `test_ordered_source_census_flags_each_form[...]` plants each form: a comparison wrapped across lines, nested parentheses, a qualified call on the right, an aliased import, `tuple(...)` around the call, and `_guard.captured_sources(...) ==`. `test_ordered_source_census_passes_the_set_forms` is the control (`sources_changed(...)`, the `set(...)` comparison and a plain assignment are not flagged). Run with the test's own function over `git show` of each file: dev d0486978dd gives exactly the seven sites, head gives none, and 62c4cc0762^ also gives the two `native_commands.py` checks TASK-41315 fixed.

**Red/mutation.** With the product diff taken out (patch out, run, patch in): all 5 new nodes failed (the three site refusals, `TypeError` for `service`, the census listing all seven sites). With only the 947/2263 checks reverted and `service` ignored: the census named those two lines and the service test failed. With the 2263 check rewritten as Black would wrap it (`captured_sources(bridge)` and `!= sources` on separate lines), the AST census names it; the first line-regex census would have missed it.

**Regression runs (load 55-90 on 18 CPUs).** Hook, plugin, reference and storage-source suites at `-n 4`: 252 passed, 10 failed (plugin preparation and reference custody). The same two files rerun serially on head and on dev d0486978dd: identical 10 failures on both. `test_native_legacy_hook_handoff.py`: 4 failed, 1 passed on both head and base (not laned; reported as a follow-up). Workflow suites (probe active): 34 passed, 23 failed; three of the failing nodes rerun on head and base: the same 2 failed (timeout in `wait_for_status`), 1 passed, on both. `Tests/Architecture` serially: 1476 passed, 5 skipped, 3 failed (`test_persistent_diagnostic_inventory::test_reviewed_diagnostic_changes_are_metadata_only`, TASK-41038; `test_timer_path_static_update_inventory` x2); the same 3 fail on dev d0486978dd. `./scripts/preflight.sh`: all checks passed.

**Review round.** The census became AST-based (above); the three new imports in `console_workflow_runs.py` are wrapped under 88 columns. `test_console_agent_storage_sources.py` at `-n 4`: 38 passed.

Files: `tldw_chatbook/Agents/activation.py`, `Agents/legacy_hook_ownership.py`, `Chat/console_plugin_hook_preparation.py`, `Chat/console_reference_preparation.py`, `Chat/console_workflow_runs.py` (import lines and the four comparison lines only), `Tests/Chat/test_console_agent_storage_sources.py`.
<!-- SECTION:NOTES:END -->
