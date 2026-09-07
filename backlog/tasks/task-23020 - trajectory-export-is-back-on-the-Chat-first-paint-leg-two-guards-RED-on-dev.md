---
id: TASK-23020
title: >-
  trajectory_export is back on the Chat first-paint leg - two guards RED on dev
status: Done
assignee:
  - '@claude'
created_date: '2026-08-27'
labels:
  - performance
  - startup
  - regression
  - dev-red
priority: high
---

## Description

`Chat.trajectory_export` is resident at `_ui_ready` again, breaking a guarantee that shipped ~24
hours earlier. Two guards are red on pristine dev, so every branch inherits them.

Three module-scope edges reach it from the Chat mount leg, each importing **one name**
(`TraceExportProfile`, a three-member `str, Enum`) that drags 1,463 LOC plus `Chat.trajectory`.
**Fixing one edge buys nothing** — all must break.

`chat_screen.py:52-57` carries a comment explicitly forbidding this; the change routed around it
through a file the comment does not name.

## Acceptance Criteria

- [x] `Tests/Performance/test_ui_ready_module_census.py::test_ui_ready_module_census_stays_at_the_pinned_size` passes
- [x] `Tests/Packaging/test_rag_boot_import_closure.py::test_chat_screen_import_does_not_execute_the_deferred_packages` passes
- [x] All three edges are broken, verified by an import tracer recording `(importer, imported)` — not by grep
- [x] The export dialogs still work; a test drives them from the deferred state
- [x] Neither guard is relaxed to accommodate the regression
- [x] The guard names the offending edge well enough that the next person does not have to trace it

## Evidence

```
UI/Screens/chat_screen.py:448
  -> Widgets/Console/console_conversation_inspector.py:114
    -> Widgets/Console/console_exchange_export_dialog.py:22 -> Chat/trajectory_export.py
                                     :25 -> Widgets/Console/trace_export_dialog.py:16,17
    -> Chat/console_exchange_export.py:18 -> Chat/trajectory_export.py
```

Introduced by `c6218918d1` (#2126). Arrives on the **mount** leg, not the import leg, which is why
the import-weight guard stays green. Import self-time 1.67-1.98 ms; the cost is the contract, not
the milliseconds.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.

## Implementation Plan

1. Confirm both guards RED on pristine `d7bb844d9b` (they were: both name
   `tldw_chatbook.Chat.trajectory_export`).
2. Re-trace the edges with an `__import__`-wrapping tracer recording
   `(importer, imported)` on BOTH legs (bare `chat_screen` import; headless
   boot to `_ui_ready`), not grep -- dev had moved ~90 commits since the
   review's trace.
3. Break every edge by relocating the shared vocabulary into light leaves
   re-imported by the heavy side (the `search_modes`/
   `chunking_engine_version` pattern), so no edge fix depends on the others
   and no copy can drift.
4. Extend the guards to NAME the breach route per file; add per-edge closure
   tests plus a deferred-state functional test of the export dialog.
5. Mutation-test by restoring one edge at a time (four mutants) and
   recording which tests kill each.

## Implementation Notes

**Edges found by the tracer on `d7bb844d9b`** (same set on the import and
mount legs; the third differs from the review's line numbers only because
dev moved):

1. `Chat/console_exchange_export.py:18` -> `Chat.trajectory_export`
   (`TraceExportProfile`, the FIRST-load edge)
2. `Widgets/Console/console_exchange_export_dialog.py:22` ->
   `Chat.trajectory_export` (same single name)
3. `Widgets/Console/console_exchange_export_dialog.py:25` ->
   `Widgets/Console/trace_export_dialog.py`, whose module scope imports the
   whole exporter (edge 4: `trace_export_dialog:16,17` ->
   `Chat.trajectory` + `Chat.trajectory_export`)

All ride `console_conversation_inspector` (module-scope import at
`chat_screen.py:448`) onto the Chat first paint.

**Shape chosen: relocation into two light leaves, re-imported by the heavy
side.**

- `Chat/trace_export_profiles.py` (new, stdlib-only): `TraceExportProfile`
  moved here verbatim; `trajectory_export.py` re-imports it, so the deferred
  family's import sites keep working and the enum stays ONE object.
- `Widgets/Console/trace_export_profile_ui.py` (new, light):
  `TRACE_EXPORT_PROFILE_COPY` / `TRACE_EXPORT_PROFILE_LABELS` /
  `full_trace_confirmation` moved here; `trace_export_dialog.py` re-imports
  (and keeps re-exporting) them.
- The two chat-leg modules import only the leaves; deferring to function
  scope was rejected because the failure mode is exactly a future
  module-scope one-liner -- the leaf shape gives the correct import an
  obvious home, and the re-import direction makes the seam-level wrong move
  (`trace_export_profile_ui` importing from `trace_export_dialog`) a loud
  circular-import failure rather than a silent re-eager.
- A side effect worth naming: the exchange-export flow now never resolves
  the trajectory engine at all -- `trace_export_dialog` also left the leg
  (it was resident pre-fix), so the leg swapped 2 heavy modules for 2 leaves
  (mount census 958 -> 957, budget 970).

**Guard naming (AC 6):** `chat_screen.py`'s forbidding comment now names
the transitive route and the two leaves; the closure guard's
`CHAT_LEG_DEFERRED_MODULES` comment documents the breach route and points
at the per-edge guard; new
`Tests/Packaging/test_exchange_export_trajectory_deferral.py` checks each
file individually so a red names the offender: per-edge closure (x3
modules), leaf-lightness (the leaves cannot re-grow the edge), one-object
identity across both sides, and a subprocess deferred-state test driving
the real dialog (mount, profile copy render, Full confirmation, projection,
clipboard disclosure) with the engine asserted absent before/during/after.
The existing chat-screen closure guard gained an on-demand identity
assertion; neither target guard was relaxed (no budget, list, or assertion
weakened -- both diffs are comment/anti-vacuity additions only).

**Verification.** Tracer on both legs post-fix: zero edges into the family;
only the three legitimate `Chat.trajectory` importers remain
(`console_chat_store`, `agent_service`, `review_selection`). Both target
guards green (census: 1 passed, 7.7s; closure file: 6 passed, 5.2s). Wider
run: 214 passed across the trajectory/exchange test files + 44
inspector/closure, 155-passed packaging/performance/architecture sweep.
Mutation results (one mutant per edge, implementation committed first,
restores via `git restore`):

- **M-A** (`console_exchange_export` -> trajectory_export): killed by 5 --
  all 3 per-edge tests, the deferred-state dialog test, the leg-wide
  closure guard; census guard also red (mount-leg net proven live).
- **M-B** (dialog -> trajectory_export): killed by 4 (dialog + inspector
  per-edge, deferred-state, leg-wide guard).
- **M-C** (dialog -> trace_export_dialog for the copy): killed by the same
  4 -- `trace_export_dialog` is itself in the per-edge FORBIDDEN set.
- **M-D** (`trace_export_profile_ui` -> trace_export_dialog): killed by 6,
  including its dedicated killer `test_the_replacement_leaves_are_
  themselves_light`, via the deliberate circular-import failure.

**Files.** New: `tldw_chatbook/Chat/trace_export_profiles.py`,
`tldw_chatbook/Widgets/Console/trace_export_profile_ui.py`,
`Tests/Packaging/test_exchange_export_trajectory_deferral.py`. Modified:
`Chat/trajectory_export.py`, `Chat/console_exchange_export.py`,
`Widgets/Console/console_exchange_export_dialog.py`,
`Widgets/Console/trace_export_dialog.py`, `UI/Screens/chat_screen.py`
(comment), `Tests/Packaging/test_rag_boot_import_closure.py` (comments +
identity assertion), two exchange-side tests repointed at the leaf.

Out of scope, found while verifying: two pre-existing architecture reds on
pristine `d7bb844d9b` (`test_chat_screen_remains_within_reviewed_projection_
without_ratchet_raise` pins ChatScreen at 17,727 lines vs 17,013 actual;
`test_closeout_evidence_explains_the_remaining_absolute_deficit`), plus the
already-documented `test_rag_citation_provenance_benchmark.py` reds.
