---
id: TASK-41631
title: "The Console live-work MCP label says \"Not wired\" instead of the unbound-store reason"
status: Done
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- console
- mcp
- ux
priority: low
references:
- tldw_chatbook/Chat/console_live_work.py
dependencies:
- TASK-41308
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** When the MCP store is unbound, the Inspector row now names the reason (TASK-41308), but the live-work summary in `Chat/console_live_work.py:378` still sets `mcp_label = "Not wired"`. The two surfaces disagree, and "Not wired" suggests a setup step the user can take, when the actual cause is that the profile's MCP settings file can't be used.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With the MCP store unbound, the live-work MCP label carries the same reason as the Inspector row
- [x] #2 Other "Not wired" cases (no store problem) keep their current label
- [x] #3 A test covers both cases
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (display copy; same reason string and predicate as TASK-41308's Inspector row).

1. Red tests first: a unit row in `Tests/UI/test_console_live_work_handoffs.py` (`from_acp_runtime_status(..., mcp_unavailable_reason=...)` gives `MCP: Off - <reason>`; empty reason keeps `Not wired` / `Connected`), and a mounted row in `Tests/UI/test_console_send_without_mcp_store.py` (bound vs unbound profiles; reads the card the builder makes and the row the sync tick rewrites).
2. `Chat/console_live_work.py`: `from_acp_runtime_status` takes `mcp_unavailable_reason: str = ""`; a non-empty reason wins over the counts, as `_mcp_inspector_row` does. `UI/Screens/chat_screen.py`: both callers (card builder, sync tick) pass `console_mcp_unavailable_reason(read_mcp_service(self.app_instance))` with lazy imports, no new method.
3. Mutation checks (drop either pass-through -> red). Update tests that hand-publish MCP counts on an unbound bootstrap app, and the docs that quote the row.
4. Covering files base vs head, preflight, `Tests/Architecture` serially.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Change.** `ConsoleLiveWorkSourceReadinessState.from_acp_runtime_status` takes `mcp_unavailable_reason` (default ""). A non-empty reason gives `MCP: Off - <reason>` (unavailable class) and wins over the counts, as `_mcp_inspector_row` does. An empty one keeps "Not wired" / "Connected". Both writers in `ChatScreen`, the card builder and the sync tick, pass `console_mcp_unavailable_reason(read_mcp_service(self.app_instance))`, the Inspector row's pure predicate. They use lazy imports and add no method. The reason string is now one module constant, `MCP/console_snapshot.UNBOUND_MCP_STORE_REASON`. The Inspector row, this row, the hub rail (TASK-41630) and the tests all read it.

**Decision.** The reason wins over a non-zero count. An unbound store never publishes counts in production, because the Send drops MCP. The two surfaces therefore cannot disagree even when a test hand-publishes counts. Review round: the reason was reworded once, for every surface, to "Console can't use the MCP settings file". The old "this profile's MCP settings file can't be used here" read wrong in the hub, where that file is edited, and was longer. Measured with the Inspector open, unbound vs bound:

| Size | Row content width | Unbound MCP row | Card rows, bound -> unbound |
|---|---|---|---|
| 235x52 | 36 columns | 2 lines | 16 -> 17 |
| 200x60 | 29 columns | 2 lines | 17 -> 18 (19 with the old copy) |
| 160x45 | 23 columns | 3 lines | 18 -> 19 |

All stay within the live-work section's 20-row cap, with no "more" hint. `test_unbound_live_work_mcp_row_fits_the_section_at_full_screen` pins the 235x52 row: 2 lines, a 17-row card, the cap of 20 and no hint. Its mutation check, a reason that wraps to 3 lines, fails it. The existing swap-geometry tests pin a bound store (commit a891153852).

**Tests.**
- `Tests/UI/test_console_live_work_handoffs.py::test_console_live_work_mcp_row_names_the_unbound_store_reason[None|0|4]`: the unit case.
- `Tests/UI/test_console_send_without_mcp_store.py::test_live_work_mcp_row_names_the_unbound_store_reason[bound|no-pinned-io|non-default-data-dir]`: mounted. It reads the row the builder makes and the row the sync tick rewrites. Bound keeps "Not wired".
- On base product code (with only the constant added so the tests import), 5 fail and the bound case passes. On head all 6 pass.
- Mutation checks on head:
  - Drop the tick pass-through: 2 failed (`ticked`).
  - Drop the builder pass-through: 2 failed (`built`).
  - Drop the reason branch: 3 unit cases failed.
- Test drift from this change (commit a891153852): `make_console_pilot` (`test_console_right_rail.py`), the Nielsen closeout replay and `test_console_renders_source_readiness_summary_without_pending_launch` built unbound apps while pinning "Not wired", "Connected" or card geometry. They now build at the profile's own data folder, where the store binds as it does by default in the app. The right-rail and Nielsen files are red on dev for the bootstrap-profile trap (`RecoveryRequired: raw_source_selection_changed`). With `bootstrap_profile` added locally (not committed), 4 rail nodes and the Nielsen node failed on this branch before the drift fix and pass after it (5 passed); base passes them.

**Covering runs, base d0486978dd vs head, same 8 files, `-n 4`, load 50-100.** The files are right_rail, collapsed_inspector_send_layout, nielsen, live_work_handoffs, recovery_docs, send_without_mcp_store, settings_tools_hub and mcp_rail. Base: 49 failed, 140 passed. Head: 41 failed, 150 passed.
- Head-only reds were 3:
  - `test_console_renders_source_readiness_summary_without_pending_launch`: real drift, now fixed.
  - `test_an_inspector_opened_after_chat_switches_matches_one_open_throughout`: passes alone; it uses a bound app.
  - `test_project_status_remains_visible_in_real_thirty_column_rail[size1]`: also red on base.
- The shared reds are the bootstrap trap (right_rail 33) and dev reds.

Preflight passed. `Tests/Architecture` was run serially; see TASK-41630.

**Files.** `tldw_chatbook/Chat/console_live_work.py`, `tldw_chatbook/MCP/console_snapshot.py`, `tldw_chatbook/UI/Screens/chat_screen.py` (+14 lines, no method, within the 24592 row), `Tests/UI/test_console_live_work_handoffs.py`, `Tests/UI/test_console_send_without_mcp_store.py`, `Tests/UI/test_console_right_rail.py`, `Tests/UI/test_unified_shell_phase6_nielsen_closeout.py`, `Docs/User_Guide/console/context-and-rag.md`, `Docs/Development/release-recovery-setup.md`.
<!-- SECTION:NOTES:END -->
