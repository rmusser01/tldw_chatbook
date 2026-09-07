---
id: TASK-16313
title: 'Trajectory screen with ledger, inspector, search'
status: Done
assignee: []
created_date: '2026-08-15 00:20'
updated_date: '2026-08-15 05:48'
labels: []
dependencies:
  - TASK-16312
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console-launched screen: DataTable ledger with turn grouping/collapse, per-record inspector (tokens incl. cache read/write, timing, tool payload), search, ADR-031 keybindings/footer. Plan task 4 in Docs/superpowers/plans/2026-08-14-console-trajectory-view.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ledger renders snapshot with collapse,Inspector shows usage+timing+payload,Search filters rows,ADR-031 footer governance passes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented in commit 45459ff50: trajectory screen with ledger/collapse/inspector/search, modal Static footer hints per video-player precedent; 22 tests green; see Implementation Notes and ADR-066

- **Approach**: read-only `TrajectoryScreen(ModalScreen[None])` rendering a
  `TrajectorySnapshot` from the pure projection (ADR-066,
  `backlog/decisions/066-console-trajectory-view-and-trace-metadata.md`); the screen
  never queries the DB itself.
- **Key files**: `tldw_chatbook/UI/Screens/trajectory_screen.py` (new);
  `Tests/UI/test_trajectory_screen.py` (22 pilot-driven tests fed by the real
  `derive_trajectory`, green — ledger/collapse, inspector contents, search, pagination,
  worker threshold, ADR-031 governance).
- **Decisions**: DataTable ledger (`cursor_type="row"`, 8 columns) with turn-header
  rows and indented nested tool rows; `t` collapse, `i` inspector (live-follows cursor),
  `e` load-earlier (dsh-style: newest `PAGE_SIZE = 500` page first, control row at top),
  `/` search (matches content/kind/model/provider AND tool payload name/args/result —
  tool output lives only in the payload; an active query reveals collapsed turns);
  safe two-stage `escape` (blur search first). Durations are computed only between two
  PROVIDED endpoints — never fabricated. Above `WORKER_THRESHOLD = 5000` records the
  initial render moves to a worker with a generation counter + `_alive` guard against
  stale/late results. **Footer hints are the modal's own one-line Static**, per the
  video-player precedent (`video_player_screen.py` hints line) — NOT
  `register_footer_shortcuts` — and stay exactly 1:1 with non-escape `BINDINGS`
  (ADR-031 governance tests enforce this); the `e earlier` hint drops when exhausted.
- **Deviations**: none of note; the screen was committed self-contained (launch wiring
  is task-16314). The stale-worker test exercises the generation guard at the seam
  because the pilot cannot deterministically interleave keystrokes with a worker build.
<!-- SECTION:NOTES:END -->
