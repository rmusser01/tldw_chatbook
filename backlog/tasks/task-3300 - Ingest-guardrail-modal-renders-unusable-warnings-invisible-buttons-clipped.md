---
id: TASK-3300
title: >-
  Ingest guardrail modal renders unusable — warnings invisible, buttons clipped
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 19:30'
labels:
  - library
  - ingest
  - ux
  - uat
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finding MI-01 (P0) of the 2026-08-07 Media Ingestion UX review (dual-agent live critique; tracking file `.impeccable/critique/2026-08-07-media-ingest-ux-options-review.md` in the main checkout). The Library ingest surface's only blocking consent dialog — `IngestGuardrailModal` (`library_screen.py:980`) — opened live as an empty black full-height column: "Some files may fail to import:" and the per-warning lines never rendered, because each warning sits in a bare `with Vertical():` (default `height: 1fr`) inside the `height: auto` modal, starving every Static (the exact "bare Container starving its sibling" defect DESIGN.md §7 names). The confirm label "Start import anyway" wraps to two lines inside `width: 14` while Cancel doesn't (misaligned baselines). The modal uses off-token `background: black; border: tall gray`. The warning line renders `({count} files)` — "(1 files)". Cancel carries `variant="error"` (a red safe action).

The user is asked to consent to a partially-doomed ingest they cannot read about — a "no hidden recovery states" violation at the surface's highest-stakes moment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 With one and with three tooling warnings, every warning line and its Copy-install-command button are visible inside a compact modal (no full-height empty band), verified by a rendered-geometry test that fails on the pre-fix CSS
- [x] #2 Both action buttons render their full labels on one line with aligned baselines
- [x] #3 Modal styling uses theme tokens (no `background: black` / `border: tall gray` literals)
- [x] #4 Warning count is grammatically correct for 1 vs many ("1 file", "2 files")
- [x] #5 Cancel is not styled as the destructive action; the confirm carries the action emphasis
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify the defect on the worktree (dev tip 023a04a48) with a mounted-modal test asserting warning Statics have nonzero rendered height — expect RED.
2. `height: auto` on the per-warning Verticals (or drop the wrapper); widen/auto-size the confirm button; replace color literals with `$panel`/`$surface` + border tokens; fix plural; swap variant emphasis.
3. Rendered-geometry assertions (region heights, label single-line) under a CSS-true harness; mutation-check by reverting the height rule and confirming RED.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD: six new tests were written first against the pre-fix code and all six
went RED on the exact defect each targets — the modal container filled the
full 32-row harness screen (`32 of 32 rows`, the live "empty black
full-height column"); `'Start import anyway' needs 19 columns but got 14`;
`(1 files)`; Cancel `variant="error"`; `black`/`gray` literals in
`DEFAULT_CSS`.

Fix (all in `IngestGuardrailModal`, `tldw_chatbook/UI/Screens/library_screen.py`):

- **Geometry (AC#1):** the per-warning `Vertical()` wrappers now carry
  `.ingest-guardrail-warning { height: auto }` — a bare `Vertical` defaults
  to `height: 1fr`, which inside the `height: auto` modal starved every
  Static to zero rendered height (DESIGN.md §7's "bare Container starving
  its sibling"). Added `max-height: 90%` on the container as a long-list
  backstop and `height: auto` on the actions row.
- **Buttons (AC#2):** `width: 14` → `width: auto; min-width: 10;
  margin-left: 1` so both labels render on one line, right-aligned with
  shared baselines.
- **Tokens (AC#3):** `border: tall gray; background: black` → `border:
  thick $primary; background: $surface` — the dominant repo modal chrome
  (21× `thick $primary`, e.g. `password_dialog.py`, `feedback_dialog.py`;
  136× `$surface`).
- **Plural (AC#4):** `({count} files)` → `file`/`files` on `count == 1`.
- **Variants (AC#5):** Cancel `variant="error"` → `"default"`; confirm
  stays `"primary"` — the repo-wide Cancel/confirm convention.

Tests: `Tests/UI/test_library_ingest_guardrail_modal.py` gained a
parametrized (1 and 3 warnings) rendered-geometry test (every warning
Static and copy button ≥ 1 rendered row, buttons not clipped below the
container, container height strictly below the screen and under a compact
bound), a single-line/aligned-buttons test, a plural test, a variant test,
and a no-color-literals CSS test. Mutation check: reverting
`.ingest-guardrail-warning` to `height: 1fr` sends both geometry tests RED
(modal balloons to 28 rows — caught by the compact bound even under
`max-height: 90%`); restored, all green.

Also repaired two pre-existing stale test helpers unrelated to this defect
(RED before this task's code change): `_minimal_library_screen()` in the
modal test file and the `__new__`-based screen in
`Tests/integration/test_library_ingest_flow.py::test_options_persist_to_config`
bypass `__init__` and did not seed `_library_ingest_preflight_generation`,
which submit now bumps → `AttributeError`. Both now seed it to 0.

Docs: `Docs/User_Guide/library/import-and-export.md` guardrail-dialog
paragraph updated ("1 file" singular) + a task-3300 "Verified against"
stamp.

Files: `tldw_chatbook/UI/Screens/library_screen.py`,
`Tests/UI/test_library_ingest_guardrail_modal.py`,
`Tests/integration/test_library_ingest_flow.py`,
`Docs/User_Guide/library/import-and-export.md`.

Final counts: `Tests/integration/test_library_ingest_flow.py` +
`Tests/UI/test_library_ingest_guardrail_modal.py` → 23 passed.

**xhigh review round 2 addendum (F3, 2026-08-08).** The round-1 fix made
each warning row render, but the modal container itself was still a plain
Vertical: with 5+ warnings it grew past `max-height: 90%` and the
Cancel/"Start import anyway" row rendered off-screen (measured: buttons at
y=40 on a 24-row screen), unreachable by mouse. The warning list now lives
in a `VerticalScroll` (`#ingest-guardrail-warnings`) between the pinned
title and the pinned action row. One trap worth recording: `max-height:
1fr` on the scroll resolves against the modal's FULL inner height without
subtracting the pinned siblings (measured: the scroll took all 17 inner
rows and still pushed the actions off), so the row budget is computed in
`on_mount` (90% of screen minus the 9 chrome rows: border 2 + padding 2 +
title 1 + actions 3 + margin 1). Tests: a 7-warning/24-row reachability
test (both action buttons' regions fully inside the screen) and an
overflow test (`max_scroll_y > 0`, title above/actions below the scroll).
Mutation check: neutralizing the on_mount cap (budget=1000) sends both
RED; restored, the modal file is 19 passed.
<!-- SECTION:NOTES:END -->
