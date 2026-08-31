---
id: TASK-18918
title: Add paged recovery viewing to Library Media Trash
status: Done
assignee: []
created_date: '2026-08-15 02:51'
updated_date: '2026-08-31 15:11'
labels:
  - library
  - pagination
  - media-trash
  - follow-up
dependencies:
  - TASK-18912
  - TASK-18913
  - TASK-18914
  - TASK-18915
  - TASK-18916
references:
  - >-
    Docs/superpowers/specs/2026-08-14-library-top-level-source-pagination-design.md
  - >-
    Docs/superpowers/specs/2026-08-30-task-18918-library-media-trash-paging-design.md
  - Docs/superpowers/plans/2026-08-30-task-18918-library-media-trash-paging.md
  - Docs/superpowers/specs/2026-08-30-library-media-return-settlement-design.md
  - Docs/superpowers/plans/2026-08-30-library-media-return-settlement.md
  - backlog/decisions/067-library-top-level-pagination-contracts.md
  - backlog/decisions/104-library-media-return-settlement-boundary.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make every deleted Media item reachable in the nested Trash recovery surface through bounded pages while preserving restore, permanent-delete, selection, and recovery semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Media Trash exposes coherent exact-total bounded pages with deterministic ordering and complete-source filtering before slicing.
- [x] #2 Restore and permanent delete reconcile or relocate the affected stable ID truthfully, clamp emptied pages, and never misreport a committed mutation as failed.
- [x] #3 Trash selection cannot remain invisibly active across page or scope changes, and stale refresh failures disable destructive actions until authoritative recovery.
- [x] #4 Loading, empty, failure, Retry, focus, back navigation, and narrow-terminal pager behavior match the established Library pagination convention.
- [x] #5 Request generations, unmount fencing, malformed envelopes, concurrent shrink, and privacy-safe diagnostics have regression coverage.
- [x] #6 Automated database/service/state and mounted Textual tests plus isolated live verification with more than 40 synthetic Trash records pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a coherent local-only database page/count/facet contract.
2. Propagate and canonically validate the exact envelope through Media services.
3. Add immutable Trash paging state plus a Trash-specific request controller.
4. Wire screen entry, paging/filter generations, Back receipt, and lifecycle fencing.
5. Render the bounded pager/filter/confirmation surface at all supported sizes.
6. Reconcile Restore and permanent deletion through the shared Media mutation owner.
7. Run focused automated/live verification, review, documentation, and closeout.

ADR required: yes

ADR paths: `backlog/decisions/067-library-top-level-pagination-contracts.md`, `backlog/decisions/104-library-media-return-settlement-boundary.md`

Reason: ADR-067 governs exact source-owned pages and stale mutation recovery; ADR-104 records the cross-module, event-driven Media return-settlement boundary required by the production-shaped cross-reader gate.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented local Media Trash as an independent, exact-total recovery surface:
the database owns one coherent count/page/facet snapshot; the local and scoped
services preserve and validate the canonical envelope; immutable state and a
Trash-only controller own filtering, generations, Retry, clamping, selection,
and request fencing; and the mounted Library surface renders 20-row pages with
Restore and confirmed permanent deletion through the existing Media mutation
interlock. A committed Restore removes the Trash row immediately and marks the
retained normal-Media page stale instead of inserting an unranked item.

The production-shaped Task 7 gate exposed a separate cross-reader return race:
same-size recomposition could retain the wrong reader presentation and clamp an
otherwise correct Media return receipt. The ADR-104 amendment projects the
Media presentation before layout, waits for producer-owned Resize evidence from
the current `LibraryMediaRowScroll`, and fences settlement by route, owner,
presentation epoch, content/layout revision, focus authority, and deadline.
Viewer returns finish on the semantic row; Trash returns finish on the captured
normal-Media control. Fixed callback-turn retries were rejected because they
remained nondeterministic in fresh processes.

Closeout preserved the original one-line Task 7 fake compatibility repair
(`_library_notes_focus_intent_generation=0`). The mounted gate then revealed two
additional stale test-fake seams, so the fake now supplies the production route-
change disarm callback and binds the return-candidate method. The live harness
was reviewed rather than recreated. Two initial harness corrections were
necessary: use the receipt's shipped `final_focus_identity` field, and exercise
viewer return before Restore because the retained page is intentionally stale
and action-gated afterward.

Quality review found five false-proof seams in that first closeout. Permanent-
delete start/success/failure logs now emit fixed operation/status/count metadata
and exception category only; they omit backing IDs, database paths, raw
exceptions, and tracebacks. A real file-backed success test and a trigger-
injected SQLite failure test cover both Loguru and stdlib logging. The live gate
now proves every SQLite target against explicit pytest/HOME/app-scratch roots
before open instead of reading real-profile bytes, asserts exact pane/content/
scrollbar/row geometry rather than positive widths, starts viewer evidence from
a distinct auto-selected row with no private pre-seed, and gives every size a
fresh event loop with a one-worker connection owner followed by zero-handle
DB/WAL/SHM verification. Painted-copy evidence uses Textual's public SVG
screenshot exporter rather than the private compositor.

The final geometry review also removed an observed-state oracle: exact sizes
now have immutable expected postures before the app mounts. The walkthrough
asserts **160×50** is wide before running its split/collapse-delta branch and
asserts **120×35**, **100×30**, and **80×24** are compact before running their
Items-priority/exclusive-optional-pane branch. Branch selection, labels, and the
final ordered `(size, layout_contract)` aggregate all come from that fixed
oracle, so a posture regression cannot relabel itself and pass.

Production and focused owners modified across the original work and ADR-104
amendment:

- `tldw_chatbook/DB/Client_Media_DB_v2.py`
- `tldw_chatbook/Media/local_media_reading_service.py`
- `tldw_chatbook/Media/media_reading_scope_service.py`
- `tldw_chatbook/Library/library_media_state.py`
- `tldw_chatbook/UI/Library_Modules/library_media_trash_browse_controller.py`
- `tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py`
- `tldw_chatbook/UI/Screens/library_screen.py`
- `tldw_chatbook/Widgets/Library/library_media_trash_canvas.py`
- `tldw_chatbook/Widgets/Library/library_media_canvas.py`
- `Tests/DB/test_client_media_trash_pagination.py`, the focused Media/Library
  state and controller suites, `Tests/Media_DB/test_media_db_v2.py`,
  `Tests/UI/test_library_media_trash.py`,
  `Tests/UI/test_library_media_return_settlement.py`, the existing cross-reader
  suites, and `Tests/Live/test_library_media_trash_paging_closeout.py`
- `Docs/User_Guide/library.md` and
  `backlog/docs/lessons-testing-evidence.md`

### Verification evidence

- Pure paging/mutation gate:

  ```bash
  PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-18918-media-trash-paging /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/DB/test_client_media_trash_pagination.py Tests/Media/test_local_media_reading_service.py Tests/Media/test_media_reading_scope_service.py Tests/Library/test_library_media_trash_state.py Tests/UI/test_library_media_trash_browse_controller.py Tests/UI/test_library_media_browse_controller.py -k 'media_trash or library_media_trash or permanently_delete_media_item or mark_stale_after_trash_restore'
  ```

  Passed **108**, deselected **191**, and reported **1 warning in 4.84s**;
  the historical count is unchanged.
- Focused privacy and live-evidence mutations:

  ```bash
  PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-18918-media-trash-paging /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Live/test_library_media_trash_paging_closeout.py::test_sqlite_path_authority_rejects_escape_before_open Tests/Live/test_library_media_trash_paging_closeout.py::test_reader_width_contract_rejects_one_column_mutation Tests/Live/test_library_media_trash_paging_closeout.py::test_viewer_target_contract_rejects_cleared_row_selection Tests/Media_DB/test_media_db_v2.py::test_permanent_delete_success_logs_only_fixed_metadata Tests/Media_DB/test_media_db_v2.py::test_permanent_delete_sqlite_failure_logs_category_without_private_values --disable-warnings --basetemp=/private/tmp/task18918-fast-green
  ```

  The two privacy nodes first failed against identifier/path/raw-exception
  logging, then passed after the production fix. The combined final gate passed
  **5 with 2 warnings in 1.93s**. Its path mutation proves rejection before the
  fake opener runs; its geometry mutation changes one allocated column; and its
  viewer mutation clears the row-produced selection.
- Fixed layout-posture oracle mutation:

  ```bash
  PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-18918-media-trash-paging /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Live/test_library_media_trash_paging_closeout.py::test_layout_posture_oracle_rejects_160_compact_before_branch --disable-warnings --basetemp=/private/tmp/task18918-layout-oracle-green
  ```

  RED failed **1 with 2 warnings in 2.05s** because a forced compact
  observation at **160×50** did not raise and instead projected the compact
  label. GREEN passed **1 with 2 warnings in 2.00s** after the helper asserted
  the immutable size oracle before returning its branch contract.
- Mounted cross-reader gate:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_media_return_settlement.py Tests/UI/test_library_media_side_by_side.py Tests/UI/test_library_media_trash.py Tests/UI/test_library_adaptive_reader_shell.py Tests/UI/test_library_adaptive_reader_closeout.py --disable-warnings
  ```

  Passed **171 with 2 warnings in 236.73s**. The first run was **169 passed,
  2 failed, 3 warnings in 239.71s** and identified only the two stale test-fake
  seams documented above; the focused repair check passed **2 with 1 warning in
  1.46s** before the full green rerun.
- Five fresh exact-return runs of
  `test_compact_media_viewer_back_survives_authoritative_recompose`:

  ```bash
  for run in 1 2 3 4 5; do PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-18918-media-trash-paging /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_library_media_side_by_side.py -k 'compact_media_viewer_back_survives_authoritative_recompose' --basetemp="/private/tmp/task18918-media-return-final-${run}"; done
  ```

  Each passed **1 / deselected 28 / 2 warnings** in **3.52s, 3.57s, 3.56s,
  3.53s, and 3.43s**.
- Live four-size command:

  ```bash
  PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/task-18918-media-trash-paging /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Live/test_library_media_trash_paging_closeout.py -s --disable-warnings
  ```

  The exact command exited successfully; a concise same-file recording passed
  **5 with 6 warnings in 73.98s** (the walkthrough plus four mutation proofs).
  At **160×50, 120×35, 100×30, and
  80×24**, every observation reported `pages=47/3`, `query=5`, `clamp=32`,
  `delete=46`, `restore=45`, exact Trash Back, viewer row return, privacy
  sentinels absent from the live-delete segment, all effective paths inside an
  explicit allowed root, and zero process handles for the target DB/WAL/SHM
  after each size. The **160×50** layout proved both optional panes plus equal
  Items/row expansion after Library collapse
  (`split-then-library-collapse-delta`); the three compact layouts proved the
  Items-priority allocation and independently hidden Library pane
  (`items-priority-exclusive-optional-pane`). The final aggregate pins those
  four `(size, layout_contract)` pairs in exact order.
- Static/inverse gate: the mounted suite included the inverse matrix. `ruff
  check tldw_chatbook/Widgets/Library/library_media_canvas.py
  tldw_chatbook/UI/Screens/library_screen.py
  Tests/UI/test_library_media_return_settlement.py
  Tests/UI/test_library_media_trash.py
  Tests/Live/test_library_media_trash_paging_closeout.py`, `python -m
  py_compile tldw_chatbook/Widgets/Library/library_media_canvas.py
  tldw_chatbook/UI/Screens/library_screen.py
  Tests/Live/test_library_media_trash_paging_closeout.py`, and `git diff
  --check` all passed. The repository virtualenv had to be prepended to `PATH`
  because bare `ruff` and `python` are not exposed by the login shell.

### ADR check

Existing ADR-067 governs the coherent exact-page and stale-recovery contract;
existing ADR-104 governs event-driven Media return settlement. Both approved
specs and both implementation plans remain linked in task references. No new
ADR is required for this verification, documentation, and test-harness closeout.
<!-- SECTION:NOTES:END -->
