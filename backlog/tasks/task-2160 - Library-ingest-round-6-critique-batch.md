---
id: TASK-2160
title: >-
  Library ingest round-6 critique batch (in-place clear confirm, picker dedupe, empty-file forecast, focus fragments)
status: Done
assignee: []
created_date: '2026-08-04 03:40'
labels:
  - library
  - ingest
  - ux
priority: high
dependencies: []
---

## Description (the why)

Round-6 dual-agent critique (snapshot `2026-08-04T03-25-53Z…`, 31/40 —
first Good-band; trend 21→24→29→25→26→31; every round-5 fix verified
holding). Owner: fix everything now.

1. **[P1] The tall-queue armed clear-confirm is STILL off-screen** (both
   agents, 8-9-row repros): arming goes through the queue-panel
   recompose, which yanks the viewport to the queue top; the round-5
   `call_after_refresh` scroll ran against the wrong moment. Also
   suspected: a double-click lands both presses.
2. **[P2] The Browse picker double-lists files** (empty.txt twice while
   the directory holds one).
3. **[P2] The forecast still promises "1 will import" for a 0-byte file**
   it just measured at "0 B", which then fails post-commit.
4. **[P2] Clicking non-interactive text focuses the scroll container and
   paints fragmented blue border dashes** (root cause of round-5's
   "stray blue tick-marks").
5. **[P3] "Clear finished" silently includes failed rows in its count;
   the details Category line leaks exception class names.**

## Acceptance Criteria (the what)

- [x] Arming Clear finished changes ONLY the button's label in place —
      no queue recompose, no scroll disturbance; the armed confirm is
      visible at the exact spot the user just clicked, at any queue
      height. A press within ~300ms of arming does not confirm
      (double-click cannot destroy).
- [x] The Browse picker lists each file exactly once.
- [x] A 0-byte file is forecast as a failure before commit (named, and
      counted in "will fail"), consistent with unsupported handling.
- [x] Clicking dead text on the canvas does not paint fragmented focus
      borders (container focus visual suppressed or made continuous).
- [x] The armed clear label names failed rows when present ("… incl. N
      failed"); the details Category line drops the parenthesized
      exception class name.

## Implementation Plan (the how)

(1) Arm in place — label-only update, no recompose, dead zone; (2) picker
atomic publish; (3) empty-file classification at analysis time + named
forecast + kind-accurate gate copy; (4) container focus paint suppressed;
(5) armed label names failed rows; category line drops class names.

## Implementation Notes

- **Arm in place (P1, third attempt, root-caused):** two rounds of scroll
  repair (2130 immediate, 2140 call_after_refresh) both lost to the
  queue-panel recompose yanking tall queues. The cure: arming changes
  ONLY the button label — no recompose, no scroll, the confirm appears
  under the finger. Live-caught follow-up: the longer label clips
  without `refresh(layout=True)` on the auto-width compact button.
  Dead zone: presses within 300ms of arming are the same gesture and do
  not confirm (three pre-existing two-press tests taught to step past
  it deliberately). Live-verified with an 8-row queue: full armed label
  at the exact click row, zero viewport movement.
- **Picker double-listing (P2, root-caused):** vendored
  `directory_navigation._load` published into `self._entries`
  incrementally; a superseded worker kept appending into the attribute
  the fresh worker had rebound (location + show-files watchers both fire
  around mount) — entries doubled up to the cooperative cancel check.
  Fixed with a local list published atomically after a final cancel
  check. Live: fixtures dir lists empty.txt exactly once.
- **Empty-file forecast (P2):** 0-byte files leave their type group at
  `analyze_path` time into `PreflightResult.empty_files`; the summary
  names them ("1 empty file will fail — empty.txt is 0 B."), the commit
  summary counts them in "will fail", solo-empty gates Start, and the
  gate line names blockers by kind ("— 1 empty file", live-caught: the
  total-files fallback used to call it "1 unsupported file").
- **Focus fragments (P2):** `LibraryIngestCanvas:focus`/`focus-within`
  paint suppressed (scoped CSS). Live: clicking dead text yields zero
  accent-blue runs in the canvas body (the only blue is the nav tab
  box).
- **P3:** armed label appends " (incl. N failed)" when failures are in
  the count; the details Category line drops the parenthesized
  exception class.

**Verification.** 285 core + 72 shell-subset targeted green; 29,614
collect. Live (fresh isolated profile): tall-queue arm-in-place, dead
text, empty forecast + gate copy, picker single-listing.

**Qodo round (PR #1313, both fixed in `141f9221f`):** (1) dead-zone
threshold named `_CLEAR_FINISHED_DEAD_ZONE_SECONDS`; (2) REAL bug — an
unstatable file was classified "0 B"/empty via `_safe_size`'s error
fallback; new `_statted_size` returns None on OSError so the file stays
in its type group and ingest surfaces the real error
(`test_unstatable_files_are_not_mislabeled_empty`).
