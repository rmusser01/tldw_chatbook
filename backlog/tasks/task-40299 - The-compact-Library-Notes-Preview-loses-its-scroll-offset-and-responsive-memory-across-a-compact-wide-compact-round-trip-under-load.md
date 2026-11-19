---
id: TASK-40299
title: "The compact Library Notes Preview loses its scroll offset and responsive memory across a compact-wide-compact round trip under load"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-13 05:30'
created_date: 2026-11-12 19:36
labels:
- follow-up
- library
- notes
- flaky
priority: low
references:
- tldw_chatbook/UI/Library_Modules/library_notes_controller.py
- Tests/UI/test_library_shell.py
- scripts/library_shell_push_deselect.txt
dependencies:
- TASK-39152
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down. Review r13l-rev1 of TASK-39152 classified both nodes as one product defect: the compact Preview's scroll offset and its responsive memory are lost across a compact -> wide -> compact round trip. The offset stays (0, 0) where (0, 2) was kept, and the deliberate scroll is overwritten.

**How often.** Load-dependent:
- TASK-39152 measured 3 of 3 whole-file runs red at load 41-51, and 6 of 7 alone in review.
- Checked at the round-13 stack top 8b633d0101, 2026-11-12, at load 13-15: both passed in a batch, and 3 of 3 alone each.

So the loss shows only when the machine is busy. It is still to be decided whether the product restores the scroll before layout settles (a real race a busy user would hit) or the test reads too early.

**Nodes.**
- `Tests/UI/test_library_shell.py::test_library_note_breakpoint_round_trips_restore_every_region_focus_role`: `preview-body (0, 0) == (0, 2)`.
- `::test_library_note_deliberate_scroll_override_replaces_responsive_memory`: "Compact Preview did not restore the deliberate scroll offset."

Both are deselected from the `library-shell-dev` push job (`scripts/library_shell_push_deselect.txt`, TASK-39153).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The notes say whether the loss is a product race or a test that reads too early, with the evidence (for example a probe of when the offset is restored against when layout settles)
- [x] #2 A product race is fixed with a test that is red under load (or under a temporary delay) before the fix; a test race waits for the restored offset instead
- [x] #3 Both nodes pass 5 of 5 alone at load 30 or more, and their lines leave `scripts/library_shell_push_deselect.txt`
<!-- AC:END -->

## Implementation Plan

1. Reproduce both nodes alone at base under load; probe every scroll restore of the compact Preview (owner range before/after) against the owner's layout over time, to decide product race versus early read.
2. If the restore chain runs before the owner is laid out, make the last restore wait for the owner's layout signal instead of a fixed number of hops.
3. Make the round-trip test wait for the restored offset (the product restores once layout lands), keep the deliberate-scroll test as the red-under-load product pin.
4. Run both nodes 5 of 5 alone at load 30+, drop their deselect lines, run the notes covering tests.

## Implementation Notes

- **Verdict (AC#1): a product race, plus one early read.** At base 26b2e677dd (2026-11-13) both nodes failed alone at load 65-75 (round-trip 3 of 3, deliberate scroll 2 of 3). A probe wrapped `_restore_library_notes_scroll_offset` (owner range and offset before/after each call) and polled the Preview every 50 ms after the compact -> wide -> compact crossing. In the failing run (load 69) all five restores of the compact chain ran while the Preview had not been laid out (`size` width 0, `virtual_size` (2, 6), `max_scroll_y` 0), so `scroll_to(y=2)` clamped to 0; the last one ran at +3421 ms. The Preview laid out later (`max_scroll_y` 296 by +8886 ms) and nothing restored again: y stayed 0 for good. In passing runs the last hop (`_restore_library_notes_scroll_after_layout`) happened to land after layout. The chain is a fixed number of `call_later`/`call_after_refresh` hops, and a first fix that retried for up to 40 more refresh hops spent all of them in about 200 ms with the owner still unlaid out (probe: 45 of 46 restores at `max_scroll_y` 0), so hop counts cannot wait for this layout. The deliberate-scroll node already waited 15 s for the offset, so its failure is the product's alone; the round-trip node also read the offset one pause after the breakpoint, which is too early for a restore that has to wait for layout.
- **Fix (AC#2).** `_restore_library_notes_scroll_offset` now returns the owner when the offset clamped. When the after-layout restore clamps, it parks `(expected, guard)` on that owner and watches its `virtual_size` once per owner (`_retry_library_notes_scroll_on_layout`); the next layout change reruns the same after-layout restore, which re-arms while it still clamps and records the presented focus as before. The restore guard still drops a retry once a newer scroll or focus intent exists. Probe after the fix, four runs at load 39-51: every run ended at y=2, the landing restore running on the layout change (in two runs after the compact state had settled).
- **Tests.** The deliberate-scroll node is the red-under-load product pin (unchanged). The round-trip node now waits for the restored offset (bounded `_wait_for_condition`) instead of reading after one pause; with the fix reverted it stays red, so the wait does not hide the race. Red at base with the waiting test (load 36-53): round-trip 3 of 3 `preview-body: (0, 0) != (0, 2)`, deliberate scroll 2 of 3 "Compact Preview did not restore the deliberate scroll offset". Green after (AC#3), each node alone, `-p no:xdist`: 5 of 5 each at load 36-57. Both lines left `scripts/library_shell_push_deselect.txt`.
- **Covering runs.** `Tests/UI/test_library_shell.py -k "breakpoint or responsive or round_trip or deliberate_scroll or compact_preview or scroll_offset"` 13 passed (`-n 4`, load 40); notes layout files listed in the report; `Tests/Architecture/test_library_notes_wiring.py` passed.
- **Size.** `library_notes_controller.py` 7425 -> 7440 (+15, measured), pin raised by exactly that with a dated comment in `Tests/Architecture/test_library_modules_size_ratchet.py`.
- Files: `tldw_chatbook/UI/Library_Modules/library_notes_controller.py`, `Tests/UI/test_library_shell.py`, `Tests/Architecture/test_library_modules_size_ratchet.py`, `scripts/library_shell_push_deselect.txt`, `Docs/User_Guide/library/notes.md`.

