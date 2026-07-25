---
id: TASK-623
title: Investigate Console transcript ghost action-row paint artifact (live-only)
status: To Do
assignee: []
created_date: '2026-07-25 10:15'
updated_date: '2026-07-25 17:30'
labels:
  - image-generation
  - console
  - ui
  - uat
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live UAT 2026-07-25: after a no-prompt generation completed while its message was (or became) selected, a selected-message-style action row rendered INSIDE the generation card's box -- sandwiched between the image area and the Style/Source/Seed detail rows -- clipped at the card's inner width (labels truncated at "Vie...", `keep`/`Save Image` unreachable). Escape (clear selection) did NOT remove it; it persisted through further selections. A fresh keyboard selection then rendered a SECOND, correctly-placed full action row below the message (two rows visible at once). A tab switch away/back (full transcript recompose) cleared it.

Investigation (this task, 2026-07-25) DISPROVED the obvious DOM-race explanation. `ConsoleTranscript._reconcile_rows` (tldw_chatbook/Widgets/Console/console_transcript.py:1153-1214) is the sole mutator of the transcript's mounted rows, fully serialized by `self._refresh_lock` (an `asyncio.Lock`, :556) plus a screen-level in-progress guard (`chat_screen.py:10958-10962`, `_console_sync_in_progress`/`_console_sync_requested`). `Widget.mount(after=X)` resolves its mount parent from `X.parent` (Textual 8.2.7 widget.py:1478-1481/1416-1422), and within `_reconcile_rows` the `previous_widget` anchor used for each row's `after=` is always the literal top-level widget just built/reused for the PRIOR row -- never a grandchild of a card -- so the action row cannot structurally land inside `ConsoleGenerationCard`'s subtree via any interleaving of `_reconcile_rows` passes. `move_child` (the reuse-in-place fast path) also hard-fails (`WidgetError`) if either widget isn't already a child of `self`, so it cannot silently reparent either. Five distinct async-interleaving reproductions were built in a real Textual `run_test()` harness (new-card-arrival racing a click; same message across two card rebuilds; genuine `asyncio.create_task` concurrency; two-message cross-selection during a card rebuild, with and without a scheduling gap) -- all five produced a single, correctly-parented action row; none reproduced nesting, duplication, or an orphan.

Two live-UAT facts, gathered after the DOM-race disproof, point away from a real mounted widget entirely and toward a GHOST PAINT ARTIFACT: (1) SGR clicks on the in-card "row" produced NO button response at any coordinates -- a real `Horizontal` of `Button`s would have responded; (2) the in-card row never showed the Guide line (`SELECTED_MESSAGE_ACTION_GUIDE`) that accompanies every real selection render (`_transcript_rows`, console_transcript.py:1104-1120 always pairs an "actions" row with an "action-help" row). Both are inconsistent with a real, mounted `Horizontal` action row (which always carries its Guide sibling and always has live Buttons) and consistent with a STALE COMPOSITOR PAINT REGION -- cells from a previously-rendered row left unrepainted over the static generation-card image area during scroll, rather than any node in the actual widget tree.

Full investigation detail (candidate-race analysis with file:line citations, all 5 repro scenarios and their negative results, Textual internals reviewed) lives in `.superpowers/sdd/task-623-report.md` (gitignored scratch notes, not part of any diff -- point future investigators there for the complete writeup before re-investigating from scratch).
<!-- SECTION:DESCRIPTION:END -->

<!-- SECTION:NOTES:END -->

<!-- SECTION:NOTES:END -->

<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reproduce the ghost action-row artifact live with paint-debug instrumentation: an assert-parent-after-mount check in ConsoleTranscript._reconcile_rows (widget.parent is self after every mount/move_child) plus captured compositor region repaints while scrolling with a message selected over a generation card.
- [ ] #2 Determine whether the artifact is a Textual compositor issue or app-side render caching: check whether the ConsoleImageRenderCache (keyed message_id:index) can hand back a card render holding stale composited cells across a browse/keep/scroll sequence.
- [ ] #3 Fix the confirmed cause, or file it upstream against Textual with a minimal repro if it is a compositor-layer bug outside this app's control.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read console_transcript.py end-to-end (select_message, refresh_messages/_reconcile_rows, _action_row/_action_row_signature, action_clear_selection, _transcript_rows, _build_row_widget/_update_row_widget) and console_generation_card.py end-to-end.
2. Read chat_screen.py's generation-completion call sites (_console_command_generate_image, _regenerate_console_generation_variant) and the transcript sync bridge (_sync_native_console_transcript_to_legacy_surface / _sync_native_console_chat_ui) to find the real production interleaving between selection and a generation-completion re-render.
3. Form a precise candidate race (action row mounted with after=<generation-card widget> as anchor; a completion re-render racing the click's deferred refresh reparents it into the card) and verify it against Textual 8.2.7's actual mount()/move_child()/remove() semantics (_find_mount_point, AwaitRemove/_prune) to confirm the mechanism is structurally possible before writing a test.
4. Build a Textual run_test() harness reproducing the interleaving and try to get a red test (widget tree assertion: action row parented under the card, or duplicated/orphaned).
5. If red: fix minimally at the root cause; if reproduction resists after several genuine attempts: stop and report the investigation findings honestly instead of shipping a speculative fix.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Re-scoped 2026-07-25: original task (fix a stale action row nesting inside the generation card) investigated and its DOM-race hypothesis DISPROVEN -- ConsoleTranscript._reconcile_rows is the sole DOM mutator, serialized by an asyncio.Lock plus a screen-level in-progress guard, and Textual's mount()/move_child() semantics cannot structurally reparent a row into ConsoleGenerationCard's subtree. Five distinct async-interleaving reproductions in a real Textual run_test() harness all failed to reproduce nesting/duplication/orphaning.

Two live-UAT facts gathered afterward (no button response on SGR clicks at the in-card row's coordinates; the in-card row never carried the Guide line that always accompanies a real selection render) point away from a real mounted widget and toward a stale compositor paint region left over a generation card during scroll, rather than a DOM defect.

No code change was shipped (no red test to justify one). Full investigation writeup -- candidate-race analysis with file:line citations, all 5 repro scenarios, Textual internals reviewed -- lives in .superpowers/sdd/task-623-report.md (gitignored scratch notes). Re-scoped ACs above point the next pass at live paint-debug instrumentation and the app-side render cache as the two remaining suspects.
<!-- SECTION:NOTES:END -->
