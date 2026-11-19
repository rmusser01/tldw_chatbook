---
id: TASK-37840
title: >-
  Console: the always-visible Temporary-chat surfaces no longer say the chat is
  not saved
status: Done
assignee:
  - '@claude'
created_date: '2026-10-30 01:30'
updated_date: '2026-10-31 01:06'
labels:
  - console
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33621.17 changed the status-strip segment to 'Temporary · traces off' and the tab-strip ◌ tooltip decode with it, both owner-specified copy. The chip's own tooltip (which still says 'not saved locally') has not been rendered since TASK-35160 turned the chips into one status line, so the only on-screen 'not saved' a Temporary chat now shows is the composer Menu's Save chat hint. A user glancing at the strip learns that traces are off but not that the whole chat is lost when the tab closes. Raised by the pre-merge review of TASK-37811; needs a UX look and an owner call on whether the always-visible copy should say both.

A second question for the same call, from the second review of TASK-37811: the segment keeps saying 'Temporary · traces off' after someone explicitly turns Capture On for the Temporary chat (This conversation or Next send). That is literally true (nothing is written while the chat stays temporary), but those sends now pause on the Save chat to capture this send card, so the strip no longer predicts what a send will do.

**Owner decision (2026-10-30).** The always-visible Temporary status segment says both: 'Temporary · not saved · traces off', and 'Temporary · not saved · capture on' once Capture is explicitly turned On for that Temporary chat (This conversation or Next send). The tab-strip ◌ tooltip decode stays consistent with it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The owner's decision on what the Temporary segment and tab marker say is recorded on this task
- [x] #2 The status-strip segment and the tab-strip marker's decode say what the decision requires, and the User Guide matches
- [x] #3 A test pins the visible copy
- [x] #4 The decision also says what the segment reads when Capture On is chosen for a Temporary chat (its sends pause), and a test pins that case
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure on this head (done 2026-10-30, mounted Console, real Trace Privacy dialog): the strip and the tab decode read 'Temporary · traces off' in all three states (default, This conversation On, Next send On), though the last two pause each/the next send. Next send On survives the pause card's Cancel and is spent by Send without capture.
2. Tests first: pure pins for the segment render and the tab decode (both cases) and for a helper that says whether a Temporary chat's next send is captured; one mounted file (stock ChatScreen/controller/store, the real Trace Privacy dialog with the Inspector's bindings) that reads the painted strip and the tab tooltip for default, This conversation On, back to Inherit, Next send On, and after Send without capture spends it. Red on this head.
3. Implement: labels in console_ephemeral; helper in console_exchange_capture; the status segment and the tab decode take the chat's resolved capture; every full Console sync re-pushes the segment.
4. User Guide: sessions-tabs-workspaces.md, context-and-rag.md, semantic-trace-capture.md.
5. Rebase onto newest dev, gates, live round in the real app.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Owner decision 2026-10-30 implemented: the status strip's Temporary segment and the tab's ◌ tooltip decode read 'Temporary · not saved · traces off', or 'Temporary · not saved · capture on' while the chat's next send resolves Capture On.

**Measured first (pre-fix head 8f7a66caf1, mounted Console, the real Trace Privacy dialog).** The strip and the tab decode read 'Temporary · traces off' in every state: default, This conversation On, and Next send On. The last two paused each send, or the next one, on Save chat to capture this send. A Next send On survives the pause card's Cancel and is spent by Send without capture.

**Approach.**
- `console_ephemeral.temporary_label(capture_on=...)` returns the two labels (`TEMPORARY_LABEL`, `TEMPORARY_CAPTURE_ON_LABEL`).
- `console_exchange_capture.temporary_chat_captures_next_send(conversation, next_send)` is the controller's own resolution for a chat that is still temporary: the Next send choice, else This conversation's, else Off. The global default never applies.
- The tab decode (`_session_tab_tooltip`) and the strip (`status_row.temporary_segment_state`, read from the same session fields) both use it.
- `ConsoleStatusChips` takes `temporary_capture_on` at compose (first frame) and in `sync_temporary_chip`, which now returns whether the segment changed.
- `_sync_native_console_chat_ui` re-pushes the segment after its tab sync on every full pass. That covers a spent Next send, and a Trace Privacy choice once the dialog closes.

**Interpretation, flagged for the owner.** 'capture on' means the next send will be captured, so it will pause. With This conversation On and Next send Off, the segment reads 'traces off' until that one send goes out, then 'capture on'.

**Deviation from the plan.** The plan also refreshed on `on_screen_resume`. Mutating that call out left the mounted test green. Pushing a dialog suspends the Console, which clears `_console_attach_reconciled`. On resume the screen first returns early into the attach reconcile. The reconcile runs the full sync pass, which does the repaint, and then calls `on_screen_resume()` again, and that second call does reach the refresh. So the refresh in the resume tail only repeated the sync pass's repaint. It was redundant, not unreachable, and was removed. (The first write-up said it never ran; the pre-merge review's spy fired once per dialog close. lessons-testing-evidence.md has the corrected entry.) Mutating the full-pass call out turns the mounted test red at the This conversation On step ('traces off'). So does dropping the Next send from the strip's resolution.

**Tests.**
- New `Tests/UI/test_console_temporary_status_copy.py`, added to the UI PR gate census (floor 1163 -> 1164).
- Two pure pins cover the segment render and tab decode in both cases, and the helper's nine-case table.
- One mounted case drives the stock ChatScreen, controller, store and file DB, and the real `ConsoleTracePrivacyDialog` with the bindings the Inspector passes. It reads the painted strip and the tab tooltip through: default; This conversation On; back to Inherit; Next send On; and a send paused, then sent without capture (the one-shot spent, the segment back to traces off). Only the provider call is a double.
- The Inspector itself is not opened in the test. Its Next Send estimate reads the workspace on the UI thread, and the harness's probe then fails teardown (lessons entry).
- Red on the pre-fix head: the painted strip read 'Temporary · traces off', plus an ImportError for the helper. Dev 1eaa6a9429 with the branch's tests copied in: 3 failed. Head: 3 passed, 6 of 6 repeat runs, about 8 s.
- `test_console_temporary_capture_flow` pins the new default literal. `test_console_ephemeral`'s copy sweeps include the new label.

**User Guide.** sessions-tabs-workspaces.md (tab strip paragraph), semantic-trace-capture.md (default and Capture On paragraphs) and context-and-rag.md (the segment's name).

**Live (head d73bad2651 + docs, 160x45 and 80x24, real Anthropic claude-haiku-4-5 via the review proxy; evidence/bd5/U07/60-73).**
- A new Temporary tab's strip read 'Temporary · not saved · traces off' (60). Hovering its tab, the tooltip ended '... Temporary · not saved · traces off.' (61).
- Inspector > c > Trace Privacy > This conversation > On > Apply ('Saved and active', 62), then Esc twice. The strip read '... capture on' (63) and so did the tab tooltip (64).
- A send paused on Save chat to capture this send with the strip still reading 'capture on' (65; log capture_enabled=true, no POST). Send without capture got the reply 'OK 71' (log capture_enabled=false, proxy POST). The strip still read 'capture on' because This conversation On stands (66).
- Inherit took it back to 'traces off' (67). Next send On made it 'capture on' (68). A send paused (69), and Cancel kept 'capture on' and gave back the draft (70). Resending and choosing Send without capture got 'OK 72'. The strip (71) and the tab tooltip (72) went back to 'traces off'.
- At 80x24 the segment is still the first thing on the strip (73).

**Gates (head on dev 1eaa6a9429; plain command; reds compared by test id against a detached dev worktree).**
- No new reds vs dev in: composer_menu (the same 2 reds), session_tab_strip, session_tab_close, session_controller, chip_strip_overflow, status_row_collapse, settings_console_status_row, rate_chip, tab_publication_read_barrier, tab_publication_maintenance, tab_mount_ownership, tab_switch_binds_composer (the same 5), poll_reconciliation, poll_tab_source_boundaries, capture_policy_dialog and ui_responsiveness (RecoveryRequired setup errors on both), exchange_capture, ephemeral, the native_chat_flow Temporary subset (the same 4), Tests/Docs (the same 4), and the module-size ratchet (the same 8).
- Two files differed between the full-file runs, and both differences are flaky on dev too:
  - test_console_workbench_contract: `test_console_active_stream_sync_skips_unchanged_chrome_and_inspector` (isolated: head failed 1 of 3, dev failed 3 of 4).
  - test_console_cost_chip_screen (a different 4-5 reds on each side): isolated, dev fails `editing_earlier_history_alerts_the_warm_cache_chip` and `ttl_timer_expires_the_chip_and_stops_itself` as head does.
- Screen size ratchet: passed (chat_screen 24,328 of 24,331; no method added). UI-ready module census: 4 passed (no module added). Boot CSS budget: passed (no CSS). ruff: clean on the changed files.

Related: like the Save chat row, the segment follows the capture policy only. TASK-38000 covers a runtime without durable capture (every send falls back to Capture Off); this branch did not check what the segment says there.

**Pre-merge review fix (2026-10-30).** The review's one blocking finding was the 'Deviation from the plan' paragraph above and its lesson, which said the removed resume refresh never ran. A spy after `apply_status_chips_position` fired three times in the mounted test, once per dialog close: the attach reconcile runs the full sync pass and then re-enters `on_screen_resume`. Re-measured and confirmed; the paragraph and lesson now say the refresh was redundant, not unreachable. No code changed for it. Also from the review: the temporary chip's docstrings in `_sync_console_retrieval_scope_row` and two composer-menu tests now say every full sync pass re-pushes it. Follow-ups: TASK-38000 gains AC#4 (the segment and tab decode say 'capture on' on a runtime without durable capture); TASK-38400 (pin the compose-time first frame); TASK-38401 (the longer segment pushes the run segment off a narrow strip; owner call); TASK-38402 (a composer-menu test that is flaky on dev).

**Merge evidence (2026-10-30, rebased onto dev 7536ac3b87).**
- New tests, plain command: test_console_temporary_status_copy 3 passed on head; 3 failed on dev with the branch's tests copied in (old 'Temporary — not saved' copy, ImportError for the helper, and the mounted strip still on the old copy). test_console_ephemeral 19 passed on head; ImportError on dev.
- Touched modules' test files (35, including the ratchets, Tests/Docs, the UI-ready census and the boot CSS budget), compared by test id against a detached dev worktree: no new failures. Two ids failed only in head's full-file runs and both are flaky on dev: test_console_composer_menu's `keeps_temporary_chip_honest` (isolated, 2 of 6 failed on dev, 2 of 4 on head; TASK-38402), and in test_console_cost_chip_screen a different set fails on each side every run (rerun: head 5, dev 6; isolated, `reverting_the_edit_clears_the_alert` failed 2 of 3 on dev).
- Size rows: chat_screen.py 24,331 -> 24,332, exactly this task's growth, with a dated owner-decision comment; console_chat_controller.py 29,038 -> 29,049, TASK-33621.17's +11. UI-ready census 4 passed; boot CSS budget passed. ./scripts/preflight.sh exits 0 (UI gate census 1169 of floor 1169; diagnostic inventory verified).
- Live on the rebased head (160x45, real Anthropic claude-haiku-4-5 via the review proxy): a new Temporary tab read 'Temporary · not saved · traces off'; This conversation On made it 'capture on'; a send paused with 'Run: Blocked — save chat first' (log capture_enabled=true); Send without capture got 'OK 81' (capture_enabled=false); Inherit took it back to 'traces off', and the tab tooltip ended 'Temporary · not saved · traces off.' (evidence 80-86).
<!-- SECTION:NOTES:END -->
