---
id: TASK-32545
title: >-
  Library Notes: Manage sync folders disables Retarget/Disconnect with a bare ○,
  renders its buttons identically focused or not, and the sync copy is
  engineering-facing throughout
status: Done
assignee:
  - '@claude'
created_date: '2026-09-13 06:47'
updated_date: '2026-09-14 16:28'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor A (B recorded the strings), personas Sam and Alex / solo operator, lasting-sync workflow. A P2 #10 minus the name placeholder, which task-32451 already owns; plus the copy that set A's heuristic 2 score (3 → 2).

**What happened.** Manage sync folders: Retarget and Disconnect are a grey "○" with no visible reason (A 69); the colour capture shows no focus difference between "Check changes" and "Pause" and one identical "enter run action" footer for every Tab (A 72). Copy across the leg: "60 applied · durable receipt recorded" (A 65; B 44), "0 managed placements" (A 62; B 43), "(name unavailable before cutover)" (32451), "Manual check failed. Review root status, then try again." (A 73), "Additional setup content is scrollable." as the scroll cue (A 60; B 43 "Additional reviewed effects are scrollable."). The server row does this right: "Unavailable - server sync-folder capability not installed" (A 59; B 39). Captures: A 59, 60, 62, 65, 69, 72, 73; B 39, 43, 44.

**Cause.** Copy sites PROVEN (`tldw_chatbook/UI/Library_Modules/library_notes_sync_controller.py`); the focus treatment is INFERRED. Improvement idea in the critique-3 ideas task: sync activation as a What / Where / Impact / Recovery review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Retarget and Disconnect carry their disabled reason as text at the control, in the grammar the server row already uses
- [x] #2 Focused Manage sync folders buttons show a shape-based cue and the footer names the focused button
- [x] #3 "durable receipt recorded", "managed placements", "cutover", "Review root status" and "… content is scrollable" are replaced by user-facing copy stating what happened and what to do next, reviewed against the Import once receipt grammar
- [x] #4 notes.md's lasting-sync chapter and its stamp are updated to the new copy
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live: Manage sync folders at 235x52 and 100x30 (Retarget/Disconnect bare circle; focus cue; footer 'enter run action').
2. RED pin file Tests/UI/test_library_notes_w4_sync_roots.py: disabled labels state their reason in the server-row grammar; focused roots buttons carry the shape cue and the footer names them; rendered sync copy contains no engineering terms.
3. Fix: _disabled_action_label for Retarget/Disconnect (+ the shared disabled-reason line), _LIBRARY_SYNC_ROOTS_ENTER_LABELS consumed by _library_focus_enter_label so the lasting_roots footer tier appends the focused control's label, and the copy replacements (applied line, skip/managed line, scroll cue, cutover line, Review root status lines).
4. Guide notes.md Manage sync folders paragraph + stamp; landing comparison vs dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:BEGIN -->
All four ACs live at 235x52 and 100x30.

**AC#1 — disabled controls state their reason.** Retarget and Disconnect were a bare grey "○" whose reason existed only in a tooltip. Both now go through `_disabled_action_label` (task-32257's grammar, reused from the import canvas): "○ Retarget unavailable — not in this release". The blocked Check on an offline/passive root gets the same treatment, and its tooltip became a "why" rather than an instruction ("the folder is disconnected" / "another Chatbook has this folder open"). The line under the list reads "Retarget/Disconnect unavailable — not in this release; nothing on disk or in Notes changes."

**AC#2 — focus.** The buttons already carried `library-canvas-action`, so `Button.library-canvas-action:focus` applies and a focused Check is bold+underlined where Pause is bold only (`roots-15-focus-check-footer.ansi`). What was missing was the footer: the `lasting_roots` tier said "enter run action" for every control. `_LIBRARY_SYNC_ROOTS_ENTER_LABELS` (next to `_LIBRARY_NOTE_EDITOR_ENTER_LABELS`) is consumed by `_library_focus_enter_label`, and the tier substitutes the focused control's own label -- "enter check changes", "enter pause" -- while disabled controls keep the generic chip because Enter does nothing there. Note: reaching these buttons takes ~23 Tabs from the rail, which is the Library screen's screen-wide Tab order, not this surface's.

**AC#3 — copy.** "durable receipt recorded" → "listed under Receipts" at BOTH sites (the apply path and `activate_root`; the second is what a first activation renders and the live walk still showed the old string), "managed placements" → "folder moves", "Additional setup/reviewed … is scrollable." → "More below — scroll.", "Unavailable until the reviewed lasting-sync cutover" → "Keeping a folder synced isn't ready on this profile yet.", "At-action receipts are unavailable" → "Per-change receipts aren't available here", and "Check failed. Review root status, then Check again." → a named reason or the exception category.

**AC#4 — guide.** `Docs/User_Guide/library/notes.md`: the Manage sync folders paragraph (disabled reasons, the Receipts section, the footer), the activation receipt wording, the failed-check copy, and a "Verified against" stamp quoting the new labels.

Deviation worth recording: the brief expected these buttons to lack `library-canvas-action`. They had it; the critique's "no focus difference" was the footer, not the paint. Tests: `Tests/UI/test_library_notes_w4_sync_roots.py` (disabled labels, focus + footer, no engineering terms across roots/review/receipt/setup surfaces), plus the sibling pins that asserted the old strings.
<!-- SECTION:NOTES:END -->

## Fix round 1 (review 2026-09-14)

**AC#3 was ticked while it was false.** "Lasting folder sync is unavailable until the reviewed cutover." was still rendered twice -- as the chooser's `status_line` and as the setup `validation_message` (painted under the pane heading and used as a tooltip). Both now use the sentence the neighbouring disabled reason already uses: "Keeping a folder synced isn't ready on this profile yet." The only "cutover" left on screen is the root row's placeholder name, which task-32451 owns. AC#3 is re-ticked on that basis.

**The sweep pin that let it through.** `test_sync_copy_uses_no_engineering_terms` built its own snapshots and supplied its own `receipt_line`, so four of its six terms were unreachable: two are controller-produced, one renders only in the `review` phase it never rendered, and "cutover" needed a `validation_message` it never populated. It now drives the controller to each phase -- inert chooser, setup validation, review, receipt, refused roots -- and sweeps what production produced; it goes red if either cutover string comes back. The 32451 placeholder is excised by name, with an assertion that it is still there so the exclusion cannot rot silently.

Also in this round (review Minor 4): "No writes yet. Sync writes appear here as they happen." was untrue -- `refresh_receipts` runs on the route that opens the list -- and now says when they are listed.

A proxy assertion in a sibling suite (`"unavailable" in status_line.casefold()`) went red on the copy change and was tightened to the exact sentence rather than reworded around.
<!-- SECTION:NOTES:END -->
