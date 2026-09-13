---
id: TASK-32545
title: >-
  Library Notes: Manage sync folders disables Retarget/Disconnect with a bare ○,
  renders its buttons identically focused or not, and the sync copy is
  engineering-facing throughout
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-13 06:47'
updated_date: '2026-09-13 15:15'
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
- [ ] #1 Retarget and Disconnect carry their disabled reason as text at the control, in the grammar the server row already uses
- [ ] #2 Focused Manage sync folders buttons show a shape-based cue and the footer names the focused button
- [ ] #3 "durable receipt recorded", "managed placements", "cutover", "Review root status" and "… content is scrollable" are replaced by user-facing copy stating what happened and what to do next, reviewed against the Import once receipt grammar
- [ ] #4 notes.md's lasting-sync chapter and its stamp are updated to the new copy
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live: Manage sync folders at 235x52 and 100x30 (Retarget/Disconnect bare circle; focus cue; footer 'enter run action').
2. RED pin file Tests/UI/test_library_notes_w4_sync_roots.py: disabled labels state their reason in the server-row grammar; focused roots buttons carry the shape cue and the footer names them; rendered sync copy contains no engineering terms.
3. Fix: _disabled_action_label for Retarget/Disconnect (+ the shared disabled-reason line), _LIBRARY_SYNC_ROOTS_ENTER_LABELS consumed by _library_focus_enter_label so the lasting_roots footer tier appends the focused control's label, and the copy replacements (applied line, skip/managed line, scroll cue, cutover line, Review root status lines).
4. Guide notes.md Manage sync folders paragraph + stamp; landing comparison vs dev.
<!-- SECTION:PLAN:END -->
