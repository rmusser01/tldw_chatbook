---
id: TASK-434
title: Roleplay workbench preserves selection across navigation
status: Done
assignee: []
created_date: '2026-07-21 09:38'
updated_date: '2026-07-23 15:35'
labels:
  - roleplay
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live: navigate Personas > Console > back and the workbench resets to "Selected: none" with a blank center pane, "Console blocked: select an item", and a collapsed preview - the working context is lost on every round-trip of the workbench-to-Console loop the design itself encourages.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Returning to the Personas screen restores the previously selected item, mode, and center view within a session
- [x] #2 Preview conversation contents survive the round-trip (until Reset or selection change)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the app never caches screens — navigate-away stores screen.save_state() in app._screen_states, return builds a FRESH PersonasScreen and calls restore_state(). PersonasScreen overrode neither, so the workbench self.state (a PersonasWorkbenchState dataclass) + the preview were lost; on_mount blanked the center.

Fix (full scope AC#1+AC#2, characters mode):
- Task 1 (primitives): PersonasPreviewPane.greeting_text property (the greeting is stored in the pane, NOT in controller.history) + PersonasPreviewController.restore_conversation(greeting, history, seeded_for) which sets seeded_for BEFORE its first await so the still-pending character-load worker's handle_character_loaded guard (:159) preserves rather than erases the restored turns.
- Task 2 (screen): save_state captures asdict(self.state) + preview {greeting, history, seeded_for}; restore_state (runs pre-mount) rebuilds PersonasWorkbenchState filtered to known dataclass fields; deferred _apply_pending_restore at the end of on_mount re-applies the selection via the existing _select_* flow (guarded so a stale/deleted entity degrades to blank center); _select_character gains a restore_preview branch that calls restore_conversation instead of reset_for_character (mutual exclusion — reset_for_character's invalidate() would wipe the restored turns).

SCOPE (whole-branch review Finding 1): restoration is GATED to active_mode=='characters'. Non-character modes (dictionaries/lore/personas) fall back to today's default characters view, because their row-render + mode-widget-visibility live only in _apply_mode which restore doesn't call — restoring them naively left an empty list + mis-toggled Preview/Try-It widgets. So AC#1 (selection/mode/center) + AC#2 (preview greeting+turns survive) are fully met for CHARACTERS mode — the task's driver and the review scenario. Full non-character-mode restore filed as follow-up TASK-488.

Verification: full SDD (per-task implement + review, whole-branch opus review). seeded_for-first ordering mutation-verified; AC#1/AC#2/stale mutation-verified non-vacuous; async race doubly-safe (seeded_for-first belt + same-Screen-message-pump serialization). 6 state tests + 320 personas regression pass. Minors kept: dead saved seeded_for field (entity_id is authoritative), no corrupt-value validation on restore (same tolerance as pre-existing, in-memory same-session only).

Files: personas_preview_pane.py, personas_preview_controller.py, personas_screen.py, Tests/UI/test_personas_preview.py, test_personas_preview_restore.py, test_personas_workbench_state.py.
<!-- SECTION:NOTES:END -->
