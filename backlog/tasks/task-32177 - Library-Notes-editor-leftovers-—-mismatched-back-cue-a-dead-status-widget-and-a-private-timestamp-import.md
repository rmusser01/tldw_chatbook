---
id: TASK-32177
title: >-
  Library Notes: editor leftovers — mismatched back-cue, a dead status widget,
  and a private timestamp import
status: Done
assignee: []
created_date: '2026-09-09 09:15'
updated_date: '2026-09-09 17:32'
labels:
  - library
  - notes
  - critique-notes-2026-09
  - rider
  - editor
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rider from the Notes critique fix wave (plan
Docs/superpowers/plans/2026-09-09-library-notes-critique-wave.md); raised in
the task review of task-32139 and the final whole-branch review. Three
small leftovers in the Notes editor surface: the New-note view and the
load-retry view still read `‹ Notes` at compact width where task-32139's
own guide correction says `‹ Back to list`; `#library-note-context-status`
is composed on mount but then permanently hidden, dead weight in the tree;
and `_parse_browser_timestamp` is imported directly from another package's
private module rather than through a public helper.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One back-cue wording rule applies everywhere at compact width,
  including the New-note and load-retry views
- [x] #2 The dead `#library-note-context-status` widget is removed, and the
  two geometry-pinning tests that reference it are updated
- [x] #3 `_parse_browser_timestamp` is exposed as a public helper and the
  cross-package import is updated to use it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Compose-time fix: New-note view (_compose_create) and load-retry view (_compose_loading), both in library_notes_canvas.py, hard-code the pre-task-32139 wording ('\u2039 Notes') at every width; switch both to the existing _library_note_back_label(self.compact) helper.
2. Remove the dead #library-note-context-status widget entirely (compose yield, the sync-loop entry, and the display=False line in library_notes_canvas.py); remove its rule from the two hand-authored CSS sources (_agentic_terminal.tcss, library_screen.py BUNDLED_CSS) and regenerate bundles via build_css; reconcile the two geometry-pin tests in test_library_shell.py and the _LIBRARY_NOTES_COMPACT_GEOMETRY entry in test_css_build_integrity.py by removing the dead node id with a recorded reason; update test_info_shows_saved_only_once to assert absence instead of display=False.
3. Rename Workspaces/conversation_browser_state.py's _parse_browser_timestamp to parse_browser_timestamp (module's own convention: public helpers have no leading underscore) and update its one cross-package importer, Library/library_notes_state.py.
4. New tests in Tests/UI/test_library_notes_riders_r_editor.py (RED then GREEN) covering the two back-cue fixes and the public-helper rename.
5. Docs stamp on Docs/User_Guide/library/notes.md; live-verify on the power profile.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Compose-time: New-note view (_compose_create) and load-retry view (_compose_loading), both in library_notes_canvas.py, were left out of task-32139's back-cue unification and stayed hard-coded '‹ Notes' at every width; both now call the existing _library_note_back_label(self.compact) helper.

#library-note-context-status: removed entirely (compose yield, the sync-loop entry, and the display=False line) rather than left hidden -- task-32142 had only hidden it. Removed its rule from both hand-authored CSS sources (_agentic_terminal.tcss and library_screen.py's BUNDLED_CSS fallback) and regenerated the bundles via build_css (screen_agentic_library.tcss, widget_defaults_{scoped,self}.tcss changed; tldw_cli_modular.tcss untouched -- the rule was never in the app-bundle tier). Reconciled by node id: the _LIBRARY_NOTES_COMPACT_GEOMETRY entry in test_css_build_integrity.py, and the two geometry pins in test_library_shell.py (test_library_note_60x20_editor_state_allocation's 'context' row-height table, test_library_note_compact_surplus_allocation_expands_only_named_owner's 'context' fixed_selectors/expected_fixed_heights) -- both pins are pre-existing baseline-broken (same stale #library-notes-row-0 selector documented in the editor-keys task-6-report.md), confirmed identical 9 FAILED names before and after this change via a throwaway detached worktree at d0ff40842f, so no re-derived heights were guessed. Updated test_info_shows_saved_only_once to assert the widget is absent instead of display is False.

Renamed Workspaces/conversation_browser_state.py's _parse_browser_timestamp to parse_browser_timestamp (the module's own convention: public helpers carry no leading underscore) and updated its one cross-package importer, Library/library_notes_state.py, plus the module's own internal caller.

New tests in Tests/UI/test_library_notes_riders_r_editor.py (RED before each fix, GREEN after): two back-cue tests each for New-note and load-retry views (compact + wide), and one asserting the public timestamp helper exists with no private cross-package import remaining.

Files: tldw_chatbook/Widgets/Library/library_notes_canvas.py, tldw_chatbook/Library/library_notes_state.py, tldw_chatbook/Workspaces/conversation_browser_state.py, tldw_chatbook/UI/Screens/library_screen.py (BUNDLED_CSS only), tldw_chatbook/css/components/_agentic_terminal.tcss + 3 regenerated bundles, Tests/UI/test_library_notes_riders_r_editor.py (new), Tests/UI/test_library_notes_wave_editor_keys.py, Tests/UI/test_library_shell.py, Tests/UI/test_css_build_integrity.py, Docs/User_Guide/library/notes.md.

Live-verified on the power profile (tmux, wide 235x52 and compact 60x24): New-note view reads '‹ Notes' wide and '‹ Back to list' compact; Info pane shows 'Saved' exactly once with no gap where the dead widget used to render.

Tests: 133 passed across test_library_notes_riders_r_editor.py + test_library_notes_wave_editor_keys.py + test_css_build_integrity.py + test_library_notes_state.py + test_console_conversation_browser_state.py. The two reconciled geometry pins remain pre-existing baseline-broken (unchanged failure set, verified against d0ff40842f).
Review round (PR #2555, Qodo 2 findings, both real, both fixed):

1. Back cues stayed wrong after a resize. Both views pick their wording in compose() only, and crossing the compact breakpoint re-runs apply_compact_presentation() instead of recomposing -- so the fix above only held for the width the view was opened at. apply_compact_presentation now rewrites #library-note-back and #library-notes-create-back in place (the editor's own pair is already rewritten from the snapshot in apply_session_state, which calls this method first). Two new tests cross the breakpoint in BOTH directions on an already-open view (test_new_note_view_back_cue_follows_the_compact_breakpoint, test_load_retry_back_cue_follows_the_compact_breakpoint) -- confirmed RED against the fix removed, GREEN with it.
2. parse_browser_timestamp, made public here, lacked the required Google-style Args:/Returns: sections; added, naming the accepted ISO-8601 spellings and the None-on-unparseable contract.

Also this round (found reviewing the removals rather than reported): deleting the first, shadowed _seed_local_source_snapshot_from_cache took its long docstring with it, and TWO call-site comments (__init__ and restore_state) defer to "the method's own docstring" for the pre-mount-safety rationale. Folded that rationale, plus Args/Returns, into the surviving definition so those pointers still resolve. Verified before agreeing to either removal: #library-note-context-status has no remaining reference anywhere (py, the four .tcss tiers, tests) and was permanently display=False since task-32142; the duplicate method was genuinely dead (Python keeps the second definition, and the survivor's now-keyword default keeps both no-arg call sites working).

<!-- SECTION:NOTES:END -->
