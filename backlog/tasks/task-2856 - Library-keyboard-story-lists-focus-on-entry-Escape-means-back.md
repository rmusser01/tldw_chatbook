---
id: TASK-2856
title: 'Library keyboard story: lists focus on entry, Escape means back'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 01:10'
updated_date: '2026-08-07 08:31'
labels:
  - library
  - keyboard
  - accessibility
  - uat-2026-08-06
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library UAT 2026-08-06 (LIB-08, prior-critique P1 now measured worse; A + B evidence at dev
`6ffa56516`).

Measured: from a fresh Library landing, the rail search box is 14 Tab stops away and the first
canvas control is 36 (Tabs 1–12 walk the top nav; 13–35 walk the entire rail). Up/Down never move
the media-list selection (7/7 checks, including directly after ‹ Back — the list is not focused).
Escape never functions as back in any detail view. "‹ Back to list" is mouse-only. Focus is
visible at most stops (bg + bold + underline), but two stops are provably invisible (Tab#35
released focus with nothing gaining it; Tab#40 produced a byte-identical capture) and the media
viewer's Author input never shows focus styling.

Keyboard-first is the product's first principle; the destination most users land on is its
slowest keyboard surface. Related open task-2520 covers the landing FOOTER advertisement; this
task covers the mechanics themselves.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Entering a list canvas (Media, Notes, Prompts, Skills) focuses its primary list; Up/Down move the selection and Enter opens it
- [x] #2 Escape returns from detail/viewer surfaces to their list, and from a list canvas focus back toward the rail (no-op only where there is genuinely nothing to leave)
- [x] #3 A direct rail-focus accelerator exists and is advertised (footer or F1), cutting the 14/36-Tab traversal
- [x] #4 Every Tab stop in the Library screen produces a visible focus change (the two invisible stops and the Author input are fixed), proven by ANSI-attribute assertions, not "something changed"
- [x] #5 Live keyboard-only walkthrough: landing → Media list → item → back → search, without touching the mouse
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify all four measured defects live at HEAD via tmux (focus-on-entry, up/down, escape, invisible focus stops, Author input).
2. AC1: focus the primary list's first row on canvas ENTRY (rail-row press) and on RETURN (every viewer/editor Back exit) for Media/Notes/Prompts/Skills; wire Up/Down to move focus between list rows in place (module-level pure function, siblings filtered by row class to skip interleaved non-row Statics); Enter already opens via Textual's native Button binding.
3. AC2: four new mutually-exclusive check_action-gated Escape bindings (media viewer back, note editor back, prompt editor back, list-canvas-focus-rail), following task-2850's exact idiom; refactor the three Button.Pressed 'Back to list' handlers to share one guarded-exit method each with their new Escape action.
4. AC3: re-verify the existing '/' rail-search accelerator (already screen-wide, already footer-advertised) satisfies the outcome; add no redundant new key.
5. AC4: add readable non-obscuring :focus CSS to the four list-row classes (previously falling back to the generic outline); re-verify the specific 'invisible Tab stop'/Author-input claims live.
6. Centralize footer-shortcut freshness by calling _register_footer_shortcuts() from compose_content() itself (every recompose), instead of chasing every editor/viewer entry call site.
7. TDD: CSS contract tests, check_action/action unit tests, Pilot-based integration tests (focus-on-entry, arrow movement, escape chain) for Media/Notes; pure-function test for the Skills-shaped interleaved-sibling case.
8. Live keyboard-only tmux walkthrough with capture-pane -e ANSI proof.
9. Backlog hygiene + report.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented all four sub-fixes for the Library keyboard story.

**AC1 (focus on entry/return + Up/Down + Enter)**: entering a list canvas (rail-row press) or
returning to one (every viewer/editor Back exit) focuses the primary list's first row via
_arm_library_list_entry_focus()/_focus_library_list_entry(). Up/Down move DOM focus between
Library list rows in place via a new module-level pure function _move_library_list_row_focus
(filters siblings by CSS class so the Skills list's interleaved non-row Static secondary lines
never break the walk; no wrap at boundaries). Enter already opens the focused row via Textual's
native Button 'enter'->'press' binding -- no new code needed.

Found and fixed a real race: _refresh_local_source_snapshot (kicked by every editor 'back to
list' exit) is an independent @work background worker whose completion can trigger a LATER
recompose (for Skills this chains into ANOTHER worker, the trust-posture reload) that rebuilds
the list's row Buttons as fresh instances and silently drops the just-set focus. A single-consume
flag lost that race (proven live, then fixed): _library_pending_list_entry_focus now stays armed
for a bounded settle window (LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS=1.0s) and is re-consulted by
compose_content() -- the one choke point every recompose passes through -- on every run, not just
the first. Disarmed immediately the moment the user presses Up/Down (manual control). Also found
_LIBRARY_LIST_ROW_CLASS_BY_ROW_ID needed CREATE_PROMPT/CREATE_SKILL entries alongside the four
BROWSE_* ids: _library_selected_row_id stays CREATE_SKILL (never reassigned to BROWSE_SKILLS)
after a freshly-created skill's editor exits back to the list, mirroring
_library_skill_editor_active's own dual-row-id gate.

**AC2 (Escape = back)**: four new check_action-gated screen-level 'escape' Bindings, following
Task 2's exact idiom -- library_media_viewer_back, library_note_editor_back,
library_prompt_editor_back (each: detail/viewer -> list, reusing the SAME guarded-exit method the
'Back to list' button calls) and library_list_focus_rail (list -> rail search box, a pure focus
hop, never navigation). The three Button.Pressed handlers were refactored to share one
_exit_library_*_guarded() method each with their new Escape action. Footer honesty: centralized
_register_footer_shortcuts() into compose_content() itself (every recompose) instead of chasing
every editor/viewer entry call site -- two new footer sets, LIBRARY_DETAIL_BACK_SHORTCUTS ('esc
back to list') and LIBRARY_LIST_SHORTCUTS ('esc focus rail').

**AC3 (rail-focus accelerator)**: re-verified live that the pre-existing '/' screen-level
accelerator (already footer-advertised on every non-search canvas) already satisfies this AC --
jumps directly to the rail search box from anywhere, including from deep inside the media viewer
after 30+ Tabs. No new key added (avoids a redundant/conflicting binding); Escape-from-list now
converges on the SAME #library-search-input target as / and F6.

**AC4 (invisible focus stops + readable focus)**: re-verified live at HEAD -- could not reproduce
the specific 'Tab#35'/'Tab#40 byte-identical' claims (a 30-press Tab walk through the viewer/
edit-form found no identical consecutive frames) or the Author-input claim (#library-media-edit-
form Input:focus already applies correctly, confirmed live with ANSI evidence: blue border +
bold + blue background on Tab). Layout has shifted materially since the UAT's older commit.
Proactively hardened the four list-row classes anyway, since AC1 now makes them the primary
keyboard-interactive surface: added .library-media-row:focus / .library-notes-row:focus /
.library-prompt-row:focus / .library-skill-row:focus (previously relying on the generic
*:focus{outline:solid} fallback) using the SAME background/color/text-style contract
.library-media-row-selected already uses -- never outline:heavy, which the round-3 lesson
documents as label-eating on 1-row compact widgets (.library-skill-row is exactly that shape).
Verified live with ANSI evidence that labels stay fully readable when focused.

**AC5 (live walkthrough)**: full keyboard-only landing->Media list(entry-focus)->item(Enter
opens)->Escape(list, refocused, footer shows 'esc focus rail')->Escape(rail, search box focused)
proven with capture-pane -e ANSI evidence at every hop. Also verified the Skills 'New skill ->
save -> Escape' race-fix scenario cleanly twice (with and without diagnostic instrumentation).

**Files changed**: tldw_chatbook/UI/Screens/library_screen.py (BINDINGS, footer shortcut sets,
check_action, new predicate/action/guarded-exit methods, on_key Up/Down, compose_content
centralization, module-level pure focus-movement function + row-id/class maps);
tldw_chatbook/css/components/_agentic_terminal.tcss (+regenerated tldw_cli_modular.tcss) for the
four list-row :focus rules; Docs/User_Guide/library.md (Keyboard & commands section + Verified-
against stamp); Tests/UI/test_non_obscuring_focus_contract.py (CSS contract test),
Tests/UI/test_screen_navigation.py (check_action/action unit tests, compose_content flag-
consumption tests, arm/disarm timer test), Tests/UI/test_library_shell.py (Pilot-based
focus-on-entry/arrow-movement/escape-chain integration tests for Media/Notes, pure-function test
for the Skills interleaved-sibling case), Tests/UI/test_library_skills_canvas.py (updated the
pre-existing footer-shortcut and skill-back tests for the new LIBRARY_LIST_SHORTCUTS/arm-timer
behavior).

**Discovered, filed as follow-up (out of scope for this task)**: task-2860 -- AppFooterStatus's
_RESERVED_GLOBAL_KEYS filter silently strips the Library screen's OWN 'F6 next pane' hint text
from every context that advertises it (LIBRARY_LANDING_SHORTCUTS/LIBRARY_GENERAL_SHORTCUTS/
LIBRARY_NOTES_FILES_SHORTCUTS, and now LIBRARY_DETAIL_BACK_SHORTCUTS/LIBRARY_LIST_SHORTCUTS too),
replacing it with the unrelated global 'F6 panes' hint. Confirmed pre-existing via direct A/B
(still fails with this task's own footer-registration change fully reverted) -- the F6 key
itself still works, only its Library-specific hint text is silently dropped. Does not block this
task (the ESCAPE mechanics this task cares about are unaffected; esc isn't a reserved key).
<!-- SECTION:NOTES:END -->
