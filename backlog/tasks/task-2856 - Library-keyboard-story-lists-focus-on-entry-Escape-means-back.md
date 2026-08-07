---
id: TASK-2856
title: 'Library keyboard story: lists focus on entry, Escape means back'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 01:10'
updated_date: '2026-08-07 09:44'
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
ROUND 2 (post-review fixes): review verdict "Needs fixes", 3 Important findings, all addressed.

(1+2) Settle-window timer could silently steal focus from a user who Tabbed/clicked away within
the window (only 2 disarm call sites existed: Up/Down and the timer). Fixed: disarm is now
interaction-driven -- on_key disarms unconditionally on ANY key (before the Input/TextArea early
return); new on_descendant_focus disarms the instant focus lands on any widget that isn't a row
of the armed list (catches mouse clicks, which never reach on_key). The timer's role is now only
"bound an IDLE list's re-fire window" -- no longer what protects user navigation. Timer duration
measured live (not guessed): two runs of the Skills "New skill -> save -> Escape" chain showed
142ms/249ms for the chained trust-posture worker; raised LIBRARY_LIST_ENTRY_FOCUS_ARMED_SECONDS
1.0 -> 2.0 (~8-14x measured worst case) with the measurement documented in code. New tests
(2 real Pilot integration tests + 5 unit tests) prove: Tab-away survives a background recompose
within the window (focus NOT stolen); 3 chained recomposes with no user input all still get
re-focused (not narrowly tuned to the one measured case). AC1's caveat is resolved by this fix,
not just noted -- the race it flagged no longer exists.

(3) AC4 was checked without meeting its own "ANSI-attribute assertions, not something changed"
bar -- the round-1 evidence leaned on a git-blame fact instead of a genuine capture, and the CSS
hardening covered a different surface than the UAT's own Tab#35/Tab#40 citations. Redid it with
the UAT's own method: scripted a full Tab walk at HEAD from a TRUE landing state (55 presses
through nav+rail+Details-chip+into canvas: 0 identical consecutive pairs of 55) and a dedicated
walk through the media viewer/edit form (45 presses: 0 identical pairs of 45), both fully
byte-compared, both pasted into the report with counts. Separately captured the Author input
focused-vs-unfocused with explicit raw ANSI bytes (border/background/text-weight all change,
label stays readable). None of the three original claims reproduce at HEAD -- AC4's checkbox
stands as checked on this stronger evidence; no amendment needed since nothing survived to amend
for.

Also fixed (promoted Minor): _exit_library_media_viewer() ran unconditionally on Escape even
mid-edit, unlike note/prompt's dirty-guarded exits. The media edit/delete-confirm/analysis-edit
sub-states have NO dirty-tracking field to veto on (unlike notes/prompts/skills) -- rather than
inventing one, Escape now steps back ONE level per sub-state, mirroring each one's own
pre-existing Cancel button exactly (#library-media-edit-cancel / -delete-cancel /
-analysis-cancel, all three already discard unconditionally, pre-dating this task). Strictly
LESS aggressive than before. The always-visible "Back to list" button's own unchanged behavior
(skips sub-states) is a deliberate, documented scope boundary, not an oversight.

Foreground test totals (no background waiting): test_library_shell.py 352 passed/1 failed;
test_screen_navigation.py+test_library_skills_canvas.py+test_non_obscuring_focus_contract.py 308
passed/1 failed; Tests/Library 1077 passed. TOTAL 1737 passed, 2 failed -- both the SAME
pre-existing unrelated failures from round 1 (task-2860's footer bug; an untouched _forms.tcss
Select:focus rule). A third failure appeared once under heavy combined-run load
(test_library_shell_ingest_canvas_live_updates_without_manual_recompose, an unrelated
media-ingest queue polling test) -- confirmed transient by re-running in isolation (passed) and
re-running the full file alone (352 passed, only the known failure).

All temporary timing/diagnostic instrumentation removed before commit; verified via grep + clean
py_compile.
<!-- SECTION:NOTES:END -->
