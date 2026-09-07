---
id: TASK-2860
title: >-
  Library footer 'F6 next pane' hint silently stripped by AppFooterStatus's
  reserved-key filter
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 07:30'
updated_date: '2026-08-07 19:13'
labels:
  - library
  - footer
  - keyboard
  - bug
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
AppFooterStatus.set_shortcut_context filters any workbench-shortcut hint whose key is in _RESERVED_GLOBAL_KEYS = {f1, f6, ctrl+p, ctrl+q} (tldw_chatbook/Widgets/AppFooterStatus.py). LibraryScreen's LIBRARY_LANDING_SHORTCUTS, LIBRARY_GENERAL_SHORTCUTS, LIBRARY_NOTES_FILES_SHORTCUTS, and (as of task-2856) LIBRARY_DETAIL_BACK_SHORTCUTS/LIBRARY_LIST_SHORTCUTS all advertise ("F6", "next pane") for the screen's own workbench pane-cycle action (action_focus_next_workbench_pane), but that hint is silently dropped from the rendered footer text and replaced by the global GLOBAL_HINTS suffix (which shows an UNRELATED "F6 panes" hint for a different, app-level F6 action). Discovered while re-running the full Tests/UI/test_library_shell.py suite for task-2856: test_landing_footer_advertises_the_landing_keyboard_story deterministically fails at HEAD (confirmed via a direct A/B: still fails with task-2856's own footer-registration change fully reverted), so this predates task-2856 and is not caused by it. The F6 KEY ITSELF still works (Textual resolves the binding regardless of footer text); only the per-screen, action-specific hint text is wrong/missing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Library landing footer's rendered shortcut_text literally includes the screen's own F6 hint copy (e.g. 'next pane'), not just the unrelated global 'F6 panes' hint
- [x] #2 Audit other screens registering an F6 (or F1/Ctrl+P/Ctrl+Q) workbench shortcut hint through AppFooterStatus for the same silent-drop, and fix or document them
- [x] #3 The ambient test_landing_footer_advertises_the_landing_keyboard_story (Tests/UI/test_library_shell.py) failure's root cause (generic 'F6 panes' silently replacing the screen's own 'F6 next pane' copy) is fixed; the test's remaining literal-string mismatch (it expects zero global-hints suffix, which contradicts the always-present-globals contract ~10 other tests enforce) is out of this task's scope per task-3022's own coordination note ('coordinate rather than patch the assertion') -- before/after delta recorded in Implementation Notes for that task to consume
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify the bug at HEAD with a direct unit-test run against AppFooterStatus (set_shortcut_context/_apply_responsive_footer).
2. Read ADR-031 (backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md) to confirm F1/F6/Ctrl+P/Ctrl+Q are meant to be app-global, always-present hints -- the fix must preserve that invariant, not remove it.
3. Redesign the dedup direction: instead of filtering the SCREEN's context actions against a hardcoded _RESERVED_GLOBAL_KEYS set (which silently drops a screen's own, more specific hint for a reserved key), render the context UNFILTERED and instead exclude from the GLOBAL half whichever reserved keys the context already covers. Make this width-tier-aware (full/compact/min global variants) so a screen's F6 hint survives even where GLOBAL_HINTS_COMPACT itself omits F6.
4. TDD: add failing tests pinning (a) a screen-supplied F6 hint survives and shows its own label, not the generic one: (b) it survives at 170/100/80 cols; (c) a genuine content duplicate for another reserved key still collapses to one occurrence. Confirm RED against unmodified AppFooterStatus.py, then GREEN after the fix.
5. Update existing direct unit tests of this exact mechanism that pinned the old (buggy) behavior: Tests/UI/test_app_footer_shortcut_context.py::test_footer_renders_workbench_shortcuts and Tests/UI/test_console_workbench_contract.py::test_console_registers_footer_workbench_shortcuts (Console's CONSOLE_WORKBENCH_SHORTCUTS also advertises F6/F1/Ctrl+P).
6. Audit other set_shortcut_context/register_footer_shortcuts callers for the same silent-drop shape (chat_screen/Console, personas_screen, library_screen confirmed affected; study/lab/evals/mcp/schedules/logs/settings screens confirmed NOT affected -- none advertise a reserved key in their own hint tuples).
7. Run the ambient Tests/UI/test_library_shell.py::test_landing_footer_advertises_the_landing_keyboard_story before and after the fix; record the delta for task-3022 (do not patch its assertion -- that task's own note says "coordinate rather than patch the assertion").
8. Live-verify in tmux at 170/100/80 cols on the Library landing: F6 visible exactly once in the footer.
9. Backlog hygiene + commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: AppFooterStatus.set_shortcut_context filtered a screen's OWN hint for any
reserved global key (f1/f6/ctrl+p/ctrl+q) out of the context BEFORE rendering, regardless
of whether the global cluster's currently-selected width variant actually said that key.
At the compact-width tier (GLOBAL_HINTS_COMPACT = "F1 · Ctrl+P · Ctrl+Q") F6 was never
mentioned at all, so a screen's F6 hint vanished from the footer entirely at that width --
not merely deduped, genuinely unadvertised (the binding still worked; only the hint text
was gone). ADR-031 confirms f1/f6/ctrl+p/ctrl+q are meant to be app-global, ALWAYS-PRESENT
hints -- the filter's intent was avoiding duplicate global hints, not censoring a screen's
own, more specific copy.

Fix (dedup direction flipped, per the brief's "deviate with recorded reason" allowance):
the context now renders UNFILTERED; the GLOBAL half instead excludes whichever reserved
keys the context already covers (AppFooterStatus._remaining_global_text), computed
per-width-tier so it stays correct through the full/compact/min degradation ladder. A
screen's own hint for a reserved key now always wins (shown once, with its real label);
an uncovered reserved key still gets its generic global hint; two screens/tests confirm
genuine duplicates still collapse to one occurrence.

The "literal F6 hint in the rendered footer" criterion and the "audit other screens"
criterion: met cleanly. The "ambient test root-cause fixed" criterion
(test_landing_footer_advertises_the_landing_keyboard_story): its root cause is fixed, but
the test's own literal expected string still does not match -- it has NO global-hints
suffix at all ("... | F6 next pane", nothing after), which is incompatible with the
"always-present globals" contract this fix (and ~10 other tests) enforce. That criterion's
own wording carves this out as acceptable scope (root cause fixed + delta recorded, not a
requirement that the test itself goes green here) -- task-3022's own note for that exact
test says "coordinate rather than patch the assertion" -- recorded delta below for that
task to consume:
  BEFORE (unmodified AppFooterStatus.py): "...n new note | F1 help · F6 panes · Ctrl+P
    palette · Ctrl+Q quit" (generic "F6 panes", screen's own copy absent).
  AFTER (this fix): "...n new note | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q
    quit" (screen's own copy present, "F6 panes" gone, but a global-hints suffix remains,
    which the test's literal string does not expect).

Audit (the "audit other screens" criterion): only three screens advertise a reserved key in their own footer hints --
library_screen.py (F6), chat_screen.py/Console (F6, Shift+F6, F1, Ctrl+P), and
personas_screen.py (F6) -- all now covered by the fix and re-verified (Console's own
direct unit test updated). study/lab/evals/mcp/schedules/logs/settings screens do not
advertise any of f1/f6/ctrl+p/ctrl+q in their own hint tuples -- confirmed via grep, no
changes needed there.

Live-verified in tmux at 170/100/80 cols on the Library landing: F6 shown exactly once at
every width. At 170 and 100 cols it is the screen's own "F6 next pane" copy; at 80 cols
the whole screen-hint cluster (not just F6) already yields to the pre-existing LIB-18
width ladder -- an emergent, honest side-effect of the context now genuinely carrying one
more real item (F6) than before, not a defect in the dedup logic itself (still only ever
one F6 shown, never a duplicate).

Side-finding: filed task-3223 for an unrelated, pre-existing ambient failure
(test_narrow_footer_collapses_but_f1_help_stays_truthful in
Tests/UI/test_settings_footer_hints.py) discovered during the regression sweep -- A/B
confirmed it fails identically against unmodified AppFooterStatus.py, so it is not caused
by this task.

Files changed: tldw_chatbook/Widgets/AppFooterStatus.py (fix);
Tests/UI/test_app_footer_shortcut_context.py (updated 1 pre-existing test pinning the old
behavior, added 5 new TDD tests); Tests/UI/test_console_workbench_contract.py and
Tests/UI/test_screen_footer_hints.py (updated 1 exact-match assertion each, same
mechanism); Docs/User_Guide/library.md (Verified-against stamp).
<!-- SECTION:NOTES:END -->
