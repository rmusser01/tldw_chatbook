---
id: TASK-3312
title: >-
  Ingest copy polish from live verify: F1 esc dupe, egress failure receipt,
  warning noun echo, panel-header focus
status: Done
assignee: []
created_date: '2026-08-08 00:30'
updated_date: '2026-08-09 04:29'
labels:
  - library
  - ingest
  - ux
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cosmetic findings from the 3300-3305 arc live verification (2026-08-08):

1. F1 help in Ingest lists escape twice ("esc: back to hub" from the shared shortcut set + "escape: Back to Library hub" from BINDINGS).
2. The queue failure receipt for an egress-blocked URL leaks a markup escape (`\[web_security]`), ends mid-sentence with a trailing comma, and is far more technical than the plain-language inline preflight line one row above (task-3305's mapping) — route it through the same plain-language treatment with the remedy intact.
3. Guardrail modal warning line can repeat its noun ("- Audio processing (1 file): Audio processing") when the feature label equals the capability hint — suppress the echo.
4. The collapsible options-panel header is a Tab stop whose focus is color-only (glyph-less) — the one focusable the task-3302 treatment missed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 F1 in Ingest lists one escape row
- [x] #2 Egress-blocked URL receipts render plain language, no markup-escape leak, no dangling comma, remedy preserved
- [x] #3 Guardrail warning lines never repeat the feature name as their own hint
- [x] #4 Focused collapsible panel headers show a glyph-level focus indicator
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. F1 dedupe: dedupe escape row where LIBRARY_INGEST_SHORTCUTS merges with BINDINGS in the help modal.
2. Egress receipt: plain-language mapping for [web_security] blocked-URL errors in short_ingest_error/expanded detail; fix markup-escape leak + dangling comma; keep remedy.
3. Warning noun echo: suppress hint==label echo at the shared warning line builder (used by inline preflight too; modal retirement in 3314 noted).
4. Collapsible header focus glyph: CSS-level indicator, no dimensional change; rebuild bundle; CSS-true render_lines test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Four live-verified copy/focus fixes, each with a RED-first pin. (1) F1 escape dupe: the footer sets spell the exit 'esc' while BINDINGS spells it 'escape', so action_show_workbench_help's raw-key dedupe kept both. New _canonical_shortcut_key (casefold + esc/escape unification) applied to both sides of the merge; one escape row now survives in every Library mode. Mutation check: reverting to raw-key dedupe sends the new F1 test RED. A stale expectation in test_screen_navigation.py (asserting the duplicate 'escape' row on the list canvas) updated to pin single-row. (2) Egress receipt: short_ingest_error (the single source for the queue row AND Home's failed-item line) now maps any 'Egress blocked' error to INGEST_EGRESS_BLOCKED_COPY -- plain language, complete sentence, bracket-free, allowed_hosts/web_security/config.toml remedy intact. Mechanism of the \[web_security] leak pinned in REPL: rich.markup.escape skips a bracket run that never closes as a tag ([remedy: ... [) while escaping the inner closed ones, and Textual 8's content markup then leaves the FIRST escape's backslash literal; the mid-sentence comma was the overlong technical line clipping. Also fixed the render mechanism itself: the queue-row / preflight-error / preflight-warning Statics now render verbatim with markup=False instead of escape-then-parse (canvas test pins a mixed-bracket error rendering byte-identical). (3) Guardrail-modal noun echo ('- Audio processing (1 file): Audio processing') suppressed with the same hint!=label casefold rule build_warning_lines already applies; the inline builder was ALREADY echo-free and pinned (test_build_warning_lines_does_not_repeat_the_label), so this is a minimal line-level fix in the modal compose, deliberately NOT a shared refactor -- task-3314 retires the modal wholesale. (4) Collapsible options-panel header focus: heavy side rails (outline-left/right) added to the existing LibraryIngestCanvas Collapsible > CollapsibleTitle:focus rule in css/components/_agentic_terminal.tcss; bundle rebuilt via build_css.py (check_bundle_sync green). Trap found: the first attempt used a descendant selector at (0,1,2), which _widgets.tcss's later app-wide 'Collapsible > CollapsibleTitle:focus { outline: none; }' silently discarded at equal specificity -- the child-combinator (0,1,3) rule wins regardless of order. CSS-true render_lines() test pins glyph change + intact title + identical region/rows. Files: tldw_chatbook/UI/Screens/library_screen.py, tldw_chatbook/Library/library_ingest_state.py, tldw_chatbook/Widgets/Library/library_ingest_canvas.py, css/components/_agentic_terminal.tcss (+ rebuilt bundle), Docs/User_Guide/library/import-and-export.md (+stamp); tests in test_library_ingest_keyboard.py, test_library_ingest_canvas.py, test_library_ingest_guardrail_modal.py, test_library_ingest_state.py, test_screen_navigation.py.
xhigh review round (2026-08-09): fix (2)'s plain-language egress receipt was TOO flat -- every
refusal collapsed to one fixed sentence saying "this address" and never which one, so a queue of
blocked URLs read as N identical rows and the expanded details could not recover the host either.
New `egress_blocked_receipt` keeps the whole register the task established (one complete sentence,
no policy jargon, no bracketed config-key syntax) and names the origin `EgressBlockedError` already
puts in its message. An origin that cannot be parsed, or one carrying square brackets (a bracketed
IPv6 literal -- the exact character class behind this task's own live `\[web_security]` incident),
falls back to `INGEST_EGRESS_BLOCKED_COPY` rather than shipping markup-hostile text. The three
task-3312 tests that asserted equality with the constant now assert the host is present and the
register is unchanged. Files: tldw_chatbook/Library/library_ingest_state.py,
Tests/Library/test_library_ingest_state.py.
<!-- SECTION:NOTES:END -->
