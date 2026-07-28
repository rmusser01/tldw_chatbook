---
id: TASK-1233
title: 'Marker glyph legend and marker-aware tab tooltips'
status: Done
assignee: []
created_date: '2026-07-28 09:30'
labels: [console, fleet-ux, uat]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expert UAT F4: the fleet's status vocabulary (● running, ◆ needs approval, ✓ finished, ✗ failed) has no legend anywhere, and tab tooltips say only "Switch to Console tab: X" even when the tab carries a marker. Recognition-over-recall failure for the core status language.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Tab (and sidebar-row) tooltips include the marker meaning when one is present ("X — waiting for approval").
- [x] #2 A legend exists in Help (rides task-1232's Agents section if both land). -- delivered by task-1232 (commit bddea6ce0/a1a9c6e50: `CONSOLE_FLEET_MARKER_LEGEND` in `chat_screen.py`'s F1 Help Agents section, asserted by `Tests/UI/test_console_fleet_discoverability.py`).
<!-- AC:END -->

## Implementation Plan

1. Add a shared `ConsoleRunMarker -> meaning` map (`CONSOLE_RUN_MARKER_MEANINGS`) plus a glyph-keyed reverse lookup (`CONSOLE_RUN_MARKER_MEANINGS_BY_GLYPH`) next to `CONSOLE_RUN_MARKER_GLYPHS` in `console_chat_models.py`.
2. Make `ConsoleSessionSurface._session_tab_tooltip` marker-aware: replace the `streaming: bool` param with the `ConsoleRunMarker` the tab already resolves, append the mapped meaning, escape the title (Tooltip renders Rich markup).
3. Make the sidebar conversation-browser row tooltip (`_compose_conversation_browser_row`) append the marker meaning via the glyph reverse lookup, and escape the title.
4. Extend the same treatment to the section/group header collapse toggle tooltips, which already existed ("Expand X"/"Collapse X") and already carry an aggregate marker glyph on the label in the collapsed/capped cases.
5. TDD: write failing tests first for each marker state (incl. NONE as a pinned no-regression case) and for markup escaping, then implement.
6. Run the three gate suites in one blocking call; verify only the known pre-existing failure remains.

## Implementation Notes

**Approach**: Added one small shared vocabulary (`CONSOLE_RUN_MARKER_MEANINGS` / `CONSOLE_RUN_MARKER_MEANINGS_BY_GLYPH` in `console_chat_models.py`) and reused it from both tooltip call sites, rather than threading new controller queries into the widgets (both already receive the marker/glyph they need).

**AC#2**: Already satisfied by task-1232 (commit bddea6ce0, round-1 fix a1a9c6e50) -- `CONSOLE_FLEET_MARKER_LEGEND` in `tldw_chatbook/UI/Screens/chat_screen.py` rides the F1 Help "Agents" section and is asserted by `Tests/UI/test_console_fleet_discoverability.py`. That legend is a single hardcoded multi-glyph string ("Status markers: ● running · ◆ needs approval · ✓ finished · ✗ failed — clears once you visit that tab.") with its own already-tested short-form wording; unifying it with the new per-marker meaning dict was not pursued because the two serve different registers (a compact scannable legend line vs. a specific in-context tooltip sentence, e.g. "waiting for approval" / "finished — unseen") and forcing one copy to match the other would have broken the shipped, tested legend text for no reader benefit. No duplication was introduced: the legend still hardcodes its own copy, and the new dict is Help-agnostic.

**AC#1**:
- `ConsoleSessionSurface._session_tab_tooltip` (`tldw_chatbook/Widgets/Console/console_session_surface.py`) now takes the tab's already-resolved `ConsoleRunMarker` (no new controller queries) instead of a bare `streaming: bool`, and appends `" — {meaning}."` when the marker has one. `ConsoleRunMarker.NONE` renders exactly the pre-task-1233 copy (byte-for-byte, pinned by test). `session.title` is now escaped with `rich.markup.escape` before interpolation, since Textual's `Tooltip` is a `Static` with markup parsing on and an unescaped title containing e.g. `"[red]"` would previously have rendered as a style tag.
- The sidebar conversation-browser row tooltip (`_compose_conversation_browser_row` in `console_workspace_context.py`) appends the same meaning (looked up by glyph via `CONSOLE_RUN_MARKER_MEANINGS_BY_GLYPH`, since the browser pipeline deliberately threads glyph strings, not the enum, to keep `Workspaces/conversation_browser_state.py` free of a Chat-layer import) and now escapes the conversation title.
- Extended the same decode + escape to the section-header and group-header collapse-toggle tooltips ("Expand X" / "Collapse X"), which already existed; picked the same collapsed-vs-capped aggregate marker the visible header label already uses, so tooltip and label never disagree. No new tooltip surfaces were invented.
- Updated `Tests/UI/test_console_session_tab_strip.py`'s existing streaming-tab test: the RUNNING tooltip fragment changed from "Run in progress" to "agent running" per the task's explicit mapping; this is an intentional copy change, not a regression.

**Modified files**:
- `tldw_chatbook/Chat/console_chat_models.py` -- new `CONSOLE_RUN_MARKER_MEANINGS` / `CONSOLE_RUN_MARKER_MEANINGS_BY_GLYPH`.
- `tldw_chatbook/Widgets/Console/console_session_surface.py` -- marker-aware, escaped tab tooltip.
- `tldw_chatbook/Widgets/Console/console_workspace_context.py` -- marker-aware, escaped row and header-toggle tooltips; new `_marker_meaning_tooltip_suffix` helper.
- `Tests/UI/test_console_session_tab_strip.py` -- updated streaming-tab assertion; added per-marker + escaping tests.
- `Tests/UI/test_console_workspace_context_rail.py` -- added row tooltip (unmarked/marked/escaped) and section-toggle tooltip tests.

**Verification (round 0)**: `Tests/UI/test_console_session_tab_strip.py` + `Tests/UI/test_console_workspace_context_rail.py` + `Tests/UI/test_console_parallel_runs.py` in one run: 100 passed, 1 failed. The failure (`test_console_workspace_context_syncs_active_conversation_marker`, a `TypeError` on `ChatScreen._sync_console_workspace_context` arity) reproduces identically on the pre-task-1233 `HEAD` (verified via `git stash`), so it is pre-existing and unrelated to this change.

## Round 1 (review response)

**Verdict addressed**: one Important, three minors.

### Important fix -- unescaped bracket-wrapped status badge silently dropped words from the rendered tooltip
Reviewer verified against this venv's `textual.content.Content.from_markup`: the sidebar row tooltip's `status_suffix = f" [{status}]"` was concatenated UNESCAPED alongside the (correctly escaped) title. `"["` is read as a style-tag start by Rich/Textual markup parsing; an unrecognized tag name (`"saved"`) is silently DROPPED from the rendered text rather than shown literally -- confirmed: `Content.from_markup("Switch to Alpha [saved] — waiting for approval").plain` renders as `"Switch to Alpha  — waiting for approval"` (the word gone, double space left behind). The round-0 pinned test asserted the raw `Button.tooltip` *markup-source* string via `==`, which happened to equal the (broken) input, locking in the bug instead of catching it.

Fix: introduced one centralized assembly point, `_marker_aware_tooltip(text, marker_glyph)` in `console_workspace_context.py`. Every sidebar tooltip (row, section-header toggle, group-header toggle) now builds its RAW, un-escaped sentence body first (title + bracket-wrapped status badge, or "Expand X"/"Collapse X"), then calls this one function, which appends the marker-meaning suffix, a trailing period, and escapes the ENTIRE assembled sentence exactly once. `ConsoleSessionSurface._session_tab_tooltip` was refactored the same way for symmetry (assemble raw, escape once at the end) even though no bracket-bearing fragment exists there today -- per-fragment escaping is exactly the fragile pattern that caused the bug, so the whole module now escapes at assembly, never per-fragment.

All round-0 pinned tests that asserted the raw `.tooltip` markup-source string were rewritten to assert the RENDERED plain text instead, via a new `_rendered_tooltip()` helper (both test files) that renders through `Content.from_markup(...).plain` -- the same path Textual's `Tooltip` widget uses at display time. This is the same idiom `test_console_native_chat_flow.py` already established elsewhere in this suite.

### Minor (a) -- untested symmetric branches
Added `test_collapsed_group_toggle_tooltip_decodes_aggregate_marker` (a collapsed workspace group borrowing its marker onto the header, mirroring the section-level test) and `test_expanded_capped_group_toggle_tooltip_decodes_aggregate_marker` (the expanded-but-past-the-row-cap variant, `capped_run_marker`) in `test_console_workspace_context_rail.py`.

### Minor (b) -- trailing-period consistency
Picked "every tooltip this module builds ends in a period" (matching the tab tooltip's pre-existing, already-pinned convention for its `ConsoleRunMarker.NONE` case, which cannot change). Applied uniformly via `_marker_aware_tooltip`: sidebar row and header toggle tooltips now end in a period in BOTH the marked and unmarked cases (previously neither did). No other test in the repo asserted the old no-period sidebar copy (confirmed via a targeted grep before changing it), so this was a safe, requested copy tweak.

### Minor (c) -- legend/meanings cross-reference
Added a "TWIN CONSTANT" comment block at both `CONSOLE_RUN_MARKER_MEANINGS` (`console_chat_models.py`) and `CONSOLE_FLEET_MARKER_LEGEND` (`chat_screen.py`), each pointing at the other and stating the register split is deliberate (compact scannable legend vs. specific in-context tooltip sentence) so a future glyph-meaning edit finds its twin instead of drifting silently.

### Additional files touched (round 1)
- `tldw_chatbook/Chat/console_chat_models.py` -- cross-reference comment only (no behavior change).
- `tldw_chatbook/UI/Screens/chat_screen.py` -- cross-reference comment only (no behavior change).

### Verification (round 1)
Same three gate suites, one blocking foreground pytest call: **102 passed, 1 pre-existing failure** (same `test_console_workspace_context_syncs_active_conversation_marker` as round 0, unrelated -- re-confirmed via `git stash` in round 0 and unchanged by round-1 edits).

## Concerns (round 1)
None blocking.
