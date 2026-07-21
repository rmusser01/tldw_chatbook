---
id: TASK-398
title: >-
  Move Console Context (staged sources) section from left rail into Inspector
  above source readiness
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 01:38'
updated_date: '2026-07-21 01:39'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console left rail currently hosts a collapsible Context section (Sources count, staged-source list, empty-state guidance). Screenshot review of the Console screen shows this content belongs with the Inspector's source-readiness surface, not in the session-focused left rail. Move the Context section into the right Inspector rail directly above the source-readiness card so all source/staging state reads in one place, keeping the left rail to Session, Model, Agent, and Details.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Staged-context tray (Sources count, staged rows, empty-state guidance) renders in the Inspector rail body directly above the source-readiness/pending-launch card
- [x] #2 Left rail no longer renders a Context section header or body, and its remaining sections (Session, Model, Agent, Details) keep collapse/expand behavior
- [x] #3 Live updates when sources are staged/unstaged keep working in the new location without introducing DB reads into compose/recompose
- [x] #4 Persisted rail-state payloads containing the removed context_open key are ignored gracefully
- [x] #5 Duplicate empty-state rendering (summary "No sources attached." plus guidance "No sources attached. Stage sources from Library.") is resolved or documented
- [x] #6 UI tests assert the new Inspector placement and absence from the left rail, and updated suites pass locally
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Recon verified: left-rail Context section = ConsoleRailSectionHeader('Context') + #console-rail-section-body-context wrapping ConsoleStagedContextTray (#console-staged-context-tray) in chat_screen.py compose_content; Inspector body = #console-inspector-rail-body containing #console-run-inspector then source-readiness/pending-launch card mounted after=#console-run-inspector anchor.
2. Move the tray: compose it in the Inspector rail body between #console-run-inspector and the live-work card, keeping id console-staged-context-tray, quiet framing, and min/max height clamps; drop the rail Context header/body entirely.
3. Repoint _apply_console_live_work_card_swap anchor to the tray (fallback to #console-run-inspector) so swapped cards mount BELOW the Context section.
4. Rail state: remove 'context' from CONSOLE_RAIL_SECTION_IDS and context_open from ConsoleRailPreferences/ConsoleRailState/coerce/serialize; stored context_open keys in config payloads are ignored gracefully by coerce (reads only known keys).
5. Fix double empty-state: ConsoleStagedContextState.empty() sets summary='No sources attached.' while the tray also renders the guidance Static -> set empty() summary to '' (widget still renders any provided summary; badge/estimate consumers already treat both as inactive).
6. Update tests pinned to the old location (session_settings, internals_decomposition, persistent_rails, workbench_contract, workspace_context_rail, first-start visibility, Chat/test_console_rail_state, Chat/test_console_display_state) and add a test asserting Inspector placement above the readiness card and absence from the left rail.
7. Run Console/Chat suites + import smoke; rebuild the CSS bundle for the new Inspector-section spacing rule.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved the Console "Context" (staged sources) section from the left rail into the Inspector rail, directly above the source-readiness card.

- `chat_screen.py` compose: the left rail's Context `ConsoleRailSectionHeader` + `#console-rail-section-body-context` block is gone; `ConsoleStagedContextTray` (`#console-staged-context-tray`, unchanged id so `_sync_console_staged_context_tray` and all live-update paths keep working) now composes in `#console-inspector-rail-body` between `#console-run-inspector` and the source-readiness/pending-launch card. Same pure display-state seam (`_build_console_staged_context_state`) - no DB reads added to compose/recompose.
- `_apply_console_live_work_card_swap` anchors swapped cards after the tray (run-inspector fallback for mid-recompose), so both the pending-launch card and the readiness card always mount BELOW the Context section.
- Rail state (`console_rail_state.py`): "context" removed from `CONSOLE_RAIL_SECTION_IDS`; `context_open` removed from `ConsoleRailPreferences`/`ConsoleRailState`/serialize. Migration: `coerce_console_rail_preferences` reads only known keys, so legacy persisted payloads carrying `context_open` are ignored gracefully (covered by a new unit test); serialize no longer writes the key.
- Inspector section conventions: Inspector cards are not collapsible and carry their own titles, so the moved section keeps the tray's own "Sources <count>" header with no new chrome (matches the source-readiness card convention). Consequence: with the Inspector collapsed (first-start default, and forced below 150 columns) staged context sits behind the badged Inspector handle; a pending launch still auto-opens the Inspector at standard widths.
- Double empty-state (user screenshot): root cause was `ConsoleStagedContextState.empty()` carrying summary "No sources attached." while the tray also renders its own "No sources attached. Stage sources from Library." guidance Static - both rendered. Fixed by making `empty()` summary "" (the tray still renders any non-empty summary; rail-badge logic already treated both as inactive; the settings modal sources label now falls back to "None"). Live-driven check shows exactly one occurrence on screen.
- CSS: new `.console-inspector-context-section` spacing rule in `_agentic_terminal.tcss` (margin/padding parity with the readiness card); bundle rebuilt via `build_css.py`.
- Docs: workbench design-doc frame diagram updated (Inspector now lists "Staged sources"; left rail lists Workspace/Model/Agent/Details).
- Tests: new `test_console_inspector_hosts_staged_context_above_source_readiness` (Inspector placement above the readiness card + absence from the left rail, DOM order and measured geometry); reworked location-pinned tests across session_settings/internals_decomposition/persistent_rails/workbench_contract/workspace_context_rail; new legacy-`context_open` coercion test. All Console suites pass locally; remaining failures (schedules/watchlists destination tests) reproduce identically on clean origin/dev.
- Follow-up candidate (out of scope): the collapsed left-rail handle still reads "Context (glyph)" with the staged-count badge; consider renaming the handle label/badge source now that staged sources live in the Inspector.
<!-- SECTION:NOTES:END -->
