---
id: TASK-718
title: Console rail section preferences are keyed per workspace+conversation and get lost
status: Done
assignee: []
created_date: '2026-07-26 17:05'
labels:
  - ux
  - console
  - config
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
[console.rail_state] keys embed both workspace id and conversation id, so section open/closed preferences reset with every new conversation and a Details toggle made moments earlier is lost after switching workspaces and back (live-verified; observed keys also paired one conversation id with two different workspaces). If per-workspace layout memory is the intent, the conversation component defeats it. Finding M7.

Source: workspace-settings UX review baseline, Docs/superpowers/qa/workspace-settings-ux-2026-07-26/report.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Rail section open/closed preferences survive switching away from and back to a workspace
- [x] #2 The persistence key strategy is documented and does not multiply entries per conversation
- [x] #3 Existing stale keys are pruned or migrated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Rewrite key-strategy unit tests to the per-workspace contract (red).
2. Change build_console_rail_preference_key to a constant ':layout' scope with the legacy ':global' key as migration fallback; update prune semantics; update mounted persistence tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`build_console_rail_preference_key` now ignores conversation/session ids (kept as parameters for API compatibility) and always returns `console_rail_state:<workspace>:layout`, with `console_rail_state:<workspace>:global` (the old no-conversation key) as fallback_value - the existing `_migrate_console_rail_fallback_preferences` plumbing adopts legacy prefs into the layout key on first read and deletes the legacy entry. `collect_prunable_console_rail_keys` now treats every conversation/session-scoped key as stale (the live_scope_ids parameter is accepted and ignored); ':layout' and ':global' are kept. Result: section toggles survive workspace switch round-trips, config no longer grows one entry per conversation, and old entries are pruned by the existing one-shot mount pass. Tests: rewrote the strategy-pinning cases in Tests/Chat/test_console_rail_state.py + _prune.py (40 green) and 6 mounted cases in Tests/UI/test_console_persistent_rails.py (36 green). Note: test_generated_console_stylesheet_includes_rail_rules is baseline-red on dev (forbidden 'border: thick $ds-action-focus;' ships in dev's own _agentic_terminal.tcss + bundle at branch base) - unrelated to this change.
<!-- SECTION:NOTES:END -->
