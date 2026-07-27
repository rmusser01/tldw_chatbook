---
id: TASK-648
title: Move provider selection to Settings Console sessions and a typed handoff
status: Done
assignee:
  - '@codex'
created_date: '2026-07-26 23:50'
updated_date: '2026-07-27 13:47'
labels:
  - architecture
  - state
  - providers
  - console
dependencies:
  - TASK-647
references:
  - backlog/decisions/006-provider-aware-generation-settings.md
  - backlog/decisions/026-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the boot-time root provider cache so persisted defaults, active Console sessions, and away-from-Console provider commands have explicit non-overlapping owners.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A Settings save updates the durable default without overwriting an active Console session whose provider source is user.
- [x] #2 An active Chat provider command changes that exact session, while an away-from-Console command stages a typed, memory-only, single-slot intent with revisioned claim, acknowledge, and release behavior.
- [x] #3 Show-current resolves the active session or persisted default; invalid providers terminate the intent and transient readiness failures release it for retry.
- [x] #4 The root provider descriptor, watcher, and legacy model-select path are removed, and provider resolution accepts explicit inputs without an application surrogate.
- [x] #5 Focused protocol, privacy, static, formatting, compile, and normal production TldwCli integration checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/006-provider-aware-generation-settings.md; backlog/decisions/026-application-session-state-ownership.md
Reason: Existing ADRs define durable defaults, active session authority, and memory-only handoff semantics; no new ADR is required.

1. Add and export the typed provider intent channel with direct protocol/privacy tests.
2. Make provider/model resolution explicit and update every production caller, including the model-search picker.
3. Apply selections to exact Console sessions through the single-slot handoff and test only the full production TldwCli for mounted behavior.
4. Remove the root cache, Settings mirror writes, and affected simplified-app tests; preserve their valid behavior in direct-function or full-production tests.
5. Run focused protocol, privacy, static, formatting, compile, and production integration gates; complete fresh specification and quality review before closure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented explicit provider selection ownership: Settings persists chat_defaults without mutating user-owned Console sessions; exact Console sessions own active provider/model state; active and away commands use a typed ConsoleProviderIntent over the memory-only single-slot PendingHandoffStore protocol. Removed the TldwCli root provider descriptor, watcher, updater, and legacy model-select binding. Replaced app-shaped provider/model resolution with explicit mappings and the narrow catalog collaborator, including the production model-search picker.

Testing uses only direct store/resolver boundaries or the normal production TldwCli with registered ChatScreen, SettingsScreen, ConsoleModelPopover, ModelSearchPicker, and real ConsoleChatStore. Removed simplified picker hosts and moved the complete app-independent handoff protocol suite from Tests/UI to Tests/State.

Review corrections: preserved the full pre-existing handoff protocol suite during the move; retained exact catalog merge-cap boundary/current-model coverage; restored the optional parakeet_mlx module-cache state after full-app imports; covered active and persisted show-current behavior, stale base-URL clearing, invalid terminal rejection, transient release/retry, exact-session preservation, and metadata-only consumer diagnostics.

ADR required: yes. Existing ADR-006 and ADR-026 apply; no new ADR was needed.

Verification: focused suite 103 passed with 3 dependency/deprecation warnings; scoped Ruff checks passed; the two pre-existing settings_screen.py F841 findings matched the asserted baseline exactly; 7 selected files were Ruff-format clean; compileall and git diff --check passed. Fresh specification and quality self-review found no remaining blockers.
<!-- SECTION:NOTES:END -->
