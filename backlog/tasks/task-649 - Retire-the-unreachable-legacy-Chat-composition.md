---
id: TASK-649
title: Retire the unreachable legacy Chat composition
status: Done
assignee:
  - '@codex'
created_date: '2026-07-26 23:50'
updated_date: '2026-07-27 14:21'
labels:
  - architecture
  - state
  - chat
  - cleanup
dependencies:
  - TASK-648
references:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/026-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the dormant ChatWindow and ChatWindowEnhanced production surface instead of preserving dead UI with a second application-state owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An import and reachability manifest proves that no registered production route constructs or imports the deleted legacy Chat composition.
- [x] #2 ChatScreen removes chat_window and _ensure_chat_window branches while native Console composition and routing remain unchanged.
- [x] #3 Legacy composition, exclusive helpers, handlers, styles, and tests are deleted; shared modules remain only for live consumers with legacy-only branches removed.
- [x] #4 No LegacyChatState, compatibility root state, or adapter is introduced, and direct import of the retired surface is not supported.
- [x] #5 Normal production Console route, action, and snapshot checks plus focused static, formatting, compile, and authorized integration checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/026-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md
Reason: Existing ADRs select native Console as the only production Chat composition; no new ADR is required.

1. Prove legacy import and route reachability in a checked-in manifest.
2. Add failing structural and full production TldwCli route checks.
3. Remove ChatScreen legacy composition branches without changing native Console ownership.
4. Delete only manifest-proven exclusive modules, styles, and surrogate tests.
5. Verify production Console behavior, structural absence, formatting, compile, and focused integration checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Retired the unreachable `ChatWindow`/`ChatWindowEnhanced` composition and its
exclusive modules, diagnostics, styles, adapters, and surrogate tests. The
registered Chat route now composes only the native Console surface; retained
shared helpers have live native/non-UI consumers and no legacy fallback.

- Reachability manifest commit: `294a7eee6`.
- Implementation commit: `94b2c558f` (47 files, 341 insertions, 10,657
  deletions).
- ADR check: existing ADR-011 and ADR-026 govern this ownership change; no new
  ADR was required.
- TDD RED: the new production-route and structural checks failed as expected
  with 3 failures and 29 passes before removal.
- Final production/ownership gate: 34 passed with 3 unrelated dependency or
  deprecation warnings in 83.01 seconds. The mounted test constructs the normal
  production `TldwCli`, uses registered production screens and real Console
  controls, and verifies snapshot restoration through the real Settings route.
- Diagnostic/privacy gate: 13 passed in 11.43 seconds. The generated inventory
  verifies 416 owners, 1,006 TASK-492 calls, 6,804 TASK-494 calls, and five
  unchanged persistent sink files. TASK-649 removed eight owners, four
  TASK-492 calls, and 213 TASK-494 calls relative to a fresh branch-HEAD
  inventory; inherited TASK-648 snapshot drift is documented separately in
  the manifest.
- Ruff lint, targeted Ruff format checks, `compileall`, CSS regeneration,
  zero-hit production source/TCSS sentinels, inventory verification, and
  `git diff --check` passed. The three large pre-existing files that are not
  globally Ruff-formatted were linted, compiled, behavior-tested, and
  diff-checked without an unrelated mass-format rewrite.
- Plan deviations were scope corrections only: `app.py` cleanup, persistent
  diagnostic inventory regeneration, a retained-helper docstring correction,
  and removal of stale deleted-module monkeypatches. No simplified or test
  application suite was run or cited as verification.
<!-- SECTION:NOTES:END -->
