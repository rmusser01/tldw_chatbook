---
id: TASK-650
title: Remove legacy Chat root reactive and worker state
status: Done
assignee:
  - '@codex'
created_date: '2026-07-26 23:50'
updated_date: '2026-07-27 15:18'
labels:
  - architecture
  - state
  - chat
  - reliability
dependencies:
  - TASK-648
  - TASK-649
references:
  - backlog/decisions/011-chatbook-workbench-ui-system.md
  - backlog/decisions/033-application-session-state-ownership.md
  - >-
    Docs/superpowers/specs/2026-07-26-tldwcli-reactive-state-decomposition-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Delete the legacy Chat root session, sidebar, prompt, character, widget, worker, and debounce state after the dormant composition no longer consumes it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every legacy Chat root reactive named by the approved specification and every writer, watcher, and dynamic reference to it are removed.
- [x] #2 _chat_state_lock, current_ai_message_widget, current_chat_worker, current_chat_is_streaming, related accessors, timers, and legacy note identifiers are removed.
- [x] #3 ChatScreen saves and restores only native Console session and rail owners and performs no root sidebar writes.
- [x] #4 Native Console worker, cancellation, transcript, and session behavior remains unchanged without a legacy streaming bridge.
- [x] #5 Normal production TldwCli Chat checks plus focused ownership, privacy, static, formatting, compile, and authorized integration checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/033-application-session-state-ownership.md; backlog/decisions/011-chatbook-workbench-ui-system.md
Reason: Existing ADRs make native Console the only Chat session and run owner.

1. Add exact removed-name AST and mounted production-app guards.
2. Remove root descriptors and companion singleton fields.
3. Delete root-wired handlers, timers, and the legacy streaming bridge.
4. Verify native Console snapshots, sessions, runs, and public cancellation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed every specified root Chat reactive, accessor, writer, watcher, timer, root-only handler, legacy streaming event, and Chat worker registration. ChatScreen now serializes and restores only native Console state; the retained TldwCli chat_wrapper is a fail-closed non-streaming adapter for CCP/media, with all callers statically guarded. Compatibility session models remain importable from their retained state/model modules, while the superseded Chat widget-session and tab-container modules remain deleted on current `dev`; no registered route, app handler, or snapshot path constructs the retired UI. Removed obsolete mock/harness/simplified-app tests and replaced the owned behavior with exact AST sentinels, direct adapter contracts, and normal production TldwCli tests for native rail/session restoration, skill-confirmation state, and visible Stop-button cancellation. The post-rebase review also made snapshot restoration reject malformed field types and expanded the adapter API documentation. ADRs 011 and 033 remain authoritative; no new ADR was needed. The authorized 129-test merge-candidate matrix is green: its first run passed 128 tests and correctly rejected the stale diagnostic digest created by the review docstring change; after the one-entry digest-only inventory update was inspected, both inventory tests passed. The installed-wheel subset separately passed 6/6. Scoped Ruff lint, formatting for the nine owned format-clean files, compileall (with two pre-existing splash-art escape warnings), diagnostic inventory verification (403 owners, 971 TASK-492 calls, 6,017 TASK-494 calls, four sink files), and `git diff --check` passed. Qodo reported zero remaining bugs or rule violations, CodeRabbit completed without a new finding, and all review threads were resolved.
<!-- SECTION:NOTES:END -->
