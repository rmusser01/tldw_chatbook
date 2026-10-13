---
id: TASK-34914
title: Personas-console-handoff-poll-should-dirty-check-before-provider-resolution
status: Done
assignee: []
created_date: 2026-10-13 14:44
labels:
- performance
- ui-perf
- wave-1-timers
dependencies: []
priority: medium
updated_date: 2026-10-13 15:56
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Personas 0.25s _poll_console_handoff_readiness timer calls _provider_send_block_reason() (full console_handoff_readiness provider resolution) every tick before comparing the result to the cached reason (personas_screen.py:7296-7313). Same work-before-gate shape as the Settings readiness poll: 4Hz provider resolution while Personas is mounted even when inputs are unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] console_handoff_readiness runs only when cheap inputs (header inputs and provider selection) change
- [x] Idle ticks skip provider resolution (test asserts resolution call count)
- [x] Handoff gating behavior unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read _poll_console_handoff_readiness + console_handoff_readiness input chain
2. Add cheap gate before provider resolution
3. Extend the personas readiness tests
4. Targeted run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- `_poll_console_handoff_readiness` (personas_screen.py) now returns before `_provider_send_block_reason()` when a cheap gate tuple is unchanged: (action-allowed, selected entity kind, gathered header inputs, chat_defaults provider/model, `subscription_credential_status(background=True)`, `subscription_readiness_revision()`). All in-memory reads; the 4Hz `console_handoff_readiness()` provider resolution (get_provider_readiness: env walk, settings projection, registry) now only runs when something can actually change it.
- Same design as the Settings poll gate landed in task-34913: `console_handoff_readiness` funnels through the same `get_provider_readiness(..., background_credentials=True)` seam and the shared subscription snapshot, so cached-status + revision covers staleness/expiry/completion transitions; the docstring's "polling starts a bounded cache refresh after TTL" behavior is preserved because `subscription_credential_status(background=True)` keeps being called per tick.
- Verification: new self-contained `Tests/UI/test_personas_handoff_poll_gate.py` (3 tests: resolve-once-then-skip, revision-bump re-resolve, entity-kind-change re-resolve) — 3/3 pass. Regression: `Tests/UI/test_roleplay_header.py` 67/67 (exercises the header/inspector readiness sync paths).
- NOTE: `Tests/UI/test_personas_subscription_readiness.py` is red on this branch HEAD for ALL its tests (verified with every tracked working-tree modification stashed — timeouts at mount, pre-existing, matches the repo's known-CI-breakage tasks). It could not serve as the verification suite; the new unit tests cover the gate semantics directly.
- ADR required: no — same polling-contract perf fix as task-34913.
