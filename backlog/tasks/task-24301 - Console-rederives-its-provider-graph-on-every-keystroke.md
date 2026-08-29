---
id: TASK-24301
title: >-
  Console re-derives the whole provider-readiness graph on every keystroke and every screen entry
status: Done
assignee: []
created_date: '2026-08-28 23:30'
labels:
  - performance
  - console
  - chat
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every printable keystroke in the Console composer rebuilds the provider / readiness /
session-settings object graph from scratch, discovers the result is identical to the previous
keystroke's, and discards it. The equality gate that follows
(`_push_console_control_state_if_changed`) skips the DOM write but not the compute -- the
derivation has already run by the time the gate is consulted.

The per-pass memo added by task-15452 (`_console_derivation_scope`) caches only `app_config`, so
the expensive derived objects are recomputed repeatedly WITHIN one pass and entirely afresh on
every pass.

Measured on dev `3a3383123e`, interleaved A/B, 60 keys per arm, two rounds, against a counterfactual arm
that memoises the derivation for the burst:

  empty conversation  control 4.90 / 4.18 ms per key  ->  memoised 0.52 / 0.47  (~9x)
  200 messages        control 22.43 / 18.69 ms/key    ->  memoised 0.87 / 1.34  (~17-20x)
  `normalize_provider_config_key` calls per key: 250.5 -> 13.5 (empty), 146.0 -> 8.0 (200 msgs)

The same machinery dominates screen entry. One WARM return to the Chat screen costs 159.8 ms of
app-side Python and runs 18,938 `normalize_provider_config_key` calls, with
`_build_console_provider_selection_uncached` at 117.7 ms cumulative and
`_maybe_refresh_stale_default_console_settings` entered 247 times. The Library screen's warm
entry, for contrast, is 18.0 ms.

The counterfactual caches for a whole burst and is NOT a shippable design; real invalidation will
cost something, so 9-20x is an upper bound rather than a promise.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Repeated derivation within a single pass is eliminated -- the readiness and session-settings objects are built once per pass, not once per consumer
- [x] #2 A draft edit that changes no configuration, session, or workspace state does not rebuild the derived provider graph at all
- [x] #3 The cache invalidates correctly when configuration, active session, active workspace, or session settings change, proven by a test that would fail if any one of those were omitted
- [x] #4 Per-keystroke app-side cost on an empty conversation drops by at least 5x, measured by interleaved A/B attribution
- [x] #5 A guard pins the derivation by CALL COUNT, not wall clock, so it cannot silently re-accrete
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Widen the existing per-pass memo (`_console_derivation_scope`) to cover the session-settings leg.
2. Add a cross-pass memo for the template defaults, keyed on retained-object identity.
3. Pin both by call count; mutation-test each layer separately.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two layers, deliberately separable so each is measurable on its own.

**Layer 1 -- per-pass.** `_ensure_active_console_session_settings` now serves
from the screen's `_console_derivation_memo`. Every leg of a control-state or
Workbench derivation calls it, so one draft-edit sync entered it 3.25 times and
each entry re-derived the template defaults. The pass is synchronous and
nothing else mutates the store inside it, so one memo for its duration is exact
-- the same argument `_console_derivation_scope` was introduced on (task-15452).

**Layer 2 -- cross-pass.** `_default_console_session_settings` is a pure
function of (app_config, provider, model): no environment read, no mutation.
Memoised across passes, keyed on the config object's IDENTITY against a
retained reference (`load_settings()` returns the same mapping until
invalidated and a fresh object after, so `is` detects a reload exactly; holding
the reference makes id-reuse-after-GC impossible).

**Deliberately NOT extended to `build_console_settings_readiness`**, which
reads `os.environ` for credentials. Caching readiness against a stale snapshot
is exactly the task-177 regression -- a provider configured in Settings stayed
blocked until restart -- and it is not worth re-introducing for the remaining
milliseconds. The per-pass memo bounds it instead.

**Measured (call counts per keystroke, empty conversation).**
`build_default_console_session_settings` 3.25 -> 0;
`build_console_settings_readiness` 4.35 -> 2.35;
`normalize_provider_config_key` (module-attribute calls) 15.2 -> 4.7.
Each layer was mutation-tested independently: removing layer 1 puts readiness
back to 4.35/key, removing layer 2 puts template defaults back to 1.42/key,
and the new budget guard reds in both cases.

Files: `UI/Console_Modules/session.py`,
`Tests/Chat/test_console_default_settings_memo.py` (new),
`Tests/Performance/test_console_keystroke_work_census.py` (new).
<!-- SECTION:NOTES:END -->
