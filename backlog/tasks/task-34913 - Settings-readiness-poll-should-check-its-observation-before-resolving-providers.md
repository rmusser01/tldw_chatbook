---
id: TASK-34913
title: Settings-readiness-poll-should-check-its-observation-before-resolving-providers
status: Done
assignee: []
created_date: 2026-10-13 14:44
labels:
- performance
- ui-perf
- wave-1-timers
dependencies: []
priority: high
updated_date: 2026-10-13 15:49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Settings 0.25s _poll_subscription_readiness timer runs get_provider_readiness (env walk + provider settings + endpoint registry + credential snapshots) on EVERY tick, and only afterwards compares the observation that gates the repaint (settings_screen.py:5103-5111). The expensive work runs 4x/second whenever Settings sits on OVERVIEW (the default landing category) or PROVIDERS_MODELS, even when nothing changed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Cheap inputs (category and provider key) are compared before any expensive readiness resolution
- [x] get_provider_readiness runs only when a cheap fingerprint changes or on explicit refresh
- [x] Poll still refreshes within one tick when a credential genuinely changes (test)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a cheap gate (category, provider, auth choice, subscription snapshot status + revision) checked BEFORE get_provider_readiness
2. Reset the gate wherever the observation resets
3. Extend Tests/UI/test_settings_subscription_readiness.py with a poll-skip test
4. Targeted test run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- `_poll_subscription_readiness` (settings_screen.py) now builds a cheap gate tuple — (category, provider, Anthropic auth-source choice, `subscription_credential_status(background=True)`, `subscription_readiness_revision()`) — and returns before `get_provider_readiness` when the gate is unchanged. `get_provider_readiness` (env walk, api_settings projection, custom-endpoint registry, record construction) no longer runs 4x/second on idle ticks; the gate components are in-memory reads.
- Design constraint discovered during testing: gating on revision alone is wrong — the credential cache going stale (TTL) flips the snapshot to `pending` BEFORE any revision bump, and the poll must surface that transition ("Not ready · checking login"). Including the cached status in the gate preserves it; the pre-existing `test_settings_mount_and_overview_refresh_without_another_edit` caught exactly this in the first iteration.
- Sound because in-panel saves and draft edits re-render the provider status lines directly (the render path at settings_screen ~17703), so the poll only needs to cover background subscription transitions — matching how chat_screen.py and console_settings_modal.py already gate on `subscription_readiness_revision()`.
- Gate resets alongside the observation (non-matching categories) and is per-screen-instance, so a fresh Settings mount re-resolves on its first tick.
- Tests: added `test_settings_poll_skips_resolution_until_the_gate_changes` to Tests/UI/test_settings_subscription_readiness.py (paused-timer idiom; asserts unchanged polls add ZERO resolutions and a revision bump re-resolves). Full file: 8/8 passed.
- ADR required: no — perf fix within an existing polling contract, pattern already established by the Console poll.
