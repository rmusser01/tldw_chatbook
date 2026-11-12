---
id: TASK-34956
title: Add-a-cheap-snapshot-gate-to-the-Console-credential-poll
status: Done
assignee: []
created_date: 2026-10-13 14:46
labels:
- performance
- ui-perf
- wave-1-timers
dependencies: []
priority: medium
updated_date: 2026-10-13 16:21
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console 0.25s _poll_console_credential_readiness timer rebuilds _active_console_settings_readiness() every tick before the (revision, status, version) snapshot compare (chat_screen.py:16648-16663). The in-tree comment already clocks this at ~8ms per tick (task-32804.3) — roughly 3% of a core while idling on the Console screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Tick cost dominated by a cheap fingerprint compare and full readiness build only runs when it changes
- [x] Credential status still updates within one tick of an actual change (test)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read _poll_console_credential_readiness + _active_console_settings_readiness cost
2. Cheap-fingerprint gate before the full readiness build
3. Unit-test the gate
4. Targeted run + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

- `_poll_console_credential_readiness` (chat_screen.py) now checks a cheap pre-gate — (cached `subscription_credential_status(background=True)`, `subscription_readiness_revision()`, connection-evidence version) — BEFORE entering the derivation scope and running the ~8ms `_active_console_settings_readiness()` build. Idle ticks no longer pay the scope admission + readiness resolve at 4Hz while the Console is visible.
- The pre-gate deliberately does NOT fingerprint the session settings object: probing `_ensure_active_console_session_settings` outside the derivation scope would bypass the task-24301 memo and can trigger its convergence write/blank-session creation outside a pass (per its docstring). Provider and draft edits refresh the send controls through their own watcher paths; this poll only owns background credential transitions, and status() catches time-based expiry without a revision bump (kept in both gates).
- The existing (revision, status, version) snapshot gate and its refresh side effects are unchanged inside the build path.
- Tests: new self-contained `Tests/UI/test_console_credential_poll_gate.py` (3 tests: idle-tick skip, revision-bump rebuild, expiry-without-revision rebuild) — 3/3 pass. Regression: `Tests/UI/test_console_switcher_local_probe.py` + `Tests/UI/test_console_turn_resend_ui.py` (the suites covering the poll and its control refreshes) — 31/31 passed.
- ADR required: no — pre-gate inside one existing poll, same pattern as tasks 34913/34914.
