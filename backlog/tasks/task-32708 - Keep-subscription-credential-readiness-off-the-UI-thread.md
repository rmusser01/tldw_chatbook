---
id: TASK-32708
title: Keep subscription credential readiness off the UI thread
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 01:07'
updated_date: '2026-09-17 02:14'
labels: []
dependencies: []
documentation:
  - Docs/Development/console-model-modal-investigation-2026-09-16.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Slow macOS credential reads must not stall Console settings or repeat immediately after timeout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console readiness remains responsive during slow file/Keychain credential reads
- [x] #2 Concurrent lookups are bounded and failed reads retain a completion-based cache TTL
- [x] #3 Credential errors stay secret-free and existing send authentication remains correct; nearby cache defects are covered
- [x] #4 Mounted Console and full-model credential status refresh on expiry and bounded cache renewal without another user action
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Reproduce UI heartbeat and timeout-cache failures; move UI readiness reads to a bounded cached background path; test pending/completed/expired/concurrent/error behavior and actual authentication callers; run scoped tests and lint.
ADR required: no
ADR path: backlog/decisions/012-provider-credential-settings-boundary.md
Reason: Preserve existing credential ownership while correcting blocking I/O and cache timing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented responsive Console subscription readiness under existing ADR-012 (backlog/decisions/012-provider-credential-settings-boundary.md); no new ADR required.

Console readiness uses one secret-free background snapshot with at most one credential reader. Both file and Keychain reads stay off the UI thread; completion revisions refresh the full settings modal and Send controls. The Keychain cache now serializes concurrent lookups and measures its five-second success/failure TTL from completion with a monotonic clock. Actual send readiness remains synchronous for existing callers; the async Console gateway awaits it in a thread so cold send authentication resolves real credentials without freezing the event loop.

Typed subscription status preserves pending/ready/expired/missing copy across the modal, rail and composer. Pending checks do not offer API-key recovery. Fixed confirmed nearby malformed-file defects: non-string tokens were treated as valid, and non-finite expiry raised OverflowError. Credential exceptions never reach UI copy or logs.

Files: LLM_Calls/anthropic_subscription.py; Chat/provider_readiness.py, console_session_settings.py, console_display_state.py, console_provider_gateway.py; UI/Screens/chat_screen.py; Widgets/Console/console_settings_modal.py and console_settings_summary.py. Added Tests/LLM_Calls/test_subscription_credential_cache.py, Tests/Chat/test_subscription_background_readiness.py and Tests/UI/test_console_subscription_readiness.py (19 regular cases).

Verification: 298 targeted subscription/provider-readiness/display/mounted-UI tests passed; 86 Console readiness/credential contract tests passed. Final focused rerun: 55 passed. Scoped Ruff lint and format passed for the owned source/test files; git diff --check passed. Only the existing requests dependency-version warning appeared. The primary agent owns final formatting/checks of shared large files. No full suite, real Keychain read, external provider call, commit or publish.

Review found synchronous readiness in non-Console Settings/wizard/persona surfaces; these remain outside this task and are reported for follow-up. Earlier broader Console-contract run encountered unrelated existing identity-field/compaction-fixture failures plus transient parallel edits; all scoped readiness tests above are green. Controlled tests prove responsiveness and authentication wiring, not real-account live authorization.

PR preparation on current dev found completion-revision-only polling misses in-cache expiry and does not start TTL refresh while idle. Add current-status polling and mounted regressions before publishing; original readiness and request-credential ownership remain unchanged.

PR integration now also polls current background subscription status alongside completion revision in Console and the full model modal. Mounted tests prove expiry without a new revision, idle TTL renewal, and automatic pending-to-ready recovery. All five credential UI cases pass on current dev; full-app cases use the existing private-profile process harness. Original request authentication and recovery guards remain unchanged.
<!-- SECTION:NOTES:END -->
