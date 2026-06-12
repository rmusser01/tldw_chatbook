---
id: TASK-107
title: Add a user-facing control to switch the runtime source to server
status: Done
assignee: []
created_date: '2026-06-12 18:31'
updated_date: '2026-06-12 22:24'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during TASK-70.6 Sync v2 QA: runtime_policy active_source can only become 'server' via app.handle_runtime_backend_changed, which no production UI invokes (tests only). With no switch affordance, Manual Sync v2 stays 'blocked: requires an active server profile' even with a configured, reachable server (binding fix in PR #516). Settings (or Home) needs an explicit local/server source control wired to handle_runtime_backend_changed with reachability/auth feedback.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 User can switch active source local<->server from the UI,Manual Sync v2 unblocks when a configured server profile is active,Switch surfaces reachability/auth state honestly
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped in PR #518 (merged to dev as 0138baca): Settings 'Switch Source / Server' + ServerSwitchModal with validated root URLs, honest per-class auth verdicts, unconditional token persistence; activation rebinds runtime policy, switches the authoritative source, and enrolls the Sync v2 profile. Verified live against tldw_server2.
<!-- SECTION:NOTES:END -->
