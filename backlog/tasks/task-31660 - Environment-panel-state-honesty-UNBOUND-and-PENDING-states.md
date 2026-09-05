---
id: TASK-31660
title: >-
  Environment panel state honesty: UNBOUND and PENDING states (stale-root P0)
status: To Do
assignee: []
created_date: '2026-09-05 07:00'
labels: [console, inspector, ux, critique-2026-09-05]
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX critique P0 (2026-09-05, dual-agent, live-measured): after a workspace
switch the Environment panel keeps ANOTHER repository's counts/branch and
still offers "Commit or push · N files" — permanently (Refresh is inert when
root is None); on cold start it asserts "No git workspace" for ~20s inside a
git worktree. Root cause: root-is-None and pre-first-fetch have no
representation in EnvironmentSnapshot; poll_tick/request_refresh return
early and the last paint stands. Owner chose state-honesty as the first
burn-down cluster. Snapshot: see .impeccable critique 2026-09-05.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 When the workspace root accessor returns None, the panel lands an explicit UNBOUND state within one poll cycle: counts, Commit-or-push, PR/checks, and Tasks suppressed; copy matches Change Review's ("No folder is bound to this conversation's workspace, so changes are not tracked here — this is not a report that nothing changed")
- [ ] #2 Before the first local-tier landing, the panel renders a PENDING state (e.g. "Checking workspace…") or stays hidden — it never renders "No git workspace" (or any negative) before a gatherer has answered
- [ ] #3 A workspace switch clears the previous root's data within one poll cycle even when the new root is None
- [ ] #4 Refresh in the UNBOUND state either re-checks the binding or is not offered; it is never a visible no-op control
- [ ] #5 Deferred-fake controller tests + screen-wiring tests cover: cold start, bound→unbound switch, unbound→bound recovery
<!-- AC:END -->
