---
id: TASK-18310
title: >-
  Centralized workspace activation seam for cross-screen Console runtime sync
status: To Do
assignee: []
created_date: '2026-08-18 15:30'
labels:
  - workspaces
  - console
priority: medium
dependencies: []
---

## Description (the why)

Activating a workspace from OUTSIDE Console (Library's "Create local workspace"
— shipped long before the create modal — and now Settings via the shared modal,
PR #1809) updates only the registry (`set_active_workspace`) and toasts
"Console now targets it". Console's deeper runtime state (chat store context,
active session, native UI) is only synchronized by Console-internal paths
(`_activate_console_session_for_workspace` at switch/create/archive). The rail
reflects the registry on the next Console visit (live-verified in TASK-17650),
but no resume-time seam activates the matching session the way an in-Console
switch does. Qodo flagged this on PR #1809 (finding 5); it is a pre-existing
design gap shared by the Library path, not a regression of the modal work —
resolving it properly means one centralized activation seam that any screen can
invoke (or a Console resume-time reconcile), instead of duplicating Console's
fragile sequence per caller.

## Acceptance Criteria (the what)

- [ ] Activating a workspace from Settings or Library leaves Console — on next visit — in the same runtime state as an in-Console switch to that workspace (session activated or created, store context set, rail synced)
- [ ] The mechanism is a single shared seam (or resume-time reconcile), not per-surface copies of Console's activation sequence
- [ ] In-Console switch/create behavior is unchanged (order-pinned tests stay green)
- [ ] Covered by a test that activates from a non-Console surface and asserts the Console-side session state after resume
