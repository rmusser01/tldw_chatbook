---
id: TASK-18925
title: 'Opt-in OS notifications for background Console events'
status: To Do
assignee: []
created_date: '2026-08-19 09:55'
updated_date: '2026-08-19 09:55'
labels:
  - console
  - agents
  - notifications
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Port of hermes-agent's native OS notifications with per-type toggles (2026-08-19 hermes-release review). Console already toasts in-app for background completions, failures, parked approvals, and auto-wakes — but if the terminal window is buried, the user misses them. Fire a desktop notification (macOS via osascript / Linux via notify-send; Windows evaluated and honestly reported) for "run finished", "run failed", and "needs approval", gated by a new opt-in setting (default OFF) with per-type toggles. Notifications carry the conversation/tab name and outcome category only — never message content, never credentials. Cadence must match the existing one-toast-per-event dedupe semantics so this adds a channel, not spam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 New setting (default OFF) with per-type toggles for run finished / run failed / needs approval; off-by-default is pinned in tests
- [ ] #2 macOS and Linux implementations work; unsupported platforms or missing binaries fail silently with a debug log and never crash or block the app
- [ ] #3 Notification body includes only conversation/tab name + outcome category; no message text, no tool output, no secrets — verified by tests over the constructed command
- [ ] #4 Notification cadence dedupes identically to the in-app toasts (one per background completion / approval park; multi-prompt drain reports once)
- [ ] #5 The notification subprocess invocation is time-bounded and failure-tolerant, and never runs on the UI/event-loop thread in a blocking way
- [ ] #6 Tests cover setting gating, per-type toggles, command construction (no content leakage), and failure tolerance; user guide documents the setting
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no.
ADR path: N/A.
Reason: opt-in presentation hook over existing notification events; no storage, sync, or data-ownership change. Security review of the subprocess surface is in-plan (scrubbed env, fixed argv, bounded timeout).

1. Notification dispatcher (macOS/Linux) with bounded subprocess + scrubbed env
2. Hook the existing completion/failure/approval-park toast sites, honoring their dedupe
3. Settings surface (Console Behavior or Notifications category) with per-type toggles
4. Tests + docs
<!-- SECTION:PLAN:END -->
