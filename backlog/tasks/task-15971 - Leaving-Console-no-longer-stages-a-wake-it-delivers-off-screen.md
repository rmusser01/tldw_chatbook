---
id: TASK-15971
title: Leaving Console no longer stages a wake — it delivers off-screen
status: To Do
assignee: []
created_date: '2026-08-14 01:20'
labels:
  - fleet
  - console
dependencies: []
priority: high
---

## Description

PR3a-2 residue-arc live verification (2026-08-13, scratch profile, real
Anthropic, dev @cb89f3ff6 + the residue fixes): a survivor was spawned in
Console, the user navigated to Library BEFORE settle, and the wake turn
FIRED anyway — run 3c0cdfaf stamped `wake_delivered_at` 2026-08-14T01:08:53Z
while the Library screen was displayed, with no `fleet_unseen` mark ever
written. Instrumentation showed the delivery hook logging `mounted=True`
on the Chat screen during Library display: on the current dev nav
lifecycle the Console screen stays resident (mounted) on nav-away, so the
controller's `_shutdown_requested` gate never trips and
`_resolve_session_id` still finds the open session — every staging
precondition the wake design relies on is absent.

This contradicts both the verified behaviour and the published contract:
PR3a-2 Task 7 scenario 2 (branch `feat/fleet-autowake` @e38e62a2f) proved
the same flow STAGING — "run settled done with stamp NULL (staged, never
fired off-screen)", durable mark written, mount-claim delivery on return —
and the spec/PR-body honesty line says completions landing while the user
is elsewhere are "recorded and delivered when Console next mounts — never
acted on invisibly in the background". A supervisor turn now runs (and
spends money) while the user is on another screen, with no visible
transcript. The regression window is the dev merge between e38e62a2f and
cb89f3ff6 (nav/screen-residency changes suspected); the residue arc's own
changes gate nothing on this path (verified: its delivery hook only arms a
repaint timer).

Evidence: `.superpowers/sdd/2026-08-13-supervisor-fleet-pr3a2-autowake/`
residue-frames/dbg.log (`delivery-hook fired: mounted=True` at
1786669812/1786669814 during Library display) and the residue report.

## Acceptance Criteria

- [ ] #1 A survivor settling while Console is not the displayed screen stages its wake (durable mark + NULL stamp) instead of delivering
- [ ] #2 The staged wake delivers exactly once when Console is next displayed
- [ ] #3 A test pins the not-displayed staging path against the resident-screen nav lifecycle (a mounted-but-undisplayed Console must not count as "in Console")
