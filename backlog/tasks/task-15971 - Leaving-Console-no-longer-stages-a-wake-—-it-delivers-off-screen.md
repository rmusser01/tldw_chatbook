---
id: TASK-15971
title: Leaving Console no longer stages a wake — it delivers off-screen
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-14 01:20'
updated_date: '2026-08-14 02:53'
labels:
  - fleet
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
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
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

**ACs rewritten 2026-08-14 to the coordinator's design ruling** (ledger
`progress.md`, wake-integrity arc): OFF-SCREEN DELIVERY IS THE INTENDED
BEHAVIOR when the Chat screen is mounted-but-hidden — the supervisor acts
immediately (that IS the auto-wake invariant; staging existed only because
the screen used to die on nav-away). The original ACs below pinned the
superseded staging behaviour and are replaced.

<!-- AC:BEGIN -->
- [ ] #1 A wake turn that completes while its conversation is not the visible/active one leaves the FLEET_UNSEEN mark set (via the named seam), so the ◈ badge points at the delivered result
- [ ] #2 A mounted-but-undisplayed Console screen's sync tick does not view-clear the mark (viewing means DISPLAYED; a resident hidden screen must not count as "in Console")
- [ ] #3 View-clear semantics are otherwise unchanged: viewing the conversation on the displayed Console clears a delivered wake's mark normally
- [ ] #4 Genuinely-unmounted staging (restart / first boot: durable mark + mount-claim + open-as-trigger, task-15864) still passes its suites untouched
- [ ] #5 The spec's honesty line and the User Guide staging story are rewritten to match the ruling
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the off-screen delivery + no-surviving-mark state RED against the ruling (real nav path)\n2. Implement: wake completing off-view sets FLEET_UNSEEN via the named seam; view-clear only when the screen is displayed\n3. Rewrite spec/User Guide honesty surfaces; update tests pinning superseded staging\n4. Keep genuinely-unmounted staging (15864) green untouched\n5. Update task-15860 with the narrowed headless case\n6. Live verify off-screen delivery -> badge -> view-clear; restart staging still delivers
<!-- SECTION:PLAN:END -->
