---
id: TASK-15971
title: Leaving Console no longer stages a wake — it delivers off-screen
status: Done
assignee:
  - '@claude'
created_date: '2026-08-14 01:20'
updated_date: '2026-08-14 06:04'
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
<!-- AC:BEGIN -->
**ACs rewritten 2026-08-14 to the coordinator's design ruling** (ledger
`progress.md`, wake-integrity arc): OFF-SCREEN DELIVERY IS THE INTENDED
BEHAVIOR when the Chat screen is mounted-but-hidden — the supervisor acts
immediately (that IS the auto-wake invariant; staging existed only because
the screen used to die on nav-away). The original ACs below pinned the
superseded staging behaviour and are replaced.

**Rationale correction, 2026-08-14 (task-16300). The ruling and every AC
below stand; the reason given for them was wrong.** "The screen used to
die on nav-away" reads as though it stopped doing so by design. It did
not: `App.switch_screen` pops only the TOP of the screen stack, so a
navigation issued while a modal sat above Chat replaced the MODAL and
left Chat resident — a violation of the invariant `app.py`'s
`_create_navigation_screen` documents, not a lifecycle change. Screens
die on nav-away again now (task-16300). Off-view delivery remains
intended for the reason that never depended on residency: **a supervisor
must act on its children's results immediately** (spec §3 invariant 5).
The off-view cases that reach it — a modal covering Console, a different
session tab active — are exactly the ones the live pass verified, and
ACs #1–#3 are unchanged by the fix. What the fix does change is AC #4's
neighbourhood: genuinely-unmounted staging now covers navigating away
from Console as well as restart/first boot.

<!-- AC:BEGIN -->
- [x] #1 A wake turn that completes while its conversation is not the visible/active one leaves the FLEET_UNSEEN mark set (via the named seam), so the ◈ badge points at the delivered result
- [x] #2 A mounted-but-undisplayed Console screen's sync tick does not view-clear the mark (viewing means DISPLAYED; a resident hidden screen must not count as "in Console")
- [x] #3 View-clear semantics are otherwise unchanged: viewing the conversation on the displayed Console clears a delivered wake's mark normally
- [x] #4 Genuinely-unmounted staging (restart / first boot: durable mark + mount-claim + open-as-trigger, task-15864) still passes its suites untouched
- [x] #5 The spec's honesty line and the User Guide staging story are rewritten to match the ruling
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the off-screen delivery + no-surviving-mark state RED against the ruling (real nav path)\n2. Implement: wake completing off-view sets FLEET_UNSEEN via the named seam; view-clear only when the screen is displayed\n3. Rewrite spec/User Guide honesty surfaces; update tests pinning superseded staging\n4. Keep genuinely-unmounted staging (15864) green untouched\n5. Update task-15860 with the narrowed headless case\n6. Live verify off-screen delivery -> badge -> view-clear; restart staging still delivers
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
IMPLEMENTED PER THE COORDINATOR'S DESIGN RULING (ACs rewritten to it, dated,
before implementation — the original ACs pinned the superseded staging
behaviour). Commits 243b19da7 (mechanism) + 9144b235e (honesty surfaces).
Mechanism: the delivery COMMIT now consults a screen-wired
wake_conversation_in_view probe — in-view keeps the historical clear; off-view
SETS the mark via the new named seam set_fleet_unseen_completion (badge revision
bumped); a raising probe keeps the mark (fail toward the badge); unwired doubles
keep the historical clear. 'Viewing' now means DISPLAYED: the sync tick's
view-clear is gated on _console_screen_displayed(), so a resident hidden
screen's tick can no longer consume the mark while the user is elsewhere (the
harness-diagnosed mechanism behind the live 'no mark ever' evidence — a
modal-atop-Chat navigation pops the modal and leaves Chat resident; filed as
task-16210, refiled as task-16300 after an id collision and FIXED there on
2026-08-14: navigation reduces the stack to its content screen first, so the
resident hidden screen is no longer reachable — see the rationale correction
above; this task's mechanism is unaffected, and the displayed-gate now guards
the modal-covers-Console case rather than the leak). Genuinely-unmounted staging (restart/first-boot, task-15864)
untouched — suites green unmodified; no pre-existing test pinned the superseded
mounted-but-hidden staging, so none needed updating. Tests: Tests/Chat/
test_console_fleet_wake_view_mark.py (4) + the 15971 half of Tests/UI/
test_console_fleet_wake_hidden_screen.py (4), each RED pre-fix on its own
assertion; mutations M3–M10 killed. Docs: spec §7 dated correction, User Guide
honest-limits + ◈ semantics rewrite + live-verified stamp, plan scenario-2
correction, task-15860 narrowed-scope update. Live: off-view completion
delivered immediately (stamped while a palette covered Console and another tab
was active), mark SURVIVED the commit, ◈ rendered on the tab, viewing cleared
it; restart staging re-verified (SIGKILL with owed wake → ◈ on the sidebar row
pre-open → one click → exactly-once delivery). Evidence: wake-integrity-report.md
+ wake-integrity-frames/ in the PR3a-2 ledger dir.
<!-- SECTION:NOTES:END -->

<!-- AC:END -->
