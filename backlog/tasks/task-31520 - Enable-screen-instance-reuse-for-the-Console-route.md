---
id: TASK-31520
title: Enable screen-instance reuse for the Console route
status: To Do
assignee: []
created_date: '2026-09-04 21:30'
labels:
  - performance
  - console
dependencies:
  - task-24452
priority: high
---

## Description (the why)

TASK-24452 landed opt-in screen-instance reuse (`ScreenRoute.reusable`:
construct once, install, suspend instead of unmount) and proved it on Home.
The measured headroom for Console is the largest in the app: a warm
installed-instance switch to ChatScreen cost 158 ms CPU against 750-820 ms
for today's fresh construction (-80%, interleaved arms, 2026-09-04), and
Console's ~559-widget re-mint per visit disappears. Enablement is gated on
a lifecycle audit: `ChatScreen.on_unmount` tears down ~a dozen subsystems
per visit (sidebar-state flush, terminal workspace detach, auto-speak,
image-edit cancels, transcript-sync/fleet-survivor/cost-TTL timers,
roleplay persistence, hands-free, realtime, dictation), and with reuse
those must become suspend/resume-aware instead -- otherwise they keep
ticking on a hidden screen. TASK-1143 F5's leave-confirmation ("navigating
away cancels runs") also changes meaning when leaving no longer unmounts.

## Acceptance Criteria (the what)

- [ ] Every `ChatScreen.on_unmount` teardown step is dispositioned for reuse: paused on `on_screen_suspend` + resumed on `on_screen_resume`, moved to app-lifetime ownership, or documented as intentionally continuing while hidden
- [ ] No Console-owned timer fires while the Console is suspended (measured, not asserted from code)
- [ ] Repeat visits to the Console resume the same instance and construct materially fewer widgets (the TASK-24452 guard pattern, applied to the chat route)
- [ ] Console switch CPU improves in an interleaved A/B against the fresh-instance baseline
- [ ] Console behaviour and focus placement are unchanged across a switch-away-and-back cycle, including active-run, pending-approval, and dictation-active states
- [ ] The TASK-1143 F5 leave-confirmation copy/behavior is reconciled with runs surviving navigation
