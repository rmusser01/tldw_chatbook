---
id: TASK-352
title: >-
  Reposition feedback toasts off the composer action cluster and stop them
  swallowing clicks
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-22 04:18'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every attach/clear/limit event pops a toast anchored bottom-right, directly over the composer's right side. While it is up (measured 5.0s per toast; one persisted >=7s across typing), the staged-chip ('5 files' etc.) and the Send/Attach/Save buttons are invisible and mouse clicks at their coordinates do nothing (verified by blind-clicking the known Attach position during a toast - the picker never opened). During sequential multi-attach the controls are hidden almost continuously; in j3-30 the composer looks completely empty while 5 files are actually staged.

Also observed independently in J4 streaming as `j4-model-catalog-toast-occludes-composer-actions`: Boot-time 'Model catalog' toast sits on top of the composer's Send/Attach/Save buttons for tens of seconds.

Also observed independently in J6 keyboard-only/small-terminal as `j6-boot-toast-occludes-composer-actions`: Model-catalog boot toast covers the composer's Send/Attach/Save buttons for ~8s.

**Repro:** Console screen (2050x1240, provider home). Click composer Attach, pick any file. Immediately look at the composer's right side: the 'X attached' toast covers the chip strip and buttons for ~5s; clicking where Attach was during that window does nothing. Attach several files in a row to see the controls stay hidden.

**Verifier note:** Confirmed: j3-30 shows the 'img5.png attached' toast sitting exactly over the staged chip and Send/Attach/Save; chat_screen.py has ~95 notify() sites (attach/clear/limit all toast) using Textual's default bottom-right 5s toasts, which do capture clicks. No ledger item or backlog task covers toast placement/occlusion. Downgraded P1→P2: occlusion is transient (~5s), keyboard path unaffected, and a click dismisses the toast so the second click lands — real friction on a core flow, not a hard block.

**Source:** Console UX expert review 2026-07-20 (finding j3-toast-occludes-composer-controls, j4-model-catalog-toast-occludes-composer-actions, j6-boot-toast-occludes-composer-actions; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J3 attachments journey. Evidence: `j3-30-five-images.png`, `j3-44-just-sent.png`, `j3-24-sixth-attempt.png`, `j4-01-initial.png`, `j4-03-sent-immediate.png`, `j4-04b-gap-5s.png`, `j6-a01-baseline.png`, `j6-a02-toast-state.png`, `j6-a18-after-c.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Feedback must not obscure the primary controls it reports on. Place toasts above the composer or in the status bar, keep them short-lived, and never let them intercept clicks aimed at controls beneath
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: Textual docks notification toasts bottom-right by default
(`ToastRack { dock: bottom; align: right bottom }`) — directly over the Console
composer's Send/Attach/Save cluster and the staged-chip strip — and toasts
intercept clicks (click-to-dismiss), so a click aimed at those controls during a
~5s toast dismisses the toast instead of pressing the button. The app had no
toast override.

Fix: a one-rule `ChatScreen.DEFAULT_CSS` docks the Console screen's toast rack to
the TOP-right (`dock: top; align: right top`). A top-docked rack can never
overlap the bottom composer cluster, so feedback no longer obscures — nor
swallows clicks aimed at — the composer controls; the only thing beneath a
top-right toast is the header status chips (read-only). Kept in `DEFAULT_CSS`
rather than the CSS bundle so it applies in BOTH the real app (which loads the
bundle + widget DEFAULT_CSS) and test harnesses (which load DEFAULT_CSS but not
the built bundle) — this is what makes it regression-testable. Scoped to
`ChatScreen` (2-type selector beats Textual's 1-type `ToastRack` default);
additive to `BaseAppScreen.DEFAULT_CSS` via the MRO (44 layout tests confirm
BaseAppScreen styling intact). Toast duration left at Textual's short default —
with the occlusion removed, the ~5s lifetime is no longer harmful; changing the
global notify timeout across ~95 sites is out of scope.

Verified: RED→GREEN regression test `test_console_toast_rack_docks_top_not_over_
composer` (mounts a ToastRack under the Console screen and asserts the CSS docks
it top — the headless notification system doesn't render toasts, so the style is
asserted directly). Served-app capture confirms the composer region is clean at
boot (previously the boot "Model catalog" toast sat over it); the boot
catalog-refresh toast can't be re-triggered live because the capture harness
aborts outbound https, but the top-dock mechanism is deterministic and proven by
the test. Files: `tldw_chatbook/UI/Screens/chat_screen.py`,
`Tests/UI/test_console_toast_placement.py`.
<!-- SECTION:NOTES:END -->
