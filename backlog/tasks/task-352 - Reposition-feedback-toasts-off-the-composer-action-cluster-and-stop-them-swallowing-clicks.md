---
id: TASK-352
title: Reposition feedback toasts off the composer action cluster and stop them swallowing clicks
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
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
- [ ] #1 Feedback must not obscure the primary controls it reports on. Place toasts above the composer or in the status bar, keep them short-lived, and never let them intercept clicks aimed at controls beneath
<!-- AC:END -->
