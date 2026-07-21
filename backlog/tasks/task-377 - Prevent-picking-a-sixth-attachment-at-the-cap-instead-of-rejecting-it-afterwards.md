---
id: TASK-377
title: Prevent picking a sixth attachment at the cap instead of rejecting it afterwards
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With 5 images staged, the attach button still opened the full file picker; I navigated and selected img6.png; the picker closed exactly like a success and only a toast said 'Attachment limit reached (5 per message).' (a toast that itself renders over the composer controls and expires in ~5s). The attach affordance gives no cue that the cap is reached.

**Repro:** Stage 5 images, click the paperclip button again, select a 6th image. Picker closes normally; only a transient toast reports the rejection.

**Verifier note:** Code-confirmed: the attach button always opens the picker; the cap is enforced only after processing, when store.add_pending_attachment returns False and a toast fires (chat_screen.py:9968-9974). No pre-picker gate, no 5/5 annotation; nothing in task-217/222/230 or the ledger records this as decided. Downgraded P2→P3: pure error-prevention polish — the rejection copy itself is precise and the dead work is one picker round-trip.

**Source:** Console UX expert review 2026-07-20 (finding j3-cap-not-prevented-at-picker; P3, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J3 attachments journey. Evidence: `j3-31-sixth-image.png`, `j3-30-five-images.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Disable or annotate the attach affordance at the cap ('5/5'), or block inside the picker before file selection, so the user never does dead work
<!-- AC:END -->
