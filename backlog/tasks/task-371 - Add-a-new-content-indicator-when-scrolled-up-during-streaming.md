---
id: TASK-371
title: Add a new-content indicator when scrolled up during streaming
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PageUp mid-stream did scroll the view up and the position was respected (no yank-back - good). But the stream kept growing below the fold silently: no unread badge, no 'jump to latest' affordance, no signal when generation finished or was stopped. 20 seconds after Stop the viewport still sat mid-message showing a sentence cut by the pane edge (j4-37), with the user unable to tell from this view whether the reply was still streaming, finished, or stopped.

**Repro:** During a streaming reply press PageUp (after clicking transcript text once) -> stay scrolled up -> no indicator of ongoing streaming below; stop the run -> still no signal from the scrolled position.

**Verifier note:** Real gap but an enhancement on top of deliberate task-298 behavior (no-yank while detached is the shipped contract; nothing in the ledger promises an unread/'jump to latest' pill). Standard chat affordance absent; user can End/PageDown back. Downgrade P2→P3: nothing misbehaves, information is merely unavailable from the scrolled position.

**Source:** Console UX expert review 2026-07-20 (finding j4-no-new-content-indicator-when-scrolled-up; P3, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J4 streaming journey. Evidence: `j4-36-just-after-stop.png`, `j4-37-stop-plus-20s.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 While detached from the bottom during streaming, show a persistent indicator ('▼ streaming below / jump to latest') that also reflects completion or interruption
<!-- AC:END -->
