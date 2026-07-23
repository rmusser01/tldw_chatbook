---
id: TASK-370
title: Mark interrupted replies as stopped and offer resume or retry
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After stopping, the partial assistant message renders exactly like a completed one: it just ends mid-sentence ('A' in run A; '...search through a potentially massive log file to find' in run B) with no 'stopped/partial' badge, no dimming, nothing. After restart the fragment still looks like a normal answer (j4-28). There is no Retry/Regenerate control anywhere in the transcript; the only hint that Regenerate/Continue exist is a text block buried in the Inspector ('Message actions: Copy, Edit, Save as..., Regenerate, Continue, Feedback, Delete / Keyboard: Tab/Shift+Tab cycle actions; Enter activates') describing an invisible message-selection keyboard flow.

**Repro:** Stop a streaming reply -> inspect the partial message in the transcript, then reload the conversation from the rail -> fragment has no interruption marker; search the transcript for any retry control.

**Verifier note:** Partially overstated: mid-session a stopped message DOES render a ' [stopped]' suffix (console_transcript.py:93-94 appends '[{status}]' for streaming/stopped/failed) — run A never reached 'stopped' (the task-227 race above) and run B's tail was below the fold (j4-36 is scrolled up), so the journey likely never saw it. The genuinely new residue: stopped status is not persisted, so after restart the fragment is indistinguishable from a finished answer (mark_message_stopped flushes content only). 'No retry affordance' is not a gap — Regenerate/Continue exist via click-select action row (shipped, message-actions-save-as-note-e2e ledger item); its discoverability is already tracked as open-message-action-affordance (P3). Downgrade to P3: only the persistent-marker gap survives verification.

**Source:** Console UX expert review 2026-07-20 (finding j4-interrupted-reply-unmarked-no-retry; P3, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J4 streaming journey. Evidence: `j4-28-click-partial-message.png`, `j4-38-after-tab-flip.png`, `j4-36-just-after-stop.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An interrupted message should carry a persistent visible marker ('stopped by user - partial') and offer an in-place Continue/Regenerate affordance right on the fragment
<!-- AC:END -->
