---
id: TASK-351
title: Reduce or mask slow first feedback after primary Console actions
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Cluster of measured instances: (a) Enter-to-send: 350ms after Enter the composer still held the full text and the transcript said 'No messages yet' (j4-03); in another session the first transcript change after Enter took 7.3s (echo of the user's own message), though a later send echoed in 1.0s; (b) rail conversation clicks: in one session clicking 'No tools: explain...' produced no visible change within 1.2s and clicking 'Write a detailed...' never opened it at all during ~10s of subsequent interaction (transcript stayed on the empty 'Chat 1'), while in the next session the same click opened in 0.6s - silent intermittent failure with no pressed/loading feedback; (c) clicking 'Inspector' during a run showed no response within 0.7s (j4-32 still collapsed) and the panel was simply found open in a later frame.

**Repro:** (a) Type a prompt, press Enter, watch composer/transcript in the first second; (b) right after app start, click a saved conversation in the rail Chats list and wait - sometimes nothing happens; (c) click 'Inspector' during a run.

**Verifier note:** Not covered by the perf ledger: task-280/259 fixed tick/DB-on-loop and transcript derivation, and no item covers first-send echo latency, silently-failing rail conversation clicks, or delayed Inspector toggle. The 7.3s worst-case echo and dead rail clicks are intermittent single-session measurements (hence medium confidence) but three independent sub-symptoms point at on-loop work in the send/resume paths (e.g. first-send agent-bridge/MCP catalog init). No pressed/loading acknowledgment on rail conversation rows is verifiable by design (plain Buttons, no busy state). P2 appropriate.

**Source:** Console UX expert review 2026-07-20 (finding j4-first-feedback-latency-cluster; P2, verdict NEW, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J4 streaming journey. Evidence: `j4-03-sent-immediate.png`, `j4-18-first-change.png`, `j4-26-partial-after-restart.png`, `j4-32-inspector-midrun.png`, `j4-33-streaming3.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every primary click/submit should acknowledge within ~100ms (pressed state, optimistic echo of the sent message, loading indicator on conversation open)
<!-- AC:END -->
