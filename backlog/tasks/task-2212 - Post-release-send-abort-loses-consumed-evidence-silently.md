---
id: TASK-2212
title: Post-release send abort loses consumed staged evidence silently
status: To Do
assignee: []
labels:
  - console
  - rag
dependencies: []
priority: low
---

## Description

After PR-4's consume-on-send releases staged evidence (successful capture), a genuine task cancellation (screen teardown mid-send) or a raise in _resolve_submit_prefill can abort the send before the message completes — the evidence is unrecoverably consumed while the strip's transient claims "Evidence sent with this message". Window is narrow (Stop is cooperative; dictionary/world-info swallow non-cancel exceptions; only true cancellation reaches it). Options: re-stage on cancellation, or soften the transient copy to "Evidence attached to the outgoing message".

## Acceptance Criteria

- [ ] A cancelled-after-capture send either restores staging or the UI never claims the evidence was sent
- [ ] The chosen behavior is pinned by a test holding the cancellation window open

## Note (2026-08-04, PR-T1 backlog audit)

The 2026-08-04 RAG re-score critique initially mapped one of its P0 findings to this task ID. PR-T1's scout and review analysis determined that critique finding describes a **different** defect than the one below (the D1/D2/D3/D4 truth-and-integrity defects fixed by this PR, recorded in task-2370 through task-2374) — this task's post-cancel evidence-loss window (a genuine task cancellation or a raise in `_resolve_submit_prefill` consuming staged evidence before the send actually lands) is a separate, still-unfixed defect. This task **remains open and valid**; status is unchanged.
