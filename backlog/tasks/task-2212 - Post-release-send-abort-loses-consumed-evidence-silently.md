---
id: task-2212
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
