---
id: TASK-31739
title: llama.cpp real-server audio snapshot reuse qualification
status: To Do
assignee: []
created_date: '2026-09-05 19:55'
labels:
  - llamacpp
  - snapshots
  - verification
  - follow-up
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow up on TASK-31552 and PR #2419: establish whether manual prompt-cache snapshots preserve useful audio prefix state with one concrete audio-capable llama.cpp runtime and model configuration. Existing live evidence covers text and vision only. This deferred qualification task does not claim audio support or authorize implementation now. Use ADR-119 and the existing live UAT report; no new ADR is needed for test-only qualification within that boundary, but production contract changes require a separate design review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A reproducible opt-in live test identifies exact runtime, model, projector if required, audio inputs, launch settings, and prerequisites without touching the user profile or unrelated servers.
- [ ] #2 Normal Chatbook Models Save, Stop, Start, and Restore actions are exercised with real audio requests; cold, native-cache, restored-same-audio, and restored-changed-audio controls distinguish reuse from incidental text-prefix caching.
- [ ] #3 Measured cache counters or equivalent runtime evidence support every reuse claim; absent evidence or unsupported audio is reported explicitly rather than treated as a passing reuse test.
- [ ] #4 The harness terminates only its owned processes, preserves user assets, and records reproducible outcomes and limitations in the snapshot guide and live evidence report.
<!-- AC:END -->

## References

- [Completed manual manager](task-31552%20-%20llama.cpp-manual-prompt-cache-snapshot-manager.md)
- [Merged PR #2419](https://github.com/rmusser01/tldw_chatbook/pull/2419)
- [ADR-119: snapshot ownership](../decisions/119-llamacpp-prompt-cache-snapshot-ownership.md)
- [Live UAT and qualification limits](../../Docs/superpowers/reviews/2026-09-05-llamacpp-slot-snapshots-uat.md)
