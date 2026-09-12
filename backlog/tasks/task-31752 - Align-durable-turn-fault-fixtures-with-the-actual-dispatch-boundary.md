---
id: TASK-31752
title: Align durable-turn fault fixtures with the actual dispatch boundary
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 20:10'
updated_date: '2026-09-05 20:14'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep durable settlement and unknown-delivery retry regressions causally aligned with the deferred provider dispatch callback instead of injecting pre-dispatch faults under post-dispatch expectations.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The settlement fixture crosses the real checkpoint boundary before issuing its token and verifies exact-token rollback while a newer token remains current.
- [x] #2 The provider-entry fault occurs after the real dispatch boundary, retains the same durable owners and warned recovery, and makes no gateway call before explicit retry.
- [x] #3 The complete affected durable-turn test file and scoped static checks pass with no runtime changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record two original failures and trace deferred before_provider_dispatch ownership under ADR-079; verify the callback-only hypothesis in memory.
2. After parent cause review, make only the two fault doubles invoke the supplied callback at the intended boundary and add state/owner assertions before faults. Preserve existing token and retry outcomes.
3. Run the complete affected file and related settlement checks, scoped lint/format, and independent review before commit.
ADR required: no
ADR path: backlog/decisions/079-console-library-conversation-authority.md
Reason: Test-only fixture correction preserving existing accepted-versus-dispatch-started authority; no runtime or contract change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected only the two post-dispatch fault doubles: invoke the real supplied before_provider_dispatch callback, verify the exact in-flight assistant and DISPATCH_STARTED checkpoint, then inject the settlement token/fault. Existing older-versus-newer token rollback, warned retry, same durable owners, and gateway zero-before/one-after assertions remain intact. Callback crossing does not claim gateway entry. Added a short incident lesson in lessons-testing-evidence. ADR required: no new ADR; preserves ADR-079 checkpoint authority.
Baseline: 2 failed/1 passed in1.67s; callback-only in-memory probe made all3 pass. Complete affected round1 and seven related files:126 passed/1 failed in103.55s (/private/tmp/tldw-31752-durable-recovery.xml), including all25 round1 cases and its1000-turn retention check. The one round2 checkpoint_transition provider_started classification failure reproduces with the exact HEAD helper:18 passed/1 failed in7.86s (/private/tmp/tldw-31752-round2-head.xml); it remains separately queued, not waived. Aggregate run emitted a459-descriptor growth warning; no broad cleanup or threshold change was made.
Whole-file Ruff lint/format and git diff whitespace passed. Parent reviewed the exact test diff and approved scoped completion with the independently proven baseline failure and resource warning explicitly qualified.
<!-- SECTION:NOTES:END -->
