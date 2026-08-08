---
id: TASK-3602
title: Correct PR 1255 audio.cpp lifecycle documentation inconsistencies
status: In Progress
assignee:
  - '@codex'
created_date: '2026-08-08 04:51'
updated_date: '2026-08-08 04:52'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/1255'
documentation:
  - Docs/superpowers/specs/2026-07-23-audio-cpp-tts-adapter-registry-design.md
  - Docs/superpowers/specs/2026-07-31-speech-tts-settings-ownership-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Align the merged audio.cpp lifecycle design documentation with its authoritative ownership and approval-state decisions so implementers do not receive contradictory guidance from PR 1255.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The Speech & TTS ownership summary states that Global Settings owns durable audio.cpp External and Managed configuration while lifecycle operations and diagnostics remain outside Settings.
- [ ] #2 The adapter-registry design distinguishes the original 2026-07-23 approval from the 2026-08-02 managed-lifecycle amendment status without implying that user-provided artifacts are already approved.
- [ ] #3 All PR 1255 review findings are answered with verification evidence, and documentation checks pass on the latest dev base.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both PR 1255 documentation inconsistencies against latest origin/dev and compare them with ADR-023, ADR-039, and the managed-lifecycle design.
2. Update the Speech & TTS ownership summary so durable External/Managed configuration and operational lifecycle ownership match the accepted ADRs.
3. Update the adapter-registry header metadata so the original approval and managed-lifecycle amendment status are explicit, including pending user-provided artifact approval.
4. Run focused text assertions, Markdown/link checks available in the repository, git diff --check, and review the complete diff against all acceptance criteria.
5. Reply to and resolve the inline review thread, report the summary finding disposition, and update the task with verification evidence.

ADR required: no
ADR path: N/A
Reason: This is a documentation-consistency correction that directly applies accepted ADR-023 and ADR-039 without changing storage, ownership, runtime, security, or interface decisions.
<!-- SECTION:PLAN:END -->
