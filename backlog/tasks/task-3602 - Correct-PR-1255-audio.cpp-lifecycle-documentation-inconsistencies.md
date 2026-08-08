---
id: TASK-3602
title: Correct PR 1255 audio.cpp lifecycle documentation inconsistencies
status: Done
assignee:
  - '@codex'
created_date: '2026-08-08 04:51'
updated_date: '2026-08-08 14:04'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/1255'
  - 'https://github.com/rmusser01/tldw_chatbook/pull/1432'
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
- [x] #1 The Speech & TTS ownership summary states that Global Settings owns durable audio.cpp External and Managed configuration while lifecycle operations and diagnostics remain outside Settings.
- [x] #2 The adapter-registry design distinguishes the original 2026-07-23 approval from the user-approved 2026-08-02 managed-lifecycle amendment without conflating written-specification approval with per-install runtime artifact selection and trust.
- [x] #3 All PR 1255 review findings are answered with verification evidence, and documentation checks pass on the latest dev base.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both PR 1255 documentation inconsistencies against latest origin/dev and compare them with ADR-023, ADR-039, and the managed-lifecycle design.
2. Update the Speech & TTS ownership summary so durable External/Managed configuration and operational lifecycle ownership match the accepted ADRs.
3. Update the adapter-registry header metadata so the original approval and managed-lifecycle amendment status are explicit, including the later user approval of the written specification without conflating it with per-install runtime artifact selection and trust.
4. Run focused text assertions, Markdown/link checks available in the repository, git diff --check, and review the complete diff against all acceptance criteria.
5. Reply to and resolve the inline review thread, report the summary finding disposition, and update the task with verification evidence.

ADR required: no
ADR path: N/A
Reason: This is a documentation-consistency correction that directly applies accepted ADR-023 and ADR-039 without changing storage, ownership, runtime, security, or interface decisions.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Resolved the merged PR 1255 documentation findings and the follow-up PR 1432 chronology finding. Updated the Speech and TTS ownership summary to match ADR-023 and ADR-039, separated the initial adapter-registry approval from the later managed-lifecycle amendment approval, and recorded the user-approved managed-lifecycle specification without conflating it with per-install runtime artifact trust. Replied to and resolved both inline review threads and corrected the PR 1255 summary disposition. Verification: rebased on current origin/dev; full unsandboxed Tests/TTS suite passed 2280 with 16 optional/live skips; focused release-evidence tests passed 2/2; exact approval and ownership assertions passed; backlog duplicate-ID guard and git diff --check passed. The sandboxed full TTS run had one local-socket PermissionError, and the identical unsandboxed command passed. ADR required: no; the changes apply existing ADR-023 and ADR-039.
<!-- SECTION:NOTES:END -->
