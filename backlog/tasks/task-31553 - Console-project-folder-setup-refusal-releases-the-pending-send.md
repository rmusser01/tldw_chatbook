---
id: TASK-31553
title: Console project-folder setup refusal releases the pending send
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 02:46'
updated_date: '2026-09-05 03:05'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live Migu provider UAT found that disabling project instructions during first send leaves Console validating forever with an empty assistant row, preventing a retry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Disable, cancel, and unavailable setup callbacks terminate the unsent run without invoking the provider.
- [x] #2 The assistant placeholder and optimistic user echo are marked unsent and the composer can retry.
- [x] #3 Targeted regressions and isolated mounted provider UAT verify recovery.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/069-console-project-instruction-local-state-and-preflight.md
Reason: Routine lifecycle repair preserving the existing explicit setup decision and provider authority boundary.
1. Reproduce disable, cancel and unavailable callback refusals through a real ConsoleChatController and assert terminal state and retryability.
2. Finalize refused setup through the existing preflight block path and mark the unsent user echo blocked without expanding provider authorization.
3. Run targeted project-instruction and provider lifecycle regressions, then retry the mounted synthetic DeepSeek send through the visible composer.
4. Record native Buddy, provider, voice and governance evidence; keep hardware-only acceptance open until observed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired setup disable/cancel/unavailable callback refusals through the existing context-preflight terminal path; unsent user/assistant rows are failed and the run releases its slot. Keyboard stashes survive a consumed accepted-hook slot; mouse sends restore their immutable snapshot only while the same visible session and unchanged edit serial still own an empty draft. Born-red regressions: 3 controller cases and 3 draft cases; targeted controller/refusal 183 passed, project-instruction/refusal 48 passed, draft 17 passed. Real isolated DeepSeek New Chat → Send → Disable → Send returned the exact synthetic response, with the original draft restored and Buddy returning to idle. ADR069 existing boundary applies; no new ADR or authority expansion. Scoped Ruff/changed-range formatting/compile/diff pass. Evidence: qa/buddy-uat-2026-09-04/followup-report.md and live-provider-voice.json.

Final post-format mounted hands-free/draft/refusal/ratchet group: 67 passed. Scoped Ruff, changed-range formatter checks and diff whitespace checks pass. All task acceptance criteria verified; actual microphone/OpenAI tests remain with the broader UAT, not this setup-refusal repair.
<!-- SECTION:NOTES:END -->
