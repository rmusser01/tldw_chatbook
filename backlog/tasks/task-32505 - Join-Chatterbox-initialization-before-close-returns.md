---
id: TASK-32505
title: Join Chatterbox initialization before close returns
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-12 15:37'
updated_date: '2026-09-12 16:31'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Chatterbox starts detached initialization, spawn and readiness work that can publish a process or initialized state after close has returned. Preserve responsive startup while ensuring closing the backend owns and settles initialization work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 initialize remains nonblocking while initialization ownership is retained through spawn, readiness and native fallback.
- [x] #2 Closing during queued startup, in-flight process acquisition or readiness cannot launch or publish resources after close completes.
- [x] #3 Cancellation and repeated close calls retain cleanup until child processes and native fallback work have settled.
- [x] #4 Targeted initialization and existing audio-delivery tests pass, with real CUDA validation repeated for the corrected registered backend.
- [x] #5 Generation queued or waiting during shutdown cannot begin native synthesis after closing starts.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR-023 TTS operation ownership applies. Reason: repair detached initialization work without changing provider contracts or supported runtime choices.
1. Preserve the bounded queued-startup, gated-spawn and stale-readiness reproductions.
2. Retain initialization ownership through process acquisition, readiness and native fallback; seal startup when close begins and join cleanup despite cancellation.
3. Add focused regression tests for close during startup, late process acquisition, stale readiness, native fallback and repeated/cancelled close; run existing Chatterbox audio-delivery tests.
4. Independently review the fix, build a clean wheel, repeat registered Chatterbox CUDA Lab/Console/cancellation/repeat validation, and retain exact evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented retained asynchronous initialization ownership across subprocess acquisition, readiness and native fallback. Closing seals startup immediately and shares cancellation-safe cleanup; failed initialization children are reaped before fallback. Nine new lifecycle regressions and 30 existing audio-delivery tests passed (39); implementation review found no actionable issues. Formatter/new-test lint passed; production Ruff diagnostics matched the unchanged baseline. Subsequent combined targeted verification passed 101 tests.
The corrected installed wheel (base 8ab21ecaf3 plus this backend fix) passed real queued-startup/close with no pending tasks and exit 0, then Chatterbox CUDA run 03 passed Lab, Console, inference-overlap cancellation, successor, three repeats and synthetic-reference audition. All 2,352 source/wheel/installed Python files matched; seven clips passed independent full-text Whisper-medium verification, preserving the original small-model 6/7 discrepancy. Final child/process/GPU/audio ownership cleared. Evidence: Docs/QA/tts-linux-cuda-2026-09-12/chatterbox/README.md and runs/chatterbox-fixed-startup-close/result.json beneath it. Changed production/test files are tldw_chatbook/TTS/backends/chatterbox.py and Tests/TTS/test_chatterbox_initialization_lifecycle.py; QA and the qualification plan document the result. All ACs are evidenced for that fixed source, but status remains In Progress pending final review and validation after rebasing onto newer dev audio-player changes. ADR required: no; existing backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md governs retained ownership.

Rebased qualification on dev a766133fc4 plus the unchanged lifecycle fix passed all three runtime/playback/cleanup checks. Current source, wheel and both installed environments match all 2,344 Python files. Chatterbox run04 default clips and reference audio passed small ASR 7/7; preserve medium 6/7 reference discrepancy and orchestration exit1 as a recognizer disagreement requiring human listening. Rebased targeted tests: 217 passed. Ruff/format, baseline diagnostic statement comparison, task-ID guard and profile-owned-path census passed; independent code/evidence review found no actionable issues. See Docs/QA/tts-linux-cuda-2026-09-12/rebased/README.md. The lifecycle fix ACs are complete; listening remains in older qualification tasks. ADR required: no; existing ADR-023 applies.

Qodo review identified an additional admission race during retained startup joining. Reopen to reproduce with a bounded concurrency test, reject generation under the generation lock once closing begins, and repeat targeted plus affected real CUDA validation before completion.

Qodo shutdown finding reproduced with two bounded regressions: a queued ready request and fallback publishing initialized state during close both reached native generation after closing began. Guards under the generation lock and after readiness waiting now reject both while retaining in-flight cleanup. All 229 targeted cases pass, including the real isolated-worker encoding protocol. Final affected Linux rerun pending.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Renumbered from TASK-32494 to TASK-32505 on 2026-09-12 after rebasing onto a766133fc4. The already-landed Cross-platform Console reply speech playback task keeps TASK-32494. A refreshed sweep of 57 origin refs and 41 worktrees found maximum task ID 32504. The Backlog CLI has no renumber command; filename/frontmatter and this task's owned references were updated together.
