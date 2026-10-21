---
id: TASK-35900
title: Repair verified Console send failures from PRs 3181 and 3185
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-21 02:17'
labels:
  - console
  - bug
  - rag
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/3181'
  - 'https://github.com/rmusser01/tldw_chatbook/pull/3185'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidate the verified discoveries and original patches by Csavoldi into one safe replacement PR. Console agent and Library sends must complete or return an actionable bounded failure without losing recoverable input, changing project-instruction preferences automatically, or weakening storage admission.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Agent sends with a selected project binding preserve owning-loop and worker-loop behavior without stranding durable acceptance.
- [x] #2 The real enhanced RAG runtime accepts normal scoped search calls and first-use startup cannot wedge retrieval indefinitely.
- [x] #3 Recovery actions release only their matching paused retrieval owner and allow empty or matching drafts while preserving later edits and attachments.
- [x] #4 Never-answerable dispatch confirmations terminate without silently disabling persisted project instructions.
- [x] #5 Repeated qualification avoids reparsing unchanged evidence but observes evidence replacement and fresh native qualification on every authority check.
- [ ] #6 Targeted regression tests and changed-file static checks pass; replacement PR and superseded PR closures credit Csavoldi.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regression coverage for owning-thread dispatch, real EnhancedRAGServiceV2 search, bounded cold-start preparation, recovery ownership, invisible confirmation expiry, and evidence-cache invalidation.
2. Apply the verified thread/RAG fixes; separate bounded initialization from search; cancel invisible confirmations without changing persisted user preferences; release only the recovery's matching retrieval preparation.
3. Cache parsed qualification evidence by a verified file generation while keeping native qualification verdicts fresh for every authorization call.
4. Run targeted suites, lint/format changed Python, review the complete diff, and update subsystem documentation.
5. Publish a replacement PR crediting Csavoldi; close #3181 and #3185 with replacement links after publication succeeds.
ADR required: no
ADR path: N/A; existing backlog/decisions/069-console-project-instruction-local-state-and-preflight.md and backlog/decisions/126-complete-local-backup-and-recovery.md apply.
Reason: repair existing send/recovery and qualification contracts without changing storage, permissions, persisted consent, or provider boundaries.
<!-- SECTION:PLAN:END -->


## Implementation Notes

Consolidates Csavoldi's discoveries and original patches from PRs #3181 and #3185. Owning-thread project guards now run directly while worker callbacks preserve Textual scheduling. Enhanced RAG search no longer forwards an unsupported argument; unsupported precomputed vectors are explicitly refused. Console initialization has a separate finite 60-second budget before the existing 5-second retrieval budget on preparation and retry paths.

Unsent-turn recovery carries the exact preparation ID, preserves empty/matching and newer drafts/attachments, and releases only its matching retrieval pause. Cancellation failure retains the recovery entry. Never-answerable dispatch decisions cancel without changing persisted consent; existing once-mounted detached-time policy stays intact. Qualification caches only parsed evidence under descriptor-verified file-generation stamps, using native Windows ChangeTime; native identity/verdict checks remain fresh. Logs contain status/error categories and exception types, not draft or exception bodies.

Targeted red/green evidence: the real Textual/stock-composition test reproduced blocked durable acceptance before the thread fix; four thread/boundary cases pass after it. Nine RAG startup/search cases pass, including stalled startup, cancellation and retry. Runtime ownership/confirmation selection: 83 passed, plus 10 affected cases after cancellation-failure hardening; removing the preparation-ID guard in memory fails the newer-owner protection test. Qualification: 23 passed, including nine new cache regressions, 11 original malformed-evidence cases and three native publication cases. One unrelated symlink test requires Windows symlink privilege and was not changed.

Independent read-only review found no actionable findings. All 11 changed/new Python files introduce zero Ruff diagnostics compared with dev (69 existing diagnostics unchanged); new files and changed ranges are formatted, and git diff --check passes. Architecture docs and the Windows private-test-parent lesson are updated. Wider adjacent-run and baseline comparison results are recorded below before publication closeout.

ADR required: no. Existing ADR-069, ADR-126 and ADR-230 govern these restored contracts; no schema, dependency, persisted-consent or authority-policy change is introduced.


### Adjacent verification and baseline controls

The wider targeted Library/automatic-preparation run finished with 159 passed, 2 skipped, 4 failed and 2 setup errors. All six failures/errors reproduce on an untouched `origin/dev` (`2d071d6d7145184ba41408ec437306cfb188ea57`) snapshot with the same private-profile setup:

- `test_text_with_attachment_is_automatic_eligible_and_preserved_until_ready`;
- `test_recovery_uses_frozen_staged_inputs_and_leaves_new_state_staged[retry]` and `[bypass]`;
- `test_concurrent_rag_queries_initialize_one_shared_service`;
- `test_notes_keyword_search_matches_plural_and_singular_variants[feedback loop]` and `[feedback loops]` (setup errors).

The first group is refused by the existing image-capability test setup before retrieval; the Library cases encounter `raw_source_selection_changed`. These controls are documented rather than changed under this task. Two real-embedding integrations are skipped because optional dependencies are absent. No full suite was run.


The exact UI resend test that exposed #3185's bad enum import was also run. On this Windows host, both replacement and untouched dev fail earlier at the same `llama.cpp unavailable` transcript wait (23.72s and 22.63s call times). This is a seventh baseline-reproduced adjacent failure, not a passing UI claim. The direct recovery/real-controller tests cover the corrected import and release path. Local evidence remains in the ignored/untracked test directories and is not shipped.
