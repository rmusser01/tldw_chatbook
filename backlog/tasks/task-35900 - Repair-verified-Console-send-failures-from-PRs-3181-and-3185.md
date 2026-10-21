---
id: TASK-35900
title: Repair verified Console send failures from PRs 3181 and 3185
status: Done
assignee:
  - '@codex'
created_date: '2026-10-21 02:17'
updated_date: '2026-10-21 05:18'
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
- [x] #6 Targeted regression tests and changed-file static checks pass; replacement PR and superseded PR closures credit Csavoldi.
- [x] #7 Recovery action storage failures remain nonfatal in the real UI worker, retain the unsent recovery, show content-free retry guidance, and retire the exact transient echo and continuation before recovery removal on retry.
- [x] #8 Mounted approval projection journeys begin only after the current Console attachment is reconciled and answerable; their original card, count, navigation and decision assertions remain enforced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regression coverage for owning-thread dispatch, real EnhancedRAGServiceV2 search, bounded cold-start preparation, recovery ownership, invisible confirmation expiry, and evidence-cache invalidation.
2. Apply the verified thread/RAG fixes; separate bounded initialization from search; cancel invisible confirmations without changing persisted user preferences; release only the recovery's matching retrieval preparation.
3. Cache parsed qualification evidence by a verified file generation while keeping native qualification verdicts fresh for every authorization call.
4. Keep the controller within its existing size ratchet by mechanically delegating the retrieval preparation routine to a focused helper; run targeted suites, lint/format changed Python, review the complete diff, and update subsystem documentation.
5. Add a real-Textual-worker regression for Restore/Discard storage failures, retain recoveries on failure, surface bounded retry guidance, and verify saved-chat SQLite failures leave cancellation retryable without leaking discarded text into later provider history before pre-merge review.
6. For CI UI10, witness completed and answerable Console attachment before mounting approval journeys, reproduce the incomplete-attachment legacy behavior, and rerun the original assertions.
7. Publish a replacement PR crediting Csavoldi; close #3181 and #3185 with replacement links after publication succeeds.
ADR required: no
ADR path: N/A; existing backlog/decisions/069-console-project-instruction-local-state-and-preflight.md and backlog/decisions/126-complete-local-backup-and-recovery.md apply.
Reason: repair existing send/recovery and qualification contracts without changing storage, permissions, persisted consent, or provider boundaries.
<!-- SECTION:PLAN:END -->


## Implementation Notes

Consolidates Csavoldi's discoveries and original patches from PRs #3181 and #3185. Owning-thread project guards now run directly while worker callbacks preserve Textual scheduling. Enhanced RAG search no longer forwards an unsupported argument; unsupported precomputed vectors are explicitly refused. Console initialization has a separate finite 60-second budget before the existing 5-second retrieval budget on preparation and retry paths.

Unsent-turn recovery carries the exact preparation ID, preserves empty/matching and newer drafts/attachments, and releases only its matching retrieval pause. Cancellation failure retains the recovery entry. Never-answerable dispatch decisions cancel without changing persisted consent; existing once-mounted detached-time policy stays intact. Qualification caches only parsed evidence under descriptor-verified file-generation stamps, using native Windows ChangeTime; native identity/verdict checks remain fresh. Logs contain status/error categories and exception types, not draft or exception bodies.

Targeted red/green evidence: the real Textual/stock-composition test reproduced blocked durable acceptance before the thread fix; four thread/boundary cases pass after it. Nine RAG startup/search cases pass, including stalled startup, cancellation and retry. Runtime ownership/confirmation selection: 83 passed, plus 10 affected cases after cancellation-failure hardening; removing the preparation-ID guard in memory fails the newer-owner protection test. Qualification: 23 passed, including nine new cache regressions, 11 original malformed-evidence cases and three native publication cases. One unrelated symlink test requires Windows symlink privilege and was not changed.

The initial independent read-only review found no actionable findings. All 11 changed/new Python files introduce zero Ruff diagnostics compared with dev (69 existing diagnostics unchanged); new files and changed ranges are formatted, and git diff --check passes. Architecture docs and the Windows private-test-parent lesson are updated. Wider adjacent-run and baseline comparison results are recorded below before publication closeout.

ADR required: no. Existing ADR-069, ADR-126 and ADR-230 govern these restored contracts; no schema, dependency, persisted-consent or authority-policy change is introduced.


### Adjacent verification and baseline controls

The wider targeted Library/automatic-preparation run finished with 159 passed, 2 skipped, 4 failed and 2 setup errors. All six failures/errors reproduce on an untouched `origin/dev` (`2d071d6d7145184ba41408ec437306cfb188ea57`) snapshot with the same private-profile setup:

- `test_text_with_attachment_is_automatic_eligible_and_preserved_until_ready`;
- `test_recovery_uses_frozen_staged_inputs_and_leaves_new_state_staged[retry]` and `[bypass]`;
- `test_concurrent_rag_queries_initialize_one_shared_service`;
- `test_notes_keyword_search_matches_plural_and_singular_variants[feedback loop]` and `[feedback loops]` (setup errors).

The first group is refused by the existing image-capability test setup before retrieval; the Library cases encounter `raw_source_selection_changed`. These controls are documented rather than changed under this task. Two real-embedding integrations are skipped because optional dependencies are absent. No full suite was run.


The exact UI resend test that exposed #3185's bad enum import was also run. On this Windows host, both replacement and untouched dev fail earlier at the same `llama.cpp unavailable` transcript wait (23.72s and 22.63s call times). This is a seventh baseline-reproduced adjacent failure, not a passing UI claim. The direct recovery/real-controller tests cover the corrected import and release path. Local evidence remains in the ignored/untracked test directories and is not shipped.


### Publication closeout

Published replacement PR #3189: https://github.com/rmusser01/tldw_chatbook/pull/3189 . Closed #3181 and #3185 only after replacement creation succeeded, with links and explicit thanks to @Csavoldi. The replacement description credits discovery, reproductions, root-cause investigation and original patches; implementation commit `b2e5e95076` includes `Co-authored-by: Csavoldi <clemsavoldi@gmail.com>`. At publication, CI remained the merge gate and no merge or auto-merge had been requested or performed.


### Pre-merge review and hardening

The user requested the requesting-code-review skill followed by rebase and merge,
and explicitly substituted that independent review for Qodo on this merge.
The fresh review reproduced two connected saved-chat failures: SQLite cleanup
could escape a fatal Textual worker, and premature CANCELLED publication could
make retry discard leave an echo that entered the next provider request.

Restore/Discard now contain storage failures at the shelf action boundary with
content-free warnings and retained recovery. Cancellation deletes the transient
echo and restores session state before publishing CANCELLED under the existing
lock; failed deletion leaves the original paused owner retryable. Four new
regressions were observed red and then green: two real Textual workers and two
real saved-chat SQLite failure/retry cases, including subsequent provider history.

The frozen retrieval preparation routine moved without behavioral changes to
`Chat/console_library_preparation.py` to respect the existing controller size
ratchet. The controller delegates through its unchanged entry point; its budget
was lowered to 29,050 lines, and the store remains at 22,338. No new ADR is needed
for this mechanical delegation or repair of existing cancellation contracts.

Final targeted verification: 110 passed, 125 deselected; the three previously
baseline-reproduced automatic-preparation cases were explicitly excluded. The
controller/store architecture checks passed four cases. Independent final review
passed 19 focused cases, verified the moved routine's normalized AST equivalence,
and reported no remaining findings and ready to merge subject to CI/thread gates.
Across all 16 changed/new Python files, Ruff introduces zero diagnostics (236
baseline/current diagnostics). New helper/tests are formatted; diff and task-ID
checks pass. No full test suite was run. Console architecture and the real-write
failure-injection lesson document the corrected boundary.


### CI UI10 attachment readiness

Required run 39536237728 on head 64c681f900 failed only UI Fast Lane (10):
`test_session_owned_pending_projection_journeys` stopped at the initial legacy
approval-card mount wait (original line 751), before count/Inspector assertions.
The seed helper witnessed composer presence without completed attachment.
Relevant test/host/attachment code is unchanged from base 2d071d6d7145.

Matched actual controller/runtime controls isolate the setup hazard: an unscoped
legacy approval armed while reconciliation is incomplete is announced headless,
retains zero payloads and does not mount after reconciliation; completing
reconciliation before admission mounts the same approval. Both reviewer and
implementer independently ran these controls successfully. This establishes the
ordering dependency; the hosted failure did not record its sampled attachment
state, so that exact scheduling history is not claimed as proven.

The test seed helper now waits, under its existing bounded wait, for the exact
current screen/runtime view, completed attachment and answerability before the
seed send. Original card, count, navigation, decision and timeout assertions stay
intact. No production code or authority policy changes. ADR required: no, this
repairs the test's precondition under the existing attachment lifecycle.

Verification: the original CI-failing legacy journey passes directly with the
new helper (1 passed, 102.69s including isolated-process setup). The whole file
has 12 passed and one failure on this Windows host, at an earlier, different
`marshalled.is_set()` wait in `_verify_late_chat_create_projection_transition`
(new line 642/original 633). That exact earlier wait also fails in full-sequence
diagnostic probes on both untouched base and pre-fix head (531.05s/533.48s).
It is recorded as a baseline limitation, not counted green; the hosted run passed
that earlier journey. Initial 180s full-sequence diagnostic attempts timed out
locally; longer CLI-only budgets enabled the comparisons without changing the
committed test timeout. No full suite was run.

Independent requesting-code-review assessment found no actionable findings in
the nine-line test-only change. Changed-file Ruff comparison introduces zero
diagnostics (16 existing diagnostics unchanged); formatting, diff and task-ID
checks pass. Auto-merge was disabled and the branch pulled with rebase before
editing, as required by the queue. Re-arm only on the reviewed new head after
thread checks; combined CI remains the final merge gate.
