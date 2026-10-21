---
id: TASK-35992
title: Repair verified Console send failures from PRs 3181 and 3185
status: Done
assignee:
  - '@codex'
created_date: '2026-10-21 02:17'
updated_date: '2026-10-21 08:50'
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
- [x] #9 Windows late chat-create projection journeys keep test-owned worker gates held until explicit release, preserve the close/navigation and sibling-decision assertions, and retire held workers during cleanup within the existing test deadline.
- [x] #10 The replacement send-repair task has a unique filename/frontmatter ID and consistent live references in the combined dev and PR task tree, with renumbering provenance preserving its original identity.
- [x] #11 The UI gate retains its exact census and existing shard count, test/job timeouts, dependencies and required aggregation, with the UI2 allocation reduced by a measured balance correction; hosted outcomes and remaining global margin limits stay explicit.
- [x] #12 The changed-source replay regression distinguishes the held narrowed sync from later FULL invocations independently of recycled frame addresses, preserving original source-currentness, task, custody, draft and deadline assertions; the separately filed first-wait mode remains explicitly unclaimed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing regression coverage for owning-thread dispatch, real EnhancedRAGServiceV2 search, bounded cold-start preparation, recovery ownership, invisible confirmation expiry, and evidence-cache invalidation.
2. Apply the verified thread/RAG fixes; separate bounded initialization from search; cancel invisible confirmations without changing persisted user preferences; release only the recovery's matching retrieval preparation.
3. Cache parsed qualification evidence by a verified file generation while keeping native qualification verdicts fresh for every authorization call.
4. Keep the controller within its existing size ratchet by mechanically delegating the retrieval preparation routine to a focused helper; run targeted suites, lint/format changed Python, review the complete diff, and update subsystem documentation.
5. Add a real-Textual-worker regression for Restore/Discard storage failures, retain recoveries on failure, surface bounded retry guidance, and verify saved-chat SQLite failures leave cancellation retryable without leaking discarded text into later provider history before pre-merge review.
6. For CI UI10, witness completed and answerable Console attachment before mounting approval journeys, reproduce the incomplete-attachment legacy behavior, and rerun the original assertions.
7. Instrument the Windows late-projection baseline failure; replace test-owned five-second self-expiry with explicit event release under existing test deadline/finally cleanup, verify close and navigation paths and failure cleanup, then review.
8. Resolve the CI integration collision by renumbering this younger task and all live references, retaining ownership provenance, and verifying uniqueness against the combined dev/PR task tree.
9. Diagnose the hosted UI2 job-cap cancellation from comparable raw shard logs; balance measured whole-file slots without changing coverage, dependencies, counts or limits; verify census/queue contracts and request independent review.
10. Repair the hosted P0 replay witness using per-invocation tokens assigned on original PY_START and retired on PY_RETURN; verify the original lifecycle without retaining frames or changing product code/deadlines, and record the separately open TASK-35713 first-wait mode.
11. Publish a replacement PR crediting Csavoldi; close #3181 and #3185 with replacement links after publication succeeds.
ADR required: no
ADR path: N/A; existing backlog/decisions/069-console-project-instruction-local-state-and-preflight.md and backlog/decisions/126-complete-local-backup-and-recovery.md and backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md apply.
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


### Windows late chat-create projection baseline

The user authorized repairing the separate Windows baseline after the CI
attachment fix. A mounted closing-journey trace establishes the cause:
the source entered its artificial hold at 4.281s, its five-second wait returned
false at 9.281s, and its worker reported `AssertionError` at 9.719s. Valid sibling
creation and Close reached the explicit release at 16.141s, 11.86s after entry.
The later `marshalled.is_set()` wait hid that earlier test-worker failure.

The source and queued-clear holds now wait for explicit release. The existing
`finally` sets both events before restoring the callback and retiring workers.
The existing test/process deadline remains the outer bound; observer waits,
worker joins, Close/navigation transitions and sibling payload/decision
assertions are unchanged. This is a test synchronization repair with no
production changes. ADR required: no; the existing attachment/lifecycle
contract is preserved. The measured trap is recorded in the testing lessons.


The original complete affected file now passes: 13 passed, 2 warnings in
649.33s (all six real mounted journeys, including both Close/navigation late
projection paths and the legacy approval journey). The command used
`--timeout=900`, the same local diagnostic allowance used for baseline
comparisons; no committed deadline was widened. This establishes the targeted
Windows behavior, not that this slow host completes the aggregate within the
default 300s budget. No full suite was run.


Forced-failure controls reached each actual held gate, raised from the owner
test while its release event was unset, then ran the original `finally` and
`_stop_workers`: all workers were retired for both source and clear holds.
Clear passed in the initial parametrized run (109.74s call); source passed
serially (1 passed, 1 deselected, 114.61s total, 106.57s call). The earlier
source attempt failed during `_seed_console`/storage teardown before reaching
the gate (`raw_resources_not_retired`); it is not counted as cleanup evidence.
These diagnostic probes remain local, outside the committed test collection.

Independent requesting-code-review found no actionable findings in the gate
repair. Changed-file Ruff comparison retains exactly 16 existing diagnostics
and introduces none; changed-range formatting, diff and task-ID checks pass.
The source/clear behavior and cleanup evidence satisfy AC9. Combined hosted CI
remains the integration gate; existing unrelated baseline cases above remain
separate and are not counted green.


## Renumbering provenance

Renumbered from TASK-35900 to TASK-35992 for PR #3189 on 2026-10-21.
The original send-repair task was created at 02:17 UTC and added in
`b2e5e95076c8530034a536c34b39ed94ccf0b82a`. The competing Console
Assistant/Scope burst follow-up was created at 01:23 UTC, added in
`9162486f63d5dee79ac8515891e5016889a6d197`, and is already on `dev`.
It retains TASK-35900 under the older/landed-owner rule (TASK-19601 and
`backlog/docs/lessons-backlog-hygiene.md`). Only this PR's identity moves.

Required run 39550959500 on head 430757403e failed only the two Backlog
filename/frontmatter uniqueness tests in PR Fast Lane (2 failed, 2818 passed,
3 skipped). All ten UI shards, Console P0 and other completed gates passed.
The branch-only uniqueness check missed the peer task added by the CI merge
with newer `dev`; both exact task files reproduce both failures locally.
At allocation, 293 distinct ref tips and 52 worktrees had maximum integer
ID 35962. The replacement uses the documented +30 lead buffer. All live
Python/doc references move with the filename and frontmatter; historical
commit IDs and the original task ID remain here as provenance.

ADR required: no. This is task metadata/reference repair with no behavior,
contract, deadline, test assertion, or dependency changes. Auto-merge was
disabled and the remote branch pulled with rebase before editing.


Verification of the task repair: all three original CI uniqueness functions
pass against the clean prospective merge's 5,583 live task files after the
rename; the full guard passes 5,590 live/archive files with Windows-safe paths.
The older peer TASK-35900 is present unchanged and this PR's old path is absent.
Targeted pytest: 7 passed, 62 deselected (three uniqueness checks and four
Console controller/store size checks). Both changed Python files retain
identical executable ASTs and have zero Ruff diagnostics; changed ranges are
formatted and diff checks pass. Independent scoped review reports no blocking
findings. This metadata repair does not require repeating unrelated UI suites;
new hosted CI remains the final integration gate.


### UI2 job-budget follow-up

Run 39559121390/job 119086478771 was cancelled at the unchanged 20-minute
job limit after 97% of UI2, with no test assertion failure reported. GitHub's
annotation explicitly names the 20m0s maximum. The preceding behavior-identical
head 430757403e passed UI2's 176 cases in 1033.85s pytest / 18m05s job time;
both runs checked out merge refs based on dev 906497d9b4. This is accumulated
shard cost on slower hosted execution, not an established product failure or
an infrastructure error that warrants a blind retry.

Raw timings from all ten preceding shards and seven completed current shard
logs (including the cancelled prefix) were replayed by occurrence, preserving
repeated filenames and validating progress-path order against each selection.
The stock replay's known repeated-filename issue is already recorded by
TASK-35834; no replay-tool change is bundled here. Completed entries of the
cancelled prefix contribute observed costs; unrun entries retain the other
run's measurements. Whole-file session-summary-save costs up to 155.42s while
the static strip-focus contract costs less than a second.

These two whole-file entries trade census slots 25/61, hence jobs 6/2, and the
session-summary criterion comment travels with its suite. All 408 entries
remain unique and run once; the other eight ordered shard selections are
identical. Per-occurrence worst replay drops UI2 from 19.370 to 16.789 pytest
minutes; UI6 rises from 11.929 to 14.510. Observed complete-job overhead tops
out at 97.153s; allowing 120s projects 18.789 and 16.510 job minutes. These are
placement estimates, not hosted pass claims. All other predicted loads stay
unchanged, including the 17.763-minute pytest maximum on job 4.

The ten-shard matrix, 20-minute job / 180-second test limits, minimal
dependencies, serial execution, test bodies, P0 entries, S1 block positions
and required failed-lane aggregation remain unchanged. TASK-35834's broader
node-distribution, three-minute global margin and three-run qualification
criteria remain open; this is a measured interim balance repair. ADR required:
no; the existing ADR-103 placement and aggregation policy is preserved.


Verification: 383 focused CI tests pass (101.19s): census failure/coverage
checks, derived workflow contracts, shard replay/count statements, Console P0
coverage and queue-pressure contracts. The standalone census checker passes
408/floor 408; exact Counter/order assertions and diff checks pass. Independent
review recomputed the occurrence timings, overhead and two projections,
confirmed the exact eight unchanged shard sequences and isolated/pure nature
of the moved suites, and found no actionable findings. New-placement hosted
timing and collection/import composition remain pending, so this is not
claimed as a hosted pass or global margin qualification. At this checkpoint
all nine other UI shards on the preceding head passed; PR Fast Lane was still
running. No full suite was run.


### CI P0 invocation identity witness

On reviewed head `8952057f6d70d60285a108d403d4373d91bd16fb`, required
run 39566575303 / Console P0 job 119110873891 passed its sandboxed step but
failed the admission-sensitive changed-source replay test (236 passed, 1 failed,
571.43s). The original line-243 predicate saw a narrowed `False` followed by
successful FULL `True` results at the same numeric frame address. This is the
already filed TASK-35713 address-reuse mechanism, not evidence that FULL replay
failed. The unchanged source was introduced at `3e618d711923`.

The observer now assigns an opaque token at the original FULL's `PY_START`,
uses frame addresses only to look up live invocations, and removes that lookup
at `PY_RETURN`. Held/results retain tokens, never original frames. Original
lock/Future ownership, narrowed failure, later different FULL success, custom
callback task, custody, draft, source-currentness and five-second assertions
remain enforced. Production callbacks, helper contracts, test census and
committed deadlines are unchanged. ADR required: no; this is a test witness
repair under ADR-103.

An independent exact-observer/predicate probe used real `sys.monitoring`,
`asyncio.Lock` and Futures without importing app code or retaining frames.
Both base and repair executed the same narrowed return plus 25 sequential FULL
returns: all 26 reused one raw frame address. The base retained one identity
and rejected its genuine False/True completions; the repair retained 26 distinct
tokens and accepted them. The live map was empty after every return. The
implementer repeated that probe successfully after formatting. This validates
the observer, not 25 complete Textual journeys.

The original five-case Windows file was also attempted: all five failed earlier
at `_held_received_record`'s unchanged ten-second reader-entry wait (line 318),
before this observer is installed, in 246.53s. A fresh native-private-profile
control using the untouched head's exact replay-test source failed at the same
earlier wait (57.40s). Neither run is counted green. The different TASK-35713
first-wait mode (original line 231) and its 25 consecutive complete dev-run
qualification remain open and unclaimed. No full suite was run.

Evidence: `.tmp-replacement-tests/ci895-console-p0-failure.log`,
`p0-frame-identity-file.txt`, `p0-frame-identity-baseline.txt`, and
`p0-frame-identity-exact-observer.py`/`.txt`. An initial scratch probe imported
UI helpers outside pytest and was stopped before producing a witness result;
it is not evidence and was replaced with the app-free AST probe. Fresh scoped
independent requesting-code-review found no actionable correctness or weakened
assertion findings. Changed-file Ruff has zero diagnostics, formatting and
diff checks pass; hosted P0 remains the integration gate.
