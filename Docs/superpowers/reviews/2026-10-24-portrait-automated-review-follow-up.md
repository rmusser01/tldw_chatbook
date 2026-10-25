# PR #3244 — automated review readiness

Marked ready at the user's request on 2026-10-25 UTC. This changes review status,
not feature qualification: full live-provider UAT has not been completed.

ADR required: no
ADR path: N/A
Reason: repair the existing reviewer's repository-specific scope; no new reviewer,
application boundary, runtime, shared organization setting or paid-review policy.

## Reproduced blocker and minimum repair

CodeRabbit's resolved Organization UI (base) path filters are `Docs/docs/**`,
`Docs/Design/**`, `Docs/Product/**`, and `Docs/Plans`. The PR's production and test
paths match none. Its incremental and full requests therefore skipped all 252
changed files and produced no review. Current CLI 0.9.1 remote and committed-local
attempts independently returned `All files are ignored`; explicit local instructions
did not replace the server's inherited review-scope configuration.

- [Effective configuration](https://github.com/rmusser01/tldw_chatbook/pull/3244#issuecomment-6394106555)
- [Full-review refusal](https://github.com/rmusser01/tldw_chatbook/pull/3244#issuecomment-6394017930)

Add only `.coderabbit.yaml` with `reviews.path_filters: ["**"]`. This repository
contains application code as well as documentation; its review scope must include
both. CodeRabbit's normal ignored binary/generated paths still apply. All other
inherited settings, including automatic-review branch policy, remain inherited.
The three-line override validates against the official schema; retain its schema
and config hashes in the QA receipt. No application source or test expectation changes.

Operational red/green validation: the reproducible pre-fix outcome is all changed
files skipped, not an empty successful review. After publication, request the real
full PR review and confirm the effective scope and actual coverage before claiming
a clean review. Validate findings against connected code, fix real defects with
regressions, and wait for updated checks. Schema validation alone cannot establish
that the hosted reviewer read the code.

## Other review services

Qodo says the workspace is out of credits; Cubic reports its monthly line allowance
exhausted. Their terminal responses are unavailable coverage, not approvals. No
purchase, credit-overage flag, shared billing change or replacement service is enabled.

## UAT and evidence limits

Installed-app journeys cover selection, authoring/retries, exact pair review,
Apply/Remove/reopen, native restore and Console presentation using deterministic
image/TTS/device boundaries. The real Pocket TTS/CoreAudio run remains historical.
There is no latest-head live image-plus-TTS journey, physical acoustic loopback or
independent human acceptance. The configured image adapter remains unsupported for
this reference path: zero live image requests, no fallback, `live_qualified=false`.
TASK-36555 stays In Progress and AC5 remains unchecked. Existing ADR-242/243 and the
qualification document retain those boundaries.

## Review-size limit after scope repair

The all-path override took effect: CodeRabbit advanced past the path-filter
refusal, then rejected 227 eligible files against its 100-file per-review limit.
The official schema exposes no file-limit adjustment. The current pass includes
all 53 changed application files, all 46 test files and the reviewer configuration
(100 tracked paths before normal binary ignores). Documentation, Backlog and QA
artifacts are outside this hosted code-review pass; they must not be labeled as
automated-reviewed. The earlier all-path receipt remains historical red evidence.

A complete 53-file application batch also started through current CLI 0.9.1 and
reported a reviewing heartbeat. This is a disclosed partial batch, not a full-PR
result. The user was offered every-file coverage versus the recommended code/test
pass; after the response window, proceeded with that stated recommendation.
The final schema validation and exact selected/outside path lists are retained in
`../qa/portrait-compositing/2026-10-24/review-readiness/code-test-scope.json`.

## Application batch — first verified findings

CodeRabbit CLI 0.9.1 reviewed the entire 53-file application subtree at `155ed23c8e`
and emitted eleven findings. These are separate from the hosted 100-path pass.
The graphics mount race is reproduced with a real graphics widget: change the
manual preview role immediately after its actual mount, then observe the permanent
Preparing surface. The regression fails before the fix and passes afterward.
After an owner-current initial mount, finish hiding the surface and adopting its
rest-pose buffer; the next tick paints the newly selected pose. Keep stale checks
on threaded pose/pixel results and the owner check after mount. The four focused
checks pass: new mounted race, stable graphics updates, late loud-to-silence
rejection and exact six-role offline cycle, including shared-budget cleanup.

Controller close is reusable (discard/drain, no terminal flag); the nonexclusive
worker is intentionally serialized by `handle`'s admission gate so Stop/Cancel
can reach admitted work; `_read_avatar_image_bytes(self, path)` has the correct
bound signature. `PreparedExpression.close` empties frames and zeroes its byte
reservation, so repeated close cannot double-release the shared budget. The
proposed exclusive worker, staticmethod and non-idempotent-close fixes do not
apply. External artwork/authority changes must retire source-bound portrait drafts under
ADR-242/243; subsequent hosted feedback correctly distinguished text-only card
changes, which must retain source-current staged work. The mounted regression
reproduced that loss and the follow-up fix checks the saved source before discard. Remaining TTS/geometry suggestions are still being
verified; no blanket acceptance of conditional findings is claimed.

### Guarded legacy speech delivery and cleanup

The hosted review confirmed that guarded hands-free legacy artifacts were cached under a lifecycle owner but cleaned under the legacy None owner. The same lifecycle on a completion with no replay file incorrectly terminated the presentation before playback. Direct legacy utterances now retain validation through generation and revalidate after delivery, while their cache and completion use the legacy ownership contract. Three regressions failed before the fix (presentation, cleanup, supersession) and pass afterward; the utterance and streaming playback modules pass 58 tests. Full-file Ruff reports existing baseline violations; changed lines are checked separately.

### Hands-free invalidation and sequencer ownership

A presentation validator may become false while its hands-free loop remains live. Finish that exact in-flight token as failed at entry, after handler initialization, or on completion; fence bookkeeping by both loop identity and current utterance token, including failure counters. The four-case regression covers all three invalidation stages plus an abandoned reply, duplicate callbacks and replacement isolation. The existing replacement-session control remains green. A broader run of this legacy wiring module exposed 43 unrelated baseline failures (mostly missing ADR-126 bootstrap-profile ownership, plus an older config mock signature); these are not reported as passing. Focused worker and sequencer checks are recorded separately.

### Repository scope after hosted feedback

The hosted 100-path pass completed at `0cf888502e97dd8a28a467ecc8f7c230ba0688ba` with four inline findings. It correctly noted that the temporary allowlist would permanently exclude workflows, tooling, manifests and governance. Restore the all-path override; keep source/test batches bounded through the CLI rather than narrowing future repository coverage. The already-completed hosted pass is historical coverage, not a promise that this large PR fits the hosted cap. Qodo/Cubic remain unavailable. Source/test CLI batches and post-fix verification remain required.

The legacy hands-free module was rerun with its exact pre-fix worker from `2a810a6314`: the same 43 baseline failures reproduced (10 passed, 4 new tests deselected). Current focused worker/old-session controls plus the pure sequencer suite pass 32 checks. The broad legacy module is not claimed green.

### Preserve source-current portrait work

A mounted control reproduced text-only external card updates dropping a staged imported pose. The refresh now checks the saved artwork off-thread and treats a source-current portrait draft as unsaved work, preserving it even when the text form is clean. Actual artwork/authority changes still retire it. The awaited check fences retirement against draft identity and epoch; a held old check cannot discard a newly prepared replacement. Four clean/dirty text versus artwork controls and the replacement race pass. The authoring/character-refresh run passes 29 tests; two unrelated older Save tests fail because their worker fixture lacks the existing `submitted_snapshot` argument. No Save code changed.
