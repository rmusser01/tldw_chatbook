# PR #3244 — automated review readiness

Marked ready at the user's request on 2026-10-25 UTC. This changes review status,
not feature qualification: full live-provider UAT has not been completed.

ADR required: no
ADR path: N/A
Reason: routine review-scope and implementation corrections follow existing
ADR-097/242/243; no new storage, provider, runtime or security boundary. Shared
organization settings and paid-review policy are unchanged.

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
reproduced that loss and the follow-up fix checks the saved source before discard. Final TTS/geometry dispositions are recorded below; no blanket acceptance of
conditional findings is claimed.

### Guarded legacy speech delivery and cleanup

The hosted review confirmed that guarded hands-free legacy artifacts were cached under a lifecycle owner but cleaned under the legacy None owner. The same lifecycle on a completion with no replay file incorrectly terminated the presentation before playback. Direct legacy utterances now retain validation through generation and revalidate after delivery, while their cache and completion use the legacy ownership contract. Three regressions failed before the fix (presentation, cleanup, supersession) and pass afterward; the utterance and streaming playback modules pass 58 tests. Full-file Ruff reports existing baseline violations; changed lines are checked separately.

### Hands-free invalidation and sequencer ownership

A presentation validator may become false while its hands-free loop remains live. Finish that exact in-flight token as failed at entry, after handler initialization, or on completion; fence bookkeeping by both loop identity and current utterance token, including failure counters. The four-case regression covers all three invalidation stages plus an abandoned reply, duplicate callbacks and replacement isolation. The existing replacement-session control remains green. A broader run of this legacy wiring module exposed 43 unrelated baseline failures (mostly missing ADR-126 bootstrap-profile ownership, plus an older config mock signature); these are not reported as passing. Focused worker and sequencer checks are recorded separately.

### Repository scope after hosted feedback

The hosted 100-path pass completed at `0cf888502e97dd8a28a467ecc8f7c230ba0688ba` with four inline findings. It correctly noted that the temporary allowlist would permanently exclude workflows, tooling, manifests and governance. Restore the all-path override; keep source/test batches bounded through the CLI rather than narrowing future repository coverage. The already-completed hosted pass is historical coverage, not a promise that this large PR fits the hosted cap. Qodo/Cubic remain unavailable. Source/test CLI batches and post-fix verification remain required.

The legacy hands-free module was rerun with its exact pre-fix worker from `2a810a6314`: the same 43 baseline failures reproduced (10 passed, 4 new tests deselected). Current focused worker/old-session controls plus the pure sequencer suite pass 32 checks. The broad legacy module is not claimed green.

### Preserve source-current portrait work

A mounted control reproduced text-only external card updates dropping a staged imported pose. The refresh now checks the saved artwork off-thread and treats a source-current portrait draft as unsaved work, preserving it even when the text form is clean. Actual artwork/authority changes still retire it. The awaited check fences retirement against draft identity and epoch; a held old check cannot discard a newly prepared replacement. Four clean/dirty text versus artwork controls and the replacement race pass. The authoring/character-refresh run passes 29 tests; two unrelated older Save tests fail because their worker fixture lacks the existing `submitted_snapshot` argument. No Save code changed.

### Remaining source-batch suggestions

The fit helper is arithmetic-only over validated source dimensions and a measured integer cell box; it performs no renderer I/O. Keep existing renderer failure fallback and clarify the builder docstring/return value rather than adding a blanket exception path around trusted geometry. `_prepared_base` uses the resolved expression key, and `source_from_resolution` canonicalizes that actual artwork (portrait to neutral, legacy/pack to resolved key and persisted references); requested/manual-state metadata is deliberately absent from PerformanceSource, so recapturing the same artwork does not change identity. Existing regions/source round trips cover the actual path. Repeated PreparedExpression.close clears frames and its byte reservation, so the alleged double-budget release does not apply.

### Completed test batch and installed follow-up

CLI 0.9.1 completed all 47 changed test files at `c4ea7272fc` and returned three minor suggestions. Bound the installed guarded-sample poll to 30 seconds while retaining its trace/status failure assertion; require busy to clear. Restrict the POSIX 0600 assertion to non-Windows platforms, retaining all evidence/privacy assertions everywhere. No compatibility rewrite is needed for asyncio.timeout: pyproject.toml requires Python >=3.12. Affected probe tests plus the fresh wheel-installed authoring/Console/reopen/Remove/reapply journey pass 11 checks in 37.33s, and all 53 changed production files match the installed wheel.

CodeRabbit acknowledged all four hosted fixes and resolved their threads. A fresh hosted incremental request was rate-limited; a bounded CLI follow-up from the prior reviewed head is running. Qodo/Cubic remain unavailable. The all-path configuration validates against the current official schema. No full test sweep, current-head live image/TTS/device qualification or independent human UAT was performed. Verification hashes, reviewed path lists and exact limits are in `../qa/portrait-compositing/2026-10-24/review-readiness/review-follow-up-verification.json`.

### Post-fix batch: additional refresh races

The bounded 15-file follow-up completed with two further findings, both reproduced. Character Save temporarily makes operation authority unavailable; source revalidation now waits for Save to settle (also if Save begins during the read), while context polling preserves the draft during that transient state. It still compares saved artwork afterward and retires actual artwork changes. No generation/Apply admission gate was weakened. The event handler rereads selection after the await and returns if runtime source, entity kind or character ID changed. Five initial controls fail before the correction; all eight expanded Save/read, selection and replacement controls pass afterward.

### Explicit successful-Save transition

A further bounded review correctly identified the successful Save finisher advancing the editor generation while keeping the editor open. Its actual mounted reconciliation reproduced draft loss beyond the in-flight-flag controls. Rebind only the trusted finisher's exact prior token to its next generation, with identical editor, token tail and editor-load token; compare saved artwork after settling. Other source/authority fields remain unchanged, in-flight old operations keep their old fences, and old sessions cannot be adopted. The waiting source check follows this explicit authority replacement. Four actual-Save source/stale-session controls plus overlap/replacement checks pass nine cases; the initial failed attempt remains in red/green evidence.

### Boot ratchet and final bounded review

CI on merge `899c48bf` (same `139333920d` dev base) measured 1033 ready modules; prior Linux and current local runs measured 1031. Reconstructing the sets isolates scheduling-dependent `console_change_review_read` and `DB.character_conversation_search` as the two extra residents. Follow ADR-097 by shedding two first-use costs: absent-conversation marker projection skips an import whose capture always declines, and token-rate tracking/formatting waits for actual assistant content or an enabled available reading. New real-boot absence pins, import-denial controls, first-use/reset checks and mounted rate flows pass. All 29 boot ratchets pass without raising ceilings; local ready count is 1030/1031.

The final bounded review from `37ed61380d` completed with **zero findings** after verifying both additional Save/read and rate-import corrections. The complete application/test batch chain covers all **54 changed application and 50 changed test paths**; documentation and QA artifacts are not claimed completely automated-reviewed. Current authoring/probe/installed/boot checks pass 44 tests in 104.58s, the rate/formatting checks pass 38 tests, and all 54 production files match the fresh installed wheel. Legacy differential lint adds no diagnostics.

The c4ea Windows recovery failure was its unchanged 30-second capture watchdog, followed by secondary shutdown cancellation; the exact journey passes locally in 36.80s. Leave that deadline unchanged and inspect the final-head Windows result. Latest-head hosted CI remains pending at publication. All four original hosted review threads are resolved; Qodo/Cubic remain unavailable and the hosted follow-up request was rate-limited. Full live-provider UAT remains incomplete, AC5 stays open and zero configured live image requests were sent.

### Hosted incremental review on a84519829e

The hosted review completed on 21 incremental paths with two inline findings and two outside-diff suggestions. All were checked with red/green controls: invalid saved-source ValueError now becomes PerformanceConflict without changing other input-error handling; a retired graphics owner clears the child image before the unadopted pose closes; idle preview samples rest and blinks independently of a live open mouth; unexpected speech reads produce one fixed debug message per live widget, with no exception text, actor/artwork/speech data or new sink. Retired/absent speech remains silent. The earlier live-owner mount adoption stays intact: reintroducing the suggested stale-pose return would restore the stuck Preparing defect, and its mounted control still passes.

Affected avatar/authoring/GIF checks pass **62 tests**; a fresh installed journey and UI-ready absence checks pass **five**, with all **54 production files** matching the installed wheel. The sole new diagnostic was reviewed before regenerating its pin, which verifies 697 owners and 7753 TASK-494 calls. The bounded follow-up found no production issue. Its timestamp suggestion was addressed with a renderer-local clock and positive open-mouth checks; the original a845 tick still fails the blink assertion, so the regression cannot pass through silent rest fallback. The expected teardown and bad-import controls retain their prior behavior. Lessons record the actual Save/mount/freshness incidents. No full suite or new live-provider/TTS/hardware UAT was performed; AC5 remains open.
