# TASK-39147: review completion recovery

Base: `a0ae30b7ed9c313fe19cb3a0da5d95ebde182385` (current-dev checkout from the November 7 comparison).
Initial scope: offline submission recovery, preserving accepted decisions, permissions, accounting and sealing. Later sections record individually authorized live scopes. The current goal authorizes completion and merge after verification; prior request allowances remain consumed.
ADR: [281](../../../../backlog/decisions/281-primary-agent-multi-model-code-review.md). No new architectural decision.

## Haiku: reproduced and repaired

The Qwen-coordinated run accepted three dispositions at revision 1. Haiku then changed those records under the already accepted ID `v1-dispositions-batch-1`; the correct `submission_conflict` refusal remained outstanding. Completion consequently refused `unresolved_submission_rejections`. Changed and empty resubmissions could not repair it. Even an exact replay returned the old acknowledgment without clearing the rejection.

Readback reuses report source authorization, so revoked source permissions/exclusions prevent disclosure of previously accepted text; callbacks remain outside publication locks.

The shared store now retains the existing immutable chunk's path alongside its digest/acknowledgment. Conflict feedback can supply the original actor-owned records as retry arguments, subject to existing result/input limits. An exact replay admits the old acknowledgment, clears only its own rejection and records the repair through a charged host receipt. It changes no decision or participant revision. Failed acknowledgment/receipt admission leaves the rejection outstanding. Changed or empty accepted-ID replacements remain refused. Other rejections, missing dispositions, complete capture and successful native termination remain sealing requirements.

A second offline regression showed that a replayed old acknowledgment could cause verifier history reduction to hide an outstanding completion rejection. Checkpoints now withhold reduction eligibility while any rejection remains. This preserves recovery instructions without changing stored transcripts or continuation ownership.

Oversized original retry payloads are not truncated or exempted from capacity checks: feedback explicitly reports `original_replay_exceeds_capacity`. The caller must retain/reconstruct its original submission in that case. This repair does not guarantee that a live model follows the guidance or that its accepted verdict is factually correct.

## DeepSeek: first boundary established; exact cause unavailable

The Luna-coordinated DeepSeek child `e8cdf550dcd84b9eb60fb00024af96f3` accepted findings at revision 1, then had three approved/dispatched failed `code_review_submit` calls at 17:51:46, 17:51:49 and 17:51:52 UTC on November 7. It stopped as `stuck`, with no coverage and an empty rejection ledger. The retained durable trace omits the failed arguments and error codes. The harness did not retain response bodies; no other full-payload trace was found.

Narrow scripted controls through the native agent service distinguish possible boundaries:

| Completion input | Result |
| --- | --- |
| Valid coverage/completion | Sealed, revision 2 |
| Contradictory `empty_findings` | Stuck, revision 1, `invalid_record` rejection |
| Missing `review_id` | Stuck, revision 1, no coverage, empty rejection ledger |
| Unknown `review_id` | Same empty-ledger failure |

The missing/unknown identity controls prove that the observed state does not identify an exception in the completion store. They are **not evidence that DeepSeek used either malformed identity**. The retained live evidence cannot choose among identity/authority refusal and an unrecorded exception. No DeepSeek-specific production change is justified or claimed. Exact historical diagnosis remains blocked on the omitted tool arguments/error; no additional live run is authorized by this work.

## Evidence and checks

Live comparison evidence is retained outside temporary storage at:
`/Users/macbook-dev/.codex/artifacts/task-35582/current-dev-review-20261107/`.
Use `report.md`, `comparison-summary.json`, the Luna evidence archive, and `qwen-coordinator/completion-failure-sequence.json`. Private settings backups are not part of this evidence and contain credentials.

New permanent regression: `Tests/Agents/test_code_review_submission_replay.py`. Native missing/unknown identity controls are in `Tests/Agents/test_code_review_reviewer_completion.py`.

Red evidence: four replay/rejection tests failed on unchanged source; a fifth test separately reproduced lost rejection history. The revoked-source and capacity-feedback controls also failed before their corrections.

| Verification | Outcome |
| --- | --- |
| Related store/service/tool/recovery/history/verification tests before review corrections | 238 passed |
| Affected replay, reports and isolation tests after permission/capacity corrections | 85 passed |
| Native missing/unknown identity controls | 2 passed |
| Permanent unlocked-callback / mid-callback exclusion controls | 2 passed |
| Scoped independent review after corrections | No remaining findings |
| F/E9 lint, changed-file formatting, diff whitespace | Passed |
| Full configured lint on changed Python files | No new findings; 8 existing baseline findings remain |

These offline controls use scripted model replies; they do not supersede the failed live outcomes or complete a live qualification. Full-suite testing and hosted CI were not run. The source changes remain local on `codex/review-completion-recovery`.

Consumption for this repair: **0 live generation requests, 0 live product child launches**. The prior comparison remains 115 requests / 6 children.

## Later authorized narrow DeepSeek diagnostic — incomplete

The user subsequently authorized one isolated DeepSeek diagnostic, followed by
offline fixes, without rerunning the full review. The harness used a scripted
coordinator, the same disposable `average()` patch, one live DeepSeek reviewer and
an eight-request physical cap. No second reviewer or verifier ran.

The initial harness launch failed before HTTP because its gateway adapter passed
unsupported `execution_provider` metadata to `chat_api_call`. Its setup child and
immutable receipt were preserved. After an offline production-builder check and
that adapter correction, the provider diagnostic made **3 generation requests**,
all HTTP 200, and reached **zero review tool invocations**. The production DeepSeek
parser rejected the response as malformed and surfaced a synthetic 502 error.

This is an earlier boundary than the historical completion failure. The harness
had disabled HTTP retries but mistakenly left the agent's `max_model_retries` at
its default of two, causing the two additional requests. Its response observer
also ran after parsing, so the rejected response bodies were not retained. Thus
the diagnostic cannot identify the malformed field or claim a DeepSeek product
fix. The earlier run's failed tool arguments remain unavailable.

Live work stopped. Offline harness corrections now filter gateway metadata, pin
and verify the child's model retries at zero, capture provider JSON before parsing,
and reject a diagnostic result that never reaches the tool boundary. The combined
offline composition/capture/retry check passed with zero network requests. There
was no subsequent live run and no production-source change in this follow-up.

Actual consumption: **3 requests, 2 product children created** (one pre-network
setup child plus one live reviewer). Five unused request slots are not carried
over. Both child runs terminated with incomplete reports; no accepted finding or
decision was replaced, and no seal was fabricated. The earlier 115/6 comparison
therefore becomes 118 requests / 8 children including this follow-up's setup child.

Credential-scanned evidence, immutable source receipts, errors and assessment:
`/Users/macbook-dev/.codex/artifacts/task-39147/deepseek-diagnostic-20261107/`.
The archived runner is the corrected offline version, not an assertion of the
exact code used by the consumed live invocation. Pytest's original exit-zero
observation is not diagnostic success. `assessment.json` records the failed
qualification of this diagnostic explicitly. TASK-39147 remains open.

## Subsequent production investigation and fixes (offline)

The retained diagnostic proves a classification defect: a local
`HostedChatProtocolError` became HTTP 502, so native retry policy issued two more
identical requests. A regression through the production adapter and native loop
reproduced **three attempts instead of one**. DeepSeek now reports a status-less,
terminal error with the validator's host-owned reason. Console's worker projection
preserves that classification. The streamed adapter applies the same mapping at
iteration time; independent review caught that initially missed boundary.
Malformed tool JSON still fails closed; no arguments/body values enter the error.
Real HTTP 502 remains retryable.

A separate contract defect was reproduced for both JSON and SSE: the finish
policy rejected an empty final answer on `length` or `content_filter`, although
DeepSeek documents nullable content for interrupted output. It now preserves the
finish state and usage without accepting interrupted tool calls. This includes
reasoning consuming the output cap. Empty `stop`, missing calls on `tool_calls`,
and truncated or otherwise interrupted tools remain refused. Reference:
[DeepSeek Chat Completions response contract](https://api-docs.deepseek.com/api/create-chat-completion/),
checked 2026-11-07. This contract regression is **not proof** of the missing live
response's exact shape.

Joined offline tests use the real DeepSeek builder/parser and native child/tool
execution with scripted HTTP bodies. One interrupted response followed by valid
findings/completion/final text seals at revision 2 with 160 accounted tokens.
Two interrupted empty responses remain stuck/incomplete at revision 0 with
64 accounted tokens. Thus usage reaches the native owner, and empty output does
not fabricate completion. The existing bounded empty-turn recovery remains in
place; no accepted decisions, permission checks or sealing requirements changed.

Validation: **68 targeted tests passed** across DeepSeek response/protocol
contracts, joined native completion/accounting, retry controls and existing hosted
adapter lifecycle/error tests. All new test files pass configured lint and
formatting. Diff whitespace and scoped F/E9 lint pass; baseline comparison finds
zero new configured-lint findings (11 existing findings across touched files).
Previously reviewed Haiku recovery checks remain valid and were not rerun.
Logs: `/private/tmp/task39147-{protocol-red2,finish-red,stream-red,final-targeted}.log`;
static comparison: `/private/tmp/task39147-static-baseline.json`.
Scoped independent follow-up review found no remaining issues; its previously
failing streamed Console regression passes after the correction. Existing
production-file formatting differences were retained to avoid unrelated changes.

This phase consumed **0 live requests / 0 live children**. The diagnostic totals
remain 3 requests / 2 children; comparison plus diagnostic remains 118 / 8.
No full review, full-suite run, hosted CI, push, merge or additional budget request.
The historical DeepSeek completion tool arguments/error and rejected diagnostic
bodies remain unavailable; AC3/AC5 stay open rather than claiming a live fix.

## Fresh authorized diagnostic: indexed tool call captured and fixed

The user's next explicit grant permitted one new narrow diagnostic with corrected
capture and no retries. Its fresh exclusive receipt is separate from every consumed
scope. Four offline production-builder controls passed before launch: valid
completion, failed-submission capture, malformed response capture and empty-response
stop. An independent reviewer checked the capture and request/child/retry guards.

Actual live consumption was **1 generation request / 1 reviewer child / 0 retries**.
The HTTP-200 raw body was saved before parsing. It contained a valid
`code_review_read` source-list call with an extra `index: 0` inside the non-streaming
`tool_calls` element. The strict parser rejected that extra field with
`Hosted Chat tool call is malformed` before any tool ran. The exact function name,
argument string and decoded arguments are retained. The body reported 2,293 tokens
(2,167 prompt / 126 completion, zero reasoning); this was not a token-limit stop.
No reviewer-completion call was reached, no report was sealed, and this is not a
successful live review or AC24 qualification.

The fix reuses the existing named tool-call allowance. `hosted_chat_request`
forwards it to non-streaming normalization, and DeepSeek permits only `index`.
Normalization drops the metadata and preserves call IDs, order, exact function
arguments and usage. Unknown extras, missing/duplicate IDs, malformed arguments,
interrupted calls and invalid allowance values remain refused. Shared defaults
and streaming index validation remain unchanged.

Offline RED: the captured shape failed both one/two-call parser cases and prevented
the native reviewer from sealing. After the fix, **84 targeted regressions pass**,
plus **one replay of the exact captured raw body**. Indexed source-list, load,
findings, completion and final response pass through the production adapter/native
owner, sealing at revision 2 with 192 accounted tokens including the preceding
interrupted turn. Repeated empty output still remains incomplete. Independent fix
review is clear; new-test configured lint/formatting, scoped F/E9 and whitespace
checks pass. No unchanged Haiku review or broad suite was repeated.

Private credential-scanned raw evidence, exact failed-call arguments, immutable
claim, runner, offline controls and repair logs:
`/Users/macbook-dev/.codex/artifacts/task-39147/deepseek-captured-20261107/`.
The minimized permanent fixture is
`Tests/LLM_Calls/fixtures/deepseek_indexed_tool_call.json`.

Combined November 7 comparison/diagnostics: **119 requests / 9 children**. Seven
unused slots from this diagnostic are not carried forward. AC5 is now satisfied
by this fresh authorized diagnostic; the earlier failed diagnostic remains recorded
above. Historical completion-call arguments remain unavailable, so AC3 remains
open rather than asserting that this new response reconstructs that older failure.
No second live diagnostic, verifier, full review, push or merge occurred.

## Authorized post-fix diagnostic: parser works; source-side read refused

The next explicit user grant authorized **one post-fix diagnostic**, capped at
eight new requests and one child, with raw-response/tool-argument capture and no
retries or full review. A fresh exclusive claim was used; no historical allowance
was reused. The runner retained its guards and added a completion gate requiring
accepted completion, native capture/terminal/seal receipts and known charged usage.
Four offline controls passed (valid, rejected completion, empty and malformed
responses); the scoped independent runner review was clear.

Actual consumption: **3 new requests / 1 child / 0 retries**. All three exact raw
responses were retained and normalized. Indexed tool calls executed successfully,
confirming the parser fix across a live tool-result continuation. Three source
reads succeeded. The fourth read asked for `source_query.side="head"` for a
`working_tree` target and was correctly refused with `invalid_source_side`.
The tool description specifies `base/staged/worktree` for that target. The
diagnostic guard stopped further generation at this first failed read (its generic
`first_failed_submission` label does not mean a completion submission occurred).

**Completion and sealing were not reached.** No findings or completion submission
was accepted; revision remained zero, terminal status was `error`, and no seal
receipt exists. The completion gate failed and the runner exited nonzero. Raw
responses reported 7,918 tokens; normalized usage was preserved, but the durable
child budget counter was unset after the forced diagnostic abort. Settled
accounting is not confirmed, and this does not establish another production defect.
An independent outcome review confirmed the first causal boundary.

Credential-scanned private evidence, exact raw responses/arguments, native receipts,
exclusive claim, accounting snapshot, runner and hashes:
`/Users/macbook-dev/.codex/artifacts/task-39147/deepseek-postfix-20261107/`.

November 7 comparison/diagnostic cumulative actual: **122 requests / 10 children**.
Five unused request slots are retired. No retry, verifier, full review, production
change, push or merge occurred in this post-fix diagnostic. Historical completion
arguments remain unavailable; AC3 remains open. This is not AC24 qualification.

## Offline source-side repair and aborted-run accounting

The next user instruction authorized only offline diagnosis and repair. The live
failure exposed two gaps: child schemas advertised `head` even for a captured
working-tree target, and the strict rejection returned no valid choices. The
source owner now derives query sides from the captured manifest for both schema
advertising and validation. Bound reviewer/verifier schemas narrow list, search
and resolve to those sides. Primary schemas remain general. Invalid queries still
fail, with capacity-bounded `allowed_sides` after source authority checks; the
host never guesses whether `head` meant staged or working-tree content.

RED: seven new schema/lifetime cases failed against the original implementation;
three authority/capacity controls already passed. GREEN: **102 affected tests
passed**, covering target kinds, both child roles, all three query operations,
expired capture, permissions, feedback capacity, native/fenced recovery, immutable
accepted decisions/acknowledgments, failure limits and receipt-dependent sealing.
The scoped independent review found no actionable issues. Changed-test lint and
formatting, scoped F/E9 and diff whitespace checks pass. Production lint comparison
has zero new findings (six unchanged baseline findings in the three touched owners).

An offline native-owner replay reproduced the captured three responses/four reads
with usage of 2,282 + 2,561 + 3,075 = **7,918 budget tokens**. Native cancellation
after the refused read persists 7,918; a diagnostic-style exception before the next
dispatch leaves the durable budget counter unknown. Both reports remain incomplete
and unsealed at revision zero. The unknown counter is the explicit ADR-131 contract
for exceptions that escape before the runtime returns its accounting. Provider raw
usage is not substituted for a completed runtime budget or rewritten as zero.
No accounting code change was necessary, and the historical abort is not relabeled
as settled.

Tests: `Tests/Agents/test_code_review_source_sides.py`,
`test_code_review_source_recovery.py`, `test_code_review_deepseek_boundary.py`,
`test_agent_budget_persistence.py`, `test_code_review_tool_provider.py`,
`test_code_review_submission_replay.py`, and `test_code_review_reviewer_completion.py`.
Private logs and review evidence:
`/Users/macbook-dev/.codex/artifacts/task-39147/source-sides-offline-20261107/`.

This repair used **zero live requests/children**; the November 7 live cumulative
actual remains 122 requests/10 children. No live diagnostic, full review, full test
suite, hosted CI, push or merge ran. Completion remains unconfirmed live; historical
completion-call evidence is still missing, so AC3 and overall task completion are
not claimed.

## Authorized post-source-side-fix diagnostic: completion accepted at the cap

A fresh explicit grant authorized one diagnostic with eight new requests, one
child, raw/tool capture, and no retries or full review. The runner was unchanged
except for its exclusive evidence root; its prior review was reused, reviewed
source hashes matched, and all four offline composition controls passed.

Actual consumption: **8 requests / 1 child / 0 retries**. All eight responses were
captured exactly, normalized and returned HTTP 200. All eight source reads and
three submission calls succeeded. The advertised list/search/resolve schemas
contained only base/staged/worktree; `invalid_source_side` did not recur. Findings,
coverage and `reviewer_complete` were accepted at revisions 1, 2 and 3 respectively.

**The diagnostic still failed its full completion gate.** Response eight submitted
completion; the physical cap blocked request nine before network dispatch. There
was no final model response or native successful termination. The report remained
incomplete with terminal `error`, complete capture and no seal receipt. No rejected
submissions remained. The terminal HTTP-500 wording wraps the local cap exception;
it is not a ninth provider response or a remote HTTP failure. An independent
outcome review confirmed this first causal boundary.

The raw/normalized responses report **46,782 tokens**. The forced exception leaves
the durable child budget counter unknown under ADR-131, so accounting settlement
is not confirmed or reconstructed from raw usage. The three accepted findings
remain originals; no verifier adjudicated them. Accepted completion is not a seal,
and this is not successful AC24 qualification.

Credential-scanned raw responses, exact tool arguments/results, native receipts,
accounting and report snapshots, immutable claim and evidence hashes are retained
privately at
`/Users/macbook-dev/.codex/artifacts/task-39147/deepseek-sides-postfix-20261107/`.
November 7 cumulative actual: **130 requests / 11 children**. This scope is
exhausted. No retry, full review, production change, push or merge occurred.

## Fresh 30-request diagnostic: missing required review identity

The next explicit grant allowed one fresh diagnostic, capped at 30 new requests
and three children, with no retries/full review. The isolated diagnostic retained
its stricter one-child guard. Only its request cap and exclusive evidence root
changed; scoped review was clear. Four offline composition controls passed, and
a fake-transport check admitted 30 requests and refused 31 without network calls.

Actual consumption: **4 new requests / 1 child / 0 retries**. All four exact raw
responses normalized. Two source reads and the findings submission succeeded.
Response four called `code_review_submit(kind="coverage")` without `review_id`,
despite that field being required in the advertised schema. Public argument
validation returned `invalid_arguments`; the no-retry guard stopped the run at
this first failure. A narrow offline replay of the captured arguments confirmed
that `review_id` was the sole missing field. Independent outcome review agreed.

The three accepted findings remain at revision 1; coverage and completion were
not accepted. Native status is `error`, the report is incomplete, and no seal
receipt exists. Raw/normalized usage totals **15,855 tokens**; durable budget
accounting remains unknown after the forced guard exception under ADR-131.
No completion/sealing/accounting success or AC24 qualification is claimed.

Credential-scanned raw responses, exact failed arguments, offline replay,
receipts, review and hashes:
`/Users/macbook-dev/.codex/artifacts/task-39147/deepseek-final30-20261107/`.
November 7 cumulative actual: **134 requests / 12 children**. The unused 26 request
and two authorized child slots were not used as retries or carried forward.
No full review, production change, push or merge occurred.

## Offline repair: omitted child review identity

The final30 capture proved that DeepSeek omitted only `review_id` on its coverage
submission after three accepted findings. The host already bound the child to
one review, but the adapter required the model to repeat that identity and returned
an empty `invalid_arguments` result on omission.

`CodeReviewService.bound_review_id` now resolves an omitted child identity through
the current actor, registered grant, context scope and active attempt. The tool
adapter uses it for reads/submissions and before source instruction path selection.
Child schemas allow omission; explicit values are never replaced, and primary
schemas and the underlying validated wire models remain strict. No source
permission, accepted record, submission revision or native sealing check is bypassed.
The existing ADR-281 is amended; ADR-131 accounting semantics are unchanged.

Verification (offline only):

- New omission/path/schema/rejection cases failed before the production fix.
  One initial test setup incorrectly prepared two reviews under one primary actor;
  it was corrected to use independent actors.
- All **193 affected tests** pass across targeted runs. The first combined run
  had 191 passes and one old missing-ID expectation; the corrected repeated-call
  cases both pass and still require `stuck`/incomplete/no seal. The added verifier
  case passes after correcting its checkpoint page assertion during scoped review.
- The **four exact retained tool payloads** replay successfully after remapping
  only ephemeral review/source identities. Three findings remain accepted;
  coverage advances to revision 2, with no premature seal.
- The actual DeepSeek request builder/parser and native runtime complete a scripted
  omission sequence at revision 3, with capture/terminal/seal receipts and **15,919
  budget tokens** (the captured four usage buckets plus two synthetic 32-token turns).
- At the same post-coverage boundary, native cancellation persists **15,855 budget
  tokens**. An escaped exception leaves the durable counter **NULL/unknown**, as
  ADR-131 specifies; both retain the three findings and remain unsealed. Raw usage
  is not substituted into the durable counter, and historical receipts are unchanged.
- Scoped review is clear. Changed tests/provider pass lint; all five changed Python
  files pass formatting; F/E9 and whitespace checks pass. The service retains its
  three pre-existing lint findings, with no new findings.

Evidence: `/Users/macbook-dev/.codex/artifacts/task-39147/review-id-offline-20261107/`.
The archive includes the exact-replay result, red/green logs, scoped patch and review,
static comparison, and source hashes. The original raw responses remain in the
separate consumed `deepseek-final30-20261107` archive.

**Zero new live requests or product children**; November 7 cumulative remains
134 requests/12 children. No live diagnostic, full review/suite, hosted CI, push or
merge. These are offline controls, not renewed live-model or AC24 qualification.

## Post-fix live DeepSeek confirmation: passed

After the user's direct request to confirm the fix, one fresh isolated diagnostic
ran with the announced 30-request/one-child ceiling and no retries. The runner
changed only its exclusive evidence directory; its reviewed transport guard and
30/31 cap control were reused. Four offline runner controls passed on the fixed
source before launch. No historical request allowance or claim was reused.

**Actual live consumption: 7 requests / 1 child / 0 retries.** DeepSeek Flash
omitted `review_id` on all six review-tool calls: three reads, findings, coverage
and reviewer_complete. All succeeded through the authenticated host binding.
Three original records were accepted; coverage and completion advanced the stream
to revision 3. The child then returned a readable original report and terminated
`done`, with complete capture and distinct native capture/terminal/seal receipts.
All seven pre-parse raw responses were retained and their hashes verified.

The durable child budget is **32,454 tokens**, matching the seven normalized usage
records (2,293 + 2,549 + 3,877 + 5,392 + 5,750 + 6,113 + 6,480). There are no
rejected submissions, fallback routes, guard stops or model retries. The live
result gate and pytest run passed; the process exited zero after fixture teardown.
The report readback is complete (`cursor=null`) and preserves all three originals.

This directly confirms missing-ID recovery, reviewer completion, native sealing
and settled accounting on the live DeepSeek path. It does **not** qualify the full
multi-model/Haiku-verifier workflow or complete AC24. Report quality is also
separate: the reviewer correctly identifies the empty-input regression, labels
boolean policy uncertainty, and includes the refuted QA comment as a non-defect
original record. No verifier adjudication or unified report was run here.

Evidence: `/Users/macbook-dev/.codex/artifacts/task-39147/deepseek-identity-confirm-20261107/`.
November 7 cumulative actual: **141 requests / 13 children**. The unused 23 request
slots are not used for a retry. No full review, production changes, push or merge
occurred during this confirmation.

## Full multi-model run after the missing-ID fix

The next explicit grant authorized one full native Console review with Luna and
DeepSeek reviewers, fresh Haiku verification, at most 60 new requests and three
children, and no retries. Luna also coordinated the same disposable `average()`
fixture. **The run failed end to end after 60 requests / 3 children / zero retries.**
Request 61 was refused before network. The native child schedule remained `[2,1]`.

R1, R2 and V1 all terminated `done` and sealed with complete capture, terminal and
seal receipts. Both originals were presented before V1 dispatched. DeepSeek
submitted three originals; Haiku disposed of every original exactly once. A
readable unified report was retained while the primary grant was valid. This
snapshot does not establish successful final delivery: after the cap stopped the
coordinator, final readback returned `stage_grant_unavailable` and the runner exited
2 with `live_original_mapping_incomplete`.

The observed remaining failure is primary coordination. Luna attempted
verifier-only batch/checkpoint operations before Haiku finished and again
afterward, receiving correct role refusals. Haiku's final response also falsely
said it had neither read source nor submitted dispositions, contradicting its own
captured successful calls and accepted completion at revision 4. The trace proves
both output/coordination problems; it does not establish their underlying
prompt/history cause or prove that Haiku's final text caused Luna's confusion.

Child durable budgets match raw provider usage: Luna **12,356**, DeepSeek
**32,904**, Haiku **55,344**. The initial coordinator also matches at **332,943**.
The aborted coordinator continuation has **211,692** weighted tokens in raw usage
but an unset durable counter; 24 automatic token reservations settled and one
remains uncertain. Fully settled accounting is therefore **not confirmed**. All
60 raw responses and usage records were retained; native shutdown left no active
runs, fleet threads, workspace connections/leases or registered chat connections.
Saved real settings and pinned source bytes remained unchanged.

Twelve narrow harness controls passed, including terminal-close raw capture,
malformed responses, compression, physical retry denial and accounting negatives.
Offline native composition used zero network calls. Scoped runner/outcome reviews
completed. These controls are separate from live evidence. An offline four-case
fixture comparison confirms only empty-input behavior changed; bool/iterator
behavior predates this patch, and their bug status depends on API intent. Luna's
empty report missed the seeded defect. No accepted-submission conflict occurred
live; its recovery remains supported by the earlier offline controls.

Private evidence and readable reports:
`/Users/macbook-dev/.codex/artifacts/task-39147/full-review60-20261107/`.
November 7 cumulative actual is **201 requests / 16 children**. The new allowance
is consumed; no retry, new production repair, full-suite/CI run, push or merge was
performed. This is not a successful new AC24 qualification.

## Coordinator handoff repair — offline follow-up

The full run exposed app-side gaps, not merely model noncompliance:

- The primary verification guide explicitly instructed verifier-only batch/checkpoint reads and disposition submissions; actual primary schemas refused them. Both packaged recipes now keep participant work in the host-composed child brief.
- Native unified output existed, but the primary report API omitted accepted dispositions. Its existing bounded `report` records now include the decision and `verifier_report_status`, under unchanged source authority, pagination and immutable cursor replay. A later reviewer retry outside V1's frozen map exposes `original_not_in_verification`, without a fabricated decision.
- Verifier history reduction removed acknowledged tool rounds without identifying the retained decisions as the same verifier's work. It now carries attempt/target identity and three bounded accepted-verdict counts. Undelivered host capacity gaps remain in the ledger but are not attributed to the verifier. This repairs the demonstrated context loss; whether Haiku's prose improves still requires live evidence.

RED reproduced wrong-role guide calls, absent disposition readback, absent checkpoint counts, the retry-original KeyError and host-gap misattribution. GREEN: 151 affected regressions (history, reports, tool provider and submission recovery), followed by seven focused handoff cases covering both review corrections. The correction review found no remaining blocker. New files/history lint pass; baseline comparison of touched service/provider/existing-test files reports 11 existing lint findings and zero new ones; F/E9 and whitespace checks pass.

No accounting policy was weakened: recorded native cancellation controls retain known usage, while exceptions escaping the runtime preserve unknown usage under ADR-131. The previous physical-cap exception was a diagnostic abort, not evidence that a normal completed child lost its usage. Historical aborted counters are not rewritten. The new live result must still have fully settled accounting.

Zero new live requests or children in this repair. November 7 actual remains 201 requests / 16 product children; the last 60/3 grant is consumed. No new full-review success is claimed. Current offline logs: `/private/tmp/task39147-handoff-tests.log` and `/private/tmp/task39147-handoff-review-fixes.log`.

## Hosted CI bundle-stamp correction

PR #3473's Tool Recovery run `42418580000` failed the same bundle byte-parity
assertion on Python 3.12 and 3.14. The unchanged assertion also failed locally.
Running the existing bundle generator changes only the final SHA-256 stamp;
the executable prefix, builder, worker behavior and module-size budgets are
unchanged. This is a generated-artifact correction, requiring no new ADR.

After regeneration, all 17 bundle/selected loopback checks and both existing
bundle-size checks pass. Generated-file lint and diff whitespace pass. Scoped
review found no blocker. Hosted CI on the corrected head is still required.
Logs and the exact artifact delta are retained at
`/private/tmp/task39147-bundle-ci-20261107/`.

The user subsequently approved one fresh full Console qualification after CI:
60 new generation requests, three children, Luna coordinator, Luna/DeepSeek
reviewers and Haiku verifier, no retries or extra generation probes. That grant
remains unused through this CI correction: zero new requests/children, November 7
actual still 201/16. Repinning this unconsumed grant adds no allowance. The earlier
failed full run and its unknown aborted accounting remain historical limitations.

## Hosted CI wizard synchronization correction

Derived Artifacts run `42423661679`, admission job `128611492184`, failed the
keyless-401 wizard test's returned-to Provider notice assertion: the strip was
empty. The test passed alone unchanged. Delaying the real failed-row mount's
completion by 250 ms reproduced the exact hosted assertion: the wait accepted
Provider's handoff failure before Model recorded its own rendered outcome, then
Back cleared the handoff and cancelled the unfinished Model render.

The test now waits for Model's recorded failure and asserts `key_required` before
returning. All original semantic assertions and the existing wait bound remain.
No production code, workflow or timeout changed. The delayed-mount control passes;
removing the actual return notice still fails the original assertion. All 23 cases
selected by the admission lane's 13 wizard selectors pass. Changed-function
formatting and whitespace pass; file lint has the same 53 baseline findings and
zero new findings. No new ADR is required for this test synchronization fix.

Logs and both scratch controls are retained at
`/private/tmp/task39147-admission-ci-20261107/`. Fresh hosted CI is still required.
The one approved full qualification remains unused: zero new requests/children,
cumulative actual 201/16. No live completion or settled-accounting claim is added.
