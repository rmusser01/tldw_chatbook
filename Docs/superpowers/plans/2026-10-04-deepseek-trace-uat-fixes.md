# DeepSeek captured-send repairs

> Execute in this session with `superpowers:executing-plans`, using targeted tests and live UAT.

**Goal:** Complete setup, select DeepSeek/deepseek-chat, and exchange three messages with capture enabled; provider failures must remain recoverable.

**Architecture:** Preserve the strict provider profile and durable ownership gates. Settle failed calls with no response as trace-owned terminal records before automatic retries. A retry may reuse only unchanged messages and system content under the existing actor/chain/source/policy proof. Preserve custody when sealing fails or cancellation interrupts its await.

**Tech stack:** Python, Textual, SQLite, hosted Chat Completions.

**Spec:** TASK-34367, TASK-34368 and TASK-34369 acceptance criteria, based on the private live UAT. IDs moved above a concurrently created main-checkout task before finalization.

**ADR required:** yes, amendment to `backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md`; provider allowances already follow ADR-179. No schema or storage authority changes.

## Constraints and review focus

- No full test sweep; preserve provider-specific strictness and credential filtering.
- Unknown dispatch, source revisions, stale chains and changed retry surfaces remain fail-closed.
- Cover complete and streamed logprobs; cover retries before and after tool execution.
- Verify real durable records and mounted controls, followed by paid live UAT using the existing authorized credential.
- Diagnose Windows stalls and compare tokenizer bytes to the Git blobs independently.

## Steps

- [x] Reproduce documented DeepSeek null/object logprobs through the actual adapter; observe parser failures.
- [x] Declare the DeepSeek-only choice allowance; verify the adapter regressions pass.
- [x] Reproduce provider rejection and automatic retry through the real controller/gateway/store/SQLite; observe the failed reservation.
- [x] Add durable failure and unchanged-surface negative controls, then repair settlement ordering and retry admission.
- [x] Run targeted provider, trace lifecycle, ownership, agent retry and mounted recovery modules; inspect lint and diff.
- [x] Repeat live setup and three-message conversation with capture enabled. Check persisted replies, trace states, context retention, original-profile hash and Windows stall diagnostics.
- [x] Record root causes, prior-fix gaps, validation and any remaining limits in task notes and lessons.

## Evidence and limits

Code starts at fetched dev `8f83422dde2a5b95da648882b8f7e09da5b41f08` in
`codex/deepseek-trace-uat-fixes`. Private receipts are under
`C:/Users/GDesktop-1/.codex/visualizations/2026/10/04/01a107fc-5504-7b82-a134-ee7a716f978d/deepseek-uat`.

- Final 12-module targeted run: 216 passed, 3 deselected, 2 warnings. Those
  three settlement assertions fail on unchanged dev too; baseline output is
  retained. No full-suite claim or full sweep.
- A subsequent real canonical-assistant fault regression passed with the
  two cancellation/custody checks (3 passed). Both wire-format ownership and
  system-content controls pass (26 cases).
- Review found lost failed-seal custody and a rendered-system bypass; both
  were fixed and covered. Re-review found no remaining Critical/Important issue.
- All six new test modules lint/format clean. The six changed production
  files retain their 50 existing Ruff diagnostics with no new diagnostic.
- Main checkout tokenizer data had CRLF transformations despite the already
  correct `-text` attributes. Restored only the five exact line-ending-only
  matches to the reviewed Git blobs; all six manifest hashes match. The original
  checkout's vendored-cache module passes all 16 tests, including loading every
  supported encoding with upstream network reads blocked.
- Live setup, DeepSeek/deepseek-chat and the three-message conversation passed.
  Two further follow-ups also completed; the last header confirms streaming On.
  All five calls are COMPLETE, all five replies and response links are durable,
  context is retained, no dispatch checkpoint remains, and owner config is unchanged.
  Receipts: `uat-verification.json`, `conversation-stream-fixed.png`.
- Removed an unused registry lookup for established stores. Its new tests
  failed before the change and pass afterward; empty startup/resume checks pass.
  Live submit acceptance fell from 6.1–7.0s on the earlier profiled runs to
  4.1s on one unprofiled run, which is an observation, not a controlled benchmark.
  That run still recorded a 5.5s guarded-directory event-loop stall and 29.1s
  to provider entry. The final streamed follow-up measured 3.9s acceptance,
  a 4.8s native storage stall and 21.4s to provider entry. Broader Windows
  storage latency remains unresolved. The test build remains open on localhost.
  The user requested a PR against dev on 2026-10-04; the fix branch is being
  published after verification, followed by a separate pause/root-cause investigation.

## PR shepherding and provider audit (2026-10-04)

- Reproduced changed retry target/settings bypass in 14 both-wire cases.
  Compare durable provider/model/endpoint/generation/response/reasoning and
  tool/literal components before dispatch; route transition remains valid.
- Six affected PR modules: 58 passed; production Ruff retains 33 unchanged
  diagnostics in trace service, changed test modules lint/format clean.
- Independent review found no remaining Critical/Important issue.
- Initial cross-provider actual-adapter probes: 10 failures across Groq, OpenRouter,
  Together, Fireworks and Cerebras; six baseline controls passed. After merging
  dev PR #3014, Together replies and usage pass in both modes: exact recheck
  reports 8 failures across the other four providers and 8 passes. Followups
  TASK-34367.1 through TASK-34367.5 remain To Do; Together AC #1 is satisfied
  independently and its remaining optional-annotation work has medium priority.
- After incorporating dev `49206beea9`, 248 targeted tests passed across the
  six PR modules plus captured-provider replay, inference presets and discovery.
  Evidence: `Docs/superpowers/qa/2026-10-04-provider-response-audit/audit.md`.
- ADR required: no new ADR. ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md. Reason: complete its existing exact-request retry contract; provider followups implement ADR-179.

Broader targeted trace runtime/service/system-prompt run: **160 passed, two failed**. Both failing cases are `test_failed_rollback_releases_observers_and_reports_ambiguous_outcome[False/True]`: the unchanged test constructs a transaction manager and exits without entering, so `_maintenance_context` is absent and masks its intended rollback/commit error. The same two failures reproduced after temporarily restoring trace_service from origin/dev; the DB and test files are byte-identical to origin/dev. The repaired service was restored afterward. These pre-existing test-fixture failures were not silently excluded or repaired in this provider PR.
## Historical PR #3017 resyncs

These records preserve the original PR heads and their then-current limits.
The current combined UAT followups are recorded in TASK-34367.1 through .5
and TASK-34368, whose additional scope remains In Progress for final integration.

Further dev resync: conflict-free rebases onto `a78a9a900b` and then `8c4dfe59a2` preserve the reviewed production/test patches. First rebase: 259 targeted passes. Console-command base: 147 passes and six generation-template profile-fixture failures, all six reproduced identically on exact dev. Both derived-artifact preflights pass. Full evidence and failed node names are in the provider audit; no full suite or new paid live requests. Follow newly merged AGENTS.md and ADR-218 merge rules (rebase with lease, disable auto before pushing, current-head Qodo review before re-arming).

Buddy-base resync: rebased onto dev `3146bbd8da` (PR #3011), retaining both appended lessons in the sole documentation conflict. Production/test patches remain identical. Six PR modules plus Home/roleplay Resume, shutdown and CI workflow contracts: **109 passed**; full derived-artifact preflight passed. Audit records exact base and log paths. Auto-merge remains disabled until current-head Qodo review.

Console chat-start base resync: rebased onto dev `74557e202a` (PR #2995), retaining all base lessons and the PR lesson in the sole appended-documentation conflict. Production/test patches remain unchanged. Six PR modules plus store/session/Resume/provider integration: **574 passed, 39 baseline profile failures**, with every exact failed node and exception reproduced on exact dev. All 58 PR cases and the full derived-artifact preflight pass. Audit records exact nodes and logs. Auto-merge remains disabled until current-head Qodo review.

Library/Notes-base resync: rebased onto dev `f922b3591e` (PR #3021), retaining all base lessons and the PR lesson in the sole documentation conflict. No production/test overlap; all twelve reviewed patches remain unchanged. Six PR modules plus CI workflow/queue/dispatch and quit-flow architecture contracts: **109 passed**; full derived-artifact preflight passed, including the 143-file UI gate census. Exact base and logs are in the audit. Auto-merge remains disabled pending current-head Qodo review.

Fireworks/discovery-base resync: conflict-free rebase onto dev `54ac2758af` (PR #3019) preserves all twelve production/test patches. Six PR modules plus shared hosted parser/handler, captured replay, offline capture tooling, presets and discovery: **445 passed**; full derived-artifact preflight passed. Archived provider probes remain **8 failed, 8 passed**; Fireworks tool-call fixes leave the audited choice-logprobs defect open. Prior head UI shard passed its single rerun and all CI passed before resync. Audit records the exact evidence. Auto-merge remains disabled pending current-head Qodo review.


Console first-reply-base resync: conflict-free rebase onto dev `f99ad86ebd` (PR #3025) preserves all three commits and twelve production/test patches by range-diff equality. File overlap in provider gateway, trace runtime and ChatScreen has separate hunks. Six PR modules plus provider/failure, trace diagnostics, first-reply/token/request planning, first-mount handoff, native-start and CI contracts: **613 passed, one baseline fixture failure**; the exact failing provider-copy node reproduces the identical hook-unavailable refusal on exact dev. All 58 PR cases and the earlier CI native-start node pass locally. Full derived-artifact preflight passes, including the 146-file UI census. Audit now records the previous two CI failures and diagnostic limits. Queue mode is now on; auto-merge remains disabled pending current-head Qodo review and fresh CI.


Switcher/finite-owner-base resync: rebased onto dev `dce291d0fa` (PR #3024), retaining all dev lessons and the complete PR lesson in the sole documentation conflict. All twelve production/test patches remain unchanged; ChatScreen overlaps only in separate hunks. Six PR modules plus finite database/read ownership, complete runtime ownership, controller/session and CI contracts: **326 passed, one existing xfailed**. Full derived-artifact preflight passed, including the 151-file UI census. The previous head's complete green GitHub CI is now archived in the audit. Auto-merge stays disabled for fresh CI/current-head Qodo review.


Voice/TTS-base resync: conflict-free rebase onto dev `5daa67c2d1` (PR #3026) preserves all three commits and twelve production/test patches by range-diff equality, with no production/test file overlap. Six PR modules plus native pocket-tts, STTS/settings model, module-size ratchet and CI contracts: **299 passed, 12 baseline failures**, all twelve reproduced at identical assertions on exact dev and all 58 PR cases passing. Full derived-artifact preflight passed, including the 151-file UI census. Previous head `56477ed370` completed all GitHub CI successfully; receipts, exact baseline nodes and logs are now in the audit. No runtime/test edits or full sweep. Queue remains on, auto-merge disabled pending fresh CI/current-head Qodo review; explicit trigger authorization is still pending after automatic approval rejection.

Portable-setup-base resync: conflict-free rebase onto dev `76d5d157aa` (PR #3030) preserves all three commits by range-diff equality and all twelve production/test files byte-for-byte, with no production/test overlap. Six PR modules plus launch options, Backup/Restore setup, startup unlock, wizard and CI contracts: **161 passed**, including all 58 PR cases. Full derived-artifact preflight passed, including the 152-file UI census. Previous head `99d3befd02` completed all GitHub CI successfully; receipts and logs are now in the audit. Auto-merge is confirmed disabled; queue remains on. Fresh CI/current-head Qodo review remain pending, and the explicit review-trigger approval question stays pending. No runtime/test edits, full sweep or paid probes.


Orchestration-base integration plan (2026-10-07): preserve all production patches when rebasing onto dev 6feb84c1d2 (PR #2918); retain both appended testing lessons. The initial targeted run exposed the PR retry oracle expecting the old generic wrapper although ADR-211 preserves ChatRateLimitError. Update only that exception expectation and assert its exact type/provider, retaining all durable-state, target-equality and refusal assertions. Recheck affected PR modules, retain exact-dev evidence for the two unrelated provider-copy failures, and record preflight and previous-head green CI receipts before a lease push. ADR required: no. ADR path: backlog/decisions/211-ephemeral-provider-failure-presentation.md (alongside ADR-097). Reason: conform the existing regression to the landed typed-failure contract.

Orchestration integration completed: initial 666 passed/42 failed; forty PR test failures traced to the landed ADR-211 typed rate-limit contract. Only the PR exception oracle changed; six PR modules now 58 passed in 46.89s. Two provider-copy failures reproduce identically on exact dev (2 failed in 0.90s). Production patches unchanged, full preflight and changed-test lint/format pass. Previous a94e CI receipts archived in audit; queue now off, auto-merge remains disabled for fresh CI/current-head Qodo review. No full suite or paid probes.
