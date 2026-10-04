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
