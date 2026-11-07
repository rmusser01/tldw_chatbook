# TASK-37144 scoped Superpowers code review

Reviewed base `e96b3507be115e9790b4d37beafe471f5785252d` through head `a0457fe1c7d9ba26d2441662bf9d5ac49a342c3a`, using the supplied immutable diff package, TASK-37144, the root plan CI-repair section, implementation report, relevant native code, and actual retained receipts. Scope is the census test, its one Perf workflow enrollment, and TASK-37144. This is a review of the new atomic CI fixture repair, not another review of the four product tasks.

## Strengths

- `Tests/Performance/test_console_keystroke_work_census.py:757` reuses the existing storage-mode hold for all typing probes. The key count, positive-readiness floor, three-builds-per-key ceiling, and production implementation are unchanged. The inspected native coordinator (`tldw_chatbook/UI/Console_Modules/console_spend_projection.py:1502`) still replaces its timer and invokes the same summary/cost callbacks.
- The control at `Tests/Performance/test_console_keystroke_work_census.py:1275` calls the real schedule and real Pilot dispatch, retains native Timer objects and ownership/coalescing, and controls elapsed delivery without replacing the production coordinator. It checks actual composer text on every key, requires 24 keys and at least 24 native timer schedules, and retains positive counted work. The finally block at line 1305 stops retained handles and drains their tasks; monkeypatch ownership and the existing private-profile decorator isolate the changes.
- The separately measured pause at `Tests/Performance/test_console_keystroke_work_census.py:768` preserves the typing counters, clears the observation counters, invokes real stop/refresh after restoring cadence, drains actual publications, and restores the typing result with pause-prefixed observations. The final 400-message control asserts positive paused readiness and zero history/snapshot/projected-row work in both phases at lines 1333-1346. This prevents the phase correction from silently dropping the pure-memory pause-history guard.
- `.github/workflows/perf-guard.yml:137` adds exactly the new node to the existing keystroke command. Removing that single line yields byte-equivalent normalized workflow text to base; existing targets, dependencies, timeout and other workflow settings remain intact.
- Task notes accurately distinguish local evidence, static source qualification, inherited formatter debt, and pending hosted CI. A new ADR is unnecessary for this bounded test-fixture correction under the existing governance.

## Evidence checked

I parsed the actual private-child XML for RED01, all three GREEN01 children, RED02 and GREEN02, and compared their properties to the consolidated receipt. All six matched. Both RED children failed solely at `4.041666666666667 <= 3` (97 builds / 24 keys), with 24 native timers and 24 delivered callbacks. Every child reported zero errors and skips. The final GREEN02 child passed with 25 typing readiness builds, 24 native timers, zero due typing deliveries, and 3 independently billed pause readiness builds. All asserted typing/pause history and snapshot counters were zero; context-estimate maxima were 0 and 1. Runtime-source properties resolve to this worktree. GREEN01's two existing guards passed; both scaling arms reported 25 builds.

I also independently evaluated the supplied static proof with its final output-file write removed. It succeeded: final `draft_pause=False` executable helper AST equals reconstructed GREEN01 (SHA256 `67a4e18ec403413d81b989e40fa6a6562692ecf930a4476b220f41183d6d4b59`), and `storage_units=True, draft_pause=False` equals the base helper. The other 23 existing functions and module assignments/ceilings remain equal. This qualifies the already-passed existing guards after the opt-in observation was added and preserves the existing storage pause/idle/revisit callback path without claiming a fresh storage run. I inspected the preservation script rather than treating its JSON assertions as sufficient evidence.

The final working census SHA256 matches `b27168b66459d914167c87f4c6c42edf9e6d521e7b96e16f25676b3f7f9bc773`. All five recorded source/workflow hashes matched. The production tree object remains `e4265ce072b5213a57d64359ebf0184d0dc761c4`, and the diagnostic inventory object remains `283912e7f06df2c37cbb1547ba0e7f9dc3a590d6`. Formatter comparison independently reproduced the same five inherited change regions and no new formatting debt. This is not a whole-file format-clean assertion.

No App tests, broad suites, remote operations or branch/index mutations were performed during this review. The only written file is this report.

## Issues

### Critical (Must Fix)

None found.

### Important (Should Fix)

None found.

### Minor (Nice to Have)

None found within the repair scope.

## Recommendations

Proceed with root-owned integration and actual current-head CI qualification. Preserve the task's pending delivery gate until the applicable hosted runs have been inspected; the local/source-equivalence evidence is sufficient for this scoped repair review but does not certify those runs.

## Declined to judge

- Final hosted Linux/Python/pytest behavior and every current-head event run: not yet qualified by the reviewed local receipts; root owns the explicit delivery gate.
- The four completed product tasks, current-dev integration, census totals and shard configuration: separately owned and reviewed work outside this three-file repair.
- A fresh local storage-mode runtime pass: not repeated because the default-false/storage-true executable path, existing callback functions and ceilings are preserved exactly; prior hosted evidence and final hosted verification remain root-owned.
- A new upper CPU/readiness budget for paused work: the plan explicitly adds positive pause and zero-history coverage without promising a paused CPU ceiling; establishing such a budget is a separate measurement decision.
- Natural wall-clock pause frequency or Timer scheduler timing accuracy: this controlled-clock regression proves phase attribution and invokes native coordinator work; it is not a timing benchmark or an exhaustive Textual Timer test.
- Fixing the five pre-existing full-file formatter regions: their exact edits are unchanged; unrelated formatting is outside this repair and no format-clean claim is made.

## Assessment

**Scoped repair: Approved.** The measured elapsed-pause work is assigned to a separate observed phase while the original per-key and storage guards remain intact. The deterministic native RED/GREEN evidence and independently checked source equivalence support the repair without production changes or budget relaxation.

**Ready to merge? No, pending delivery gates.** Actual current-head hosted CI remains root-owned and unqualified here; this review does not authorize a merge. No code fix is required by this scoped review.
