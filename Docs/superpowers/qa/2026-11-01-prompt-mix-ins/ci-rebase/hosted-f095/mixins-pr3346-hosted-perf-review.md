# Independent hosted Performance contract review

Verdict: no actionable Critical, Important, or Minor findings in the three-file patch against `f09597cdd5a2b5af1b07dea7eabe2eb6d33d9c80`.

Read-only review of settled working-tree source in `/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook`. No edits to product/tests, runtime tests, or performance runs were performed by this reviewer. The implementer signalled these three source files settled before closeout.

## Connected behavior checked

- The compiled cold-FULL mechanism executes the real selected ChatScreen method/guard bodies. Upstream FULL now calls `_sync_console_temporary_chip()` after awaited tab publication. The added fixture callback asserts a recorded surface publication and zero outstanding tab-sync calls, records exactly one invocation, and retains the existing deferred-replay and delayed-scheduling assertions. It is an observed ordering seam, not an unconditional noop substituted for production work.
- A stable real, immutable `ConsoleAttachVisit` replaces the obsolete `object()` visit. Production now invokes `same_visit` and checks active equality after the preparation await, so the fixture must supply that value contract. This particular cold-deferral path returns before `can_complete` and therefore does not fabricate or test runtime attachment authority. The callback preventing progress beyond deferred projection remains in place. Source digest, installed-body compile, teardown, task ownership, and prohibited-import checks are unchanged.
- Production persistence classifies exactly the two mix-in attempt message references as forbidden locators: `user_message_id` and `assistant_message_id`. The inventory change from 28 to 30 reflects those references. New tests assert their precise inclusion in both the inventory and seeded distribution. Neither reference is exempted or removed from reconciliation.
- There are 26 forbidden locator observations: two checkpoint columns at 44 rows and 24 others at 38 rows, yielding exactly 1,000. Every large bucket remains exactly 100 times baseline, totalling 100,000. Shared-table columns still receive identical counts and the existing grouping/conflicting-count validation remains unchanged. These are locator observations, not a claim of 1,000 distinct physical rows.
- Warmups, trial counts, ABBA order, SQL planner inspection, SQLite parity, transaction timing, fresh-commit and uncertain-retry paths, and all timing thresholds remain unchanged. The strict `xfail(raises=AssertionError)` remains unchanged; no new skip or waiver was introduced.

## Retained execution evidence inspected

The implementer's `/private/tmp/mixins-f095-perf-contracts/final.log` and `final.xml` report **26 passed, 1 xfailed in 3.64 seconds**. I read those artifacts without rerunning them.

I also read `run_observed.py`: its collection wrapper calls the original history gate, writes its returned report, and returns that same object to the unchanged test assertions. The retained `history-report.json` confirms 30 inventory locators, 1,000/100,000 observations, 10 warmups, 40 trials, ABBA measurement order, and scan_count 14. Consequently the unchanged assertion sequence reaches `assert report["scan_count"] == 0` after its preceding structural assertions pass; this is an inference from the captured returned data and source, not a separately captured traceback.

Both baseline and large plans show covering-index SEARCH for both mix-in columns. The seven scanning columns per corpus are the two hook-continuation receipt message columns, two machine-followup receipt message columns, and response-rule drafts/fixtures/validations message columns. The existing xfail reason still says eight; that stale descriptive count predates this patch and was not silently presented as current measurement. The gate remains an expected failure for the existing query-plan defect, not a green performance-threshold claim.

## Limits

This is fixture/benchmark-contract review, not native UI qualification or proof that all performance tests pass. The strict expected failure remains visible. No production authorization, schema, index, source fence, or runtime scheduling code is changed by these three files.

## Reviewed working-tree SHA-256

- `Packaging/speculative_voice_history_gate.py`: `f46c7be363dbc92bfe0944815e4d2c957c955461c14648b358658ae6e39db176`
- `Tests/Performance/test_console_tab_sync_source_ownership.py`: `0539d72e72677fb60738fd89229da26754c85014f50165f3b48f4840314b84f8`
- `Tests/Performance/test_speculative_voice_latency.py`: `26d598cecb97fece6313916ebc7d554bf8aed456bee88d3b55e8f5e2e33fe48e`
