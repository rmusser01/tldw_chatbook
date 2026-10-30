# TASK-38101 qualification evidence

Implementation base: `a60e027f5db751f57f0256d6b132d08ca32f5679` (cost reporting).
The owner approved one full queue-front qualification plus one full background
qualification. ADR-103 and ADR-218 amendments govern the change.

## Local evidence

- All seven parsed application job definitions match the base exactly after
  removing their old preflight dependency and changing only UI max-parallel
  from 17 to `inputs.queue_front && 17 || 8`. Commands, selectors, dependencies,
  environment, 28 shards and timeouts are conserved.
- Targeted CI, queue, selector, census, replay, reporting and packaging run:
  **806 passed, 28 deselected**. Deselected cases exhaustively re-deal unchanged
  shard counts; no full application suite was run. The minimal environment
  warns that optional asyncio/timeout plugin configuration is not loaded.
- The real inline classifier executes under Node in behavior tests. Initial
  graph/classifier tests failed against the old workflow. Missing earlier-PR
  metadata produced six additional RED witnesses, then passed after repair.
- Independent review found permissive `Date.parse` and empty eligibility values.
  Eight actual-script probes failed, then passed with canonical UTC timestamp
  round-trip and eligibility validation. The resulting classifier suite has
  **55 passing tests**. No other actionable source defect was reported.
- A read-only live GitHub probe ran the actual classifier's GraphQL page and
  candidate reread: observed queue front #3264 received `queue_front=true`.
  This used local authenticated API access; it does not prove hosted token
  permissions or scheduler behavior.
- A further raw-graph RED/GREEN check requires the wrapper verdict to use
  `always()`, so cancellation does not skip its non-success rejection. Existing
  individual lane verdicts remain unchanged apart from their output references.
  The post-correction admission/aggregation run passed all **150 tests**.
- Ruff lint/format and Git whitespace checks cover the changed Python files.
  Source guards inspect the actual caller and callee; their helper does not
  rewrite conditions, dependencies or result expressions.

## Tool limitations

Actionlint 1.7.12 reports eight schema diagnostics: one for `concurrency.queue`
and seven for `jobs.<id>.result`. Both features are documented by GitHub; the
result-context limitation is also tracked in upstream issue #343 / PR #724.
Running with only those two exact diagnostic patterns excluded reports no
additional findings. No repository lint suppression was added. This is not a
claim of an unqualified actionlint pass or a substitute for hosted validation.

- [Native concurrency queues](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency)
- [Reusable workflow call keywords](https://docs.github.com/en/actions/reference/workflows-and-actions/reusing-workflow-configurations)
- [Reusable jobs context](https://docs.github.com/en/actions/reference/workflows-and-actions/contexts#jobs-context)
- [Actionlint result-context issue](https://github.com/rhysd/actionlint/issues/343)

## Hosted qualification still required

Keep this PR draft and TASK-38101 In Progress until hosted evidence verifies:

1. The current PR run accepts the native queued reusable call, exposes all seven
   results and reports the unchanged required context with all 34 application
   jobs represented. Observe prefixed job names and the cost report.
2. A second background qualification waits while the first holds its slot; a
   verified front can use the separate slot. UI execution respects 8 / 17 limits.
   Preflight and queue-control jobs remain outside those slots.
3. A failed-job-only retry reacquires admission, retains successful lane results
   and only reruns failed work plus dependent aggregation. Do not intentionally
   corrupt a production PR merely to manufacture this evidence.
4. Missing, failed, skipped or cancelled wrapper/lane results cannot report a
   successful required check. Native overflow cancels excess pending calls;
   local source guards alone do not certify that hosted cancellation path.

No throughput, runner-minute or queue-latency improvement is claimed before
rollout measurements. The 37-job bound covers application jobs using the new
graph; old PR versions, preflight, control jobs and unrelated workflows remain
outside it. Classification is a snapshot, not a preemptive scheduler or an
account-wide runner reservation.
