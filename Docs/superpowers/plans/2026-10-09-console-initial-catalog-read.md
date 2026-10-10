# Initial Console catalog operation — bounded follow-through

Task: TASK-34601, AC10; OPT28/73. Integration owner: Codex root.
ADR required: no new ADR.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md and backlog/decisions/126-complete-local-backup-and-recovery.md.
Reason: consolidate redundant method-entry proofs inside one existing finite operation; retain fresh initial input, all real read/write gates, final proof, native retirement and custom behavior.

## Evidence and scope

The earlier two-path catalog experiment remains rejected: its final quiet BA pair reversed the initial apparent gain. It added a final check to composition as well as removing three method-entry proofs from initial capture. Since then OPT88 has removed ordinary builtin-only composition's catalog dependency. This experiment changes only initial capture; `_CapturedLocalCatalog.read_bundle` remains as received. The original rejection and source archive remain available.

On clean bae450ce5e, mcp-maximum-detail-1 qualifies18 original returned bodies and42 matching QPC stages with zero observer issues, overflow or unfinished captures. Warm permission checked-read intervals are0.116482/0.132144s; catalog checked-read intervals0.535214/0.446020s; inventory0.005888/0.007528s. No audit body executes. Cold catalog is0.110222s, so scheduling/host variability matters: these inclusive intervals are ceilings, not removable native work or predicted savings. Receipt SHA256709edd6d4015c3cccd4a6891cf38f18bc9b72d2ed9863bcb7c7859bd4a5b127a.

## Implementation and ownership

1. Test lane adapts the preserved original native count control to initial capture only. Root establishes RED: four scope checks plus one existing final check versus intended one plus one, after original payload, readable/operation/file gates, one acquisition, leases and descriptor retirement have succeeded.
2. Only after RED, product lane owns MCP/local_store.py and MCP/console_snapshot.py. Reuse the archived shared original load-body extraction, pure bundle projection and private issued-owner helper. Preserve readable-before-dynamic-reader lookup and the actual migration writer. Reuse defining callback records and existing input/native guard qualification; do not duplicate the load implementation or introduce a new generic framework.
3. Initial `_CapturedSources` consumes that helper through its existing checked read. Unknown/custom inputs retain the preceding route. Effect/uncertainty marking covers the final check and retirement. Do not alter composition, permission loading, raw/storage infrastructure, watch policy, TTLs or source lifetimes.
4. Test lane owns initial count/behavior controls and the narrow passive source-drift barrier adaptation in test_console_snapshot_source_contracts.py. Preserve the current OPT88 plugin corrections. Root owns integration, documentation and every local test/timing run; no lane runs native work concurrently.
5. Root runs targeted parity, missing/corrupt/inactive/migration, custom reader, held-read cancellation, source drift and original worker-ownership controls. Test barrier changes must still hold after actual original native admission and preserve mutation/refusal/retirement assertions. Independent review and scoped lint/format checks precede comparison.
6. Compare frozen candidate against exact bae450ce5e product sequentially with the original full-default, saved-turn probe, then a reversed confirmation pair if needed. Retain every raw sample and failed run. No deadline/budget changes, early imports, native overlap or speed claim from instrumented elapsed time. Reject and restore if improvement is not supported or implementation complexity grows beyond the already-reviewed extraction.

## Limits

The extraction still adds roughly200 net product lines of qualification and owned-body plumbing; changing only one consumer does not make those correctness obligations disappear. A small uncertain gain does not justify retaining it. No new policy, public API, process cache, retained snapshot or across-await lease is proposed. Ordinary one-second Send and physical sub100ms feedback acceptance remain open.


## Outcome (2026-10-09)

Rejected; no initial-catalog implementation retained. The original source fails only the causal four-versus-one scope assertion (initial-catalog-red-1), after real payload and ownership/retirement proof. Candidate integrated controls pass77 with two new test expectation failures; both expected only the external governance gates copied from composition. Initial capture also calls the inventory gate. The corrected exact ordered three-gate expectation passes both originalbae and candidate2/2, preserving all custom argument/thread/result assertions. Candidate total is79 distinct passing cases. Source/HEAD stayed stable, static checks pass and independent review found no actionable correctness issue.

Quiet sequential ABBA followed by BA confirmation, cold/warm/warm seconds:

| Run | Seconds |
| --- | --- |
| A1 baseline | 4.176367 / 2.848485 / 2.315082 |
| B1 candidate | 3.874678 / 2.672952 / 2.362119 |
| B2 candidate | 3.719994 / 2.800344 / 2.378922 |
| A2 baseline | 4.398010 / 2.730120 / 2.522113 |
| B3 candidate | 3.728209 / 2.768415 / 2.341416 |
| A3 baseline | 6.314893 / 4.866397 / 4.480464 |

All18 saved turns settle with zero checkpoints; source hashes/HEAD are stable and no native overlap is detected. Every run within each arm loads identical sources. The only normalized product differences are the two candidate files; three other raw hash differences are line endings only.

Initial ABBA warm means2.603950/2.553584s differ by50.37ms (1.93%). Cold means4.287188/3.797336s are suggestive but too few to establish reliable cold gain. A3 slows broadly across preparation, commit and postcommit phases; its cause is unproven. The six-sample aggregate22.46% is not a causal saving. These results do not justify approximately213 net product lines of qualification and effect-handling machinery. Independent review agrees. Preserve the count/behavior proof separately from the adoption decision.

Exact rejected source/tests and manifest are retained in task artifacts initial-catalog-candidate/rejected-final; initial-catalog-comparison.json SHA329f9cc41a657c190d93d4c6071b2ebc0041eac31e37c73022e9437495fa43c1 includes every raw run and interpretation. The two product files and existing source-contract test are restored to publishedbae, and both experimental test files removed after verifying the archive. OPT88 and the retained context-publication route remain shipped. OPT28/73 remains available for later review, not scheduled work or a promised saving.
