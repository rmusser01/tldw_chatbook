# Character display: omit a determined precheck

Task: TASK-34601, AC18, OPT91. Root owns integration and all native execution.

ADR required: no new ADR.
ADR path: backlog/decisions/225-console-send-preparation-and-io-ownership.md and backlog/decisions/226-console-polling-and-full-state-reconciliation.md.
Reason: one presentation comparison is already determined by resident identity; the complete original refresh retains every read and lifetime boundary.

## Decision

Use `_stock_character_refresh_readers` to qualify the original presentation route.
If its stored fingerprint is absent or its database/current-character identity or
open-conversation ID already differs from the captured resident fields, the
outer `_capture_scope` cannot return an equal fingerprint. Enter the existing
refresh directly. Do not transfer a captured scope, retain a handle, extend TTL,
add a worker, or use presentation evidence for an action.

If stock qualification fails, the fingerprint shape is unknown, or those resident
fields match, keep the outer fresh metadata probe. In particular a database
revision/authority change with unchanged ambient identity still needs that probe.
Existing `refresh` owns its generation increment, fresh reader selection,
pair/groups/pair read, post-await refusal, errors and exact connection retirement.

## Work and verification

1. Verification lane owns only Tests/Backup_Recovery/test_character_refresh_finite_batch.py.
   Update the real changed-display control to require one finite callback/two
   metadata pairs, preserving actual rows and physical retirement. Add focused
   identity-change and unchanged/custom controls where existing coverage lacks
   them. Root establishes original-source failure before the product guard.
2. Controller lane owns only UI/Console_Modules/character_context.py. Prepare the
   minimal resident check using the existing reader object; no abstraction or new
   qualification framework. Root applies the candidate after RED.
3. Shared lane reviews the exact shortcut and existing fallback/effect semantics
   independently and owns the existing source-controls and completed-memo test
   adaptations. Root owns the queue/admission entry controls and runs final
   integrated targeted controls once source is
   frozen, then a sequential quiet comparison. Record work counts separately
   from whole-Send timing and preserve negative samples.

Current attribution cannot distinguish outer scope from refresh because ancestry
truncates at metadata. No prior saving or affected-branch frequency is claimed.
The existing changed-display test demonstrates two finite callbacks where the
first result only selects an already-required refresh. This small simplification
can be retained for reduced redundant work; it does not meet the overall latency
target without corresponding timing evidence.

## Qualification in progress

Original product2b5e6eef1b fails the intended one-callback assertion after actual
row equality and handle retirement (character-precheck-red-1). The candidate
passes58 of59 integrated cases; the remaining metadata-error control assumed
two prechecks across two attempts. After the first failure clears the fingerprint,
the second now correctly uses refresh directly. Its assertion now proves exact
per-attempt reads and absence of memo, with the original real SQL fault and repair.
The focused rerun passes all6 selected controls in21.03s, yielding59 distinct
qualified cases with unchanged product source; no deadline changed. Both runs
record unchanged source/HEAD. All6 changed Python files pass lint and format
checks. Two independent source reviews found no blocker.

In the original/candidate causal controls, the direct action remains4 callbacks,
3 pairs and1904 native opens. Changed presentation falls2→1 callbacks,3→2 pairs,
2→1 newly opened/retired handles and663→405 native opens. Instrumented callback
wall times are not whole-Send acceptance. Candidate and baseline use the same
profile layout and all original body/source/retirement assertions.

## Sequential quiet result

| Sample | Cold Send s | Warm Send2 s | Warm Send3 s |
|---|---:|---:|---:|
| Baseline A1 |8.551652|4.898911|4.449660|
| Candidate B1 |5.424602|5.754672|29.089330|
| Candidate B2 |6.786114|4.495246|4.515487|
| Baseline A2 |7.257562|5.168478|3.693676|

Warm means4.552681→10.963684s establish no latency improvement. B1's third
Send is retained, including4.779s before controller,8.229s through save,14.723s
saved-to-trace and1.357s trace-to-provider. No cause is assigned and no favorable
retry is selected. The other candidate warm samples also do not establish an
improvement over the baseline range. This branch retains the small determined-
comparison omission for the independently qualified native-work reduction, not
as evidence of an elapsed-time fix or regression-free performance. The unexplained
long turn remains an active investigation in TASK34601.

All12 turns save and settle with3 complete traces/links and zero checkpoints
per process; source/HEAD remain stable and no native overlap is detected. Only
character_context.py differs in normalized loaded product source. B1 additionally
loads Workspaces/change_retention.py; common loaded bytes are identical within
each arm. The membership variation is retained; corresponding file bytes differ
only by line endings. Other raw differences are also newline-only.

Receipt: character-precheck-comparison.json SHA256 84fc2c3c6f71b0c8a51e643ab9e3a0809fa13f887b5fd99efed568856eab97e9.
The full-default profile uses a stubbed provider and headless UI; no physical
frame or percentile claim follows. The under-one-second task remains open.
