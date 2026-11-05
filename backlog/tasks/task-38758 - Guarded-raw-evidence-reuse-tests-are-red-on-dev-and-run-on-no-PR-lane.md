---
id: TASK-38758
title: "Guarded raw-evidence reuse tests are red on dev and run on no PR lane"
status: To Do
assignee: []
created_date: 2026-11-05 07:11
labels:
- follow-up
- tests
- backup-recovery
priority: medium
references:
- Tests/Backup_Recovery/test_guarded_raw_evidence_reuse.py
- Tests/Backup_Recovery/test_mcp_source_lifetimes.py
- tldw_chatbook/Backup_Recovery/generation_witnesses.py
dependencies:
- TASK-38685
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 7 of the 2026-10-23 follow-up burn-down. Checked on dev bb80905eb3, 2026-11-04.

Run on its own (`-n 4`, through the scratch-profile wrapper), `Tests/Backup_Recovery/test_guarded_raw_evidence_reuse.py` gives 40 failed and 14 passed. The tests warm a reader, count the admission derivations made by the next read, and require that count to be zero. The counters still record derivations, so the failures are in these warm-up preconditions:
- "oracle cache never engaged: {...}" at :162 (24 nodes, the `test_complete_witness_and_pause_reuse_matches_original_mutation_verdict[...]` family);
- "warm reader N still derives" at :75 (5 nodes, `test_actual_warm_mcp_readers_skip_all_admission_derivations`);
- "the actual source must be warm before the pause stimulus" at :887 (`test_actual_permission_parse_pause_matches_original_refusal[True/False]`);
- and others: `test_witness_cache_covers_every_input_of_the_complete_original_reader`, `test_warm_witness_requirements_have_exact_original_refusals[...]`, `test_warm_bound_*` and `test_concurrent_full_witness_reads_cannot_demote_confirmed_evidence`.

TASK-38685's notes (PR #3402) list these as red on dev and untouched by that task. They also record a contradiction. With its warm precondition removed, `test_actual_permission_parse_pause_matches_original_refusal` gets `DID NOT RAISE` under both the old and the scoped stub. It expects the kill-switch write to be refused with `storage_locally_paused` when a pause begins right after the parse (:899-903). `Tests/Backup_Recovery/test_mcp_source_lifetimes.py::test_permission_rmw_admitted_before_pause_finishes` (:52) pins the opposite for the same stimulus: the admitted read-modify-write completes across the pause. Both cannot hold.

The file's last two commits are the 2026-11-02 public-name refactors (b5e0256e72, b3874e98bd). Whether the reds date from them is not established. No PR lane runs the file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each red node is classified with evidence: a stale expectation (naming the commit that changed the behaviour on purpose), test-harness drift, or a lost evidence-reuse cache (a product regression, fixed or filed as its own task)
- [ ] #2 The parse-pause test and `test_permission_rmw_admitted_before_pause_finishes` agree on one recorded answer for a pause that begins after the read-modify-write's parse; the pin that loses is changed, with the reason in the notes
- [ ] #3 The file passes 3 of 3 `-n 4` runs
- [ ] #4 The file runs on a PR lane, and its measured time in that lane is recorded in the notes
<!-- AC:END -->
