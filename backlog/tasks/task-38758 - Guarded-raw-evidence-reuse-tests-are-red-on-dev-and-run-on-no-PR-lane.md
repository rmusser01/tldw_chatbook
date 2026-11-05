---
id: TASK-38758
title: "Guarded raw-evidence reuse tests are red on dev and run on no PR lane"
status: Done
assignee: ['@claude']
created_date: 2026-11-05 07:11
updated_date: 2026-11-05 23:26
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
- [x] #1 Each red node is classified with evidence: a stale expectation (naming the commit that changed the behaviour on purpose), test-harness drift, or a lost evidence-reuse cache (a product regression, fixed or filed as its own task)
- [x] #2 The parse-pause test and `test_permission_rmw_admitted_before_pause_finishes` agree on one recorded answer for a pause that begins after the read-modify-write's parse; the pin that loses is changed, with the reason in the notes
- [x] #3 The file passes 3 of 3 `-n 4` runs
- [x] #4 The file runs on a PR lane, and its measured time in that lane is recorded in the notes
<!-- AC:END -->

## Implementation Plan

1. Read every red node's failure and trace the counted derivations to the reuse guards (`_witness_metadata_current`, `_pause_metadata_current`); name the commit that added them.
2. Re-anchor the harness's derivation counter so it observes the seams in place (code identity) instead of replacing them, which the guards treat as non-stock readers.
3. Re-run the file; classify any node still red (harness drift, stale expectation, lost cache) with evidence, and fix a lost cache in product code red-then-green.
4. Reconcile the parse-pause test with `test_permission_rmw_admitted_before_pause_finishes` on one recorded answer.
5. Run the file 3x under `-n 4`, choose a PR lane step from measured dev job durations, add the file, time it with the lane's exact command, and update the `Tests/CI` lane pins.

## Implementation Notes

**Approach.** No product change: the evidence-reuse caches engage. The reds were the harness, plus one expectation that a later commit made stale on purpose.

**AC#1, classification (base 29592ecc9b: 40 failed, 14 passed under `-n 4`).**
- **Harness drift, 38 nodes**, caused by 3e618d7119 (2026-10-15). That commit added `_witness_metadata_current` (`generation_witnesses.py`) and `_pause_metadata_current` (`admission.py`). They decline reuse when any reader the reuse would skip is not the stock function. `_derivations` counted by replacing exactly those readers (`bootstrap.startup_permission`, `_control_records`, `_registry`, `Admission._read`, `_groups`) with call-through wrappers, so installing the counter turned the cache off and then counted the fallback. Evidence: a temporary `return False` in `_witness_metadata_current` reproduces the dev witness signature `{_control_records: 4, startup_permission: 1, _registry: 3}` on 23 nodes. The same in `_pause_metadata_current` reproduces the pause signature `{_read: 2, _groups: 2, _control_records: 2}` on 22 nodes. Their union is all 40 dev reds. Witness guard: the 12 `[...-witness]` mutation-verdict nodes, `test_witness_cache_covers_every_input...`, `test_warm_witness_requirements...[bytes/mode/replace]`, `test_complete_historical_chain...`, `test_concurrent_full_witness_reads...`. Pause guard: the 12 `[...-pause]` nodes, `test_actual_warm_native_probe...`, `test_warm_bound_config_companions...`, `test_warm_bound_history...`. Both guards: `test_actual_warm_mcp_readers...[0-4]` (0/2/3 need both, 1/4 only the pause guard) and the two parse-pause nodes' warm precondition. Fix: `_derivations()` now counts the same seams by code identity with `sys.monitoring` PY_START, without replacing them, and a fixture frees the tool id. `private_paths._open_verified_parent` is one closure shared by `bootstrap` and `raw_participants`, so that seam is counted only for callers in `raw_participants`, as the wrapper did. Positive controls: the file's `test_witness_content_still_requires_the_original_settle_margin` requires derivations and passes. A temporary `_EVIDENCE_REUSE = False` before the counted read makes every `test_actual_warm_mcp_readers...` node red, with all seven seams counted (`_open_verified_parent: 1` each).
- **Stale expectation, 2 nodes** (`test_actual_permission_parse_pause_matches_original_refusal[True/False]`, after the warm precondition). 3e618d7119 changed `recovery_activation._require_store_write` to `observed(path, retained=raw._mcp_observation(...))`, now `raw.mcp_observation`. The write's binding check therefore reuses the lease the RMW was admitted under, instead of the fresh acquisition that the old comment said refuses after a pause. The same commit added the "accepted mutation reacquired after pause" assertion to `test_permission_rmw_admitted_before_pause_finishes`.
- **Lost evidence-reuse cache: none.**

**AC#2.** The recorded answer is the lifetimes pin's: an RMW admitted before a pause that begins after its parse completes and publishes. The losing pin is renamed `test_actual_permission_parse_pause_matches_original_admitted_write[True/False]`. It now asserts one pause and a persisted `kill_switch` of True under both reuse settings, with the reason in a comment. Negative control: dropping `retained=` from `_require_store_write` turns both of its nodes and `test_permission_rmw_admitted_before_pause_finishes` red with `RecoveryRequired: storage_locally_paused`.

**AC#3.** 3 of 3 `-n 4` runs passed (54 passed in 12.7 s, 14.1 s and 15.3 s; load average 19-22). Serial with `--timeout=180`: 54 passed in 28.8 s.

**AC#4.** Lane: dreams-boundary's sandboxed step "Run Dreams and Guardian boundary contracts", appended after the TASK-36532 rider. I read 25 successful dev pull_request runs (2026-10-31 to 2026-11-05). This job took median 6.2 / max 7.2 min of its 20, and the step 1.8 / 2.4. The others had less room: PR Fast Lane 31.6 / 34.8 of 35; admission 20.6 / 24.9, admission-2 26.4 / 28.5 and admission-3 19.0 / 24.1, all of 35; Console P0 18.1 / 19.5 of 20; Dreams tracking 7.1 / 10.2 of 20. The file is sandboxed, not bootstrap_profile. Measured with the step's exact command (all 20 targets in order, `--timeout=180 --tb=short`, serial, through the scratch wrapper, load 18-27), the step gave 277 passed in 178.3 s, and this file took 21.9 s of it (junit sum, 54 tests). Pins updated: `DREAMS_BOUNDARY_TARGETS` (`Tests/CI/test_ci_queue_pressure_contract.py`) and `INHERITED` (`Tests/CI/test_response_rules_critical_gate.py`). Removing either entry fails its contract. Tests/CI: 1199 passed, 1 skipped. Not verified on Linux: the lane's ubuntu run will be the first.

**Files.** `Tests/Backup_Recovery/test_guarded_raw_evidence_reuse.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `Tests/CI/test_response_rules_critical_gate.py`, `backlog/docs/lessons-testing-evidence.md` (the wrapping-counter trap).
