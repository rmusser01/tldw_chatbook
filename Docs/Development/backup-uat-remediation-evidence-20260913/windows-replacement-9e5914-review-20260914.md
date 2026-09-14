# Windows replacement verification — exact 9e5914

Run 34825668804 completed with failure. No source, workflow, fixture, or running session was changed during this review.

- Artifact SHA index: **54/54 verified**.
- Source receipt: clean **9e5914fca140e762b66ec4a7beb77cf2af450ca0**.
- Two installed receipts contain 2,866 files each. All 2,474 comparable Python files match the checked-out source receipt in each installation and the exact Git revision: 32 byte-exact Git blobs, 2,442 with Windows CRLF checkout normalization. No other differences. This verifies the downloaded receipts; installed binaries/wheels were not downloaded and independently rehashed.
- Native JUnit: **42 passed, zero skips**, 1.479 seconds.
- Product JUnit: **31 cases: 11 passed, one failed, 19 setup errors, zero skips**, 2,452.181 seconds. Summary and both JUnit files agree.

## Exact causal finding

All 20 Persona cases fail to reach their backup/rollback assertions because the synthetic seed calls `publish_persona_visual`. The first traceback ends at exact-revision `tldw_chatbook/Persona_Visual/publication.py:179`: `_posix_guards_available()` is false. Its definition at lines 1279–1285 explicitly requires `os.name == "posix"`. The test seed call is `Tests/Backup_Recovery/test_created_persona_subtree_rollback.py:19`.

The direct lifecycle test fails after 39.948 seconds. A separate shared seed fails after 40.142 seconds; its exception accounts for all 19 negative-case setup errors (nine graph cases and ten retirement cases). These are not 20 independent rollback failures and are not timeouts. Preserve the legitimate publisher guard. The smallest relevant correction is a portable test-only seed representing already-existing valid Persona artwork/repository data, followed by the unchanged real owner validation, capture, replacement, reopen, and rollback assertions. Native Windows acceptance of that corrected seed remains required.

The ordinary explicit replacement passes (688.030 seconds); default replacement and later rollback pass (1,562.124 seconds). All nine Evals retention cases pass (12.563–13.164 seconds each).

## Case accounting

- PASSED 688.030s `Tests.Backup_Recovery.test_f9_replacement_workflow::test_full_f9_replacement_after_explicit_safety_and_credential_review`
- PASSED 1562.124s `Tests.Backup_Recovery.test_default_service_replacement::test_default_profile_service_replacement_and_later_rollback`
- FAILED 39.948s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_nested_persona_pack_survives_reopen_and_later_rollback`
- ERROR 40.142s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_mapping_refuses_unproved_graph[source-parent]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_mapping_refuses_unproved_graph[source-root]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_mapping_refuses_unproved_graph[source-core]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_mapping_refuses_unproved_graph[current-parent]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_mapping_refuses_unproved_graph[current-root]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_mapping_refuses_unproved_graph[current-core]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_mapping_refuses_unproved_graph[current-config]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_mapping_refuses_unproved_graph[duplicate-id]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_mapping_refuses_unproved_graph[foreign-owner]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_retirement_keeps_ancestor_exclusion[mapping-alone]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_retirement_keeps_ancestor_exclusion[ancestor-container]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_retirement_keeps_ancestor_exclusion[ancestor-retire]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_retirement_keeps_ancestor_exclusion[ancestor-not-restored]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_retirement_keeps_ancestor_exclusion[restore-inside]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_retirement_keeps_ancestor_exclusion[candidate-inode]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_retirement_keeps_ancestor_exclusion[stale-generation]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_retirement_keeps_ancestor_exclusion[ancestor-file]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_retirement_keeps_ancestor_exclusion[ancestor-missing]`
- ERROR 0.001s `Tests.Backup_Recovery.test_created_persona_subtree_rollback::test_native_persona_retirement_keeps_ancestor_exclusion[ancestor-replaced]`
- PASSED 12.857s `Tests.Backup_Recovery.test_eval_rollback_retention::test_known_original_without_rollback_coverage_refuses[False]`
- PASSED 12.563s `Tests.Backup_Recovery.test_eval_rollback_retention::test_preserved_canonical_eval_needs_no_retained_source_authority[canonical-preserved]`
- PASSED 13.164s `Tests.Backup_Recovery.test_eval_rollback_retention::test_original_eval_retained_and_captured_without_incoming_owner_or_candidate`
- PASSED 13.098s `Tests.Backup_Recovery.test_eval_rollback_retention::test_rolled_back_eval_requires_current_provenance_and_exact_path[plan]`
- PASSED 13.050s `Tests.Backup_Recovery.test_eval_rollback_retention::test_rolled_back_eval_requires_current_provenance_and_exact_path[terminal]`
- PASSED 12.810s `Tests.Backup_Recovery.test_eval_rollback_retention::test_rolled_back_eval_requires_current_provenance_and_exact_path[association]`
- PASSED 13.013s `Tests.Backup_Recovery.test_eval_rollback_retention::test_rolled_back_eval_requires_current_provenance_and_exact_path[missing]`
- PASSED 13.070s `Tests.Backup_Recovery.test_eval_rollback_retention::test_rolled_back_eval_requires_current_provenance_and_exact_path[alias]`
- PASSED 13.145s `Tests.Backup_Recovery.test_eval_rollback_retention::test_rolled_back_eval_requires_current_provenance_and_exact_path[unrelated]`

## Preserved evidence

`/private/tmp/uat-windows-9e5914-replacement/verification.json` records all checksum and count checks; `git-blob-verification.json` records exact-revision source comparison; `case-accounting.json` lists all 31 outcomes. The downloaded artifact retains original sanitized logs, JUnit, and receipts. Full failure bodies are separately preserved under `junit-failures/`; the first direct trace is `db94cce0b1ea7c71.txt`, and the first shared setup trace is `6b5d77b5526523fc.txt`.

This result belongs solely to 9e5914. It does not verify subsequent Console retry or ACL-fixture corrections. No tests were dispatched, canceled, skipped, or given longer deadlines by this review.
