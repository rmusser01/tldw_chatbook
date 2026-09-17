# Legacy config suite: exact HEAD baseline comparison

Read-only investigation of the two uncommitted config lock-order changes. No repository or fixture edits; no guard changes.

Result: no newly failing test was observed. The exact requested 68-case selection produces **52 failed / 16 passed** in both the current working tree (7.70s) and exact HEAD d3082ef98211e3ad42273ac2d2dcbbf43b6d93cc for both affected modules (7.51s). All 68 test node outcomes match, not merely the totals. This is baseline classification, not a claim the legacy suite is healthy.

Evidence and fidelity:
- Current: /private/tmp/uat-existing-config-order.log and /private/tmp/uat-existing-config-order.xml.
- Final baseline: /private/tmp/uat-existing-config-order-baseline-complete.log and .xml.
- Machine comparison: /private/tmp/uat-existing-config-order-comparison.json.
- Exact baseline files: /private/tmp/uat-config-order-baseline-1g5clrqh/tldw_chatbook/config.py and Backup_Recovery/config_participants.py. Both bytes were checked against git show of the explicit revision before execution.
- Module SHA-256: config.py 78f20102d319dfbff54b25fe6b48dabc84a19446939d6639b567041a5d969625; config_participants.py 796843ab6626c1a142f7262a40368ba62ab8608d2ded3a1b46fe198e928bd7c2.
- Temporary sitecustomize loader /private/tmp/uat-config-order-legacy-overlay/sitecustomize.py substitutes only those exact source bytes. Module names, original repository __file__, compilation filenames, package relationships, and normal remaining imports are preserved. It also reaches the test-launched child interpreters through their existing inherited PYTHONPATH. No product file is replaced on disk. Traceback source excerpts may display working-tree lines because their filenames intentionally remain the real paths; source execution receipts disambiguate this.
- Per-process receipts /private/tmp/uat-existing-config-order-baseline-process-*.json prove both HEAD modules executed in the pytest parent. The two first-failing child interpreters executed HEAD config and exited at startup scope refusal before importing config_participants. The tests then terminate their other child in their existing cleanup; no completion receipt is claimed for those terminated children.

Compact cause classification:
- 22 failures directly report raw_source_selection_changed. Tests/conftest.py sets a bootstrap selector during collection, then changes TLDW_CONFIG_PATH in its per-test fixture (around1060); raw snapshot and delete fixtures further replace that selection. raw_participants._participant_state (around127) rejects the discrepancy against the already installed source participant. Same refusal exists on HEAD.
- 1 failure is the existing test_runtime_snapshot_takes_rebuild_lock_before_file_lock TrackingLock double, which implements only __enter__/__exit__. HEAD's operation already calls the FILE lock's acquire(timeout=...), so the double fails with missing acquire before this change; current operation encounters the equivalent missing method on REBUILD first.
- 2 cross-process writer tests exit before readiness with recovery_scope_uncertain on both revisions, before the changed participant operation is imported. They do not reach the concurrency assertion.
- Remaining 27 are assertion outcomes (false mutation returns, mutation-phase/result mismatches, empty worker results, etc.), largely downstream of the same refusal path. Their exact individual pass/fail outcomes also match baseline; this investigation does not claim an independent root-cause proof for every derivative assertion.

The requested comparison provides no evidence of a new regression from the two pending module edits. Keep these baseline fixture failures distinct from the newly passing native lock-order regression; do not relax admission to make the inherited tests pass.
