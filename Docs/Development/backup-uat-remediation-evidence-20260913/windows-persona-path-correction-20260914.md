# Windows Persona safety-copy readback fixture

TASK-32562; sole test-only edit after handoff commit 18f7321c0a. Frozen file: `Tests/Backup_Recovery/test_created_persona_subtree_rollback.py`, SHA256 `55d0294bc5cf8101cb079075e2c9338741cc91b91161fd59b8c61b552806f140`.

Verified Windows d308 evidence records `_LATER` line 67 `len(matches)==1` failure after committed rollback, verified recovery-copy entry, and archive validation. `_REOPEN` previously serialized `str(path.relative_to(pack))`; nested paths use backslashes on Windows, while manifest `relative_path` always uses slash separators. The matching assertion therefore could not locate the correct safety-copy payload.

The only edit serializes that saved relative key with `.as_posix()`. Payload bytes/hashes, owner matching, rollback execution, ordinary reopen, subsequent capture, all 20 test bodies, other embedded scripts, deadlines, and every product file are unchanged. No new helper or mirror test was added.

Minimal validation `/private/tmp/uat-persona-relative-path-validation.json` proves the exact sole replacement, embedded script compilation, unchanged test bodies/other scripts, Windows original false match → corrected true match, and unchanged POSIX match/native path equivalence. Ruff is clean; Bandit baseline/current unchanged; diff check clean. Reports `/private/tmp/uat-persona-relative-path-{ruff,bandit,bandit-baseline}.json`.

This is not a native Windows pass claim. The existing Windows replacement selection must rerun the genuine lifecycle. No commit/push performed.
