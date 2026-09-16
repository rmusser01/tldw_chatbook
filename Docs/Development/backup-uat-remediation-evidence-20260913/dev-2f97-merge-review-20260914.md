# Final dev2f97 integration review

APPROVED for the scoped service sweep gate and repository-probe maintenance handling. No actionable finding. No repository edits or full-app tests performed.

Reviewed the merge in progress from ours `15c1541f77780c7897a11ba136d738bad679e9b4` and incoming `2f97a42c9aa9cc737cf304f927e6d861824e1b3a`.

## Product correctness

The File Notes constructor retains our maintenance-source registration, inspection flag, and recovery methods while adding upstream's one-service sweep state. `scan` performs the destructive hidden tombstone sweep only inside the already-authorized replica refresh branch. Otherwise it returns the original activation warning without changing sweep state or retained bytes. Existing reconcile fallback uses that same inspection-only scan when execution/pairing permission is absent. Admitted reconcile and the incoming hidden-file/genuinely-deleted-file distinction remain unchanged. No cache, schema, native transaction, ownership or permission bypass is added.

The new repository probe handles only the exact built-in RuntimeError with the sole `file_notes_maintenance_paused` argument. It lets a later ordinary status refresh retry by resetting the completed-probe marker, only if that marker still belongs to this binding. It does not publish Git confirmation, touch a newer binding, enqueue another worker while paused, or pretend the refused query was admitted. Unrelated errors and cancellation re-raise identically. Existing discover native/maintenance checks and admitted child lifetime remain unchanged.

AST comparison against both parents finds only the expected service constructor/scan and workspace constructor/probe as methods synthesized by this merge. The workspace constructor is the union of the prior runtime state and incoming repository-probe/editor state. Other methods match a parent.

## Verification

Root's final 33-case suite is terminal: 33 passed in 44.39s, `/private/tmp/uat-dev-2f97-final-lifetimes.log`.

After that completion, independent focused command ran the two new parameterized tests (eight cases), using an explicit private basetemp:

```
python -m pytest \
  Tests/Backup_Recovery/test_file_notes_maintenance.py::test_repository_probe_preserves_maintenance_and_error_lifetimes \
  Tests/Backup_Recovery/test_notes_recovery_review.py::test_hidden_replica_tombstone_requires_fresh_pairing_before_sweep \
  -q --basetemp=/private/tmp/uat-dev-2f97-review-fixtures
```

**8 passed in 9.28s**, exit 0. Log: `/private/tmp/uat-dev-2f97-review-independent.log`. Existing Requests dependency warning retained.

The tombstone cases use the real isolated restore, activation store and exact fresh pairing review. Both scan/reconcile preserve retained bytes with either inactive ownership or the insufficient owner-flag-only approval; explicit fresh pairing then permits the sweep and leaves disk bytes unchanged. Unrelated owners remain inactive.

The probe cases invoke the actual workspace method with the real session owner and Git service in fixed-profile children. They prove pause refusal, cleared retry marker, native repository confirmation after resume, preserved newer binding, exact unrelated exception identity, and cancellation propagation. They do not mount a Textual worker or assert a complete UI lifecycle; the scope is the corrected callback boundary and real service admission. No product bypass is used to obtain a pass.

Preserved substantive RED: tombstones 4 failed in 7.26s; native probe 2 failed / 2 passed in 3.85s. The earlier unqualified shared-profile setup failure is not counted as product RED.

## Source receipt

Exact SHA-256 values are in `/private/tmp/uat-dev-2f97-final-review-hashes.json`:

- File Notes service: `2fb8e3bc925e80ac506012ec0556734a45c88de0fa008928d538389195155b0e`
- Workspace: `4f4c30e584e3da45c9bb48bfd489fd188522dbc6b5eb489a23d22234f22ba570`
- Maintenance tests: `adf2c874d470d4c65b9d81136a5dfdec115a25c9d30643470f0f44d1972764a8`
- Recovery review tests: `d1c73dfa5fcc4263e58f6cb57c47e544a84eeb890f3a309f0d1b21ab2de2513e`

This approves the measured merge-integration scope. It does not claim native Windows acceptance, complete upstream UI coverage, or replace root's static and installed-app verification.

Final receipt refresh: the sole subsequent maintenance-test change wraps the new LibraryFileNotesWorkspace import. Reversing that exact formatting reproduces the reviewed 728c0625 SHA; ASTs are identical. No test rerun was needed.
