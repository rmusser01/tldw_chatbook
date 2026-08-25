---
id: TASK-21531
title: >-
  Four dev reds from test doubles that production outgrew - Notes library and
  lasting-sync flow
status: Done
assignee: []
created_date: '2026-08-23'
labels:
  - testing
  - dev-red
  - stale-doubles
priority: medium
---

## Description

Four tests are red on pristine dev because their doubles were never updated when the
production signatures they stand in for changed. They are not testing anything real in this
state, and they sit in the red baseline every branch inherits. Same class as the test-suite
programme's "stale test doubles production has outgrown" finding.

## Acceptance Criteria

- [x] `Tests/Notes/test_notes_library_unit.py::TestNotesInteropService::test_get_db_new_instance` and `::TestLibraryNotesInteropDelegates::test_get_db_new_instance` pass, with the double updated to the real constructor signature rather than the assertion relaxed
- [x] Both `Tests/UI/test_library_notes_lasting_sync_flow.py::test_activation_result_routes_to_truthful_receipt_or_root_recovery` parameterisations pass, with the `activate_root` double matching the production signature
- [x] Each repaired double is verified to still fail when the behaviour it guards is broken -- a double updated only far enough to stop raising is not a fix
- [x] A brief note records whether either double had drifted far enough that it was passing vacuously before the signature change

## Evidence (verified first-hand on dev 022b67fc7, 2026-08-23)

```
pytest Tests/Notes/test_notes_library_unit.py -k test_get_db_new_instance
  -> 2 failed, 62 deselected
pytest Tests/UI/test_library_notes_lasting_sync_flow.py -k activation_result_routes
  -> 2 failed, 5 deselected  (TypeError at test_library_notes_lasting_sync_flow.py:164)
```

Mechanisms: the Notes doubles' mock `CharactersRAGDB` signature was outgrown by TASK-19900's
`console_library_migration_seed` argument; the sync-flow double's `activate_root` is missing a
now-required positional argument.

Surfaced by the TASK-21129 implementer while A/B-baselining its own reds against dev.

## Implementation Plan

1. Re-verify all four reds on the working base (`a71e62e4b`) before changing anything.
2. Read the real production signatures, not the filing's summary of them.
3. Repair each double so it tracks the real seam, and add the assertion that a
   drifted-but-silent version would still be missing.
4. Mutate the production behaviour each repaired test claims to guard and prove
   it goes red.
5. Record the vacuity findings.

## Implementation Notes

All four reds reproduced on `a71e62e4b`. One of the two filed mechanisms was
wrong, which is itself the finding:

**Notes doubles (as filed).** `_get_db` gained
`console_library_migration_seed=load_console_library_migration_seed()` in
TASK-19900 and the expected-call literal never followed. The patch already used
`spec=True`, which is signature-aware for *assertion matching* but applies no
call-time check, so the drift surfaced only as a mismatched expected call.
Repair: `autospec=True` (call-time signature enforcement) plus a patched seed
loader returning a **non-default** sentinel
(`ConsoleLibraryMigrationSeed(auto_retrieve_on_send=True)`), so the assertion
proves the *loaded* seed is forwarded rather than that some seed object was
passed.

**Sync-flow test (filing was wrong).** The `activate_root` *double* was fine —
`_Runtime.activate_root(self, _root_id, _authorization)` already had both
parameters. The `TypeError` came from the test's own call to **production**:
`LibraryNotesSyncController.activate_root(root_id, observation_token)` gained a
required `observation_token`. Adding an argument alone would not have fixed it:
activation is now gated on a current, activation-typed migration review whose
plan and review both carry the runtime's observation token, so the call would
have short-circuited and left the phase at `review`. The test now drives
`check_migration("root-1")` first, asserts `review.activation is True`, then
activates with the runtime's own token.

**Vacuity findings.**
- Notes: *not* vacuous before the drift. `spec=True` supplies `_spec_signature`,
  so the pre-drift expected call was genuinely matched; it went red exactly when
  it should have.
- Sync flow: the `result1` (`failed` → `roots`) parameterisation **was**
  structurally vacuous on `accepted`. Proven: with
  `result = await self._runtime.activate_root(...)` replaced by a fabricated
  `NotesSyncControlResult(False, "failed", "review_changes")` and the new
  runtime-call assertion removed, that parameterisation still passed — the
  runtime was never called and the test could not tell. The new
  `assert runtime.activation_calls == [("root-1", OBSERVATION_TOKEN)]` closes it.

**Mutation results (each applied to production, then reverted).**

| Mutation | Result |
| --- | --- |
| Drop `console_library_migration_seed=` from `Notes_Library._get_db` | 2 failed |
| Hardcode `ConsoleLibraryMigrationSeed(auto_retrieve_on_send=False)` instead of loading it | 2 failed |
| Pass an unknown kwarg (autospec signature check) | 2 failed (`TypeError`) |
| `phase=... "roots" if recovery` → `"review"` | 1 failed (`result1`) |
| Receipt line → `"No changes were applied."` | 1 failed (`result0`) |
| Fabricate the activation result, never call the runtime | 2 failed (1 of which passed before the new assertion) |

**Counts.** `Tests/Notes/test_notes_library_unit.py` 62 passed / 2 failed → 64
passed. `Tests/UI/test_library_notes_lasting_sync_flow.py` 5 passed / 2 failed →
7 passed. With `Tests/UI/Library_Modules/test_library_notes_sync_controller.py`:
199 passed, 0 failed.

**Files.** `Tests/Notes/test_notes_library_unit.py`,
`Tests/UI/test_library_notes_lasting_sync_flow.py`. No production changes.
