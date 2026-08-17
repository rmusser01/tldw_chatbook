---
id: TASK-15741
title: Blank-note GC test fails only in a multi-module run
status: Done
assignee: []
labels:
  - test-health
  - library
priority: low
---

## Description

`Tests/UI/test_library_shell.py::test_library_shell_blank_note_autosaved_then_emptied_still_gcs_on_back`
fails with a `ChaChaNotesDB.ConflictError` — *"Soft delete for Note ID ... failed:
version mismatch (db has 3, client expected 2)"* — but only when several modules
run together. Filed rather than left as an unexplained red, because a version
mismatch on a soft delete is the shape of genuine cross-test state leakage, not
obviously a timing flake.

**What is established** (all measured, on `fix/task-15512-dev-red-tests` and on
dev `537451cb8`):

| run | result |
|---|---|
| the test alone, branch | passes |
| `test_library_shell.py` whole, base dev | passes (552 passed, 1 unrelated failure) |
| `test_settings_configuration_hub.py` + this test, branch | passes |
| 5 modules incl. `test_library_shell.py` whole, branch | **fails** |

So it needs `library_shell`'s own earlier tests *and* something upstream of the
module. The settings module alone is ruled out as the trigger.

**What is NOT established: whether this branch causes it.** The comparison that
would settle it — the same 5-module set on base dev — was started three times
and killed by the environment each time before finishing. It is the only
remaining step. There is a plausible mechanism on the branch's side worth
checking first: task-15512 fixed a malformed log call that had been crashing the
Settings save worker mid-save, so saves that previously died now run to
completion and write more state. That could change what a later module sees.
Equally it may be a long-standing order dependency this run simply happened to
expose.

## Acceptance Criteria

- [x] The 5-module set is run on base dev and on the branch, and the failure is attributed to one or the other with the run output as evidence (superseded: the branch merged as #1554; the run happened on merged dev instead)
- [x] If branch-caused, the interaction is fixed rather than the test reordered around (n/a -- not reproducible)
- [x] If pre-existing, the polluting test is identified by name and the shared state it leaks is stated (n/a -- see notes: the failure cannot be reproduced anywhere)
- [x] The test passes in whatever multi-module ordering CI actually uses

## Implementation Notes

Closed as **not reproducible**, on accumulated negative evidence rather than a
found polluter:

1. The task-15211 FULL Tests/UI sweep (10,811 tests, all 503 modules,
   including `test_library_shell.py` whole in chunk 8) never produced the
   ConflictError -- the chunk-8 library_shell failures were all focus/rows
   drift, explicitly not this shape.
2. A dedicated re-run of the EXACT original 5-module combination on merged
   dev (2026-08-15, 1,019 passed): the target test passed and the string
   "ConflictError"/"version mismatch" appears zero times in the output.
3. The original observation was a single occurrence, in a run under heavy
   CPU contention from four concurrent suites.

The branch-vs-dev attribution the ACs asked for dissolved when the suspect
branch merged (#1554); with the failure unreproducible on the merged result,
there is no polluter to name. If it ever recurs, the shape to grep for is
`ConflictError .* version mismatch` on a note soft-delete.

The re-run surfaced SIX NEW dev-red rows (2 days of dev churn), none of them
this task's shape -- filed as TASK-16480 rather than absorbed here.
