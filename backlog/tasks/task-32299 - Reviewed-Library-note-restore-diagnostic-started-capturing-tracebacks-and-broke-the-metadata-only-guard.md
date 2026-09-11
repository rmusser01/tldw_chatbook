---
id: TASK-32299
title: >-
  Reviewed Library note restore diagnostic started capturing tracebacks and
  broke the metadata-only guard
status: Done
assignee: []
created_date: '2026-09-10 20:34'
updated_date: '2026-09-10 20:34'
labels:
  - library
  - notes
  - tests
  - security
  - test-health
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Architecture/test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only`
is red on dev (reproduced at `6a79651ee8` and at `98704acc28`) with:

    tldw_chatbook/UI/Library_Modules/library_notes_controller.py:
    'Failed to restore a Library note' captures exception or stack details

The diagnostic sits in that test's `REVIEWED_METADATA_ONLY_DIAGNOSTICS`
registry: a diagnostic whose shape has been through security review may later
change its *metadata* (message wording, fields, level) but may never start
capturing exceptions or stack details, because those reach a persistent log
file and a rendered frame can carry the very user data the review cleared it
of. PR #2553 (task-32144, commit `593961cb9c`) answered a Qodo "lost
tracebacks" finding by turning the call into
`logger.opt(exception=True).warning(...)`, which is exactly the change the
registry forbids. The peer PR #2567 (task-32199) had just re-homed this row
from `Library_Window.py` to the controller after the wave-8 notes
decomposition, so the row itself is current — only the call's shape drifted.

The privacy stake is not theoretical for this call in particular: the frame it
logs from holds the restored note record and the note id, and the repo already
treats a rendered traceback in the log file as a leak of frame locals
(`app.py` model-catalog loader: "No traceback: the log file sink runs with
diagnose=True, which would dump frame locals ... into the log";
`Logging_Config.py` task-2119 comment).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `test_reviewed_diagnostic_changes_are_metadata_only` passes, and the
  whole of `Tests/Architecture/test_persistent_diagnostic_inventory.py` is
  green
- [x] #2 A failed Library note restore is still diagnosable from the log: the
  warning names the exception type
- [x] #3 No traceback or frame locals from the restore failure reach the log
  sinks
- [x] #4 `scripts/check_persistent_diagnostic_inventory.py` exits 0 with the
  re-pinned manifest, and the drift was reviewed statement-by-statement before
  re-pinning
- [x] #5 `Tests/UI/test_library_notes_riders_trash.py` stays green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the failure on a worktree at `origin/dev` and read the contract
   the test enforces (registry + field extraction + exception-capture check).
2. Choose between the two sanctioned shapes: keep the row and make the call
   metadata-only again, or drop the row the way `canvas sync failed` was
   dropped in `51533602c4`.
3. Apply the smaller one, update the registry's field tuple, re-pin the
   diagnostic inventory after reading the statement diff.
4. Run the inventory module in full, the checker, and the trash rider tests;
   separate any pre-existing dev reds from anything this change caused.
<!-- SECTION:PLAN:END -->

## Renumbering provenance

Drafted as task-32296, then task-32297; both ids were taken by peer
sessions filing concurrently (32296 a chat_screen ScreenStackError guard,
32297 `fix/console-env-poller-teardown-guard`, created 2026-09-10 20:24 and
already pushed; 32298 is claimed by an unpushed Library reader-suite task).
Older arrival keeps the id, so this task took the next free id, 32299,
before its first push. No id but 32299 ever reached a remote from here.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Kept the reviewed row and made the call metadata-only again, rather than
following the `canvas sync failed` precedent of leaving the registry. That
precedent was justified by "not a persistent sink"; this call *is* one of the
Library note paths whose frame holds the note record, and the registry row for
it was deliberately re-homed a day earlier by task-32199 — dropping it would
have re-litigated that review instead of honouring it.

The call is now the house shape already used twice in this same file
(`Notes import check failed; error_type={}`):

    except Exception as exc:  # noqa: BLE001 - degrade to a notice
        logger.warning(
            "Failed to restore a Library note; error_type={}",
            type(exc).__name__,
        )

That keeps the Qodo finding's substance — a restore failure is no longer a
bare, causeless line — without the traceback: the exception *type* is
metadata, which the registry explicitly allows (the field tuple moves from
`()` to `("type(exc).__name__",)`, which is what "reviewed drift may change
metadata" means), while the rendered frame is what it forbids.

The second call PR #2553 added, `Failed to read the Library notes trash`, was
checked against the same test and left alone: it is not in
`REVIEWED_METADATA_ONLY_DIAGNOSTICS`, so no reviewed shape is being changed,
and the inventory pin already covered it (checker exit 0 on dev before this
change). It is worth a separate look by whoever owns the sink-privacy sweep —
its frame holds a page of soft-deleted notes — but that is a new review, not
this red.

Verification: `Tests/Architecture/test_persistent_diagnostic_inventory.py`
69 passed, 1 skipped (was 1 failed at the same tree);
`scripts/check_persistent_diagnostic_inventory.py` exit 0 after `--write`
(one owner row changed, 10 -> 10 calls, statement diff read: one warning
removed, one added);
`Tests/UI/test_library_notes_riders_trash.py` 15 passed.
`Tests/UI/test_library_notes_reader.py` is 13 failed / 21 passed both with and
without this change (baseline worktree at `origin/dev` `684cf77e01`) — a
pre-existing dev red, untouched here.

Modified files: `tldw_chatbook/UI/Library_Modules/library_notes_controller.py`,
`Tests/Architecture/test_persistent_diagnostic_inventory.py`,
`Docs/security/production-diagnostic-inventory.json`.
<!-- SECTION:NOTES:END -->
