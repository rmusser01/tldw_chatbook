---
id: TASK-19936
title: 'change_review debug line interpolates the raw [console] workspace_root path'
status: In Progress
assignee: []
created_date: '2026-08-22 10:30'
updated_date: '2026-08-29 00:40'
labels:
  - privacy
  - diagnostics
  - change-review
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`UI/Screens/change_review_screen.py` logs the raw, unredacted `[console]
workspace_root` configuration value when the untracked-writable disclosure
cannot normalise it:

    logger.debug(
        f"change_review: skipping untracked-writable disclosure "
        f"for {raw!r}: {exc}"
    )

`raw` is a user-configured filesystem path, so this writes the location of
the user's project tree — commonly including their home directory and
therefore their account name — into a diagnostic. That is the class of
diagnostic TASK-15103 removed 49 instances of, and the same class the
persistent-diagnostic inventory exists to catch on the way in.

Surfaced during the TASK-19572 inventory review while repairing unrelated
test failures. Left unfixed deliberately: it arrived with PR #1941 and
belongs to that change's author, not to the test-repair branch that found
it. Severity is low — `debug` level, and the path is the user's own
workspace rather than third-party content — which is why this is filed
rather than hotfixed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The diagnostic no longer interpolates the raw path value; it reports
      the failure with metadata only (e.g. `type(exc).__name__`), matching
      the convention the neighbouring repaired call sites use.
- [x] #2 A reviewer can still tell WHICH disclosure was skipped and why, so the
      line keeps its debugging value rather than being deleted outright.
- [x] #3 `Docs/security/production-diagnostic-inventory.json` is regenerated
      and the row for `change_review_screen.py` reflects the new text.
- [x] #4 `Tests/Architecture/test_persistent_diagnostic_inventory.py` passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: `backlog/decisions/029-local-private-data-boundary.md`

Reason: TASK-19936 is folded into TASK-19864 and applies ADR-029's existing producer-side diagnostic privacy boundary without changing storage, ownership, or user-facing behavior.

1. Replace the raw configured workspace-root and exception diagnostic with a stable root fingerprint, disclosure operation metadata, and exception type.
2. Preserve the validation-failure early return and the existing untracked-writable banner behavior.
3. Regenerate the schema-3 diagnostic inventory and dependent fixture, then verify the owner and inventory architecture gates.
4. Mutation-check the original raw `raw!r` seam and retain TASK-19864's final
   verification evidence while this task remains In Progress pending explicit user
   acceptance of the process deviation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Folded into TASK-19864 final implementation and verification.

- **Process Deviation:** TASK-19936's own In Progress status and Implementation Plan were added during closeout, after its code change. TASK-19864's previously committed and approved detailed plan covered the Change Review seam before implementation, but that does not retroactively cure the TASK-19936 timing mistake. TASK-19936 remains In Progress, and final Done status awaits explicit user acceptance of this administrative deviation.
- The untracked-writable validation diagnostic now records `operation=untracked-writable`, a stable `root_sha256` fingerprint, and `exception_type`; it emits neither the raw configured workspace root nor the raw exception body.
- The reviewer can still identify the skipped disclosure and failure class. The validation-failure early return remains unchanged, and the existing user-visible warning banner is still produced only for a valid untracked writable root.
- [Schema-3 inventory](../../Docs/security/production-diagnostic-inventory.json) regeneration contains no governed-owner path candidate for `change_review_screen.py`; the no-write checker reports exact artifact synchronization.
- [Owner architecture coverage](../../Tests/Architecture/test_diagnostic_path_privacy.py) and [inventory architecture coverage](../../Tests/Architecture/test_persistent_diagnostic_inventory.py) passed within the expanded 621-test focused suite, including the Change Review push and Console context modal modules omitted from the earlier recorded command. The original `raw!r` seam was mutation-restored and made `test_untracked_writable_validation_failure_logs_safe_metadata` fail while preserving its empty-banner early-return assertion; restoration returned the node green.
- No full suite was run. No new ADR was required; this applies [ADR-029](../decisions/029-local-private-data-boundary.md). No new lessons entry was warranted.
<!-- SECTION:NOTES:END -->
