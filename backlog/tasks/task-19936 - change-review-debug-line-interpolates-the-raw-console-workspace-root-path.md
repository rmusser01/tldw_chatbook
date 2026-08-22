---
id: TASK-19936
title: change_review debug line interpolates the raw [console] workspace_root path
status: To Do
assignee: []
created_date: '2026-08-22 10:30'
labels:
  - privacy
  - diagnostics
  - change-review
dependencies: []
priority: low
---

## Description

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

## Acceptance Criteria

- [ ] The diagnostic no longer interpolates the raw path value; it reports
      the failure with metadata only (e.g. `type(exc).__name__`), matching
      the convention the neighbouring repaired call sites use.
- [ ] A reviewer can still tell WHICH disclosure was skipped and why, so the
      line keeps its debugging value rather than being deleted outright.
- [ ] `Docs/security/production-diagnostic-inventory.json` is regenerated
      and the row for `change_review_screen.py` reflects the new text.
- [ ] `Tests/Architecture/test_persistent_diagnostic_inventory.py` passes.
