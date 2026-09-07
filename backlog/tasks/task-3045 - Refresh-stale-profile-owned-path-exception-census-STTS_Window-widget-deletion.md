---
id: TASK-3045
title: >-
  Refresh stale profile-owned-path exception census (STTS_Window widget
  deletion)
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 13:09'
updated_date: '2026-08-07 13:12'
labels:
  - testing
  - baseline
  - tts
dependencies: []
references:
  - backlog/decisions/040-profile-owned-state-and-shared-asset-paths.md
  - >-
    backlog/tasks/task-2951 -
    task-1266-AC4-is-false-on-dev-TTSPlaygroundWidget-was-restored-and-never-re-deleted.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-2951's TTSPlaygroundWidget deletion (PR #1405) removed the last executable occurrences the ADR-040 profile-owned-path sentinel had approved for tldw_chatbook/UI/STTS_Window.py, leaving two stale exception rules and a red architecture gate on dev. Refresh the frozen census to match current source without changing runtime behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every stale exception rule caused by the TTSPlaygroundWidget deletion is removed from the approved census.
- [x] #2 Any occurrence newly missing an exception rule (e.g. ported to SpeechPlaygroundPane) is explicitly reviewed and classified under ADR-040, not silently blessed.
- [x] #3 The production profile-owned-path inventory exactly matches current dev source.
- [x] #4 The focused profile-owned-path architecture tests pass.
- [x] #5 No production runtime behavior changes are introduced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/040-profile-owned-state-and-shared-asset-paths.md
Reason: This refresh applies the existing ADR-040 profile-owned-path classification and updates the frozen census to match code that already deleted the sole classified owner; it makes no new path-ownership decision.

1. Run the failing test/checker to get the exact reconciliation diff (reconcile_inventory output) between current source and the frozen APPROVED_EXCEPTIONS census.
2. For each stale rule, confirm the referenced function/class is actually gone (not renamed/moved) via git log/grep; for any unapproved-occurrence or count-mismatch problem, read the actual call site and classify it under ADR-040's disposition table before adding a rule.
3. Edit only scripts/check_profile_owned_path_inventory.py's APPROVED_EXCEPTIONS tuple to match: remove confirmed-dead rules, add only rules for deltas that are genuinely explained by the deletion/porting -- anything unexplained is reported as a finding, not blessed.
4. Re-run the focused test and full Tests/Architecture/ suite; confirm the diagnostic-inventory gate from the prior commit is still green too.
5. Record the per-delta accounting, check all acceptance criteria, and close the task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Second reviewed refresh on the same branch, same method: diffed the checker's deterministic reconcile output (reconcile_inventory) against the frozen APPROVED_EXCEPTIONS census embedded in scripts/check_profile_owned_path_inventory.py. Result: exactly 2 problems, both "stale exception rule" for tldw_chatbook/UI/STTS_Window.py (`function:_chatterbox_profile_choices` and `function:_higgs_profile_choices`, both `join:.config/tldw_cli`) -- zero "unapproved occurrence" and zero count-mismatch problems, so there was nothing new to classify.

Confirmed root cause: task-2951's TTSPlaygroundWidget deletion (PR #1405, commit f560217fb) removed the class and both functions entirely from STTS_Window.py (grep confirms zero remaining references) -- not a rename or move. The functionally-equivalent code already lives in tldw_chatbook/UI/Speech/speech_catalog_mixin.py's SpeechCatalogMixin (composed by SpeechPlaygroundPane, the widget stts_events.py now exclusively mounts), which already carries its OWN separate approved exception rules for the identical context names -- added in an earlier commit (c908ca896, predating this drift window) when the mixin/pane was built, so no new rule was needed for the port; only the two dead STTS_Window.py rules needed removing. This is a straight bless of a deletion, not a porting classification decision.

Fix: removed the two stale ExceptionRule entries for tldw_chatbook/UI/STTS_Window.py from APPROVED_EXCEPTIONS. No other source edits.

Verification: checker exits 0. Tests/Architecture/ full: 30/30 passed (was 28 passed/2 failed), including both this task's gate and TASK-3035's diagnostic-inventory gate together (18 passed when run jointly). ruff check + format --check clean on the touched script. Repo-wide `pytest --collect-only -q`: 31873 tests collected, 0 errors (same count as before). Diff hygiene: exactly two changes -- the script and this task file.
<!-- SECTION:NOTES:END -->
