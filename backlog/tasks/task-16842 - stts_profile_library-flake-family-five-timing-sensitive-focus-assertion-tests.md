---
id: TASK-16842
title: 'stts_profile_library flake family: five timing-sensitive focus-assertion tests'
status: To Do
assignee: []
created_date: '2026-08-16'
labels:
  - test-health
  - tts
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_stts_profile_library.py` carries a pre-existing flake family that two
independent reviews hit and characterized (task-15772 review round 2, PR #1691; task-15771
review F4, PR #1699). It is broader than any single pair of tests: across the 15772
reviewer's three full-file runs, **five distinct tests** flaked —

- `test_reference_export_defaults_sanitized_and_bundle_requires_ack`
- `test_windows_clone_export_keeps_sanitized_default_and_disables_bundle`
- `test_delete_shows_advisory_count_but_repository_conflict_is_final`
- `test_unavailable_profile_disables_playground_action_with_clear_recovery`
- `test_import_warns_before_picker_and_stale_successor_requires_reconfirm`

— and the first **reproduced standalone** (single test, own process):
`AssertionError: assert (None is not None)` on `app.focused.id == "bundle-warning-ack"`
after `_wait_until` confirmed the button was *mounted*. So the root cause looks like each
test's own internal focus-settle race (mounted ≠ focused yet), not cross-test pollution.
The 15771 review saw the family degrade under machine load (3 failed normally, 14 failed
in a run at 2x wall-clock). At dev `ee741cf10` the file has had no stabilization commit
since (last touched by 15772's own fix), and one standalone re-run of the first test
passed — consistent with intermittency, not with a fix having landed.

Root-cause the focus-settle pattern (likely one shared helper/idiom around
export/bundle-ack focus) and make the family deterministic — condition-polls on the
actual focus state, not mounted-state proxies or fixed sleeps (the repo's GGUF-settle
lessons apply).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The mechanism of the focus-settle race is identified and stated (not just retried around)
- [ ] #2 All five named tests pass repeatedly under load (e.g. 10 consecutive full-file runs, at least a few under parallel CPU load), with the run evidence recorded
- [ ] #3 No fixed-duration sleep is introduced as the fix
<!-- AC:END -->
