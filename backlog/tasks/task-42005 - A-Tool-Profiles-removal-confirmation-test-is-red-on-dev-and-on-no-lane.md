---
id: TASK-42005
title: "A Tool Profiles removal-confirmation test is red on dev and on no lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-19 19:39
updated_date: 2026-11-20 17:23
labels:
- follow-up
- tests
- settings
priority: low
references:
- Tests/UI/test_settings_tool_profiles.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-41526 (round 17). `test_settings_tool_profiles.py::test_removal_confirmation_does_not_interpolate_the_profile_id` fails with `assert '' == 'Remove cancelled'` (~1165). TASK-36387 recorded it as red at dev but owns no fix, and the file is on no PR lane. The product copy it quotes exists (`settings_screen.py` ~6466-6477); the test patches `host.push_screen_wait`, so this looks like test drift.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The cause is named (test drift or product defect) and fixed
- [x] #2 The file runs on a PR lane
<!-- AC:END -->

## Implementation Plan

1. Read the failure: the test's `choose` asserts inside the removal worker, whose `exit_on_error=False` swallows it, so only the empty result shows.
2. Trace the confirmation message's history (88ae92a234 moved escaping into ConfirmationDialog) to name the cause.
3. Fix the drifted assertion so it pins today's contract (the caller passes the id verbatim) and fails visibly outside the worker; show it red under a temporary pre-escape in the source.
4. Append the file to the UI census tail, raise MINIMUM_FILES, and record its serial time with the UI lane's flags.

## Implementation Notes

Cause: test drift. 88ae92a234 (TASK-32802.2) made `ConfirmationDialog` render its title and message literally and removed the callers' pre-escaping, including `escape_markup(display_profile_id)` in Settings' removal flow. The test still asserted the escaped contract (`profile_id not in screen.message`) inside the mocked `push_screen_wait`; that assertion failed in the removal worker, which has `exit_on_error=False`, so the worker died before setting a result and the only visible failure was `assert '' == 'Remove cancelled'`. No product change.

Fix: the test is renamed `test_removal_confirmation_names_the_profile_id_verbatim` (was `test_removal_confirmation_does_not_interpolate_the_profile_id`, which named the old escaped contract). The mocked dialog only records what it was shown; after the run the test asserts one `ConfirmationDialog` whose message names the id verbatim (`Remove [bold]local-profile[/bold]?`) and the cancelled result. Literal rendering itself is the dialog's contract, pinned by `Tests/Widgets/test_confirmation_dialog_literal_text.py`. Review rev1: the two sibling removal tests (`test_removal_worker_confirms_and_uses_the_rendered_revision`, `test_removal_worker_rechecks_cancellation_after_confirmation`) asserted `isinstance(screen, ConfirmationDialog)` inside the same kind of mock; they now record the dialog and assert its type after the run.

Red/green: red at base (`assert '' == 'Remove cancelled'`). With `escape_markup(...)` temporarily restored in `settings_screen.py` (run through pytest, then reverted) the new assertion fails visibly: `assert 'Remove [bold]local-profile[/bold]?' in 'Remove \\[bold]local-profile\\[/bold]? ...'`. Green after. Rev1: with the removal flow temporarily pushing a `dict(...)` instead of the dialog, all three removal tests fail visibly on `assert [<class 'dict'>] == [<class '...ConfirmationDialog'>]`; reverted, 3 passed.

Lane: appended whole at the UI census tail (`MINIMUM_FILES` 2439 -> 2440). Charge: 30 tests (all `@private_profile_test`), 30 passed in 190.5 s serially with the lane's flags (`-p no:xdist --timeout=180 --durations=0 --durations-min=0 --tb=short`), load 20-27.

Files: `Tests/UI/test_settings_tool_profiles.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.

