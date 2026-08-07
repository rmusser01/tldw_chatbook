---
id: TASK-2710
title: Audit remaining super().on_mount() calls over BaseAppScreen
status: Done
assignee:
  - '@claude'
created_date: '2026-08-06 20:35'
updated_date: '2026-08-07 04:04'
labels:
  - ui
  - tech-debt
dependencies:
  - task-2610
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-2610's root cause: Textual dispatches EVERY `on_mount` along the MRO for one Mount
event, so a subclass handler calling `super().on_mount()` runs the parent handler twice.
That crashed Lab ▸ Speech because `LabFrameScreen.on_mount` mounts widgets. Roughly twenty
other screens/widgets still call `super().on_mount()` over `BaseAppScreen` (grep
`super().on_mount()` under `tldw_chatbook/`); today that only duplicates the base's log
line, so they are harmless-by-luck — but the moment anyone adds real work to
`BaseAppScreen.on_mount` (or to any intermediate class), every remaining call site
detonates the same way. `BaseAppScreen.on_mount`'s docstring now states the contract;
this task removes the latent calls so the contract is enforced by absence, not by
discipline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No `super().on_mount()` call remains in tldw_chatbook code whose parent chain includes a class that defines `on_mount` (Third_Party/ excluded)
- [x] #2 Each removal is verified not to change behavior (the parent handler still runs exactly once via the dispatcher)
- [x] #3 A guard (test or lint rule) prevents new `super().on_mount()` calls from being introduced over BaseAppScreen
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Confirm Textual 8.2.7 dispatch semantics by reading message_pump.py::_get_dispatch_methods: the dispatcher walks self.__class__.__mro__ (derived-class-first) and separately yields+awaits each class's OWN on_mount if defined in that class's __dict__ -- so a parent's on_mount ALWAYS runs once via the dispatcher regardless of whether a subclass also calls super().on_mount() inline; the only question per site is ORDERING (does the subclass's own logic wrongly assume the parent's real work already ran, given the dispatcher runs child-class-first then parent).
2. Grep every real `super().on_mount()` call site under tldw_chatbook/ (excluding Third_Party/ and comment-only mentions). Classify each by tracing its parent chain's on_mount:
   (a) BaseAppScreen-derived screens where BaseAppScreen.on_mount is just a log line -> remove the redundant call.
   (b) mixins/parents whose on_mount does real work -> verify whether the subclass's own logic depends on that work having already run (ordering assumption); if yes, add a small hook the parent calls after its own work, move subclass logic into the hook instead of on_mount+super() (per TASK-2610's pattern); if no dependency, remove the call.
   (c) dead/unused mixins -> remove (no behavior to preserve).
3. Implement fixes per file.
4. Add Tests/Architecture/test_on_mount_super_guard.py: AST census over UI/Screens + UI/Wizards + Widgets building a transitive BaseAppScreen-subclass name closure, asserting none of those classes define on_mount with a super().on_mount() call. Mutation-test by temporarily re-adding one call and confirming the census fails, then restore via Edit.
5. Run targeted tests for every touched screen/wizard module + repo-wide --collect-only + ruff on touched files.
6. Write report to .task-2710-report.md (git-excluded), tick ACs, add Implementation Notes, mark Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed 21 redundant super().on_mount() calls and hook-fixed 1 ordering-dependent
case, confirmed against Textual 8.2.7's actual dispatch mechanics (message_pump.py
_get_dispatch_methods: walks the MRO derived-class-first, independently dispatches
every class's own on_mount regardless of super() calls -- so a parent handler
always runs exactly once via the dispatcher; the only real risk is ORDER).

18 BaseAppScreen screens + EnhancedFileDialog (Widgets/enhanced_file_picker.py):
parent's on_mount has no ordering dependency on the subclass -> removed cleanly.
2 dead/unused mixins (TooltipMixin, ToastMixin): removed -- both were mixed into
nothing, so the call was a latent AttributeError/duplicate-execution trap waiting
for a future consumer.

1 real ordering-dependent case found (not previously flagged): WizardContainer/
SetupWizardContainer (UI/Wizards/BaseWizard.py + FirstRunSetupWizard.py).
WizardContainer.on_mount does real work (show_step(0), nav setup, validation
timer) and SetupWizardContainer's own on_mount explicitly depended on that work
having already run before its own _refresh_active_ids()/update_progress() (its
docstring said so). Naive removal would have flipped the order (dispatcher runs
the subclass first). Fixed the way TASK-2610 fixed Lab>Speech: added
WizardContainer._post_mount_hook() (no-op, called once at the end of its own
on_mount), and SetupWizardContainer now overrides that hook instead of
on_mount()+super(). Concretely this was ALSO a live (if harmless) bug before the
fix: show_step(0) -- and its on_hide/on_show pair and the 0.1s validation timer --
ran twice per wizard mount.

AC#3 guard: Tests/Architecture/test_on_mount_super_guard.py, an AST census (repo's
established style, e.g. Tests/DB/test_private_sqlite_inventory.py) over
UI/Screens+UI/Wizards+Widgets building a transitive BaseAppScreen-subclass name
closure and flagging any on_mount in that closure containing a super().on_mount()
call. Includes a tmp_path self-test proving the detector fires (incl. through a
transitive grandchild) and never flags unrelated classes. Mutation-tested against
production code: temporarily re-added a super().on_mount() call to
StatsScreen.on_mount, confirmed the guard fails, restored via Edit (not git
checkout), reconfirmed green.

Verification: 293+26 wizard tests (incl. the exact compose_failed/on_mount
ordering path), 149 file-picker tests, 129+299+29+323+337+72+104+75 screen/UI
tests, Tests/test_application_state_ownership.py (54, incl. an AST census that
names WorkflowsScreen/SchedulesScreen.on_mount by symbol), repo-wide
--collect-only (31737 tests, 0 errors), and a ruff before/after diff on all 23
touched files showing byte-identical pre-existing findings (43, same codes) with
zero new issues. 4 test failures found across the run, all independently
confirmed pre-existing/unrelated to this change: 1 CSS-content assertion in
test_non_obscuring_focus_contract.py (checks _forms.tcss, never touched), 3
AttributeErrors in test_library_screen.py from a documented test-fixture shortcut
(_minimal_ingest_screen bypasses __init__ via object.__new__) missing two
unrelated attributes, and 1 cross-file test-isolation flake
(test_mcp_destination_mode_chip_syncs_to_restored_mode passes alone and in its own
file, fails only combined with two other files in one session).

Files: 18 UI/Screens/*.py + scheduling/schedules_workbench.py (removed super()
call), UI/Wizards/BaseWizard.py (+_post_mount_hook), UI/Wizards/
FirstRunSetupWizard.py (on_mount -> _post_mount_hook override),
Widgets/enhanced_file_picker.py, Widgets/tooltip.py, Widgets/toast_notification.py
(removed super() call), Tests/Architecture/test_on_mount_super_guard.py (new).
Full per-site census with category (a)/(b)/(c) breakdown, dispatch-mechanics
citations, and gate output in .task-2710-report.md (worktree-local, git-excluded).

Review follow-up (3 minors, all folded in, stacked commit e23b27fa2):
1. Guard was BaseAppScreen-only and missed the one real bug found
   (WizardContainer). Added test_no_wizardcontainer_subclass_calls_super_on_mount
   reusing the existing root= parameter; mutation-tested against
   SetupWizardContainer (re-added the pattern, confirmed failure, Edit-restored).
2. Pinned show_step(0) running exactly once per wizard mount behaviorally
   (not just via the AST guard): test_show_step_runs_exactly_once_on_wizard_mount
   monkeypatch-counts WizardContainer.show_step through a real pilot mount;
   mutation-tested the same way (assert [0, 0] == [0] on the reintroduced bug).
3. Report's test_library_screen.py writeup wrongly attributed all 3
   pre-existing failures to 2 shared attributes; corrected -- 3 different
   attributes, one per test (verified by re-running the third test and
   reading its traceback: _library_ingest_clear_finished_armed).

Guard file + wizard file re-run together post-fix: 101 passed (4 + 97).
<!-- SECTION:NOTES:END -->
