# TASK-19642.1 Skills Harness Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the 29 assigned Skills Library/import integration failures by aligning their test harness and interactions with the accepted Library lifecycle and Skill-editor contracts.

**Architecture:** Change only the two Skills integration test owners plus task evidence. Reuse the existing returning-user Library app wrapper and bounded Textual wait helpers, make optional Prompt/Study/Quiz seams inert, and drive the current clean/dirty/delete/create lifecycle instead of changing production behavior.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest/pytest-asyncio, Ruff, Backlog.md.

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`

**Reason:** ADR-076 already owns fresh-profile Starter filtering; TASK-19025 owns the current Skill-editor lifecycle. This repair changes neither contract.

---

## Scope and file ownership

**Modify**

- `Tests/Skills/test_skills_library_flow.py` — returning-user app factory, inert optional owners, bounded navigation helpers, and current Skill-editor interactions.
- `Tests/Skills/test_skills_import.py` — returning-user app factory and bounded Skills/Import navigation.
- `backlog/tasks/task-19642.1 - Repair-recurring-Skills-Library-and-import-harness-failures.md` — checked acceptance criteria, exact evidence, and implementation notes.

**Modify only if the incident is not already recorded adequately**

- `backlog/docs/lessons-testing-evidence.md` — one concise incident note that a truthful shared fresh-profile factory is not automatically a valid returning-user destination harness.

**Do not modify**

- `tldw_chatbook/` production code.
- ADR-076, Library lifecycle defaults, Skill trust/security behavior, service ownership, or the shared `Tests/UI/app_factory.py` defaults.
- Unrelated fixed pauses or unrelated Skills tests.

## Task 1: Repair the full-Library harness contract

**Files:**

- Modify: `Tests/Skills/test_skills_library_flow.py:35-153`
- Modify: `Tests/Skills/test_skills_import.py:34-90`

- [ ] **Step 1: Reconfirm representative RED failures**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Skills/test_skills_import.py::test_import_real_superpowers_skills_lands_trust_pending \
  Tests/Skills/test_skills_library_flow.py::test_saving_a_trusted_skill_warns_and_requeues_needs_review \
  --tb=short
```

Expected: both fail. The import node cannot find `#library-row-browse-skills` under a newly admitted profile; the clean trusted-skill node observes no `Saved.` status because Save is not lifecycle-available until an edit exists.

- [ ] **Step 2: Import the existing returning-user wrapper and wait helpers**

In both files, remove the direct import:

```python
from Tests.UI.app_factory import _build_test_app
```

Import `_build_test_app` from `Tests.UI.test_library_shell` instead. Add only the wait helpers each file uses:

```python
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _wait_for_display,
    _wait_for_library_shell,
    _wait_for_selector,
)
```

`test_skills_import.py` does not need `_wait_for_display`; keep its import list smaller.

- [ ] **Step 3: Make non-Skills optional count owners inert**

Extend the existing shared helper in `test_skills_library_flow.py` without creating a new fake class:

```python
def _wire_empty_non_skill_services(app) -> None:
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.prompt_scope_service = object()
    app.study_scope_service = object()
    app.study_quiz_scope_service = object()
```

This retains real Skills/trust/policy owners and takes the production-supported missing-optional-seam path for decorative counts.

- [ ] **Step 4: Replace only the failing navigation pauses with observable waits**

Change `_open_skill_editor` to wait for the rail row, requested Skill row, and mounted editor:

```python
async def _open_skill_editor(screen, pilot, skill_name: str) -> None:
    browse = await _wait_for_selector(
        screen, pilot, "#library-row-browse-skills"
    )
    assert isinstance(browse, Button)
    browse.press()
    skill_row = await _wait_for_selector(
        screen, pilot, f"#library-skill-row-{skill_name}"
    )
    assert isinstance(skill_row, Button)
    skill_row.press()
    await _wait_for_selector(screen, pilot, "#library-skill-name")
    assert screen._library_skill_detail is not None
```

Change `_open_skills_import_row` similarly:

```python
async def _open_skills_import_row(screen, pilot) -> None:
    browse = await _wait_for_selector(
        screen, pilot, "#library-row-browse-skills"
    )
    assert isinstance(browse, Button)
    browse.press()
    import_button = await _wait_for_selector(
        screen, pilot, "#library-skills-import"
    )
    assert isinstance(import_button, Button)
    import_button.press()
    await _wait_for_selector(screen, pilot, "#library-skills-import-path")
```

Do not change unrelated pause loops.

- [ ] **Step 5: Clear the two in-scope Ruff findings**

Remove the unused local `textual.app.App` import in the bootstrap-modal test. Split the semicolon-separated pauses in `test_list_mode_unlock_refreshes_snapshot_not_just_posture` onto separate lines. Make no other formatting-only edits.

- [ ] **Step 6: Run the import owner and representative Library navigation nodes**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Skills/test_skills_import.py \
  Tests/Skills/test_skills_library_flow.py::test_open_skill_row_populates_editor_fields_and_save_bumps_version \
  Tests/Skills/test_skills_library_flow.py::test_uninitialized_trust_store_list_still_shows_needs_review_glyph \
  Tests/Skills/test_skills_library_flow.py::test_orphaned_manifest_is_one_click_resetup \
  --tb=short
```

Expected: 16 passed. No missing Skills row and no local Prompt/Study/Quiz
backend exception in emitted failure output.

- [ ] **Step 7: Commit the harness repair**

```bash
git add Tests/Skills/test_skills_import.py Tests/Skills/test_skills_library_flow.py
git commit -m "test(skills): repair Library harness posture"
```

## Task 2: Align legacy interactions with the accepted Skill lifecycle

**Files:**

- Modify: `Tests/Skills/test_skills_library_flow.py:307-1188`

- [ ] **Step 1: Run the full flow owner after Task 1 and record residual RED nodes**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Skills/test_skills_library_flow.py --tb=short
```

Expected: the profile/row failures are gone. Residual failures must be limited to tests still driving a clean Save, a hidden clean Delete, or the superseded create-save copy. If a different production defect appears, stop and return to systematic debugging rather than broadening this plan.

- [ ] **Step 2: Make the trusted-skill save perform a real edit**

Immediately before pressing Save in `test_saving_a_trusted_skill_warns_and_requeues_needs_review`, change a live field and wait for Save to become visible:

```python
screen.query_one(
    "#library-skill-description", Input
).value = "Reviews a diff after saving"
save = await _wait_for_display(screen, pilot, "#library-skill-save")
assert isinstance(save, Button)
assert screen._library_skill_dirty is True
save.press()
```

Keep the durable post-save trust assertions unchanged.

- [ ] **Step 3: Reveal clean Delete through More actions in every deletion scenario**

For the clean saved-skill delete paths in:

- `test_delete_skill_returns_to_list_and_decrements_count`
- `test_skill_editor_opens_under_real_runtime_policy_enforcer`
- `test_delete_cancel_preserves_edits_typed_during_confirm`
- `test_derived_flag_cleared_when_snapshotting_populated_description`

replace direct hidden-Delete presses with the real presentation path:

```python
more = await _wait_for_display(
    screen, pilot, "#library-skill-more-actions"
)
assert isinstance(more, Button)
more.press()
delete = await _wait_for_display(screen, pilot, "#library-skill-delete")
assert isinstance(delete, Button)
delete.press()
confirm = await _wait_for_display(
    screen, pilot, "#library-skill-delete-confirm"
)
assert isinstance(confirm, Button)
assert screen._library_skill_confirming_delete is True
```

Only the scenarios that actually delete should press `confirm`. In
`test_delete_cancel_preserves_edits_typed_during_confirm`, keep the current
ordering of entering confirmation, typing, and cancelling. In
`test_derived_flag_cleared_when_snapshotting_populated_description`, move the
typing step *after* the clean editor enters confirmation, then press Cancel so
the production cancel handler snapshots the populated description and clears
`description_derived`; a dirty editor correctly has no visible More/Delete
path under TASK-19025. Update that test's outdated recompose wording and assert
the current in-place presentation contract:

```python
assert screen._library_skill_editor_state.description_derived is False
hint = screen.query_one("#library-skill-description-hint", Static)
assert hint.display is False
```

The hint stays mounted because the current lifecycle patches visibility
without recomposing; DOM absence is no longer the contract.

- [ ] **Step 4: Pin the accepted first-save trust guidance**

Update the two create-save assertions:

```python
assert status_text == (
    "Saved. Review trust before using this Skill with the agent."
)
```

Apply this to:

- `test_library_shell_create_skill_save_creates_and_increments_count`
- `test_library_shell_create_skill_save_arrives_needs_review_with_panel_primed`

Keep their persisted record, rail count, and trust-state assertions unchanged.

- [ ] **Step 5: Run the exact lifecycle-focused nodes**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Skills/test_skills_library_flow.py::test_saving_a_trusted_skill_warns_and_requeues_needs_review \
  Tests/Skills/test_skills_library_flow.py::test_delete_skill_returns_to_list_and_decrements_count \
  Tests/Skills/test_skills_library_flow.py::test_skill_editor_opens_under_real_runtime_policy_enforcer \
  Tests/Skills/test_skills_library_flow.py::test_library_shell_create_skill_save_creates_and_increments_count \
  Tests/Skills/test_skills_library_flow.py::test_library_shell_create_skill_save_arrives_needs_review_with_panel_primed \
  Tests/Skills/test_skills_library_flow.py::test_delete_cancel_preserves_edits_typed_during_confirm \
  Tests/Skills/test_skills_library_flow.py::test_derived_flag_cleared_when_snapshotting_populated_description \
  --tb=short
```

Expected: 7 passed.

- [ ] **Step 6: Commit the lifecycle alignment**

```bash
git add Tests/Skills/test_skills_library_flow.py
git commit -m "test(skills): follow current editor lifecycle"
```

## Task 3: Prove the repair and its inverse

### Implementation-time RED amendment

The first full two-file gate after Tasks 1 and 2 produced `2 failed, 32 passed`.
Both nodes failed identically in isolation, so this was not order pollution or
timing noise:

- `test_uninitialized_trust_shows_setup_state_and_bootstrap_enables_approve_flow`
  expected healthy trust actions to remain expanded after bootstrap, but the
  accepted compact trust panel now requires `View details` first;
- `test_library_shell_create_skill_row_opens_blank_editor` queried the removed
  basic-mode `#library-skill-allowed-tools` input, while allowed tools now use
  the advanced `#library-skill-tool-picker` selection list.

Before continuing the original Task 3 steps, align only these two stale
assertions with their passing production-owner contracts. In the bootstrap
test, assert `#library-skill-trust-view-details`, press it, and wait for the
Unlock control before retaining the Unlock/Review/Approve assertions. In the
blank-create test, assert the advanced region is initially hidden, press
`#library-skill-editor-mode`, wait for the advanced region to display, then
assert the allowed-tools `SelectionList` has no selected values and the
captured-value `Static` is empty. Re-run the two nodes and commit only the
existing flow test as `test(skills): follow progressive editor disclosure`.
This remains test-only and changes no accepted UI, trust, or policy behavior.

**Files:**

- Verify: `Tests/Skills/test_skills_import.py`
- Verify: `Tests/Skills/test_skills_library_flow.py`
- Verify: `Tests/UI/test_library_skills_canvas.py`

- [ ] **Step 1: Run the complete touched two-file gate**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Skills/test_skills_import.py \
  Tests/Skills/test_skills_library_flow.py \
  --tb=short
```

Expected: 34 passed. This includes all 29 assigned nodes plus the five incumbent passing nodes; do not claim broader Skills or repository coverage.

- [ ] **Step 2: Run the direct production owner nodes**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_skills_canvas.py::test_skill_editor_clean_saved_mode_renders_navigation_actions_only \
  Tests/UI/test_library_skills_canvas.py::test_skill_editor_lifecycle_exposes_only_valid_primary_actions \
  Tests/UI/test_library_skills_canvas.py::test_handle_library_skill_delete_enters_confirm_state \
  Tests/UI/test_library_skills_canvas.py::test_create_save_success_consumes_scroll_receipt_after_recompose \
  Tests/UI/test_library_skills_canvas.py::test_mark_dirty_clears_stale_saved_status \
  Tests/UI/test_library_skills_canvas.py::test_skill_editor_healthy_trust_is_compact_until_details_are_requested \
  Tests/UI/test_library_skills_canvas.py::test_skill_editor_advanced_tool_picker_is_bounded_unique_and_lossless \
  Tests/UI/test_library_skills_canvas.py::test_library_skill_mode_switch_is_targeted_and_remembered \
  --tb=short
```

Expected: 12 passed because the lifecycle test is parameterized into five cases.

- [ ] **Step 3: Run static checks without bulk formatting**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  Tests/Skills/test_skills_import.py \
  Tests/Skills/test_skills_library_flow.py
git diff --check
```

Expected: both commands exit 0.

Record, but do not “fix,” the known baseline:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  Tests/Skills/test_skills_import.py \
  Tests/Skills/test_skills_library_flow.py
```

Expected baseline: exit 1 and both files reported as requiring whole-file reformat. Confirm the task did not add a third file or broaden formatter drift; do not bulk-reformat.

- [ ] **Step 4: Prove the three root-cause inverses one at a time**

Use `apply_patch` for each temporary mutation, run only its named test, then immediately restore with `apply_patch` before continuing:

1. Temporarily import `_build_test_app` directly from `Tests.UI.app_factory` in `test_skills_import.py`. Run `test_import_real_superpowers_skills_lands_trust_pending`; expected failure is the missing Skills row under the fresh profile.
2. Temporarily remove the More-actions reveal from `test_delete_skill_returns_to_list_and_decrements_count`. Run that node; expected failure is that the visible lifecycle never exposes/enters the Delete confirmation as asserted.
3. Temporarily restore the old `status_text == "Saved."` assertion in `test_library_shell_create_skill_save_creates_and_increments_count`. Run that node; expected failure shows the current trust-review guidance.

After restoring each mutation, run `git diff --check` and confirm
`git status --short` is empty.

- [ ] **Step 5: Re-run the touched two-file gate after inverse restoration**

Run the Task 3 Step 1 command again.

Expected: 34 passed.

## Task 4: Close TASK-19642.1 with exact evidence

**Files:**

- Modify: `backlog/tasks/task-19642.1 - Repair-recurring-Skills-Library-and-import-harness-failures.md`
- Modify only if needed: `backlog/docs/lessons-testing-evidence.md`

- [ ] **Step 1: Decide and record the lesson outcome**

Search `backlog/docs/lessons-testing-evidence.md` for an existing returning-user/fresh-profile harness incident. If none exists, add one concise entry naming the repeated TASK-19579/TASK-19642.1 incident and the evidence: the shared factory correctly admitted a fresh profile, while destination integration tests assumed the full returning-user rail. If an equivalent lesson exists, do not duplicate it; state that in Implementation Notes.

- [ ] **Step 2: Update task acceptance criteria and notes**

Check all three ACs only after Task 3 is green. Add Implementation Notes containing:

- the four confirmed drift classes;
- the test-only files changed and explicit no-production-code result;
- exact `34 passed` two-file evidence and `12 passed` owner evidence;
- Ruff and `git diff --check` results;
- the pre-existing two-file formatter baseline, without claiming formatter success;
- all three inverse failures and successful restoration;
- ADR-076 reuse and no-new-ADR decision;
- lesson outcome.

- [ ] **Step 3: Mark the task Done through Backlog.md**

Run:

```bash
backlog task edit 19642.1 -s Done
```

Verify:

```bash
backlog task 19642.1 --plain
```

Expected: status Done, all ACs checked, plan/notes/ADR evidence present.

- [ ] **Step 4: Commit closeout documentation**

```bash
git add \
  'backlog/tasks/task-19642.1 - Repair-recurring-Skills-Library-and-import-harness-failures.md' \
  backlog/docs/lessons-testing-evidence.md
git commit -m "docs: close TASK-19642.1"
```

If the lessons file was not changed, omit it from `git add`.

- [ ] **Step 5: Run final scoped verification from clean HEAD**

Run Task 3 Steps 1-3 again after the closeout commit, then:

```bash
git status --short
git diff --check origin/dev...HEAD
```

Expected: empty status, focused tests/Ruff green, and no whitespace errors. Do not run the repository-wide suite.

## Execution and review handoff

After implementation, use `superpowers:verification-before-completion`, then `superpowers:requesting-code-review`. Address technically valid review feedback through `superpowers:receiving-code-review`. Push and create the PR only after the focused clean-HEAD evidence is green; address Qodo/PR comments, then use `superpowers:finishing-a-development-branch` before merge.
