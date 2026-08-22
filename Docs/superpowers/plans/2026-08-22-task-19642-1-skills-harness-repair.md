# TASK-19642.1 Skills Harness Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the 29 assigned Skills Library/import integration failures and eliminate the trust-header ordering and stale-worker races exposed by independent verification.

**Architecture:** Keep the completed harness repair in the two Skills integration owners, move the existing trust-posture refresh across the Skills-canvas mount boundary, and cancel that posture worker group when the service disappears. Preserve strict targeted sync, route/generation/focus guards, and the no-whole-screen-fallback contract; use deterministic owner tests instead of retries, sleeps, tokens, or a new reconciliation abstraction.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest/pytest-asyncio, Ruff, Backlog.md.

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`

**Reason:** ADR-076 and TASK-19025 already own the Library and Skill-editor contracts, and the existing entry-reconciliation contract already requires automatic workers to project only into their mounted owner. Moving one refresh trigger after that mount is a routine sequencing fix, not a new architectural policy.

---

## Scope and file ownership

**Modify**

- `Tests/Skills/test_skills_library_flow.py` — returning-user app factory, inert optional owners, bounded navigation helpers, and current Skill-editor interactions.
- `Tests/Skills/test_skills_import.py` — returning-user app factory and bounded Skills/Import navigation.
- `Tests/UI/test_library_entry_compose_once.py` — deterministic mounted-owner ordering and stale-worker overlap regressions.
- `Tests/UI/test_library_skills_canvas.py` — editor-mode no-sync regression preserving live draft, widget identity, and focus.
- `tldw_chatbook/UI/Screens/library_screen.py` — start the existing Skills trust-posture refresh after destination mounting and supersede it when the service disappears.
- `backlog/tasks/task-19642.1 - Repair-recurring-Skills-Library-and-import-harness-failures.md` — checked acceptance criteria, exact evidence, and implementation notes.

**Modify only if the incident is not already recorded adequately**

- `backlog/docs/lessons-testing-evidence.md` — one concise incident note that a truthful shared fresh-profile factory is not automatically a valid returning-user destination harness.

**Do not modify**

- Production code outside `tldw_chatbook/UI/Screens/library_screen.py`.
- ADR-076, Library lifecycle defaults, Skill trust/security behavior, service ownership, or the shared `Tests/UI/app_factory.py` defaults.
- `_load_library_skills_trust_posture` guards, `_sync_library_canvas` fallback semantics, or entry-reconciliation ownership contracts.
- Unrelated fixed pauses or unrelated Skills tests.

## Execution state

Tasks 1-3 below are already implemented, committed, and reviewed; they remain
as provenance for the repaired 29-node harness. Do not repeat their edit or
inverse steps. Resume at Task 4 from the clean plan commit.

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
git diff --check origin/dev...HEAD
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

## Task 4: Start Skills trust posture only after its canvas mounts

**Files:**

- Modify: `Tests/UI/test_library_entry_compose_once.py`
- Modify: `Tests/UI/test_library_skills_canvas.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py:17327-17360`

### Quality-review stale-worker amendment

The first no-service fix cleared and repainted the mounted Skills list but did
not supersede an earlier `library_skills_trust_posture` worker. Add
`test_missing_trust_service_supersedes_in_flight_posture_worker`: gate a real
posture callable in its `asyncio.to_thread` hop, remove the trust service, run
the no-service refresh, release the old callable, and assert the old worker is
cancelled and no already-clear list repaint occurs. Verify RED first as a stale
`["ready"]` projection, then minimally call
`self.workers.cancel_group(self, "library_skills_trust_posture")` before the
existing unconditional clear. Do not add a request token unless this real
Textual cancellation fails to block publication.

### Qodo no-op repaint amendment

Repeated no-service refreshes currently strict-sync an already-empty Skills
list even when no trust header is mounted. Add
`test_missing_trust_service_already_clear_list_skips_repaint` first and verify
RED with one sync. Preserve worker-group cancellation and state clearing, but
strict-sync only when the prior posture is nonempty or a trust header is still
mounted. Keep `allow_screen_fallback=False` and all route, generation, focus,
security, and runtime-policy behavior unchanged.

- [ ] **Step 1: Add the deterministic mounted-owner RED test**

Add this owner test near the existing Skills posture reconciliation tests:

```python
@pytest.mark.asyncio
async def test_skills_rail_starts_trust_posture_after_canvas_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.skills_scope_service = _FakeSkillsScopeService(
        available=[{"name": "code-review"}]
    )
    screen = LibraryScreen(app)
    screen.restore_state(
        {"library_selected_row_id": LIBRARY_ROW_BROWSE_CONVERSATIONS}
    )
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        await _wait_for_library_shell(active_screen, pilot)
        await active_screen.workers.wait_for_complete()
        observed_owner_state: list[bool] = []

        def record_refresh() -> None:
            observed_owner_state.append(
                bool(active_screen.query("#library-skills-canvas"))
            )

        monkeypatch.setattr(
            active_screen,
            "_refresh_library_skills_trust_posture",
            record_refresh,
        )

        await active_screen._select_library_rail_row(
            LIBRARY_ROW_BROWSE_SKILLS
        )
        await _wait_for_selector(
            active_screen, pilot, "#library-skills-canvas"
        )

        assert active_screen._library_selected_row_id == (
            LIBRARY_ROW_BROWSE_SKILLS
        )
        assert active_screen._library_skills_view == "list"
        assert observed_owner_state == [True]
```

The spy is installed only after initial Library workers settle, so it observes
the rail-entry trigger rather than snapshot initialization.

- [ ] **Step 2: Run the owner test and verify deterministic RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_entry_compose_once.py::test_skills_rail_starts_trust_posture_after_canvas_mount \
  --tb=short
```

Expected: fail with `observed_owner_state == [False]`; the existing trigger
runs while the outgoing route still owns the canvas.

- [ ] **Step 3: Move the existing trigger after destination mounting**

Delete the pre-mount block that calls
`self._refresh_library_skills_trust_posture()` before the shell is built. After
the existing targeted replacement/whole-screen fallback completes, add:

```python
if (
    self._library_selected_row_id == LIBRARY_ROW_BROWSE_SKILLS
    and self._library_skills_view == "list"
):
    # The strict posture projection requires its retained owner to exist.
    self._refresh_library_skills_trust_posture()
```

Do not modify `_load_library_skills_trust_posture`, `_sync_library_canvas`, or
their route/generation/focus and fallback semantics.

- [ ] **Step 4: Run the deterministic owner test and verify GREEN**

Run the Step 2 command.

Expected: 1 passed.

- [ ] **Step 5: Run the directly related reconciliation owners**

This gate owns both follow-up lifecycle regressions: the mounted list must
clear a stale header when the trust service disappears, while a background
snapshot in editor mode must clear cached posture without syncing the canvas or
losing the live draft, widget identity, or focus.

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_library_entry_compose_once.py::test_skills_rail_starts_trust_posture_after_canvas_mount \
  Tests/UI/test_library_entry_compose_once.py::test_skills_rail_without_trust_service_clears_mounted_header \
  Tests/UI/test_library_entry_compose_once.py::test_missing_trust_service_supersedes_in_flight_posture_worker \
  Tests/UI/test_library_entry_compose_once.py::test_missing_trust_service_already_clear_list_skips_repaint \
  'Tests/UI/test_library_entry_compose_once.py::test_automatic_entry_worker_composes_screen_once_and_routes_in_place[skills-size0]' \
  'Tests/UI/test_library_entry_compose_once.py::test_automatic_entry_worker_composes_screen_once_and_routes_in_place[skills-size1]' \
  Tests/UI/test_library_entry_compose_once.py::test_stale_skills_posture_cannot_project_after_route_switch \
  Tests/UI/test_library_entry_compose_once.py::test_stale_skills_generation_cannot_project_on_the_same_route \
  Tests/UI/test_library_entry_compose_once.py::test_skills_posture_sync_composes_focus_with_render_completion \
  Tests/Skills/test_skills_library_flow.py::test_orphaned_manifest_is_one_click_resetup \
  Tests/UI/test_library_skills_canvas.py::test_missing_trust_service_snapshot_preserves_open_skill_draft \
  --tb=short
```

Expected: 11 passed. The compose-once cases must still report no screen-level
recompose after the mounted owner exists.

- [ ] **Step 6: Run scoped static checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_entry_compose_once.py \
  Tests/UI/test_library_skills_canvas.py \
  Tests/Skills/test_skills_import.py \
  Tests/Skills/test_skills_library_flow.py
git diff --check
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_entry_compose_once.py \
  Tests/UI/test_library_skills_canvas.py \
  Tests/Skills/test_skills_import.py \
  Tests/Skills/test_skills_library_flow.py
```

Expected: Ruff check and `git diff --check` exit 0. Formatter check retains the
pre-existing exit 1 baseline for all five large files; record it and do not
bulk-format.

- [ ] **Step 7: Prove the ordering and editor-lifecycle inverses**

Use `apply_patch` to move the refresh block back before shell construction,
run only the Step 2 owner test, and confirm it fails with
`observed_owner_state == [False]`. Restore with `apply_patch`, rerun the owner
test to 1 passed. Then temporarily restore the unconditional no-service Skills
canvas sync and run
`test_missing_trust_service_snapshot_preserves_open_skill_draft`; confirm it
fails with one recorded sync against the open editor. Restore the list-mode
guard, rerun that regression to 1 passed, then require `git diff --check` green.
Finally remove only the no-service `cancel_group` call and run
`test_missing_trust_service_supersedes_in_flight_posture_worker`; require RED
with `["ready"]`. Restore cancellation, rerun to 1 passed, and require the
worker to be cancelled with no posture projected. Temporarily restore an
unconditional no-service list sync and run
`test_missing_trust_service_already_clear_list_skips_repaint`; require RED with
one recorded sync, then restore the stale-state/header guard.

- [ ] **Step 8: Commit the ordering repair**

```bash
git add \
  tldw_chatbook/UI/Screens/library_screen.py \
  Tests/UI/test_library_entry_compose_once.py \
  Tests/UI/test_library_skills_canvas.py
git commit -m "fix(library): mount Skills before trust refresh"
```

- [ ] **Step 9: Run the exact two-file gate three consecutive times**

Run this command three separate times and record every result:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/Skills/test_skills_import.py \
  Tests/Skills/test_skills_library_flow.py \
  --tb=short
```

Expected each time: 34 passed. If any run fails, keep TASK-19642.1 open and
return to systematic debugging.

## Task 5: Close TASK-19642.1 with exact evidence

**Files:**

- Modify: `backlog/tasks/task-19642.1 - Repair-recurring-Skills-Library-and-import-harness-failures.md`
- Modify only if needed: `backlog/docs/lessons-testing-evidence.md`

- [ ] **Step 1: Decide and record the lesson outcome**

Search `backlog/docs/lessons-testing-evidence.md` for an existing returning-user/fresh-profile harness incident. If none exists, add one concise entry naming the repeated TASK-19579/TASK-19642.1 incident and the evidence: the shared factory correctly admitted a fresh profile, while destination integration tests assumed the full returning-user rail. If an equivalent lesson exists, do not duplicate it; state that in Implementation Notes.

- [ ] **Step 2: Rebase onto current development**

Keep the task In Progress. Run:

```bash
git fetch origin
git rebase origin/dev
```

Resolve only conflicts in this task's owned files. If current development has
changed the mounted-owner contract, stop and return to design review instead of
forcing the old ordering patch through.

- [ ] **Step 3: Run the complete post-rebase focused gate**

After rebase, run Task 4 Steps 5-6, explicitly including
`test_skills_rail_without_trust_service_clears_mounted_header` and
`test_missing_trust_service_supersedes_in_flight_posture_worker`, plus
`test_missing_trust_service_snapshot_preserves_open_skill_draft`. Run the
exact two-file command from Task 4 Step 9 three consecutive times, and run the
12 editor-owner cases from Task 3 Step 2 together with
`test_missing_trust_service_snapshot_preserves_open_skill_draft` and
`test_missing_trust_service_supersedes_in_flight_posture_worker` as a 14-case
editor/trust lifecycle gate. Then run:

```bash
git diff --check origin/dev...HEAD
git status --short
```

Expected: `11 passed` ordering/reconciliation and lifecycle-regression owners, three separate
`34 passed` two-file runs, `14 passed` editor/trust lifecycle owners, Ruff green, the recorded
five-file formatter baseline only, no whitespace errors, and an empty status.
Do not run a repository-wide suite.

- [ ] **Step 4: Obtain final independent code review**

Use `superpowers:verification-before-completion` and
`superpowers:requesting-code-review` against `origin/dev...HEAD`. Address all
Critical and Important findings through `superpowers:receiving-code-review`,
rerun the affected focused gates, and obtain approval before changing task
status.

- [ ] **Step 5: Update task acceptance criteria and notes**

Check all three ACs only after Steps 3-4 are green. Add Implementation Notes containing:

- the four confirmed test-contract drift classes, mounted-owner ordering race,
  and no-service editor lifecycle regression;
- all modified files, including the narrow production sequencing change;
- three consecutive `34 passed` two-file runs, `14 passed` editor/trust lifecycle owners, and
  `11 passed` ordering/reconciliation and lifecycle-regression owners;
- Ruff and `git diff --check` results;
- the pre-existing five-file formatter baseline, without claiming formatter success;
- all six representative causal RED/inverse failures and successful restoration;
- explicit confirmation that the repaired two-file Skills gate output contained
  no Prompt/Study/Quiz backend exception or local Library snapshot failure
  warning; do not extend that claim to the separate reconciliation-owner
  harness;
- ADR-076 reuse and no-new-ADR decision;
- lesson outcome.

- [ ] **Step 6: Mark the task Done through Backlog.md**

Run:

```bash
backlog task edit 19642.1 -s Done
```

Verify:

```bash
backlog task 19642.1 --plain
```

Expected: status Done, all ACs checked, plan/notes/ADR evidence present.

- [ ] **Step 7: Commit closeout documentation**

```bash
git add \
  'backlog/tasks/task-19642.1 - Repair-recurring-Skills-Library-and-import-harness-failures.md' \
  backlog/docs/lessons-testing-evidence.md
git commit -m "docs: complete TASK-19642.1"
```

If the lessons file was not changed, omit it from `git add`.

- [ ] **Step 8: Run final scoped verification from clean HEAD**

Run Task 3 Steps 1-3 and Task 4 Steps 5-6 again after the closeout commit, then:

```bash
git status --short
git diff --check origin/dev...HEAD
```

Expected: empty status, focused tests/Ruff green, and no whitespace errors. Do not run the repository-wide suite.

If this clean-HEAD gate fails, immediately return the task to In Progress,
uncheck ACs 1-2, record the failure, and resume systematic debugging.

## Execution and review handoff

Push and create the PR only after Task 5 is complete.
Inspect Qodo and all PR review threads, apply only technically valid feedback
through the receiving-review workflow, rerun the affected focused gates, and
resolve the threads. If review changes code or exposes a failed acceptance
criterion, immediately reopen TASK-19642.1, uncheck the affected ACs, and repeat
Task 5's focused verification/review/closeout sequence. Use
`superpowers:finishing-a-development-branch` before merging.
