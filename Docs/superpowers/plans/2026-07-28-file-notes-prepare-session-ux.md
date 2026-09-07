# File Notes Prepare Session UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn File Notes Session Git into a legible, note-centered `Prepare session for commit` workflow without changing its session-only Git safety behavior.

**Architecture:** Keep `LibraryFileNotesGitPanel` presentation-only and derive every new label, status, count, and recovery instruction from existing immutable row/status models. Keep `FileNotesSessionOwner` as the sole root/repository/session authority; `LibraryFileNotesWorkspace` adds only a keyed last-action presentation record and responsive CSS classes. No Git service, owner, database, schema, or command behavior changes.

**Tech Stack:** Python 3.11+, Textual 8.2.7, Rich cell measurement, pytest 8.4.2, existing real mounted Textual harnesses.

**Backlog:** [TASK-1235](../../../backlog/tasks/task-1235%20-%20Polish-File-Notes-prepare-session-for-commit-UX.md)

**Specification:** [File Notes Prepare Session UX Design](../specs/2026-07-28-file-notes-prepare-session-ux-design.md)

**Source UAT:** [File Notes Session Git acceptance critique](../../../.impeccable/critique/2026-07-28T15-38-30Z__ok-widgets-library-library-file-notes-git-panel-py.md)

**Depends on:** TASK-1213

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a presentation-only repair conforming to ADR-035, ADR-033, ADR-011, and the File Notes disk-authority decision in `backlog/decisions/029-file-notes-disk-authority.md`. It changes no persistence, ownership, Git service contract, security boundary, dependency, or long-lived application structure.

---

## Execution Environment and Scope

Run every command from:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/file-notes-move-tombstone
```

Use the main checkout's environment explicitly:

```bash
../../.venv/bin/python -c "import pathlib, tldw_chatbook, textual, pytest; print(pathlib.Path(tldw_chatbook.__file__).resolve()); print(textual.__version__, pytest.__version__)"
../../.venv/bin/python -m ruff --version
git --version
```

Expected package path: this worktree. Verified baseline versions are Python
3.12.11, Textual 8.2.7, pytest 8.4.2, Ruff 0.15.22, and Git 2.39.5.

The pre-change focused baseline is:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_library_file_notes_git.py
```

Expected: `49 passed`.

Use strict red/green TDD. Run each named test immediately after writing it and
confirm it fails because the approved behavior is absent before modifying
production code.

Per the approved task boundary, do not run full CI, the multi-hour full pytest
suite, network/remotes, or a broad soak/performance harness. Run the complete
focused UI file plus targeted Ruff, compile, and diff checks before completion.

## File Structure

- Modify `tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py`:
  note-intent rows, semantic/recovery copy, title/help hierarchy, independent
  current/last-action surfaces, authority clearing, counts, bounded geometry,
  and non-obscuring focus.
- Modify `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`:
  owner-snapshot-keyed last-action presentation, contract-true summaries,
  authority transition projection, quiet wide editor actions, and narrow
  toolbar layout.
- Modify `Tests/UI/test_library_file_notes_git.py`: all new automated coverage;
  reuse its existing real panel/workspace harnesses and complete fake service.
- Modify
  `backlog/tasks/task-1235 - Polish-File-Notes-prepare-session-for-commit-UX.md`
  before implementation to link this plan, then after verification to check
  acceptance criteria and add concise implementation notes.

Do not modify `file_notes_session_owner.py`, `file_notes_git_service.py`,
`file_notes_service.py`, SQLite code, application navigation, global CSS, or
Git configuration.

## Planning Checkpoint

Before Task 1, link this plan from TASK-1235, record the ADR determination in
the task's `## Implementation Plan`, and commit the plan, task update, and
approved-spec wording correction together. Production implementation begins
from that clean documentation checkpoint.

## Task 1: Make the Git Panel Note-Centered and Authority-Safe

**Files:**

- Modify:
  `tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py:1-614`
- Modify: `Tests/UI/test_library_file_notes_git.py:320-806`

- [ ] **Step 1: Write failing note-intent, semantic-state, hierarchy, and authority tests**

Extend the existing `_row()` helper with optional `latest_action`,
`source_path`, and `destination_path` arguments. Reuse `_PanelHarness`,
`_status`, `_text`, and `_wait_until`; do not add another harness or fake.

Add a parametrized row projection test:

```python
@pytest.mark.parametrize(
    ("latest_action", "verb"),
    [
        ("created", "CREATED"),
        ("modified", "EDITED"),
        ("moved", "MOVED"),
        ("deleted", "DELETED"),
        ("restored", "RESTORED"),
    ],
)
@pytest.mark.asyncio
async def test_rows_project_note_intent_on_a_separate_primary_line(
    latest_action: str,
    verb: str,
) -> None:
    row = _row(
        "unstaged",
        latest_action=latest_action,
        source_path="folder/before.md",
        destination_path=(
            "folder/after.md" if latest_action == "moved" else None
        ),
        stage_action="stage",
    )
    panel = LibraryFileNotesGitPanel()
    async with _PanelHarness(panel).run_test() as pilot:
        panel.render_status(_status(row))
        await pilot.pause()
        primary = _text(
            panel.query_one(".file-notes-git-row-primary", Static)
        )
        semantic = _text(
            panel.query_one(".file-notes-git-row-secondary", Static)
        )
        assert primary.startswith(verb)
        assert "folder/before.md" in primary
        if latest_action == "moved":
            assert "-> folder/after.md" in primary
        assert semantic == "READY TO STAGE · Git: unstaged"
```

Add real `coalesce_session_changes()` cases for create→delete,
delete→restore, edit→clean, and chained move. Assert `DELETED`, `RESTORED`,
`EDITED · NO ACTION`, and original-source→final-destination respectively.

Extend `test_row_action_table_is_driven_by_row_policy` with the exact semantic
token and recovery fragment for every existing row state. Its second rendered
line must contain:

| State | Required copy |
| --- | --- |
| `unstaged` | `READY TO STAGE · Git: unstaged` |
| `owned` | `STAGED · by Chatbook` |
| `owned_newer_edits` | `UPDATE AVAILABLE · newer note edits are not staged` |
| `owned_topology_changed` | `UPDATE REQUIRED · stage the moved note before unstaging` |
| `external_staged`, `external_partial` | `BLOCKED · already staged outside Chatbook` and `Refresh` |
| `clean` | `NO ACTION · matches HEAD` |
| `ignored`, `conflict`, unsafe/unsupported variants | `BLOCKED`, the specific reason, external next step, and `Refresh` |
| `unavailable` | `BLOCKED`, restore-Git next step, and `Refresh` |
| `error` | `FAILED`, reason, retry, and `Refresh` |

Extend `test_panel_renders_repository_scope_and_complete_file_state` to assert:

```text
Prepare session for commit
Session paths only · stages complete file state
Up/Down Select | Tab Actions | Enter Run | Esc Back
Status: CURRENT · READY
```

Add
`test_selected_and_bulk_labels_report_selection_and_independent_counts`.
With two stage-eligible rows and one unstage-eligible row, assert:

```python
assert _text(
    panel.query_one("#file-notes-git-selected-note", Static)
).startswith("Selected note: ")
assert str(
    panel.query_one("#file-notes-git-stage-all", Button).label
) == "Stage all (2)"
assert str(
    panel.query_one("#file-notes-git-unstage-all", Button).label
) == "Unstage all (1)"
```

Add a parametrized authority-loss test. Render two ready rows, select the
second, then call `render_untrusted()` or `render_unavailable()`. Assert
`panel.rows == ()`, `selected_group_id is None`, no rendered row widgets remain,
and no selected/bulk mutation is usable. Keep the existing stale/error test and
change it to assert rows remain while
`#file-notes-git-status` contains `STALE` or `STALE · ERROR` and
`#file-notes-git-action-status` remains independent.

- [ ] **Step 2: Run the new panel semantics tests and verify RED**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_library_file_notes_git.py -k "rows_project or coalesced or row_action_table or panel_renders_repository or selected_and_bulk_labels or authority_loss or stale_and_error"
```

Expected: FAIL because rows are one Git-first line, title/help/current status
do not exist, bulk labels have no counts, and authority-loss renderers retain
old rows.

- [ ] **Step 3: Implement the minimal panel projections**

Replace `_ROW_STATE_LABELS` with local view copy and add:

```python
_CHANGE_VERBS = {
    "created": "CREATED",
    "modified": "EDITED",
    "moved": "MOVED",
    "deleted": "DELETED",
    "restored": "RESTORED",
}
```

Keep raw model paths untouched. Build display-only move text with ASCII `->`
from `source_path` and `destination_path`, passed through the existing
control-character sanitizer.

Render `_SessionGitListItem` as exactly two `Static` children:

```python
yield Static(primary, classes="file-notes-git-row-primary", markup=False)
yield Static(
    semantic_and_recovery,
    classes="file-notes-git-row-secondary",
    markup=False,
)
```

Use the existing `disabled_reason` only as bounded detail; append the
state-specific next action from the approved table. Reserve cells for the
semantic token and complete recovery phrase first, then middle-elide only the
diagnostic detail to the remaining width. Never allow a long reason to hide
`Refresh`, `Return to the editor`, or the other exact next action. Do not alter
row policy, eligibility, grouping, or stable IDs.

Change `compose()` to mount, in order:

```text
Back/Refresh/Trust
Prepare session for commit
Repository/branch
Session paths only · stages complete file state
Up/Down Select | Tab Actions | Enter Run | Esc Back
scrollable rows or empty state
current status
last action
selected-note detail
selected actions
bulk actions
```

Use `#file-notes-git-status` for current freshness/recovery. Keep
`#file-notes-git-action-status` only for the latest action, hidden when empty.
Add `set_current_status()`, `set_last_action()`, and `clear_last_action()`.
Every `render_*`, `mark_stale()`, and `set_mutating()` path updates only current
status.

`render_status()` computes independent stage/unstage counts and uses:

```python
stage_count = sum(row.stage_eligible for row in self._rows)
unstage_count = sum(row.unstage_eligible for row in self._rows)
current = (
    "Status: CURRENT · READY — "
    f"{stage_count} can be staged · {unstage_count} can be unstaged."
)
```

Use `Status: STALE`, `Status: STALE · ERROR`,
`Status: TRUST REQUIRED`, `Status: UNAVAILABLE`, and
`Status: UPDATING INDEX` for the corresponding existing states.

Add one `_clear_rows()` helper that empties `_rows`, selection, and the real
`ListView` through the existing row-render generation. Call it from untrusted,
discovery-unavailable, non-repository, and unavailable-status paths. Checking
may retain rows only when the workspace explicitly passes
`retain_rows=True`; the panel never infers authority.

In `_update_actions()`, use `Stage`, `Stage update`, and `Unstage` for selected
buttons and set `Stage all (S)` / `Unstage all (U)` independently. Update the
two-line selected-note detail from the current row.

- [ ] **Step 4: Run the panel semantics tests and verify GREEN**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 5: Write failing focus, elision, and fixed-region geometry tests**

Strengthen `test_panel_action_controls_fit_with_focus_at_24_cells` and
parameterize the supported sizes `(150, 42)`, `(70, 28)`, `(70, 24)`, and
`(40, 20)`. Exercise both untrusted and ready/update states. After focusing
each visible button, assert its complete `render().plain` label, content width,
and region remain inside the panel.

Add `test_prepare_session_fixed_regions_remain_visible_at_40_by_20`. Render a
long moved path, enough rows to overflow, a long row failure reason, a long
current-status diagnostic, and a long failed last-action result. Assert
repository, selected-note, current-status, and last-action widgets are visible,
at most two rows high, and end within the panel. Assert Back and eligible
actions remain visible, both ends of the elided path remain, every exact
recovery action is still visible, and the list is the only flexible-height
region.

Add one helper-level test for cell-aware middle elision, including a wide-cell
path, proving the result never exceeds the requested Rich cell width and keeps
both ends.

- [ ] **Step 6: Run the geometry tests and verify RED**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_library_file_notes_git.py -k "focused_controls or action_controls_fit or fixed_regions or middle_elide"
```

Expected: FAIL because heavy outlines replace focused labels, paths are not
middle-elided, and fixed status/action regions can fall below `40×20`.

- [ ] **Step 7: Implement bounded cell geometry and non-obscuring focus**

Use `rich.cells.cell_len` and `split_graphemes` in a small local
`_middle_elide_cells(text, width)` helper. Reserve three cells for `...`, take
whole graphemes from both ends within the remaining cell budgets, and return
the original string when it already fits.

On list-item mount/resize, update the primary line against the real row width.
Use the same helper for selected-note detail. Keep each row exactly two cells
high.

In panel-local `DEFAULT_CSS`:

```css
.file-notes-git-row:focus,
.file-notes-git-row.-highlight,
LibraryFileNotesGitPanel Button:focus {
    background: $ds-focus-bg;
    color: $ds-focus-fg;
    text-style: bold underline;
    outline: none;
}
```

Repository, selected-note, current-status, and last-action regions get
`max-height: 2` and hidden overflow. The list gets `height: 1fr`,
`min-height: 1`, and no vertical margin that can push fixed regions below the
viewport. Before updating any bounded semantic/current/last-action widget,
fit its diagnostic portion to the widget's real cell budget while reserving
the semantic prefix and complete recovery suffix. Keep the unabridged result
in `_GitLastAction`; truncation is display-only and recomputed after resize.

Replace the fixed `width <= 48` action-stack guess. Sum the Rich cell width
plus existing button chrome for the currently visible labels in each header,
selected, and bulk row; set `-stack-actions` only when a row does not fit.
Recompute after resize and after `_update_actions()` changes visibility/labels.

- [ ] **Step 8: Run the complete panel slice and verify GREEN**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_library_file_notes_git.py -k "panel or row_action_table or rows_project or coalesced or selected_and_bulk or authority_loss or stale_and_error or focused_controls or action_controls_fit or fixed_regions or middle_elide"
```

Expected: PASS.

- [ ] **Step 9: Commit the panel slice**

```bash
git add tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py Tests/UI/test_library_file_notes_git.py
git commit -m "fix(notes): clarify prepare-session Git panel [TASK-1235]"
```

## Task 2: Keep Feedback Current and Quiet Competing Editor Actions

**Files:**

- Modify:
  `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py:45-100, 186-215, 270-305, 560-780, 890-925, 999-1245, 2020-2197`
- Modify: `Tests/UI/test_library_file_notes_git.py:822-2022`

- [ ] **Step 1: Write failing last-action ownership and summary tests**

Use the existing complete `_FakeGitService`; do not assert on the fake itself.
Assert real workspace/panel behavior.

Add direct `_git_action_summary()` cases for:

- successful Stage and Unstage with one and two coalesced groups;
- one moved group with two endpoints counting as one session note;
- zero-effect success;
- blocked, stale, error, and uncertain results;
- a `result.message` plus clean/blocked counts.
- a nominal success with a deliberately mismatched captured action key.

Only certain success with at least one affected group may contain:

```text
1 session note staged; Chatbook targeted only eligible session paths.
2 session notes unstaged; Chatbook restored only its owned session entries.
```

Every other state omits both promises, retains material counts, and includes an
exact recovery ending in Refresh. Existing bulk tests continue to assert
nonzero already-staged, skipped, clean, and blocked counts.

The mismatched-success case must not be retained or presented by the workspace
and must never produce promise copy.

Add `test_session_change_invalidates_last_action_before_refresh`:

1. complete a successful action and its normal refresh;
2. block the next status;
3. record `late.md`;
4. call `_refresh_session_changes()`;
5. assert the last-action static clears immediately and current status becomes
   stale before refresh finishes.

Update `test_hidden_action_summary_is_presented_after_reopen` to prove a
same-binding/repository/change-snapshot result survives hiding, reopening, and
its normal status-generation change.

Add `test_selected_root_change_clears_rows_and_last_action`: after ready rows
and a successful last action, call the real `set_root()` for another temporary
root and assert rows, selection, and last action clear.

Extend the repository-retrust test to assert old rows and last action are gone
before the replacement trust prompt is accepted.

Add `test_refresh_failure_keeps_stale_error_separate_from_last_action`: after a
proven action, make the next status task raise and assert the old result remains
under last action while current status becomes `STALE · ERROR`, rows remain,
and mutations are disabled.

- [ ] **Step 2: Run the feedback tests and verify RED**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_library_file_notes_git.py -k "action_summary or summary_counts or hidden_action_summary or session_change_invalidates or selected_root_change or repository_retrust or refresh_failure"
```

Expected: FAIL because one unkeyed string overwrites freshness, survives
authority/session changes, and lacks checked promise/recovery rules.

- [ ] **Step 3: Implement owner-snapshot-keyed last-action presentation**

Add one local immutable view record beside `_GitActionSummaryContext`:

```python
@dataclass(frozen=True, slots=True)
class _GitLastAction:
    binding: SessionBinding
    repository: RepositoryIdentity
    changes: tuple[SequencedSessionChange, ...]
    text: str
```

Replace `_git_action_detail` with `_git_last_action: _GitLastAction | None`.
Add helpers that derive the current key only from the owner's immutable
snapshot and compare all three fields. Do not include Git status generation or
staging ownership because the action and automatic postflight refresh
legitimately change them.

Capture binding, complete trusted `RepositoryIdentity`, and exact
`snapshot.changes` after draft flush/save/transition checks and immediately
around synchronous Stage/Unstage admission. Clear the prior presentation only
after a newer action is admitted. Pass the captured key into
`_render_git_action()`.

Retain/display a result only when the captured key still equals a fresh owner
snapshot. Validate/clear it from `_refresh_session_changes()`,
`_rehydrate_git_presentation()`, `_render_git_status()`, root publication, and
untrusted/unavailable discovery. A status-generation-only change must not
clear it.

Route Git blockers, trust loss, and discovery recovery through the panel's
current-status API. Route only checked action results through
`set_last_action()`. Replace `settle the draft` with the exact editor recovery
from the specification.

Change `_git_action_summary()` so a promise is emitted only when:

```python
service_reported_checked_success
and affected_group_count > 0
and action_key_still_matches
```

`service_reported_checked_success` is exactly
`result.state == "success"` under the existing `GitActionResult`/ADR-035
contract: the Git service is the sole authority and emits `success` only after
repository/`HEAD` postflight and Stage ownership publication or owned-baseline
Unstage restoration have all verified. Do not duplicate those Git/index safety
checks in the presentation layer. The workspace-owned proof is the fresh
binding, complete `RepositoryIdentity`, and exact session-change-key match; a
mismatch suppresses the entire obsolete result, not just its promise.

Use coalesced staged/unstaged group counts, correct singular/plural nouns, and
append only material nonzero bulk counts. Do not return `result.message` early;
preserve it beside counts and add state-specific recovery for blocked, stale,
error, uncertain, and zero-effect results. Keep the unabridged summary in the
workspace record; the panel's bounded display projection protects the complete
recovery phrase when diagnostic text is long.

When a different root binding is atomically published, clear the panel rows
and last-action presentation on the UI thread. Repository change already
passes through untrusted/unavailable rendering, which performs the same clear.

For checking, pass `retain_rows=True` only when the current owner snapshot
contains a prior status with the same binding generation and complete trusted
repository identity. The panel remains presentation-only.

- [ ] **Step 4: Run the feedback tests and verify GREEN**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 5: Write failing wide-editor quieting and narrow-toolbar tests**

Extend `test_workspace_retains_files_search_and_git_modes_with_back_focus` or
add
`test_wide_prepare_session_quiets_and_restores_editor_toolbars_without_remount`.
At `150×42`, open a note and retain the `TextArea` identity, body, cursor, and
selection. Enter Prepare session mode and assert:

- both `.file-notes-toolbar` rows are hidden;
- breadcrumb, save state, and the same editable `TextArea` remain visible;
- typing still changes the retained editor;
- Back restores both toolbars immediately without remount or draft loss.

Add `test_narrow_editor_actions_keep_complete_labels_at_40_by_20`. At `40×20`,
open a note so Editor is visible. For every visible editor action, assert its
region is inside the editor pane and its rendered label is complete. Explicitly
assert `Protect`, then toggle protected state and assert `Unprotect`.

- [ ] **Step 6: Run the editor layout tests and verify RED**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_library_file_notes_git.py -k "quiets_and_restores or narrow_editor_actions or retains_files_search_and_git_modes"
```

Expected: FAIL because wide Git mode leaves both editor toolbars active and
the first toolbar clips `Protect` at `40×20`.

- [ ] **Step 7: Implement retained toolbar quieting and three-column narrow layout**

In workspace-local `DEFAULT_CSS`:

```css
LibraryFileNotesWorkspace.-prepare-session-wide .file-notes-toolbar {
    display: none;
}

LibraryFileNotesWorkspace.-stack-editor-actions .file-notes-toolbar {
    layout: grid;
    grid-size: 3;
    grid-columns: 1fr 1fr 1fr;
    height: auto;
}

LibraryFileNotesWorkspace.-stack-editor-actions .file-notes-toolbar Button {
    width: 1fr;
}
```

At the end of responsive/navigator synchronization, toggle
`-prepare-session-wide` only when Git mode is visible and the workspace is not
narrow. Do not hide the path breadcrumb, save state, editor, or editor status,
and never recompose `_editor_widget`.

After responsive layout and after `_update_controls()` changes
`Protect`/`Unprotect`, schedule one `_sync_editor_action_layout()` call after
refresh. Compare the editor pane's real content width with each toolbar's
natural label-cell total (`cell_len(label) + 4` existing chrome per button).
Toggle `-stack-editor-actions` only when a row cannot fit. The deterministic
three-column grid yields two rows plus one row at `40×20`; do not fully stack
eight buttons vertically.

- [ ] **Step 8: Run the complete focused UI file and verify GREEN**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_library_file_notes_git.py
```

Expected: all 49 baseline tests plus the new TASK-1235 tests pass.

- [ ] **Step 9: Commit the workspace slice**

```bash
git add tldw_chatbook/Widgets/Library/library_file_notes_workspace.py Tests/UI/test_library_file_notes_git.py
git commit -m "fix(notes): keep prepare-session feedback current [TASK-1235]"
```

## Task 3: Focused Verification, Live UAT, Review, and Backlog Closeout

**Files:**

- Verify:
  `tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py`
- Verify:
  `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`
- Verify: `Tests/UI/test_library_file_notes_git.py`
- Modify:
  `backlog/tasks/task-1235 - Polish-File-Notes-prepare-session-for-commit-UX.md`

- [ ] **Step 1: Run focused automated verification**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_library_file_notes_git.py
../../.venv/bin/python -m compileall -q tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
../../.venv/bin/python -m ruff check tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py Tests/UI/test_library_file_notes_git.py
../../.venv/bin/python -m ruff format --check tldw_chatbook/Widgets/Library/library_file_notes_git_panel.py tldw_chatbook/Widgets/Library/library_file_notes_workspace.py Tests/UI/test_library_file_notes_git.py
git diff --check origin/dev...HEAD
```

Expected: every command exits 0 with no warnings introduced by TASK-1235.

- [ ] **Step 2: Run the bounded live terminal acceptance pass**

Create a disposable local Git repository outside the worktree with two tracked
Markdown notes and one unrelated tracked file. Stage an edit to the unrelated
file before launching Chatbook. Use a temporary, untracked Textual harness that
mounts the real `LibraryFileNotesWorkspace`, an in-memory
`FileNotesReplica`, and `build_file_notes_session_owner()` against the
disposable notes root. Do not add this harness to the repository.

Launch it in an isolated tmux server:

```bash
tmux -L task1235-uat new-session -d -s chatbook-uat -x 150 -y 42 "../../.venv/bin/python tmp/task1235_uat.py /private/tmp/<exact-created-repository>/notes"
tmux -L task1235-uat capture-pane -p -t chatbook-uat -S -200
```

Exercise only:

1. edit two note bodies and wait for `Saved`;
2. open `Prepare session for commit`, decline trust once, then accept;
3. verify the title, note verbs, separate current/last-action lines, and
   selected/bulk count labels;
4. Stage all and confirm the checked promise copy;
5. verify with `git diff --cached --name-only` that both session notes and the
   previously staged unrelated file are staged;
6. Unstage all and verify only the unrelated file remains staged;
7. resize/capture at `70×28`, `70×24`, and `40×20`;
8. at `40×20`, verify focused labels, fixed feedback, Escape focus restoration,
   and complete `Protect`/`Unprotect`;
9. at `150×42`, verify editor toolbars quiet during preparation, typing remains
   active, and Back restores the same editor/toolbars.

Resize and capture with:

```bash
tmux -L task1235-uat resize-window -t chatbook-uat -x 70 -y 28
tmux -L task1235-uat resize-window -t chatbook-uat -x 70 -y 24
tmux -L task1235-uat resize-window -t chatbook-uat -x 40 -y 20
tmux -L task1235-uat capture-pane -p -t chatbook-uat -S -200
```

Record concise observed results in TASK-1235 implementation notes. Stop the
isolated tmux server and delete only the exact temporary harness/repository
created for this pass.

- [ ] **Step 3: Request focused code and specification review**

Use `superpowers:requesting-code-review` with:

- TASK-1235 acceptance criteria;
- the approved design and this plan;
- `origin/dev...HEAD`;
- focused test/lint/compile/UAT evidence.

Address only concrete in-scope findings. Any behavior fix follows a new
red/green cycle. Re-run Step 1 and the affected UAT slice after corrections.

- [ ] **Step 4: Reconcile TASK-1235**

Check all eight acceptance criteria, add concise `## Implementation Notes`,
and set the task to Done through the Backlog CLI only after:

- focused tests, compile, Ruff, and diff checks pass;
- live UAT passes;
- review findings are resolved;
- no production changes remain uncommitted.

Implementation notes must name the two production widgets, focused test file,
promise truth condition, exact viewport coverage, unrelated-index preservation,
review result, and the existing ADRs. Reaffirm:

```text
ADR required: no
ADR path: N/A
Reason: presentation-only repair conforming to ADR-035/033/011/029.
```

- [ ] **Step 5: Commit closeout documentation**

```bash
git add 'backlog/tasks/task-1235 - Polish-File-Notes-prepare-session-for-commit-UX.md'
git commit -m "docs(notes): complete prepare-session UX [TASK-1235]"
git status --short --branch
```

Expected: the branch is clean and TASK-1235 is Done.
