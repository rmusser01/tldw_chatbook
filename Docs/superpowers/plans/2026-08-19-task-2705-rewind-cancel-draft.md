# TASK-2705 Rewind Draft Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume a successfully handled argument-free `/rewind` invocation without losing text typed after Enter or clearing a changed/replaced composer.

**Architecture:** Add one narrow argument-free `/rewind` branch at the existing Console send/command boundary. Keyboard sends keep the already-captured command stash separate; visible-Send cleanup is guarded by composer identity, `edit_serial`, generation, and dispatched text. `_console_command_rewind` reports whether it opened the modal so refusal and exception paths can preserve the invocation.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest/pytest-asyncio, Ruff, MyPy

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a localized command-draft cleanup bug fix within the existing Console/composer contracts; it changes no storage, service boundary, dependency, security policy, or long-lived architecture.

---

## File map

- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: add the argument-free `/rewind` dispatch exception, guarded visible-Send cleanup, rollback, and boolean modal-open result.
- Modify `Tests/UI/test_console_rewind_restore.py`: add mounted product-path and race regressions beside the existing rewind wiring coverage.
- Modify `Docs/User_Guide/console/branching-and-rewind.md`: remove the resolved TASK-2705 workaround.
- Modify `Docs/superpowers/specs/2026-08-19-task-2705-rewind-cancel-draft-design.md`: retain the independently reviewed focus correction and final design evidence.
- Modify `backlog/tasks/task-2705 - Console-rewind-Esc-cancel-leaves-command-text-in-composer.md`: track this plan, evidence, acceptance criteria, notes, and Done status.
- Modify this plan document: check steps as evidence is collected.

### Task 1: Prove and fix ordinary `/rewind` consumption

**Files:**
- Modify: `Tests/UI/test_console_rewind_restore.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:16790-16845`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:17374-17395`

- [ ] **Step 1: Add mounted helpers and failing command-route tests**

Add a synchronous Enter helper using the same real keypress seam as `test_console_send_draft_snapshot.py`:

```python
from textual.events import Key
from textual.widgets import Button

from tldw_chatbook.Widgets.Console import ConsoleComposerBar


def _press_enter_synchronously(console) -> None:
    console.on_key(Key(key="enter", character="\r"))
```

Add mounted tests over `ConsoleHarness` and `_seed_u1_a1_u2_a2` for these outcomes:

```python
@pytest.mark.asyncio
async def test_keyboard_rewind_cancel_consumes_command_and_preserves_late_text():
    # Mount a real ChatScreen, seed one active-path USER prompt, focus the
    # composer, load `/rewind`, deliver Enter synchronously, then insert
    # `next draft` before yielding to the message pump.
    # Assert ConsoleRewindModal is topmost before pressing Escape.
    # After dismissal assert draft_text() == "next draft" and focus is composer.


@pytest.mark.asyncio
async def test_visible_send_rewind_never_mind_consumes_command_and_focuses_composer():
    # Load `/rewind`, click the real #console-send-message, assert the real
    # modal opened and the composer is empty, select a row, click the real
    # #console-rewind-action-cancel, then assert disabled Send is rejected as
    # an opener and the existing fallback focuses the composer.


@pytest.mark.asyncio
async def test_rewind_restore_replaces_late_text_with_full_selected_prompt():
    # Open via synchronous Enter, insert post-Enter text, select a row, click
    # Restore, and assert the selected full prompt replaces the live draft.


@pytest.mark.asyncio
async def test_rewind_no_prompts_restores_keyboard_command_before_late_text():
    # With no USER rows: synchronous Enter, then insert `next`; assert no modal,
    # exact draft `/rewindnext`, and the existing warning.


@pytest.mark.asyncio
async def test_rewind_with_args_keeps_existing_restore_before_dispatch_behavior():
    # `/rewind anything` remains outside the cleanup branch. Assert its modal
    # opens and cancellation leaves the invocation in the composer.
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_console_rewind_restore.py \
  -k 'keyboard_rewind_cancel or visible_send_rewind or rewind_restore_replaces_late or rewind_no_prompts_restores or rewind_with_args'
```

Expected: the cancellation/consumption cases fail because `/rewind`
remains/restores in the composer. Restore replacement, refusal, and
non-empty-args cases are non-regression controls and should pass (or expose only
test-fixture errors that must be corrected before production edits).

- [ ] **Step 3: Implement the minimal successful/refused dispatch contract**

In `_send_console_message_from_visible_action`, before the ordinary command restore, recognize only:

```python
argument_free_rewind = (
    parse.kind == KIND_COMMAND
    and parse.name == REWIND_COMMAND_NAME
    and parse.args == ""
)
```

For that branch, do not restore a keyboard stash before calling the rewind
handler. Reset `_console_unknown_send_armed`, call `_console_command_rewind`,
restore the keyboard stash on `False`, clear the visible-Send invocation on
`True`, and return `False`. Leave every other command on the current
restore-before-dispatch path. This first slice is deliberately the smallest
ordinary-flow GREEN; Task 2 writes failing race/exception tests before adding
the final revision guard and rollback.

Change `_console_command_rewind` to return `False` after the existing no-row warning and `True` after the existing `push_screen` call:

```python
async def _console_command_rewind(self, parse: CommandParse) -> bool:
    ...
    if not rows:
        self.app_instance.notify("Nothing to rewind.", severity="warning")
        return False
    ...
    self.app.push_screen(ConsoleRewindModal(prompts=rows), callback=_apply_choice)
    return True
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command.

Expected: all selected tests pass; mounted assertions prove the modal actually opened before each dismissal.

- [ ] **Step 5: Run adjacent rewind behavior tests**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_rewind_modal.py \
  Tests/UI/test_console_rewind_restore.py
```

Expected: pass; Restore, Summarize, no-row warning, modal results, and active-path behavior remain unchanged.

- [ ] **Step 6: Commit the ordinary behavior slice**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_rewind_restore.py
git commit -m "fix(console): consume handled rewind drafts"
```

### Task 2: Harden rollback and stale-composer safety

**Files:**
- Modify: `Tests/UI/test_console_rewind_restore.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:16825-16850`

- [ ] **Step 1: Add failing rollback and stale-composer tests**

Add direct-through-mounted-screen tests that still exercise `_send_console_message_from_visible_action`:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["keyboard", "visible-send"])
async def test_rewind_modal_launch_failure_preserves_draft(source, monkeypatch):
    # Seed a USER row. Arrange either a real keyboard stash plus late text or a
    # live visible-Send draft. Monkeypatch app.push_screen to raise RuntimeError
    # only for ConsoleRewindModal. Await the real send route under pytest.raises
    # and assert the exact draft is preserved/restored.


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["identity", "edit-retype", "generation"])
async def test_visible_rewind_cleanup_preserves_a_changed_composer(
    mutation, monkeypatch
):
    # Replace `_console_command_rewind` with an async success seam that performs
    # exactly one mutation before returning True:
    # - identity: make `_console_composer_or_none` resolve a replacement;
    # - edit-retype: insert then delete one character so bytes match but
    #   edit_serial advances;
    # - generation: load the same bytes so generation advances.
    # Assert the live `/rewind` bytes remain and `_clear_console_composer_draft`
    # was not invoked.
```

- [ ] **Step 2: Run the safety tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_console_rewind_restore.py \
  -k 'modal_launch_failure_preserves or cleanup_preserves_a_changed_composer'
```

Expected: launch failure loses the keyboard stash and/or stale visible cleanup clears a changed composer under the basic Task 1 implementation.

- [ ] **Step 3: Add the minimal rollback and revision guard**

Use the established revision precedent already in `ChatScreen._submit_console_native_draft`:

```python
opening_composer = composer if stash is None else None
opening_revision = None
if opening_composer is not None:
    opening_revision = (
        opening_composer.edit_serial,
        opening_composer.capture_draft_snapshot().generation,
        draft,
    )

opened = False
try:
    opened = await self._console_command_rewind(parse)
finally:
    if not opened and composer is not None:
        composer.restore_stashed_draft(stash)

if opened and opening_composer is not None and opening_revision is not None:
    current = self._console_composer_or_none()
    current_snapshot = (
        current.capture_draft_snapshot() if current is opening_composer else None
    )
    if (
        current is opening_composer
        and current.edit_serial == opening_revision[0]
        and current_snapshot is not None
        and current_snapshot.generation == opening_revision[1]
        and current.draft_text() == opening_revision[2]
    ):
        self._clear_console_composer_draft()
```

Do not add a helper, token class, composer API, or generic command policy.

- [ ] **Step 4: Run the safety tests and verify GREEN**

Run the Step 2 command.

Expected: all parameterized cases pass, including same-text edit/retype.

- [ ] **Step 5: Re-run all TASK-2705 behavior tests**

Run the Task 1 Step 5 command plus:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_console_send_draft_snapshot.py
```

Expected: all pass; the general Enter snapshot/restore contract is unchanged.

- [ ] **Step 6: Commit the safety slice**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_rewind_restore.py
git commit -m "fix(console): preserve rewind drafts across launch races"
```

### Task 3: Documentation, verification, review, and task closeout

**Files:**
- Modify: `Docs/User_Guide/console/branching-and-rewind.md:158-159`
- Modify: `Docs/superpowers/specs/2026-08-19-task-2705-rewind-cancel-draft-design.md`
- Modify: `backlog/tasks/task-2705 - Console-rewind-Esc-cancel-leaves-command-text-in-composer.md`
- Modify: `Docs/superpowers/plans/2026-08-19-task-2705-rewind-cancel-draft.md`

- [ ] **Step 1: Remove the resolved User Guide workaround**

Delete only:

```markdown
- Esc-cancelling the Rewind menu leaves "/rewind" sitting in the composer —
  clear it with `Ctrl+U`. (task-2705)
```

- [ ] **Step 2: Run the bounded final test matrix**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_command_grammar.py \
  Tests/Chat/test_console_rewind_modal.py \
  Tests/UI/test_console_rewind_restore.py \
  Tests/UI/test_console_send_draft_snapshot.py \
  Tests/UI/test_console_modal_dismissal.py
```

Expected: pass, with only pre-existing dependency/deprecation warnings documented by exact message.

- [ ] **Step 3: Run targeted static checks**

Run:

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/UI/test_console_rewind_restore.py
../../.venv/bin/python -m ruff format --check \
  Tests/UI/test_console_rewind_restore.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/UI/Screens/chat_screen.py
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/UI/test_console_rewind_restore.py
git diff --check
```

Expected: no new diagnostics. If a large legacy file is baseline-red, prove exact provenance from `origin/dev` and do not rewrite unrelated lines.

- [ ] **Step 4: Request independent code review and address all P0-P2 findings**

Review the implementation commits against the approved design and task AC. Re-run only tests related to any correction.

- [ ] **Step 5: Complete task hygiene**

Check every AC only after evidence exists. Add concise Implementation Notes covering the command-route branch, rollback/stale guard, files, tests, static results, ADR decision, and review. Update this plan's checkboxes. If a new generalizable incident occurred, update the relevant `backlog/docs/lessons-*.md`; otherwise state that no lesson change was needed.

- [ ] **Step 6: Mark TASK-2705 Done through the exact CLI id**

```bash
backlog task 2705 --plain
backlog task edit 2705 -s Done
backlog task 2705 --plain
```

Expected: both reads resolve exactly TASK-2705; final status is Done and all ACs are checked.

**Step 7: Commit closeout documentation**

After Steps 1-6 are genuinely complete and checked, make the closeout commit.
This commit action is deliberately not a checkbox: checking it after the commit
would dirty the plan again, while pre-checking it would be false evidence.

```bash
git add \
  Docs/User_Guide/console/branching-and-rewind.md \
  Docs/superpowers/plans/2026-08-19-task-2705-rewind-cancel-draft.md \
  Docs/superpowers/specs/2026-08-19-task-2705-rewind-cancel-draft-design.md \
  'backlog/tasks/task-2705 - Console-rewind-Esc-cancel-leaves-command-text-in-composer.md'
git commit -m "docs(console): complete TASK-2705 rewind cleanup"
```

## Post-plan branch handoff

After the closeout commit, do not edit this plan again. Use
`superpowers:finishing-a-development-branch` to confirm a clean worktree,
summarize exact evidence, and present integration options. Do not push, open a
PR, or merge without the user's selected branch-finish option.
