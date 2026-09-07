# TASK-397 Command Palette Selection Race Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make fast keyboard selection execute the command visible under the command-palette highlight even while Textual providers are still returning results.

**Architecture:** Characterize Textual 8.2.8's clear/rebuild/highlight-reset race with a fake clock and gated providers. Add one `StableCommandPalette` compatibility subclass that cancels gathering only when a command-list action targets an actionable visible snapshot, then make `TldwCli` open that subclass through Textual's existing action contract.

**Tech Stack:** Python 3.11+, Textual 8.2.8, pytest/pytest-asyncio, Ruff, MyPy, GitHub CLI

**ADR required:** no

**ADR path:** N/A

**Reason:** This is a localized compatibility shim around an exactly pinned framework component. It preserves provider and application boundaries and introduces no durable storage, sync, security, dependency, or service-contract decision.

---

## File map

- Create `tldw_chatbook/UI/stable_command_palette.py`: the single app-owned compatibility subclass.
- Create `Tests/UI/test_command_palette_selection_race.py`: deterministic stock characterization, compatibility behavior, and construction tests.
- Modify `tldw_chatbook/app.py`: import the subclass and override only command-palette construction.
- Modify `backlog/tasks/task-397 - Command-palette-fast-DownEnter-can-dismiss-without-running-the-command.md`: track the plan, upstream issue, evidence, ACs, notes, and Done status.
- Modify `Docs/superpowers/specs/2026-08-20-task-397-command-palette-selection-race-design.md`: record settled upstream evidence only if implementation disproves a design assumption.
- Modify this plan document: check steps only after their evidence exists.

No provider, dependency-pin, keybinding, CSS, or general command-palette documentation file changes are planned.

### Task 1: Build the deterministic stock-Textual characterization

**Files:**
- Create: `Tests/UI/test_command_palette_selection_race.py`

- [x] **Step 1: Add a fake clock, gated providers, and callback recorder**

Define a minimal harness around real Textual palette widgets and workers:

```python
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable

import pytest
from textual.app import App, ComposeResult
from textual.command import CommandList, CommandPalette, Hit, Provider
from textual.widgets import Static
from textual.widgets.option_list import Option


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float = 1.0) -> None:
        self.now += seconds


class PaletteProbe:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.release_batch = asyncio.Event()
        self.release_late = asyncio.Event()
        self.batch_waiting = asyncio.Event()
        self.late_waiting = asyncio.Event()
        self.cancelled = asyncio.Event()

    def callback(self, name: str) -> Callable[[], None]:
        return lambda: self.calls.append(name)


PROBE: PaletteProbe


async def wait_event(event: asyncio.Event) -> None:
    await asyncio.wait_for(event.wait(), timeout=1.0)


async def wait_until(pilot, predicate, *, attempts: int = 50) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause()
    pytest.fail("mounted palette condition was not reached")


class ControlledProvider(Provider):
    async def search(self, query: str) -> AsyncIterator[Hit]:
        if query != "logs":
            return
        try:
            yield Hit(0.90, "first", PROBE.callback("first"))
            yield Hit(0.80, "second", PROBE.callback("second"))
            PROBE.batch_waiting.set()
            await PROBE.release_batch.wait()
            yield Hit(0.70, "batch", PROBE.callback("batch"))
            PROBE.late_waiting.set()
            await PROBE.release_late.wait()
            yield Hit(0.60, "late", PROBE.callback("late"))
        except asyncio.CancelledError:
            PROBE.cancelled.set()
            raise


class PaletteHarness(App[None]):
    COMMANDS = {ControlledProvider}

    def __init__(self, palette_type: type[CommandPalette]) -> None:
        super().__init__()
        self.palette_type = palette_type

    def compose(self) -> ComposeResult:
        yield Static("calling screen", id="calling-screen")

    def on_mount(self) -> None:
        self.push_screen(self.palette_type(id="--command-palette"))
```

Keep every provider gate bounded with `wait_event`; use `wait_until` for mounted
state rather than sleeping for the expected condition. Add palette state to the
failure message if the first RED needs more diagnostic detail.

- [x] **Step 2: Add a passing stock characterization test**

Patch the exact imported clock used by Textual:

```python
@pytest.mark.asyncio
async def test_stock_palette_refresh_resets_a_navigated_highlight(monkeypatch):
    global PROBE
    PROBE = PaletteProbe()
    clock = FakeClock()
    monkeypatch.setattr("textual.command.monotonic", clock)

    async with PaletteHarness(CommandPalette).run_test() as pilot:
        await pilot.press("l", "o", "g", "s")
        await wait_event(PROBE.batch_waiting)
        clock.advance()
        PROBE.release_batch.set()
        command_list = pilot.app.screen.query_one(CommandList)
        await wait_until(pilot, lambda: command_list.option_count == 3)
        assert PROBE.late_waiting.is_set()

        await pilot.press("down")
        assert command_list.highlighted == 1
        clock.advance()
        PROBE.release_late.set()
        await wait_until(pilot, lambda: command_list.option_count == 4)

        assert command_list.highlighted == 0
        await pilot.press("enter")
        await wait_until(pilot, lambda: not CommandPalette.is_open(pilot.app))
        await wait_until(pilot, lambda: PROBE.calls == ["first"])
        assert PROBE.calls == ["first"]
```

The exact hit counts may be adjusted only to match observed Textual queue ordering; the test must retain two visible commands, a still-pending provider, an explicit post-navigation refresh, and the highlight-reset assertion.

- [x] **Step 3: Run the stock characterization**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_command_palette_selection_race.py \
  -k stock_palette_refresh_resets
```

Expected: PASS on unmodified Textual 8.2.8, proving Down visibly chose `second`
but the forced refresh reset selection and Enter ran `first`. Record that the exact
no-command symptom did not reproduce under this deterministic ordering while the
wrong-command selection race did. Do not generalize that one result into ruling out
other live orderings.

- [x] **Step 4: Commit the diagnostic harness**

```bash
git add Tests/UI/test_command_palette_selection_race.py
git commit -m "test(ui): characterize Textual palette refresh race"
```

### Task 2: Freeze an actionable visible result snapshot

**Files:**
- Create: `tldw_chatbook/UI/stable_command_palette.py`
- Modify: `Tests/UI/test_command_palette_selection_race.py`

- [x] **Step 1: Add and run a failing API-presence test**

Add this test without importing the missing module at collection time:

```python
from importlib.util import find_spec


def test_stable_palette_api_exists() -> None:
    assert find_spec("tldw_chatbook.UI.stable_command_palette") is not None
```

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_command_palette_selection_race.py::test_stable_palette_api_exists
```

Expected: one assertion failure because the compatibility module does not exist.

- [x] **Step 2: Add the pass-through API and verify the API test GREEN**

Create `tldw_chatbook/UI/stable_command_palette.py` with only:

```python
"""Compatibility command palette with stable keyboard selection."""

from textual.command import CommandPalette


class StableCommandPalette(CommandPalette):
    """App-owned compatibility boundary for Textual command selection."""
```

Run the Step 1 node again.

Expected: PASS. This establishes a valid import/API boundary before behavioral RED.

- [x] **Step 3: Add failing stable-selection and early-navigation tests**

Import `StableCommandPalette` and add mounted tests that use the same fake clock and gates:

```python
from tldw_chatbook.UI.stable_command_palette import StableCommandPalette


@pytest.mark.asyncio
async def test_stable_palette_runs_the_navigated_command_once(monkeypatch):
    global PROBE
    PROBE = PaletteProbe()
    clock = FakeClock()
    monkeypatch.setattr("textual.command.monotonic", clock)

    app = PaletteHarness(StableCommandPalette)
    async with app.run_test() as pilot:
        await pilot.press("l", "o", "g", "s")
        await wait_event(PROBE.batch_waiting)
        clock.advance()
        PROBE.release_batch.set()
        command_list = app.screen.query_one(CommandList)
        await wait_until(pilot, lambda: command_list.option_count == 3)
        await wait_event(PROBE.late_waiting)

        await pilot.press("down")
        assert command_list.highlighted == 1
        clock.advance()
        PROBE.release_late.set()
        await wait_until(
            pilot,
            lambda: PROBE.cancelled.is_set() or command_list.option_count == 4,
        )
        assert PROBE.cancelled.is_set()
        assert command_list.option_count == 3
        assert command_list.highlighted == 1

        await pilot.press("enter")
        await wait_until(pilot, lambda: not CommandPalette.is_open(app))
        await wait_until(pilot, lambda: PROBE.calls == ["second"])
        assert PROBE.calls == ["second"]

        await pilot.pause()
        assert PROBE.calls == ["second"]
        assert not CommandPalette.is_open(app)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key", ["down", "up", "pageup", "pagedown", "ctrl+home", "ctrl+end"]
)
async def test_navigation_before_first_result_does_not_cancel_gathering(
    monkeypatch, key
):
    global PROBE
    PROBE = PaletteProbe()
    clock = FakeClock()
    monkeypatch.setattr("textual.command.monotonic", clock)

    app = PaletteHarness(StableCommandPalette)
    async with app.run_test() as pilot:
        await pilot.press("l", "o", "g", "s")
        await wait_event(PROBE.batch_waiting)
        command_list = app.screen.query_one(CommandList)
        assert command_list.option_count == 0

        await pilot.press(key)
        assert not PROBE.cancelled.is_set()
        clock.advance()
        PROBE.release_batch.set()
        await wait_until(pilot, lambda: command_list.option_count == 3)
        assert not PROBE.cancelled.is_set()


@pytest.mark.asyncio
async def test_navigation_during_stale_no_matches_does_not_cancel_new_query(
    monkeypatch,
):
    global PROBE
    PROBE = PaletteProbe()
    clock = FakeClock()
    monkeypatch.setattr("textual.command.monotonic", clock)

    app = PaletteHarness(StableCommandPalette)
    async with app.run_test() as pilot:
        palette = app.screen
        assert isinstance(palette, StableCommandPalette)
        command_list = palette.query_one(CommandList)
        command_list.clear_options().add_option(
            Option("No matches found", disabled=True, id=palette._NO_MATCHES)
        )
        palette._list_visible = True

        replacement_worker = palette._gather_commands("logs")
        palette._action_command_list("cursor_up")
        assert not replacement_worker.is_cancelled

        await wait_event(PROBE.batch_waiting)
        clock.advance()
        PROBE.release_batch.set()
        await wait_until(pilot, lambda: command_list.option_count == 3)
        assert not PROBE.cancelled.is_set()


@pytest.mark.asyncio
async def test_settled_multi_hit_selection_runs_once(monkeypatch):
    global PROBE
    PROBE = PaletteProbe()
    clock = FakeClock()
    monkeypatch.setattr("textual.command.monotonic", clock)

    app = PaletteHarness(StableCommandPalette)
    async with app.run_test() as pilot:
        await pilot.press("l", "o", "g", "s")
        await wait_event(PROBE.batch_waiting)
        clock.advance()
        PROBE.release_batch.set()
        await wait_event(PROBE.late_waiting)
        clock.advance()
        PROBE.release_late.set()
        command_list = app.screen.query_one(CommandList)
        await wait_until(pilot, lambda: command_list.option_count == 4)

        await pilot.press("down", "enter")
        await wait_until(pilot, lambda: not CommandPalette.is_open(app))
        await wait_until(pilot, lambda: PROBE.calls == ["second"])
        assert PROBE.calls == ["second"]


@pytest.mark.asyncio
async def test_escape_closes_without_running_a_command(monkeypatch):
    global PROBE
    PROBE = PaletteProbe()
    clock = FakeClock()
    monkeypatch.setattr("textual.command.monotonic", clock)

    app = PaletteHarness(StableCommandPalette)
    async with app.run_test() as pilot:
        await pilot.press("l", "o", "g", "s")
        await wait_event(PROBE.batch_waiting)
        clock.advance()
        PROBE.release_batch.set()
        await wait_until(
            pilot, lambda: app.screen.query_one(CommandList).option_count == 3
        )

        await pilot.press("escape")
        await wait_until(pilot, lambda: not CommandPalette.is_open(app))
        assert PROBE.calls == []
```

- [x] **Step 4: Run the behavioral tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_command_palette_selection_race.py \
  -k 'stable_palette or navigation_before or stale_no_matches or settled_multi or escape_closes'
```

Expected: the pass-through subclass fails the pending-provider stable-selection test because Textual refreshes/resets the acted-on snapshot. Early-navigation, settled, and Escape cases are non-regression controls and must pass or expose fixture errors before the override is added.

- [x] **Step 5: Implement the minimal compatibility override**

Create `tldw_chatbook/UI/stable_command_palette.py`:

```python
"""Compatibility command palette with stable keyboard selection."""

from textual.command import CommandList, CommandPalette


class StableCommandPalette(CommandPalette):
    """Freeze an actionable result snapshot when keyboard selection begins."""

    def _action_command_list(self, action: str) -> None:
        command_list = self.query_one(CommandList)
        if (
            self._list_visible
            and command_list.option_count
            and command_list.get_option_at_index(0).id != self._NO_MATCHES
        ):
            self._cancel_gather_commands()
        super()._action_command_list(action)
```

Do not add timers, provider coordination, callback invocation, configuration, or copied Textual code.

- [x] **Step 6: Run the Task 2 tests and verify GREEN**

Run the Step 4 command.

Expected: all selected tests pass. The stable selection test must prove callback identity/count and cancellation; the two early-navigation families must prove later results actually appear.

- [x] **Step 7: Run the stock/stable characterization together**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_command_palette_selection_race.py
```

Expected: PASS. The stock row resets after the forced late refresh; the stable row cancels before that refresh and runs the acted-on callback once.

- [x] **Step 8: Commit the compatibility slice**

```bash
git add tldw_chatbook/UI/stable_command_palette.py \
  Tests/UI/test_command_palette_selection_race.py
git commit -m "fix(ui): stabilize command palette keyboard selection"
```

### Task 3: Make TldwCli open the stable palette

**Files:**
- Modify: `tldw_chatbook/app.py:80-90`
- Modify: `tldw_chatbook/app.py:5173-8300`
- Modify: `Tests/UI/test_command_palette_selection_race.py`

- [x] **Step 1: Add failing construction-contract tests**

Use a minimal fake app and patch the inherited open-state check:

```python
def test_tldw_cli_constructs_the_stable_palette(monkeypatch):
    app = MagicMock()
    app.use_command_palette = True
    monkeypatch.setattr(StableCommandPalette, "is_open", lambda _app: False)

    assert "action_command_palette" in TldwCli.__dict__
    TldwCli.action_command_palette(app)

    palette = app.push_screen.call_args.args[0]
    assert type(palette) is StableCommandPalette
    assert palette.id == "--command-palette"


@pytest.mark.parametrize("enabled, already_open", [(False, False), (True, True)])
def test_tldw_cli_does_not_open_a_duplicate_or_disabled_palette(
    monkeypatch, enabled, already_open
):
    app = MagicMock()
    app.use_command_palette = enabled
    monkeypatch.setattr(
        StableCommandPalette, "is_open", lambda _app: already_open
    )
    TldwCli.action_command_palette(app)
    app.push_screen.assert_not_called()
```

- [x] **Step 2: Run the construction tests and verify RED**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_command_palette_selection_race.py \
  -k tldw_cli
```

Expected: FAIL because `TldwCli` inherits Textual's stock action and does not define the stable construction contract.

- [x] **Step 3: Add the narrow TldwCli override**

Import the compatibility class with the other app UI dependencies:

```python
from tldw_chatbook.UI.stable_command_palette import StableCommandPalette
```

Add one action method to `TldwCli`:

```python
def action_command_palette(self) -> None:
    """Open the app's stable Textual command palette."""
    if self.use_command_palette and not StableCommandPalette.is_open(self):
        self.push_screen(StableCommandPalette(id="--command-palette"))
```

- [x] **Step 4: Run construction and mounted behavior tests**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_command_palette_selection_race.py
```

Expected: PASS.

- [x] **Step 5: Run adjacent palette/provider regressions**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/UI/test_command_palette_basic.py \
  Tests/UI/test_command_palette_providers.py \
  Tests/UI/test_command_palette_shell_routes.py \
  Tests/UI/test_command_palette_selection_race.py
```

Expected: the branch-only race file passes. If one of the three pre-existing files
fails, create a disposable detached `origin/dev` worktree and rerun only those three
pre-existing files there; the new race file cannot be part of a baseline command
because it does not exist on `origin/dev`:

```bash
git worktree add --detach /private/tmp/tldw-task397-origin-dev origin/dev
```

From `/private/tmp/tldw-task397-origin-dev`, run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q \
  Tests/UI/test_command_palette_basic.py \
  Tests/UI/test_command_palette_providers.py \
  Tests/UI/test_command_palette_shell_routes.py
```

After recording exact parity, remove the disposable worktree from the repository root:

```bash
git worktree remove /private/tmp/tldw-task397-origin-dev
```

Do not broaden to unrelated test directories.

- [x] **Step 6: Run focused static checks**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/UI/stable_command_palette.py \
  Tests/UI/test_command_palette_selection_race.py \
  tldw_chatbook/app.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/UI/stable_command_palette.py \
  Tests/UI/test_command_palette_selection_race.py
../../.venv/bin/python -m mypy --follow-imports=skip \
  tldw_chatbook/UI/stable_command_palette.py
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/UI/stable_command_palette.py \
  tldw_chatbook/app.py
git diff --check
```

Expected: new/changed code passes. If whole-file `app.py` reports inherited
diagnostics, run the exact Ruff command against `app.py` in the disposable
`origin/dev` worktree described in Step 5 and compare only that pre-existing file.
Do not include the new module/test in the baseline command and do not reformat
unrelated `app.py` lines.

- [x] **Step 7: Commit application integration**

```bash
git add tldw_chatbook/app.py Tests/UI/test_command_palette_selection_race.py
git commit -m "fix(app): use stable command palette selection"
```

### Task 4: Report upstream and close TASK-397

**Files:**
- Modify: `backlog/tasks/task-397 - Command-palette-fast-DownEnter-can-dismiss-without-running-the-command.md`
- Modify: this plan document
- Modify only if assumptions changed: `Docs/superpowers/specs/2026-08-20-task-397-command-palette-selection-race-design.md`

- [x] **Step 1: Search upstream before filing**

```bash
gh issue list --repo Textualize/textual --state all \
  --search 'command palette refresh highlight selection race' --limit 100
```

Expected: either identify an exact existing issue to link or establish that a new report is warranted. Record the search terms and result; do not file a duplicate.

Evidence: searched `command palette refresh highlight selection race`,
`CommandPalette highlight reset`, `command palette wrong command`, and
`command palette async provider selection` across open and closed Textual issues.
The only nearby reports, #4705 and #3714, concern unrelated help/scroll/order and
duplicate-ID clearing defects, so a new report was warranted.

- [x] **Step 2: File or link the upstream issue**

If no exact issue exists, use `gh issue create --repo Textualize/textual` with a standalone reproduction derived from the passing stock characterization. Include Textual `8.2.8`, Python version, fake-clock/gated ordering, exact expected/actual highlight and callback identity, source behavior, and the app workaround. State whether the original no-command symptom reproduced or whether only the narrower highlight reset was confirmed.

Expected: a durable GitHub issue URL. If repository permissions prevent creation,
preserve the complete ready-to-file body in TASK-397, leave AC #2 unchecked and the
task In Progress, commit the implemented mitigation/evidence if otherwise ready, and
stop with an explicit filing handoff. Do not run the Done transition or claim task
completion until a durable existing or newly filed issue URL is linked.

Evidence: filed [Textual issue #6701](https://github.com/Textualize/textual/issues/6701)
with Textual 8.2.8, Python 3.12.11, the fake-clock/gated-provider reproduction,
expected and actual callback identities, the batch refresh mechanism, and the local
workaround. The report explicitly says the deterministic run confirmed a
wrong-command race but did not reproduce the original no-command symptom.

- [x] **Step 3: Run the final related verification matrix**

Repeat Task 3 Steps 4-6 on the settled branch. Capture exact pass/fail counts, warnings, durations, and any identical `origin/dev` baseline proof.

Evidence: the four exact related palette files passed with `90 passed, 1 warning in
9.13s`; the warning was the existing Requests dependency-version warning. Focused
Ruff, Ruff format, MyPy, compileall, and `git diff --check` all passed. No baseline
comparison was needed because no focused check failed.

- [x] **Step 4: Request independent cumulative review**

Use `superpowers:requesting-code-review` over
`$(git merge-base origin/dev HEAD)..HEAD`. During closeout that merge base resolved to
`1bf7f234e`; `origin/dev` had advanced by 93 commits after the worktree was created,
so literal `origin/dev..HEAD` included unrelated upstream changes and was not a valid
TASK-397 review range. Require review of the protected Textual seam,
actionable/no-match guard, deterministic non-vacuity, callback exactly-once behavior,
app construction, upstream report honesty, scope, and task evidence. Resolve all
P0-P2 findings before closeout.

Evidence: independent cumulative review approved the protected compatibility seam,
guards, non-vacuous behavior tests, exactly-once callback contract, app construction,
upstream report, scope, and task evidence. The review's sole P2 corrected the stale
literal `origin/dev..HEAD` instruction to the merge-base-scoped range above;
re-review approved with no open P0-P2 findings.

- [x] **Step 5: Complete documentation and task hygiene**

Update TASK-397 with:

- all ACs checked only from evidence;
- the upstream issue URL or honest ready-to-file fallback;
- `## Implementation Plan` linking this executable plan;
- concise `## Implementation Notes` listing the shim, app integration, tests, tradeoff, commits, and verification;
- ADR `no` rationale; and
- whether a lessons entry was warranted.

Check completed plan steps. Then resolve the exact task and set it Done via CLI only after every DoD item is satisfied:

```bash
backlog task 397 --plain
backlog task edit 397 -s Done
backlog task 397 --plain
```

Evidence: TASK-397 now records all checked ACs, the reviewed plan, concise
implementation/verification notes, issue #6701, the ADR-no rationale, and the
no-lessons/no-user-guide rationale. `backlog task edit 397 -s Done` completed, and
`backlog task 397 --plain` resolves the exact task as Done with all three ACs checked.

- [x] **Step 6: Commit closeout documentation**

```bash
git add \
  'backlog/tasks/task-397 - Command-palette-fast-DownEnter-can-dismiss-without-running-the-command.md' \
  Docs/superpowers/plans/2026-08-20-task-397-command-palette-selection-race.md \
  Docs/superpowers/specs/2026-08-20-task-397-command-palette-selection-race-design.md
git commit -m "docs(ui): complete TASK-397 palette race mitigation"
```

Evidence: the task record and executable plan were committed with the prescribed
`docs(ui): complete TASK-397 palette race mitigation` message; the unchanged design
spec required no closeout edit because implementation did not disprove an assumption.

- [x] **Step 7: Verify final repository state**

```bash
git status --short
git diff --check "$(git merge-base origin/dev HEAD)..HEAD"
git log --oneline --decorate -6
backlog task 397 --plain
```

Expected: clean worktree; the merge-base-scoped diff contains only TASK-397 commits;
TASK-397 uniquely resolves to Done with every AC checked; no implementation or
review evidence is overstated. The merge-base range is required because `origin/dev`
advanced by 93 commits after branch creation; a literal `origin/dev..HEAD` diff would
mix those unrelated upstream changes into the closeout check.

Evidence: `git status --short` was empty, merge-base-scoped `git diff --check` and
`git show --check` passed, the log contained only the eight scoped TASK-397 commits,
and `backlog task 397 --plain` resolved the task as Done with all three ACs checked.
