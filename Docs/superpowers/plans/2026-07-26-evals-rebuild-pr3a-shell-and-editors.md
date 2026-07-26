# Evals Rebuild PR 3a — Shell and Editors

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Evals card hub with a Console-styled three-pane workbench you can author and configure a bench in. The results grid follows in PR 3b.

**Architecture:** `EvalsScreen` becomes a normal `BaseAppScreen` with the house three-pane workbench — library rail, detail pane, inspector — driven by *selection state*, not by mounting `Screen` objects inside a `Container`. That architecture is why the current hub renders an empty body.

**This is PR 3a of two.** It delivers the runner's preflight results, retires the hub, and builds the shell, the bench editor, and the snippet editor. PR 3b adds the results grid, its lenses, empty states, and the stylesheet cleanup. Splitting here keeps each half independently reviewable: 3a is a screen you can author in, 3b is a screen you can read results in.

**Tech Stack:** Python 3.11+, Textual, pytest. No new dependencies.

## Global Constraints

- Base branch: `origin/dev` at `006b31ed2` (PR 1 #922 and PR 2 #924 both merged).
- **A git worktree has no `.venv`.** Use the primary checkout's interpreter with cwd set to the worktree:
  ```bash
  cd <worktree> && /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest ...
  ```
  Verify `python -c "import tldw_chatbook; print(tldw_chatbook.__file__)"` resolves **inside** the worktree before the first test run, or tests verify the wrong tree.
- **`pytest Tests/UI` cannot run in one call** — 5,250+ tests, ~51 minutes, exceeds a hard 10-minute per-call cap. Per-task gate:
  ```bash
  python -m pytest Tests/UI --collect-only -q          # 0 collection errors
  python -m pytest Tests/UI/test_evals_screen.py Tests/UI/test_evals_deletion_guard.py -q
  python -m pytest Tests/Evals -q                       # engine must stay green
  python -c "import tldw_chatbook.app"
  ```
  Collection alone is not sufficient — some UI tests read source files off disk by path and fail only at runtime. Full-suite runs are the controller's job.
- **Do not modify `tldw_chatbook/Evals/word_bench/` except in Task 1.** The engine is merged and reviewed.
- **`Tests/UI/test_evals_deletion_guard.py` and its 19-entry tuples must stay green.** PR 3 extends them (Task 2), never rewrites them.
- Design-system contract: use `.ds-*` shared classes and `$ds-*` tokens; assert **readable status text**, never colours; support `.density-compact` and `.density-comfortable`; `ds-status-badge` colour lives in app-tier CSS, never widget `DEFAULT_CSS`.
- The `timeout` command is not available. Do not push or open a PR without explicit authorization.

## Facts this plan is built on

**The hub is already broken.** Verified during PR 1 by capturing the screen on the branch and on a baseline worktree: `DestinationHeader` and `LabModeStrip` render, the body is **empty**, on both. `EvalsWindowV3` mounts fine in isolation (`EvalNavigationScreen` plus 8 buttons), so the failure is shell integration — Textual `Screen` objects mounted inside a `Container`. **There is no working behaviour to preserve parity with.**

**Top-1 is an unstable reading.** Two identical requests seconds apart returned the top two tokens in opposite rank order, magnitudes stable to ~0.002. The Top-1 lens must mark near-ties rather than presenting a bare winner.

**Divergence is not a bound.** PR 2's whole-branch review disproved the original claim. The number is comparable and reproducible; the grid must **not** render it with a leading `≥`.

**The mode-strip slot is taken.** `LabModeStrip` (Models | Speech | Evals) occupies it, so Evals-internal navigation is the library rail, not a second strip.

## File Structure

| File | Responsibility |
|---|---|
| `tldw_chatbook/Evals/word_bench/runner.py` | **Modified** (Task 1): return preflight results |
| `tldw_chatbook/UI/Screens/evals_screen.py` | **Rewritten**: three-pane shell, selection state |
| `tldw_chatbook/UI/Evals/library_rail.py` | Benches / Datasets / Runs, collapsible, counts |
| `tldw_chatbook/UI/Evals/bench_editor.py` | Bench detail + target table + readiness |
| `tldw_chatbook/UI/Evals/snippet_editor.py` | Snippet table, whitespace flags, import |
| `tldw_chatbook/UI/Evals/results_grid.py` | The grid, lenses, baseline |
| `tldw_chatbook/UI/Evals/inspector.py` | Readiness / stats / run meta / focused-cell detail |
| `tldw_chatbook/UI/Evals/evals_state.py` | Selection state and the screen's view model |
| `tldw_chatbook/css/features/_evals.tcss` | New sheet, `$ds-*` only |

**Deleted:** `UI/Evals/navigation/`, `UI/Evals/screens/`, `UI/Evals/widgets/`, `evals_window_v3.py`, `UI/evals_window_v2.py` (~2,700 lines), and the 12 Evals-only selectors in `_evaluation_unified.tcss` (verified still unused by surviving code).

---

### Task 1: Return preflight results from the runner (TASK-703)

The engine computes a `PreflightResult` per target and discards everything except `.canary`. Without `state`, `k_returned`, and `detail`, this screen cannot render the readiness badges, recovery callouts, or effective-K header the spec requires — it would have to re-run preflight and might get a different verdict than the run used.

**Files:**
- Modify: `tldw_chatbook/Evals/word_bench/runner.py`, `tldw_chatbook/Evals/word_bench/storage.py`
- Test: `Tests/Evals/word_bench/test_runner.py`, `Tests/Evals/word_bench/test_storage.py`

**Interfaces:**
- Produces: `WordBenchRunner.run(...) -> RunOutcome`, a frozen dataclass with `group_id: str` and `preflight: dict[str, PreflightResult]`. The snapshot gains a `preflight` key; `load_grid`'s returned dict gains `preflight: dict[str, PreflightResult]`.

- [ ] **Step 1: Write the failing tests**

Add to `Tests/Evals/word_bench/test_runner.py`:

```python
@pytest.mark.asyncio
async def test_run_returns_preflight_results_per_target(db, config, targets, snippets):
    """PR 3 renders readiness from these; re-running preflight could disagree."""
    task_id = save_bench(db, config)
    runner = WordBenchRunner(db, lambda t: FakeClient([], canary="degenerate"))
    outcome = await runner.run(config, targets, snippets, task_id)

    assert outcome.group_id
    assert set(outcome.preflight) == {t.id for t in targets}
    for result in outcome.preflight.values():
        assert result.state == "ok"
        assert result.canary == "degenerate"
        assert result.is_warned is True
```

Add to `Tests/Evals/word_bench/test_storage.py`:

```python
def test_snapshot_carries_preflight_so_a_reloaded_grid_can_explain_a_column(
    db, config, targets, snippets
):
    """A grid opened next week must still say why a column is empty, without
    re-contacting the provider."""
    from tldw_chatbook.Evals.word_bench.models import PreflightResult

    task_id = save_bench(db, config)
    preflight = {
        targets[0].id: PreflightResult(state="ok", k_returned=20, canary="pass"),
        targets[1].id: PreflightResult(
            state="unreachable", k_returned=None, canary="unchecked",
            detail="connection refused",
        ),
    }
    group_id, _ = create_run_group(
        db, task_id, config, targets, snippets, preflight=preflight
    )

    grid = load_grid(db, group_id)
    assert grid["preflight"][targets[1].id].state == "unreachable"
    assert grid["preflight"][targets[1].id].status_label == "Unavailable"
    assert grid["preflight"][targets[1].id].detail == "connection refused"
```

- [ ] **Step 2: Run them and confirm they fail**

```bash
cd /private/tmp/tldw-evals-pr3
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Evals/word_bench -q
```

Expected: `AttributeError` on `outcome.group_id` (run returns a bare str), and `TypeError` on the unexpected `preflight=` kwarg.

- [ ] **Step 3: Implement**

In `runner.py`, add above `WordBenchRunner`:

```python
@dataclass(frozen=True)
class RunOutcome:
    """What a run produced, beyond the cells themselves.

    ``preflight`` is returned rather than discarded because the screen renders
    readiness badges, recovery callouts, and the effective-K header from it.
    Re-running preflight to recover this would double the canary calls and
    could report a verdict the run itself never saw.
    """

    group_id: str
    preflight: dict[str, PreflightResult]
```

Change `run()` to collect `results[target.id] = result` in the preflight loop, pass `preflight=results` to `create_run_group`, and return `RunOutcome(group_id=group_id, preflight=results)` on both the normal and cancelled paths.

In `storage.py`, give `create_run_group` a `preflight: Optional[Mapping[str, PreflightResult]] = None` parameter, serialize it into the snapshot as plain dicts, and have `load_grid` rehydrate it into `PreflightResult` objects under a `"preflight"` key (defaulting to `{}` for run groups written before this change).

- [ ] **Step 4: Confirm green**

```bash
python -m pytest Tests/Evals/word_bench -q
```

Expected: all pass, two more than before.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Evals/word_bench/runner.py tldw_chatbook/Evals/word_bench/storage.py Tests/Evals/word_bench/
git commit -m "feat(evals): return preflight results from the word bench runner

TASK-703. run() computed a PreflightResult per target and discarded
everything but the canary, so the screen could not render readiness
without re-running preflight and risking a different verdict than the
run used. Now returns a RunOutcome and snapshots the verdicts, so a
grid reopened later still explains why a column is empty."
```

---

### Task 2: Retire the card hub

**Files:**
- Delete: `tldw_chatbook/UI/Evals/navigation/`, `tldw_chatbook/UI/Evals/screens/`, `tldw_chatbook/UI/Evals/widgets/`, `tldw_chatbook/UI/Evals/evals_window_v3.py`, `tldw_chatbook/UI/Evals/README.md`, `tldw_chatbook/UI/evals_window_v2.py`
- Modify: `Tests/UI/test_evals_deletion_guard.py`, `Tests/UI/test_non_obscuring_focus_contract.py`, `tldw_chatbook/UI/Screens/evals_screen.py`, `tldw_chatbook/app.py`

**Interfaces:**
- Produces: the guard's tuples grow from 19 to 25 entries. Task 3 replaces `evals_screen.py`'s body; this task only reduces it to a stub that renders the header and strip with an empty panel.

- [ ] **Step 1: Confirm the reachability gate**

```bash
cd /private/tmp/tldw-evals-pr3
for m in eval_nav_screen nav_bar breadcrumbs quick_test evaluation_browser progress_dashboard evals_window_v3 evals_window_v2; do
  hits=$(grep -rn --include="*.py" -E "(from|import)[[:space:]]+[A-Za-z0-9_.]*\b${m}\b" tldw_chatbook/ Tests/ \
    | grep -v -E "tldw_chatbook/UI/Evals/|tldw_chatbook/UI/evals_window_v2\.py")
  echo "${m} :: ${hits:-NONE}"
done
```

Expected: only `evals_window_v3` and `evals_window_v2` print hits, from `UI/Screens/evals_screen.py` and `UI/Evals/__init__.py`. Anything else — **stop and report BLOCKED**.

- [ ] **Step 2: Extend the deletion guard**

Append these six paths to `REMOVED_MODULES` and their stems to `REMOVED_STEMS` in `Tests/UI/test_evals_deletion_guard.py`, preserving the existing 19 entries and their order:

```python
    "tldw_chatbook/UI/Evals/evals_window_v3.py",
    "tldw_chatbook/UI/evals_window_v2.py",
    "tldw_chatbook/UI/Evals/navigation/eval_nav_screen.py",
    "tldw_chatbook/UI/Evals/navigation/nav_bar.py",
    "tldw_chatbook/UI/Evals/screens/quick_test.py",
    "tldw_chatbook/UI/Evals/widgets/progress_dashboard.py",
```

Run it and confirm the six new path cases **fail** (files still exist) while the stem cases pass.

- [ ] **Step 3: Reduce the screen to a stub and delete the hub**

Replace `evals_screen.py`'s body so it renders the `DestinationHeader`, `LabModeStrip`, and an empty `.ds-panel` — and **remove the `escape` and `1`-`6` bindings**, which existed only for the hub. Then:

```bash
git rm -r tldw_chatbook/UI/Evals/navigation tldw_chatbook/UI/Evals/screens tldw_chatbook/UI/Evals/widgets
git rm tldw_chatbook/UI/Evals/evals_window_v3.py tldw_chatbook/UI/Evals/README.md tldw_chatbook/UI/evals_window_v2.py
```

Also remove `EvalsWindowV3` from `app.py`'s container list and the `"evals-window"` entry from its window-id list. Re-resolve both by symbol — line numbers drift.

- [ ] **Step 4: Fix the contract test's dangling subject**

`Tests/UI/test_non_obscuring_focus_contract.py` reads `EVAL_NAV_SCREEN` off disk by path. That file is now gone, so delete the constant and `test_evals_navigation_card_focus_is_non_obscuring_and_ordered_after_type_borders`. **Keep** `EVALUATION_UNIFIED` — that stylesheet survives until Task 8.

- [ ] **Step 5: Verify**

```bash
python -m pytest Tests/UI/test_evals_deletion_guard.py Tests/UI/test_non_obscuring_focus_contract.py -q
python -m pytest Tests/UI --collect-only -q | tail -3
python -c "import tldw_chatbook.app; print('OK')"
```

Expected: guard 50 passed; the contract file's 9 pre-existing CSS-bundle failures unchanged; 0 collection errors.

- [ ] **Step 6: Commit**

```bash
git add -u && git add Tests/UI/test_evals_deletion_guard.py
git commit -m "refactor(evals): retire the card hub

~2,700 lines. The hub never rendered inside the app shell -- it mounted
Textual Screen objects inside a Container, and PR 1 confirmed by
before/after capture that its body was empty on dev. Extends the
deletion guard from 19 to 25 entries."
```

---

### Task 3: Screen shell and selection state

**Files:**
- Create: `tldw_chatbook/UI/Evals/evals_state.py`, `tldw_chatbook/UI/Evals/library_rail.py`, `tldw_chatbook/css/features/_evals.tcss`
- Modify: `tldw_chatbook/UI/Screens/evals_screen.py`, `tldw_chatbook/css/build_css.py`
- Test: `Tests/UI/test_evals_screen.py`

**Interfaces:**
- Produces:
  - `EvalsSelection` — frozen dataclass with `kind: Literal["none","bench","classic","dataset","run_group"]` and `id: str | None`
  - `EvalsViewModel` — loads benches, datasets, and run groups from `EvalsDB`; exposes `benches()`, `classic_tasks()`, `datasets()`, `run_groups()`
  - `LibraryRail` widget posting `EvalsSelectionChanged(selection)`
  - Stable IDs: `#evals-shell`, `#evals-workbench`, `#evals-library-pane`, `#evals-detail-pane`, `#evals-inspector-pane`, `#evals-primary-action`

- [ ] **Step 1: Write the failing test**

`Tests/UI/test_evals_screen.py`:

```python
"""Evals screen shell. The old hub rendered an empty body because it mounted
Screen objects inside a Container; these tests pin that the replacement
actually puts widgets on screen."""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen


@pytest.mark.asyncio
async def test_screen_mounts_the_three_pane_workbench(evals_app):
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen
        assert screen.query_one("#evals-workbench")
        assert screen.query_one("#evals-library-pane")
        assert screen.query_one("#evals-detail-pane")
        assert screen.query_one("#evals-inspector-pane")


@pytest.mark.asyncio
async def test_workbench_body_is_not_empty(evals_app):
    """The regression that motivated this PR: the hub rendered header and
    strip with nothing beneath."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        pane = evals_app.screen.query_one("#evals-library-pane")
        assert list(pane.children), "library pane rendered no children"


@pytest.mark.asyncio
async def test_library_rail_shows_three_sections_with_counts(evals_app):
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        labels = [
            w.renderable.plain if hasattr(w.renderable, "plain") else str(w.renderable)
            for w in evals_app.screen.query(".evals-rail-section-label")
        ]
        joined = " ".join(labels)
        for section in ("Benches", "Datasets", "Runs"):
            assert section in joined


@pytest.mark.asyncio
async def test_primary_action_names_its_object_when_a_bench_is_selected(
    evals_app, seeded_bench
):
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=seeded_bench)
        await pilot.pause()
        action = evals_app.screen.query_one("#evals-primary-action")
        assert "loaded-nouns" in str(action.label)


@pytest.mark.asyncio
async def test_primary_action_is_disabled_with_a_reason_when_nothing_is_selected(
    evals_app,
):
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        action = evals_app.screen.query_one("#evals-primary-action")
        assert action.disabled is True
        assert action.tooltip, "a disabled primary action must say why"


@pytest.mark.asyncio
async def test_escape_and_bare_digits_are_no_longer_bound(evals_app):
    """Both existed only for the retired card hub."""
    bound = {b.key for b in EvalsScreen.BINDINGS}
    assert "escape" not in bound
    assert not bound & {"1", "2", "3", "4", "5", "6"}
```

Add an `evals_app` fixture in `Tests/UI/conftest.py` (or a local one) that builds a minimal Textual `App` hosting `EvalsScreen` against a `:memory:` `EvalsDB`, and a `seeded_bench` fixture creating one word bench via `storage.save_bench`. Follow the existing UI-test harness conventions in that directory rather than inventing a new one.

- [ ] **Step 2: Confirm RED**, then implement `evals_state.py`, `library_rail.py`, and the rewritten `evals_screen.py`.

The screen composes exactly the house pattern:

```python
    def compose_content(self) -> ComposeResult:
        with Vertical(id="evals-shell"):
            yield DestinationHeader(
                WorkbenchHeaderState(
                    title="Evals",
                    subtitle="Run and review evaluation jobs.",
                    status="ready",
                ),
                id="evals-destination-header",
            )
            yield LabModeStrip(active_route="evals", id="lab-mode-strip")
            with Horizontal(
                id="evals-workbench", classes="ds-panel destination-workbench"
            ):
                yield LibraryRail(
                    self._view_model,
                    id="evals-library-pane",
                    classes="destination-workbench-pane",
                )
                yield Vertical(
                    id="evals-detail-pane", classes="destination-workbench-pane"
                )
                yield Vertical(
                    id="evals-inspector-pane",
                    classes="destination-workbench-pane ds-inspector",
                )
```

**No `Screen` subclass may be mounted inside any of these containers.** Detail and inspector content is swapped by removing and mounting plain widgets on selection change. That rule is the entire point of this task.

Register `_evals.tcss` in `build_css.py`'s manifest and regenerate the bundle — never hand-edit `tldw_cli_modular.tcss`.

- [ ] **Step 3: Verify and commit**

```bash
python -m pytest Tests/UI/test_evals_screen.py -q
python -m pytest Tests/UI --collect-only -q | tail -3
python -c "import tldw_chatbook.app; print('OK')"
```

```bash
git add tldw_chatbook/UI/Evals/ tldw_chatbook/UI/Screens/evals_screen.py tldw_chatbook/css/ Tests/UI/
git commit -m "feat(evals): Console-styled three-pane Evals workbench

Selection state replaces the hand-rolled Screen-inside-Container stack,
so shell Escape works normally and the body actually renders."
```

---

### Task 4: Bench editor and readiness inspector

**Files:** create `bench_editor.py`, `inspector.py`; test `Tests/UI/test_evals_bench_editor.py`

**Interfaces:** consumes `EvalsSelection`, `word_bench.storage.load_bench`, `word_bench.models.PreflightResult`. Produces `BenchEditor` and `EvalsInspector` widgets, and `#evals-run-bench` as the run control.

Selecting a word bench shows name, dataset, prompt mode, top-K, probes, and a target table. The inspector shows per-target readiness, a call/time estimate, and the run action.

**Requirements the tests must pin:**

- Readiness badges use the **contract's readable labels** — `Ready`, `Unavailable`, `Blocked` — from `PreflightResult.status_label`. Assert the text; never assert a colour.
- A **warned** target (canary `degenerate`) renders `Ready` **plus** a `.ds-recovery-callout` naming the target and what it produced. A warned target is runnable; the callout explains, it does not block.
- A `Blocked` target renders a callout with owner, problem, and next action.
- Selecting a **classic** task shows a read-only detail with its run history and the sentence *"Running classic tasks is not available in this slice."* — no run control.
- The estimate shows call count and time; cost appears only for paid providers.
- `ds-status-badge` colour is asserted **in the app CSS bundle**, not in widget `DEFAULT_CSS` — a bundle rule outranks widget CSS regardless of specificity.

---

### Task 5: Snippet editor and import

**Files:** create `snippet_editor.py`; test `Tests/UI/test_evals_snippet_editor.py`

**Interfaces:** consumes `EvalsDB` inline datasets; produces `SnippetEditor` and `#evals-import-snippets`.

**The whitespace flag is this editor's headline feature, not a nicety.** `"The protestors were"` and `"The protestors were "` produce entirely different next-token distributions — with the trailing space, the leading-space variants that dominate the first case become impossible. A user comparing two snippets where one has a stray space would read a large divergence as a finding about the model.

**Requirements the tests must pin:**

- Anomalous whitespace — leading, trailing, or interior runs — renders a highlighted `␣` and raises a warning. Normal text carries **no marker**, so the marker means something wherever it appears.
- **Only exact duplicates are flagged**, after whitespace normalization. Minimal pairs differing by one word *are the instrument*; flagging them would warn on every well-formed word bench and train users to ignore the warning strip.
- The count column is **characters**, not tokens. There is no client-side tokenizer and a token count would be a guess rendered as fact.
- Import accepts one-snippet-per-line text, CSV with a `text` column plus optional `group`, and JSON. Each snippet gets a UUID at authoring time.
- `group` drives grid row grouping and the group-mean aggregate — assert it round-trips.

---

### Task 6: Live verification (PR 3a)

No code. Uses the `verify` skill.

- [ ] Launch the app in this worktree with a scratch `TLDW_CONFIG_PATH`, navigate Lab → Evals.
- [ ] **Capture the screen and diff it against the same capture on `origin/dev`.** This gate is the *inverse* of PR 1's: there, identical captures proved nothing broke. Here the captures must **differ** — on `dev` the Evals body is empty, so a matching capture would mean the new screen failed to render too.
- [ ] Confirm the three panes render, the library rail lists its three sections with counts, and selecting a bench populates the detail pane and the readiness inspector.
- [ ] Confirm the primary action names its object when a bench is selected and is disabled with a stated reason otherwise.
- [ ] Confirm no CSS warnings about missing selectors in the log.
- [ ] `Ctrl+1`..`Ctrl+0` **cannot** be verified through tmux — `send-keys` has no ASCII encoding for ctrl+digit. Assert those in a unit test; never conclude from a tmux probe.

Running a bench end-to-end is PR 3b's gate, not this one — there is no grid to render yet.

---

## Notes for the reviewer

- **The hub was already broken on `dev`** — empty body, verified by before/after capture during PR 1. There is no working behaviour to preserve parity with, so "matches the old screen" is not a valid review standard.
- **Task 3's rule is the point of the PR:** no `Screen` subclass may be mounted inside any workbench container. Detail and inspector content swaps by removing and mounting plain widgets on selection change.
- Schema is unchanged. `Evals_DB` is touched only by Task 1's snapshot serialization.
- The results grid does not exist yet. A reviewer should not expect `results_grid.py`, lens controls, or export in this PR.
