# Evals UAT Fix Batch (tasks 1476–1481) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the execution loop the 2026-07-30 live UAT found broken: wire the Run Bench primary action, explain failed runs at run level, keep creation affordances after first use, fix the export dialog's filename input, disambiguate rail rows, and land the copy/polish batch.

**Architecture:** All work lands inside the existing Evals slice: engine helper in `UI/Evals/sample_bench.py`, screen wiring in `UI/Screens/evals_screen.py`, grid callout in `UI/Evals/results_grid.py`, rail affordances in `UI/Evals/library_rail.py`, dialog fix in `Third_Party/textual_fspicker/file_dialog.py`. No schema changes, no new modules except tests.

**Tech Stack:** Python ≥3.11, Textual, pytest (run with plain output — never `-q`, it suppresses FAILED lines in this repo).

## Global Constraints

- The design spec `Docs/superpowers/specs/2026-07-25-evals-console-rebuild-design.md` governs; the relevant clauses are quoted inside each task.
- User-facing strings: `markup=False` on every Static/Button carrying data-derived text; em-dashes (`—`), never ASCII `--`, in rendered copy; status never conveyed by colour alone.
- Blocked/readiness vocabulary: `.ds-status-badge` + `.ds-recovery-callout`, always visible, never hover-only.
- No network calls from render paths (`compose`/watchers). Execution goes through a Textual worker: handed as a **callable**, `exclusive=True` with a `group=`, guard flag against double-dispatch — mirror `_create_sample_bench_worker`'s documented pattern exactly.
- loguru uses `{}` placeholders; exceptions via `logger.opt(exception=True)`.
- Tests: `.venv/bin/python -m pytest Tests/Evals/word_bench Tests/UI -p no:randomly` style invocations, plain output. Real in-memory `EvalsDB`, fake capture clients (`Tests/Evals/word_bench/test_runner.py` `FakeClient` convention). New UI assertions must check **painted** content where feasible (existing tests show the pattern), and every behavioural test must fail against the pre-change code (mutation check).
- Do not touch `tldw_chatbook/css/tldw_cli_modular.tcss` (generated bundle); feature CSS lives in `tldw_chatbook/css/features/_evals.tcss`.
- Backlog task files `backlog/tasks/task-1476…1481` must be updated (status, plan, notes, AC checkboxes) as each task completes.

---

### Task 1: `run_existing_bench` engine helper

**Files:**
- Modify: `tldw_chatbook/UI/Evals/sample_bench.py`
- Test: `Tests/Evals/word_bench/test_run_existing_bench.py` (new)

**Interfaces:**
- Consumes: `word_bench.storage.load_bench(db, task_id) -> BenchConfig`; the dataset-snippet reader used by `snippet_editor.py`'s read path; the eval_models row lookup `bench_editor.py` uses to resolve target rows (the one that renders `(deleted target <id>) — unresolvable`); existing `_default_client_factory`, `_mark_orphaned_runs_cancelled`, `WordBenchRunner`.
- Produces: `RunBenchResult` dataclass (`task_id: str`, `run_group_id: str`) and
  `async def run_existing_bench(view_model, app_config, task_id, *, client_factory=None, progress=None, cancel_token=None) -> RunBenchResult` — Task 2 calls exactly this.

- [ ] **Step 1: Write failing tests** in `Tests/Evals/word_bench/test_run_existing_bench.py`. Build a real in-memory `EvalsDB`, create a dataset + snippets + a model row + a saved bench (via `storage.save_bench`), then:
  - `test_runs_saved_bench_with_fake_client`: `await run_existing_bench(...)` with a `FakeClient` factory returns a `RunBenchResult` whose `run_group_id` resolves via `view_model.run_group_by_id`, and the grid has one cell per snippet×target.
  - `test_rerun_after_failure_creates_new_run_group`: run once with a factory whose client raises connection errors (cells persist as errors), run again with a working fake — two distinct run groups exist; the first's cells are untouched (no cross-run cache, per spec "Execution").
  - `test_unresolvable_target_raises_runtime_error`: bench whose `target_ids` references a deleted model row → `RuntimeError` naming the target id; no run group is created.
  - `test_missing_bench_raises_runtime_error`, `test_unavailable_service_raises_runtime_error` (`view_model.db is None`).
- [ ] **Step 2: Run tests, confirm they fail** (`run_existing_bench` not defined).
- [ ] **Step 3: Implement** in `sample_bench.py`, beside `create_and_run_sample_bench` (it shares `_default_client_factory` and `_mark_orphaned_runs_cancelled`):

```python
@dataclass(frozen=True)
class RunBenchResult:
    """What running an existing bench produced."""

    task_id: str
    run_group_id: str


async def run_existing_bench(
    view_model: EvalsViewModel,
    app_config: Optional[Mapping[str, Any]],
    task_id: str,
    *,
    client_factory: Optional[Callable[[Target], CaptureClientLike]] = None,
    progress: Optional[ProgressFn] = None,
    cancel_token: Optional[CancelToken] = None,
) -> RunBenchResult:
    db = view_model.db
    if db is None:
        raise RuntimeError("The evaluation service is unavailable.")
    config = load_bench(db, task_id)          # raises ValueError for a missing/non-bench task; wrap into RuntimeError with the bench name/id
    targets = _resolve_targets(db, config)    # RuntimeError naming any unresolvable target id
    snippets = _load_snippets(db, config.dataset_id)  # RuntimeError if the dataset is gone or empty
    factory = client_factory or _default_client_factory(app_config)
    runner = WordBenchRunner(db, factory)
    try:
        outcome = await runner.run(
            config, targets, snippets, task_id,
            progress=progress, cancel_token=cancel_token,
        )
    except asyncio.CancelledError:
        _mark_orphaned_runs_cancelled(db, task_id)
        raise
    return RunBenchResult(task_id=task_id, run_group_id=outcome.group_id)
```

  `_resolve_targets` and `_load_snippets` are small private helpers over the lookups named in Interfaces; reuse existing readers — do not hand-roll SQL. Follow the module's docstring conventions (Args/Returns/Raises, and the hard-vs-cooperative cancellation note referencing `_mark_orphaned_runs_cancelled`).
- [ ] **Step 4: Run the new tests plus the whole `Tests/Evals/word_bench/` suite — all pass.**
- [ ] **Step 5: Commit** `feat(evals): add run_existing_bench engine helper (task-1476)`.

---

### Task 2: Wire the primary action

**Files:**
- Modify: `tldw_chatbook/UI/Screens/evals_screen.py`
- Test: `Tests/UI/test_evals_screen.py` (extend the existing screen-test module; if the suite name differs, extend whichever module holds the current `_primary_action_state` tests)

**Interfaces:**
- Consumes: `run_existing_bench` from Task 1; the sample-bench worker pattern already in this file (`_on_sample_bench_requested`, `_create_sample_bench_worker`, `_set/_reset_sample_bench_running_ui`).
- Produces: an enabled primary action for bench selections; run-in-flight UI state Task 6's rail labels may observe via a new `run_groups()` status field (Task 6 is independent — no shared code).

Spec (governs this task): "**The primary action names its object.** The header action reads `Run loaded-nouns v1` when a bench is selected, and is disabled with a stated reason otherwise." / Execution: worker, row-major fill, grid doubles as the progress view.

- [ ] **Step 1: Write failing tests:**
  - bench selected + bench exists → `_primary_action_state()` returns `(f"Run {name}", False, <ready reason>)` — button enabled; the Blocked badge/callout do **not** render for this branch (they render only when `disabled`).
  - all other branches unchanged (dataset / run_group / missing bench / none) — still disabled with their current reasons.
  - the string `"isn't wired up yet"` no longer exists anywhere in `tldw_chatbook/`.
  - pressing `#evals-primary-action` with a bench selected (via `run_test()` and a fake `client_factory` injected the same way `_sample_bench_client_factory` is faked today) creates a run group and switches selection to it.
  - a second press while running is a no-op (guard flag), mirroring the sample-bench double-dispatch test if one exists.
- [ ] **Step 2: Run tests, confirm the wiring tests fail.**
- [ ] **Step 3: Implement:**
  - In `_primary_action_state`, the found-bench branch returns `(f"Run {name}", False, f"Runs {name} against its configured targets.")`. Delete the "isn't wired up yet; that lands with the results grid in a later PR." copy. Update the function docstring — its "every branch is currently disabled" paragraph is now false and must go.
  - In `_compose_inspector_pane`'s primary-action block, keep the Blocked badge + `.ds-recovery-callout` **only when `disabled`**; when enabled, yield the button alone (tooltip carries the ready reason).
  - Add `@on(Button.Pressed, "#evals-primary-action")` → guard `self._bench_run_running` → `self.run_worker(self._run_bench_worker, exclusive=True, group="evals-run-bench")`. The handler resolves the selected bench id first and stores it on the instance for the worker (never trust selection not to move mid-flight).
  - `_run_bench_worker`: mirrors `_create_sample_bench_worker` structure exactly — flag set/reset in `finally`, `CancelToken` held on the instance, `asyncio.CancelledError` re-raised after logging, `Exception` → `logger.opt(exception=True)` + `notify(f"Could not run the bench: {exc}", severity="error")`. Progress callback updates the primary-action button label to `f"Running… ({done}/{total})"` with `disabled=True`, restored on exit (QueryError-guarded, like `_reset_sample_bench_running_ui`). Success → `notify("Bench run finished.", severity="information")` + `self.select(kind="run_group", id=result.run_group_id)`. Reuse `self._sample_bench_client_factory` as the injectable factory seam (rename it to `_bench_client_factory` only if the rename stays mechanical; otherwise leave the name and note it).
- [ ] **Step 4: Run the screen suite + `Tests/Evals/word_bench/` — all pass.** Mutation check: revert the `disabled=False` change, confirm the gating test fails, restore.
- [ ] **Step 5: Commit** `feat(evals): wire the primary action to run the selected bench (task-1476)`.

---

### Task 3: Run-level failure summary in the results grid

**Files:**
- Modify: `tldw_chatbook/UI/Evals/results_grid.py`
- Test: extend the existing `ResultsGrid` test module under `Tests/UI/` (or `Tests/Evals/`, wherever `results_grid` tests live)

**Interfaces:**
- Consumes: the grid's already-loaded cell data (each failed cell carries a reason string — the same one the cell inspector renders as `Failed: <reason>`).
- Produces: `#evals-grid-failure-callout` (`.ds-recovery-callout`), rendered between the header state line and the canary callout.

- [ ] **Step 1: Write failing tests:**
  - all cells failed (fake grid data, all `unreachable`) → callout text is exactly `All 4 cells failed — unreachable. Check that the target's server is running and reachable, then run the bench again.` (count and reason interpolated; “dominant reason” = most frequent failure kind, ties broken by first-seen).
  - mixed (1 of 4 failed) → `1 of 4 cells failed — unreachable.` (no next-step sentence; the run is usable).
  - zero failures → the callout is absent from the DOM.
- [ ] **Step 2: Run tests, confirm they fail.**
- [ ] **Step 3: Implement**: derive counts + dominant reason where the grid already iterates cells to build rows (single pass, no second load); yield the callout conditionally with `markup=False`. The failed-cell count already exists for the `meta` line — reuse that accounting, do not count twice.
- [ ] **Step 4: Run the grid suite — pass.** Mutation check: make the callout unconditional, confirm the zero-failure test fails, restore.
- [ ] **Step 5: Commit** `feat(evals): explain failed runs at run level (task-1477)`.

---

### Task 4: Persistent rail creation affordances

**Files:**
- Modify: `tldw_chatbook/UI/Evals/library_rail.py`
- Test: extend the existing `LibraryRail` test module

Spec (governs this task): "Each of `Benches` and `Datasets` carries its own creation affordance in the section header — a new bench and a new snippet set are reachable without first finding an empty state."

- [ ] **Step 1: Write failing tests:**
  - rail with 1 bench + provider configured → `#evals-create-sample-bench` still present.
  - rail with 1 dataset → `#evals-rail-new-dataset` and `#evals-rail-import-dataset` still present.
  - rail with 0 of each → current empty-state hints still render (existing tests keep passing).
  - collapsed section → its affordances are not displayed (they live in the section body, which collapses).
- [ ] **Step 2: Run tests, confirm the two non-empty cases fail.**
- [ ] **Step 3: Implement**: render the action rows unconditionally at the top of each section body: the Benches body always yields the sample-bench button (when the provider gate passes — gate logic unchanged), the Datasets body always yields `_dataset_empty_actions()`'s row (rename the helper `_dataset_actions` and update its docstring — it is no longer empty-only). Empty-state *copy* (hints like "Start here — …", "No datasets yet.") remains empty-only. Keep ids unchanged so `on_button_pressed` wiring is untouched.
- [ ] **Step 4: Run the rail suite — pass.**
- [ ] **Step 5: Commit** `fix(evals): keep rail creation affordances after first use (task-1478)`.

---

### Task 5: Export dialog filename input

**Files:**
- Modify: `tldw_chatbook/Third_Party/textual_fspicker/file_dialog.py` (and `parts/` CSS if the width rule lives there)
- Test: `Tests/UI/test_fspicker_keyboard_save.py` (new)

Known facts: `file_dialog.py:85` seeds the Input with the default filename; `:146` already confirms on `Input.Submitted`; `:107-109` focuses the input when a file row is highlighted. The two defects: the Input's rendered width collapses (observed ~5 cells at 235×52 — inspect the `BaseFileDialog InputBar { Input { … } }` CSS at `:42` and the InputBar layout in `parts/`), and initial focus lands on the directory list so Enter activates `..`.

- [ ] **Step 1: Write failing tests** (drive a real `FileSave` via `run_test()` at a wide terminal size):
  - on mount, the Input's value equals the default filename **and** its rendered width is ≥ half the InputBar width (assert via the widget's `region.width` — this is the painted check).
  - on mount, `dialog.focused` is the filename Input (FileSave only; FileOpen keeps list focus).
  - pressing Enter immediately after mount dismisses the dialog with the default path (keyboard-only export path).
- [ ] **Step 2: Run tests, confirm width/focus fail.**
- [ ] **Step 3: Implement**: give the Input `width: 1fr` (or fix the flex rule that starves it) in the dialog CSS; in `FileSave`'s mount path, focus the filename Input. Do not change `FileOpen` behaviour or directory navigation. Note the deviation from upstream in `ENHANCEMENTS.md` (the vendored copy documents local changes there).
- [ ] **Step 4: Run the new tests + any existing fspicker tests (`safe_tests.py`) — pass.**
- [ ] **Step 5: Commit** `fix(fspicker): usable filename input and save-focused Enter (task-1479)`.

---

### Task 6: Run-row status glyph + timestamp

**Files:**
- Modify: `tldw_chatbook/UI/Evals/evals_state.py`, `tldw_chatbook/UI/Evals/library_rail.py`
- Test: extend the view-model and rail test modules

Spec mock (governs the shape): `● 14:31 run` / `✓ 14:02 run` / `✗ 13:55 run`.

- [ ] **Step 1: Write failing tests:**
  - `EvalsViewModel.run_groups()` rows gain `"status"`: `"running"` if any run in the group is running, else `"cancelled"` if any is cancelled, else `"completed"` (derived in the same pivot pass — `list_runs` rows already carry `status`).
  - rail run-row label for a completed group created at 14:02 is `✓ 14:02 · <task_name>`; cancelled → `✗ <time> · <name>`; running → `● <time> · <name>`. Time is `HH:MM` from `created_at` (document the assumption: `created_at` is the DB's stored local format; parse defensively, fall back to the raw string on parse failure — never crash the rail over a timestamp).
  - bench rows are unchanged.
- [ ] **Step 2: Run tests, confirm they fail.**
- [ ] **Step 3: Implement**: status roll-up inside the existing `run_groups()` loop; run-row `row_label` becomes the glyph+time format. Glyphs are single-width characters (●/✓/✗ — verify with `Rich`'s cell width if in doubt; never emoji, they are double-width in this app's terminal, a repeated past defect).
- [ ] **Step 4: Run both suites — pass.**
- [ ] **Step 5: Commit** `feat(evals): run rows carry status glyph and time (task-1480)`.

---

### Task 7: Copy and polish batch

**Files:**
- Modify: `tldw_chatbook/UI/Screens/evals_screen.py`, `tldw_chatbook/UI/Evals/results_grid.py`, `tldw_chatbook/UI/Evals/snippet_editor.py`, `tldw_chatbook/UI/Screens/lab_frame.py`
- Test: extend the affected suites

Five items from task-1481:

- [ ] **Step 1: Write failing tests** for the observable ones:
  - no rendered Evals string contains `"library rail"` (sweep the module constants; replacement copy says "Catalog rail").
  - no rendered Evals string contains ASCII ` -- ` (the two `evals_screen.py` empty-state strings switch to em-dashes).
  - single-target run group + Δ baseline lens → the grid state line (or a callout) contains `needs at least two targets`; the all-cells-say-"baseline" rendering is gone for that case.
  - snippet table: header matches row shape (either the header names only the columns the rows visually align to, or rows align to the header — implementer's choice; the test pins whichever is chosen).
- [ ] **Step 2: Run tests, confirm they fail.**
- [ ] **Step 3: Implement** the four copy/rendering fixes, plus correct the stale comment at `lab_frame.py:91` (Escape claim — state what is actually true: the shell handles Escape; EvalsScreen defines no binding).
- [ ] **Step 4: Run the full Evals-adjacent test set — pass.**
- [ ] **Step 5: Commit** `fix(evals): UAT copy and polish batch (task-1481)`.

---

## Final steps (controller, not a task)

- Update `backlog/tasks/task-1476…1481` to Done with Implementation Notes; leave 1482–1484 To Do.
- Whole-branch review, then live verification per the `verify` skill: fresh scratch profile, dead-server run → failure callout renders → start a server on the configured port → **Run Bench from the bench row succeeds** (the exact loop UAT found broken), plus keyboard-only export.
