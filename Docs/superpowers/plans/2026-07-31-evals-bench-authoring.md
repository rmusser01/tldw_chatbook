# Evals Bench Authoring (task-1482) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make word benches authorable in the Evals UI — create a bench against any dataset, edit its fields, pick targets, Duplicate/Delete — so the Probe and Δ baseline lenses become reachable with real user data.

**Architecture:** All inside the existing Evals slice. Storage grows `duplicate_bench` + rename hygiene; `BenchEditor` becomes a form (explicit Save; widget state is destroyed by any recompose, so saves recompose deliberately and nothing else may); the rail gains "+ New bench" following `_create_new_dataset`'s in-widget pattern; inspector gains Duplicate/Delete under the primary action. A hardening prerequisite escapes every surface user-authored names will reach.

**Tech Stack:** Python ≥3.11, Textual, pytest (plain output, never `-q`).

## Global Constraints

- Spec `Docs/superpowers/specs/2026-07-25-evals-console-rebuild-design.md` governs; "Bench configuration and portability" defines the editor field set: name, description, dataset, prompt mode, top-K, probe list, targets. **Deliberate deviation, recorded here:** dataset is chosen at CREATE time and read-only in the editor — `storage.save_bench` deliberately refuses dataset changes on the edit path (storage.py:43-52's documented FK hazard). Changing a bench's dataset = Duplicate + retarget (future) or new bench.
- Every name a user can author is a markup hazard: any new rendering surface takes `markup=False` (Static) or `escape_markup` (Button labels / tooltips / dialog messages). Task 1 hardens the existing surfaces BEFORE authoring arms them.
- All bench/target creates use `sample_bench._unique_name(base)` — `eval_tasks.name` is `UNIQUE` with **no deleted_at exemption** (Evals_DB.py:153); `ConflictError` must be caught at every save/rename/duplicate seam and surfaced as a toast, never propagated.
- `Tests/UI/test_evals_bench_editor.py:451-460` pins that `bench_editor.py` and `inspector.py` never mention `capture_client` / `WordBenchRunner` / `CaptureClientLike` — all persistence goes through `word_bench.storage`, never runner imports.
- `select()` → `refresh(recompose=True)` destroys all widget state. The editor saves via explicit button → storage write → `select(kind="bench", id=...)`; no auto-save, no reactive writes.
- Target picker lists `provider="llama_cpp"` rows ONLY — `WordBenchCaptureClient` ignores `target.provider` and always posts to the configured llama.cpp URL (capture_client.py:206-227); listing other providers would silently mis-route.
- Workers: exclusive groups, callable dispatch, guard flags — unchanged from the fix batch. No new workers in this plan (all writes are local DB).
- Tests: real in-memory `EvalsDB`; red-before-green for every behavioural claim; mutation-check at least one core assertion per task; painted-geometry (`region`) checks for new layout at realistic sizes. Foreground pytest via the main checkout's venv from the worktree root, `-p no:randomly`, plain output.
- CSS in `css/features/_evals.tcss` only; regenerate the bundle via `build_css.py`; new Input/Select styling follows `#evals-grid-controls > Select` (`_evals.tcss:340`), not the legacy `.form-*` classes.
- Backlog files `task-1482` (and `task-1513`'s description, Task 1) updated as work lands.

---

### Task 1: Markup hardening prerequisite

**Files:**
- Modify: `tldw_chatbook/UI/Evals/library_rail.py`, `tldw_chatbook/UI/Evals/bench_editor.py`, `tldw_chatbook/UI/Evals/snippet_editor.py`, `tldw_chatbook/UI/Evals/notify_mixin.py`
- Modify: `backlog/tasks/task-1513 - …md` (description: Evals slice landed here)
- Test: extend `Tests/UI/test_evals_empty_states.py`, `Tests/UI/test_evals_bench_editor.py`, `Tests/UI/test_evals_snippet_editor.py`

**Interfaces:** none new — same rendering, safe.

- [ ] **Step 1: Failing tests.** A bench, dataset, and classic task each named `x[/]y …` (a) render literally in rail row Buttons (`.label.plain` contains `[/]`, no MarkupError); (b) bench editor Statics (`#evals-detail-bench-name`, description, dataset line, probes line) render a `[/]`-bearing name/probe literally; (c) snippet editor heading likewise; (d) `NotifyMixin._notify` routed through a fake app records `markup=False` (mirror the fix batch's `_FakeAppInstance.notify` that parses markup — a `[/]` message must not raise).
- [ ] **Step 2: Confirm red** (rail tests raise MarkupError or mis-render; notify test raises).
- [ ] **Step 3: Implement:** `escape_markup(...)` inside `_bench_row_label` / `_classic_row_label` / `_dataset_row_label` (library_rail.py:121-130; import exists at :67, matching `_run_group_row_label` :230). `markup=False` on bench_editor.py:101-103, :105, :110-113, :119-122 and snippet_editor.py:476-480. `notify_mixin.py:35-40`: pass `markup=False`.
- [ ] **Step 4: Suites green** (`test_evals_empty_states.py`, `test_evals_bench_editor.py`, `test_evals_snippet_editor.py`, `test_evals_screen.py`).
- [ ] **Step 5: Update task-1513** description ("Evals-package surfaces hardened in task-1482 Task 1; remaining scope = other screens + repo-wide convention") and **commit** `fix(evals): escape user-authored names on every Evals surface (task-1482 prep)`.

### Task 2: Run completion stops yanking a moved selection

**Files:**
- Modify: `tldw_chatbook/UI/Screens/evals_screen.py`
- Test: extend `Tests/UI/test_evals_screen.py`

Why here: once the editor is a form, the old behavior — a completing worker unconditionally `select(run_group)` — would destroy a half-edited form when a background run finishes.

- [ ] **Step 1: Failing tests:** (a) press Run on bench A, navigate to dataset D while the paused fake run is in flight, release — the completion toast fires but selection REMAINS `dataset/D` (no yank); (b) press Run and stay on bench A, release — selection moves to the new run group (unchanged happy path). Same pair for the sample-bench worker.
- [ ] **Step 2: Confirm red** (case (a) currently ends on the run group).
- [ ] **Step 3: Implement:** both workers capture the launched selection (`_bench_run_task_id` exists; sample worker captures similarly) and on success only `select(run_group)` when `self._selection` still points at the launched bench (or already at one of its run groups); otherwise `notify("Bench run finished — see the Runs section.", markup=False)` only. Update both worker docstrings.
- [ ] **Step 4: Suites green**; mutation check: drop the guard, (a) fails, restore.
- [ ] **Step 5: Commit** `fix(evals): run completion no longer yanks a moved selection (task-1482 prep)`.

### Task 3: Storage seams — duplicate, rename hygiene, conflict mapping

**Files:**
- Modify: `tldw_chatbook/Evals/word_bench/storage.py`, `tldw_chatbook/DB/Evals_DB.py`
- Test: `Tests/Evals/word_bench/test_storage_authoring.py` (new)

**Interfaces (produces):**
- `storage.duplicate_bench(db, task_id) -> str` — loads via `load_bench` (lenient), saves a copy named `_unique_name(f"{config.name} copy")` with identical dataset/mode/top_k/probes/target_ids; returns the new task id. Raises `RuntimeError` (readable) if the source is missing.
- `storage.save_bench` unchanged in signature; documentits ConflictError propagation.
- `Evals_DB.update_task` applies the same name hygiene as `create_task` (strip + control-char filter + blank rejection — Evals_DB.py:552-560 parity).

- [ ] **Step 1: Failing tests:** duplicate copies every config field and shares the dataset (snippets not copied); duplicate of a duplicate gets a fresh unique name; deleted-bench name still blocks an exact-name create (pin the trap) while `_unique_name` sidesteps it; `update_task(name="ctrl\x07char")` stores the filtered name; `update_task(name="")` rejects; renaming onto a live name raises `ConflictError`; renaming onto a soft-deleted bench's name raises `ConflictError` (pin the trap for the UI layer).
- [ ] **Step 2: Red** (duplicate_bench undefined; hygiene asymmetry).
- [ ] **Step 3: Implement** per Interfaces; hygiene refactored into a shared private helper used by both create_task and update_task.
- [ ] **Step 4: `Tests/Evals/word_bench/` green.**
- [ ] **Step 5: Commit** `feat(evals): bench duplicate and rename hygiene at the storage seam (task-1482)`.

### Task 4: "+ New bench" rail affordance

**Files:**
- Modify: `tldw_chatbook/UI/Evals/library_rail.py`, `tldw_chatbook/css/features/_evals.tcss` (+ bundle regen)
- Test: extend `Tests/UI/test_evals_empty_states.py`

**Interfaces:**
- Consumes: `save_bench` (create branch), `_unique_name`, `EvalsSelectionChanged` (the `_create_new_dataset` pattern, library_rail.py:710-725).
- Produces: `#evals-rail-new-bench` Button in the Benches section body, both branches (beside library_rail.py:558-559 and :625), in a `Horizontal` row mirroring `_dataset_actions`.

Behavior (pinned):
- Enabled whenever ≥1 dataset exists. Dataset binding: the currently selected dataset if `selection.kind == "dataset"`, else the newest dataset. Creates `BenchConfig(name=_unique_name("Untitled bench"), prompt_mode="raw", top_k=20, dataset_id=<chosen>, target_ids=())` via `save_bench`, toasts `Bench created against <dataset name>.` (escaped/`markup=False` via `_notify`), posts `EvalsSelectionChanged` selecting the new bench. In-widget, no worker, `ConflictError`/`Exception` → `_notify` error.
- Zero datasets → the button renders disabled with tooltip AND an adjacent one-line hint `Create or import a dataset first.` (never a silent no-op; the fix-batch convention).
- NO provider gate — creating a bench writes only DB rows. This also closes the 1478-noted latent cell (benches exist + gate fails → an affordance now always renders).
- The dataset dead-end copy (evals_screen dataset blocked reason) gains the pointer: `Datasets are run from within a bench; use + New bench in the Catalog rail to create one against this dataset.`

- [ ] **Step 1: Failing tests:** button present in both bench-section branches; press with a dataset selected → bench created bound to THAT dataset and selected; press with none selected → newest dataset; zero datasets → disabled + hint rendered; blocked-reason copy updated (exact string).
- [ ] **Step 2: Red.** **Step 3: Implement.** **Step 4: Green** + bundle sync. **Step 5: Commit** `feat(evals): + New bench creates a draft against a chosen dataset (task-1482)`.

### Task 5: BenchEditor becomes a form

**Files:**
- Modify: `tldw_chatbook/UI/Evals/bench_editor.py`, `tldw_chatbook/css/features/_evals.tcss` (+ bundle regen)
- Test: extend `Tests/UI/test_evals_bench_editor.py`

**Interfaces:**
- Consumes: `load_bench`, `save_bench(db, config, task_id)`, `BenchConfig` validation, ConflictError.
- Produces: editable widgets — `#evals-bench-name` (Input), `#evals-bench-description` (Input), `#evals-bench-prompt-mode` (Select raw/chat, `allow_blank=False`), `#evals-bench-top-k` (Input, numeric), `#evals-bench-probes` (TextArea, **one probe per line, whitespace preserved exactly** — leading spaces are the instrument), `#evals-bench-save` (Button "Save"), `#evals-bench-revert` (Button "Revert"); dataset stays a read-only Static with the create-time-only note in its tooltip. `BenchEditor.Saved(bench_id)` message posted after a successful save; the screen handles it with `select(kind="bench", id=…)`.

Behavior (pinned):
- Save reads widgets → builds `BenchConfig(strict=True)` with the CURRENT stored `target_ids` (targets edited in Task 6) → `save_bench(..., task_id)`; `ValueError` (validation) and `ConflictError` (name) render in `#evals-bench-form-error` (`.ds-recovery-callout`, `markup=False`) — the form keeps its state (no recompose on failure). Success posts `Saved` → recompose re-reads.
- top-K parse failure ("abc") → the same callout, exact text `Top-K must be a whole number of 1 or more.`
- Probes round-trip: a probe line `" Sure"` (leading space) survives save→reload byte-identical; the saved view renders probes with visible `␣` markers reusing snippet_editor's marker convention.
- Prompt-mode switch revalidates targets via `Target.is_valid_for_mode` at save time; an invalid combination renders the callout naming the offending target (currently unreachable — every stored target has neither prefix nor system_prompt — but the seam is wired and tested with a hand-built Target).
- Editing is display-only until Save; Revert = `select(bench)` recompose.

- [ ] **Step 1: Failing tests:** field round-trip (name/description/mode/top-k/probes incl. leading-space probe); validation callout exact strings (top-K, blank name via DB rejection mapped to callout, ConflictError rename); no recompose on failed save (widget state persists — assert the typed value survives); Saved message → screen re-selects; ␣ markers painted.
- [ ] **Step 2: Red.** **Step 3: Implement.** **Step 4: Green** + mutation check (drop ConflictError catch → rename test fails). **Step 5: Commit** `feat(evals): bench editor edits name, mode, top-K, and probes (task-1482)`.

### Task 6: Targets — add and remove

**Files:**
- Modify: `tldw_chatbook/UI/Evals/bench_editor.py`, `tldw_chatbook/UI/Evals/evals_state.py`, `tldw_chatbook/css/features/_evals.tcss` (+ bundle)
- Test: extend `Tests/UI/test_evals_bench_editor.py`

**Interfaces:**
- Produces: `EvalsViewModel.llama_targets()` → `db.list_models(provider="llama_cpp")` (empty-safe); per-target-row `Remove` Button (`#evals-bench-target-remove-{index}` — index-derived like the rows, bench_editor.py:137-144); `#evals-bench-add-target` (Select over `llama_targets()` rows labelled `escape_markup(f"{name} ({model_id})")` + an `Add` Button); when NO llama_cpp models exist, a `#evals-bench-create-target` Button `Create target from configured llama.cpp server` reusing `resolve_sample_target(view_model, app_config, create=True)`'s creation shape with a user-visible unique name.

Behavior (pinned):
- Add/Remove mutate a staged `target_ids` list in the editor (part of form state, saved with Save — one write path, Task 5's). Duplicate adds are rejected inline (strict BenchConfig would refuse anyway — surface it at click: `Target already on this bench.`).
- Removing the last target is allowed (a draft state); the readiness panel and run gating already handle target-less benches (`No targets configured yet.`).
- Target rows keep their readiness status text; staged (unsaved) adds render status `Not yet checked`.

- [ ] **Step 1: Failing tests:** add from picker → staged row renders → Save persists (reload shows it); remove → Save persists; duplicate add rejected with the exact inline text; zero-models state renders the create-target button, pressing it creates an eval_models row and stages it; picker labels escape markup-bearing model names.
- [ ] **Step 2: Red.** **Step 3: Implement.** **Step 4: Green**; the source-scan pin (no capture_client/WordBenchRunner imports) still passes — `resolve_sample_target` lives in sample_bench, importable by the SCREEN, so the create-target press posts a message the screen handles, keeping bench_editor clean of sample_bench imports if the pin requires; verify against the actual pin and choose the minimal wiring. **Step 5: Commit** `feat(evals): bench targets are editable with a llama.cpp picker (task-1482)`.

### Task 7: Duplicate and Delete

**Files:**
- Modify: `tldw_chatbook/UI/Screens/evals_screen.py`
- Test: extend `Tests/UI/test_evals_screen.py`

**Interfaces:**
- Consumes: `storage.duplicate_bench` (Task 3), `db.delete_task`, `ConfirmationDialog` (`Widgets/confirmation_dialog.py` — the Watchlists `push_screen_wait` + `escape_markup` convention, watchlists_collections_screen.py:2117-2135).
- Produces: `#evals-duplicate-bench` and `#evals-delete-bench` Buttons composed after the primary action, **bench-selection branch only** (insert after evals_screen.py:810 inside a bench guard).

Behavior (pinned):
- Duplicate → new bench selected, toast `Duplicated as <new name>.` (`markup=False`).
- Delete → confirm dialog (`Delete bench?` / message with `escape_markup(name)`, cancel primary); confirmed → `delete_task` → `select(kind="none")` → toast `Bench deleted. Its runs remain in the Runs section.` (states the provenance decision).
- Delete while THIS bench's run is in flight (`_bench_run_running and _bench_run_task_id == selection.id`) → button disabled with reason `A run of this bench is in flight.` (the in-flight vocabulary from the fix batch). Duplicate stays enabled.
- Both handlers use the public-shaped-callback convention so tests bypass the modal (snippet_editor.py:577 precedent).

- [ ] **Step 1: Failing tests:** buttons render only for bench selections; duplicate creates+selects (name = `<src> copy <hex>`); delete-confirmed removes from rail, selection → none, runs remain listed AND still open in the grid; delete-cancelled is a no-op; in-flight delete disabled with reason; dialog message contains the escaped name literally.
- [ ] **Step 2: Red.** **Step 3: Implement.** **Step 4: Green** + mutation check (drop the in-flight guard → its test fails). **Step 5: Commit** `feat(evals): duplicate and delete benches from the inspector (task-1482)`.

### Task 8: End-to-end — the lenses come alive

**Files:**
- Test: `Tests/UI/test_evals_authoring_e2e.py` (new)
- Modify: `backlog/tasks/task-1482 - …md` (ACs + notes at the end)

- [ ] **Step 1: The E2E test (fake capture client, real in-memory DB, real screen):** import a 4-snippet dataset via the rail → `+ New bench` against it → editor: rename, add TWO targets (create-target path then picker), add probes `" Sure"` + `" I"`, top-K 20, Save → Run → grid renders; `l` to Probe lens shows probe columns with data (not `n/a`); `l` to Δ baseline shows a real Spread column with values; `✓` run row in the rail. Assert painted content per house convention.
- [ ] **Step 2: Confirm it fails on any missing seam, then passes.**
- [ ] **Step 3: Full sweep** (`Tests/UI/test_evals_*`, `Tests/Evals`, fspicker, bundle sync).
- [ ] **Step 4: Commit** `test(evals): authoring end-to-end through both cross-target lenses (task-1482)`.

---

## Final steps (controller, not a task)

- Live verification (verify skill): scratch profile + live llama server — author a bench on an imported dataset in the real app, two targets, probes; run; drive all five lenses; duplicate; delete; confirm the ✓✗ glyph on an all-failed authored bench.
- Whole-branch review (seams: editor-state vs recompose, worker vs editor, Delete vs runs, name escaping end-to-end).
- Backlog: task-1482 Done with notes; file follow-ups found en route (known already: snippet `+ Add`/`Export…` from the spec's dataset mock; target steering (prefix/system_prompt) authoring — the spec's "llama+prefix" rows — unreachable until authored).
- PR to dev; baseline reconciliation if the suite drifted.
