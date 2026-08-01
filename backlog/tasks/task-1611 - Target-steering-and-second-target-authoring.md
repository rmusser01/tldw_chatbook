---
id: TASK-1611
title: Target steering and second-target authoring
status: Done
assignee: []
created_date: '2026-07-31 15:10'
updated_date: '2026-08-01 00:52'
labels:
  - evals
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Whole-branch review of task-1482, Important 2 (scope boundary). The UI can create exactly ONE eval_models row per install (the bench editor's create-target button renders only when zero llama_cpp models exist; the sample bench reuses rather than creates), so the Δ baseline Spread and per-target Probe comparisons stay single-column for real users. The design spec's bench mock shows steering variants ("llama+prefix") as distinct targets; `Target.prefix`/`system_prompt` and the snapshot format already support them but nothing writes them (`eval_models.config` is the natural home). This task adds: creating additional targets from the editor (name + optional prefix/system_prompt against the configured server), making `Target.is_valid_for_mode` production-reachable, and rewording the mode-revalidation copy that currently names settings with no UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A second (and Nth) target can be created from the bench editor when models already exist
- [x] #2 A target can carry a prefix (raw mode) or system prompt (chat mode), persisted and used by the capture request
- [x] #3 Prompt-mode switching revalidates steered targets with user-readable copy
- [x] #4 Multi-target Δ column baseline is reachable end-to-end through the UI
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Engine/storage: steering persists on eval_models.config (prefix/system_prompt); target resolution builds steered Targets; capture client already consumes them (write side only)
2. UI: "+ New target" in the editor's targets section works when models exist (name + mode-appropriate steering field); revalidation copy reworded to name real UI
3. E2E: two targets (base + prefix-steered) through the UI -> column-mode Δ baseline renders real spread; live verification against the real server
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**T1 (engine/storage seam, commits 2e6beaa13 + d9a79095):** `storage.model_steering(model_row) -> (prefix, system_prompt)` reads `eval_models.config` (absent-safe, empty-string normalizes to `None`, raises `ValueError` naming the row's id if both are set or `config` is a non-mapping). Both Target-resolution sites (`sample_bench._resolve_targets` for "Run Bench", and the inline build in `create_and_run_sample_bench`) call it, so every run path builds `Target.prefix`/`system_prompt` from the row's own config rather than leaving the fields permanently `None`. `run_existing_bench` wraps its resolve-and-run block so a corrupt row's `ValueError` surfaces as the same `RuntimeError` a mode mismatch already did. Convention: steering is immutable per `eval_models` row (no `update_model`) — a differently-steered variant is always a new row.

**T2 (bench-editor UI, commits 1c9fa2145 + 733be4ba):** `bench_editor.py`'s "+ New target" mini-form now renders unconditionally (`_build_create_target_control`), not only in the old zero-`llama_cpp`-models state — a bench author can mint an ADDITIONAL, differently-steered target even when rows already exist. The mini-form is a Name `Input` plus ONE steering `Input` picked by the current prompt mode (`#evals-target-prefix` raw / `#evals-target-system-prompt` chat), posted as `CreateTargetRequested` and handled by `evals_screen.py` via a direct `EvalsDB.create_model` call (never `sample_bench.resolve_sample_target`, which reuses an existing row first — wrong once this control's job is minting an additional one). `is_dirty()` covers the mini-form's own typed-but-uncreated state so a background worker completing mid-edit degrades to a toast instead of silently discarding it. The targets section was restructured into one scrollable `#evals-bench-targets-body` (row table + Add picker + create form) after a live-verified pane-containment failure at 4+ targets — the table's own box no longer gets squeezed to a 1-row floor.

**T3 (E2E, this commit):** `Tests/UI/test_evals_steering_e2e.py`, a new file (kept separate from `test_evals_authoring_e2e.py` rather than extended — that suite's own second target is deliberately DB-seeded, since at task-1482 time there was no UI path to a second row at all; folding a fully-UI-authored scenario into the same file would make its docstring tell two contradictory target-creation stories). Drives the whole loop with no DB seeding of targets: import a dataset -> "+ New bench" -> "+ New target" pressed twice through the real UI (target 1 blank/unsteered, target 2 with a raw-mode prefix typed into `#evals-target-prefix`) -> Save -> Run -> explicitly switches `#evals-baseline-selector` to COLUMN mode with target 1 as baseline -> asserts a real, nonzero Spread and that the two targets' Top-1 cells differ. The fake capture client is keyed on `target.prefix`/`target.system_prompt` truthiness (not target name/id, unlike every other fake in this suite) — the crux: this is what proves the Δ actually comes from steering reaching the `Target` at run time, not merely from the grid telling two differently-named columns apart. Also asserts the run snapshot (`config_overrides.snapshot.targets`) persisted target 2's prefix byte-exact, and that target 1's snapshot entry carries no prefix/system_prompt — T1's snapshot seam, this time reached through a fully UI-authored path rather than `test_run_existing_bench.py`'s direct DB write.

Mutation check (post-commit, this task): edited the fake capture client's `capture()` to ignore `target.prefix`/`target.system_prompt` (always the unsteered distribution) — killed the Top-1-difference assertion and the Spread-nonzero assertion (both correctly, `first_cell == second_cell` and `spread_text` read `"0.00"`); restored via `git checkout` against the committed baseline, `git diff --quiet` confirmed byte-identical.

**What live verification still owes:** this task's E2E is fully simulated (a fake capture client, in-memory DB) — no run against a real llama.cpp server exercised a UI-authored steered target end-to-end. The Implementation Plan's step 3 named "live verification against the real server" as part of this task's own scope; that step was not performed here (no live server was available in this environment) and should be picked up as a follow-up before this feature is considered fully user-verified, mirroring this program's own established practice of a dedicated live-verification pass (see e.g. the evals-bench-authoring program's own "LIVE VERIFICATION COMPLETE" entry for task-1482).
<!-- SECTION:NOTES:END -->
