---
id: TASK-26974
title: Clean Ruff formatter debt for ruff-generation-media
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-02 11:40'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
priority: medium
---

<!-- TASK-26000-BATCH: ruff-generation-media -->
<!-- TASK-26000-PATHS-SHA256: b0cc9fcff57cfad5c74c3d5716b4d517cc4fce29a71dfa0c3f11230eebfd30b0 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-generation-media` Ruff formatter batch at the owner boundary recorded as: Image/video generation and playback surfaces with direct tests.. The focused test surface recorded by TASK-26000 is `["Tests/Image_Generation", "Tests/Media_Creation", "Tests/Media_Playback", "Tests/Video_Generation"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Image_Generation/test_adapter_registry.py",
  "Tests/Image_Generation/test_capabilities.py",
  "Tests/Image_Generation/test_cold_start.py",
  "Tests/Image_Generation/test_comfyui_image_adapter.py",
  "Tests/Image_Generation/test_comfyui_workflow_assets.py",
  "Tests/Image_Generation/test_comfyui_workflow_distribution.py",
  "Tests/Image_Generation/test_config_loader.py",
  "Tests/Image_Generation/test_contracts.py",
  "Tests/Image_Generation/test_demo_screen.py",
  "Tests/Image_Generation/test_fal_adapter.py",
  "Tests/Image_Generation/test_gemini_adapter.py",
  "Tests/Image_Generation/test_http_client.py",
  "Tests/Image_Generation/test_image_format_utils.py",
  "Tests/Image_Generation/test_listing.py",
  "Tests/Image_Generation/test_live_backends.py",
  "Tests/Image_Generation/test_modelstudio_adapter.py",
  "Tests/Image_Generation/test_novita_adapter.py",
  "Tests/Image_Generation/test_openrouter_adapter.py",
  "Tests/Image_Generation/test_package_skeleton.py",
  "Tests/Image_Generation/test_prompt_refinement.py",
  "Tests/Image_Generation/test_request_validation.py",
  "Tests/Image_Generation/test_sd_cpp_adapter.py",
  "Tests/Image_Generation/test_swarmui_adapter.py",
  "Tests/Image_Generation/test_together_adapter.py",
  "Tests/Image_Generation/test_worker.py",
  "Tests/Media_Creation/test_generation_templates.py",
  "Tests/Media_Playback/test_stream_resolve.py",
  "Tests/Video_Generation/test_adapter_registry.py",
  "Tests/Video_Generation/test_comfyui_adapter.py",
  "Tests/Video_Generation/test_comfyui_workflow_assets.py",
  "Tests/Video_Generation/test_comfyui_workflow_distribution.py",
  "Tests/Video_Generation/test_config_loader.py",
  "Tests/Video_Generation/test_config_projection.py",
  "Tests/Video_Generation/test_contracts.py",
  "Tests/Video_Generation/test_minimax_adapter.py",
  "Tests/Video_Generation/test_request_validation.py",
  "Tests/Video_Generation/test_video_metadata.py",
  "Tests/Video_Generation/test_video_store.py",
  "Tests/Video_Generation/test_worker.py",
  "tldw_chatbook/Image_Generation/__init__.py",
  "tldw_chatbook/Image_Generation/adapter_registry.py",
  "tldw_chatbook/Image_Generation/adapters/comfyui_image_adapter.py",
  "tldw_chatbook/Image_Generation/adapters/fal_image_adapter.py",
  "tldw_chatbook/Image_Generation/adapters/gemini_image_adapter.py",
  "tldw_chatbook/Image_Generation/adapters/image_format_utils.py",
  "tldw_chatbook/Image_Generation/adapters/modelstudio_image_adapter.py",
  "tldw_chatbook/Image_Generation/adapters/novita_image_adapter.py",
  "tldw_chatbook/Image_Generation/adapters/openrouter_image_adapter.py",
  "tldw_chatbook/Image_Generation/adapters/stable_diffusion_cpp_adapter.py",
  "tldw_chatbook/Image_Generation/adapters/swarmui_adapter.py",
  "tldw_chatbook/Image_Generation/adapters/together_image_adapter.py",
  "tldw_chatbook/Image_Generation/capabilities.py",
  "tldw_chatbook/Image_Generation/config.py",
  "tldw_chatbook/Image_Generation/exceptions.py",
  "tldw_chatbook/Image_Generation/http_client.py",
  "tldw_chatbook/Image_Generation/listing.py",
  "tldw_chatbook/Image_Generation/prompt_refinement.py",
  "tldw_chatbook/Image_Generation/request_validation.py",
  "tldw_chatbook/Image_Generation/worker.py",
  "tldw_chatbook/Media_Creation/generation_templates.py",
  "tldw_chatbook/Media_Playback/stream_resolve.py",
  "tldw_chatbook/Video_Generation/adapter_registry.py",
  "tldw_chatbook/Video_Generation/adapters/comfyui_video_adapter.py",
  "tldw_chatbook/Video_Generation/adapters/minimax_video_adapter.py",
  "tldw_chatbook/Video_Generation/config.py",
  "tldw_chatbook/Video_Generation/request_validation.py",
  "tldw_chatbook/Video_Generation/video_store.py",
  "tldw_chatbook/Video_Generation/video_templates.py",
  "tldw_chatbook/Video_Generation/worker.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [x] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [x] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [x] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [x] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [x] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [x] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [x] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

1. Reconcile assigned paths against current `origin/dev` (existence + format state) and against the TASK-26000 evidence JSON `cleanup_records` `paths_sha256` (mechanical ownership check).
2. Snapshot each assigned path's `ast.dump` (parse with `type_comments=True`, normalize `TypeIgnore.lineno`, `include_attributes=False`) before formatting.
3. Run Ruff 0.15.22 `ruff format` on exactly the assigned paths.
4. Snapshot after-ASTs; require per-file equality with the before-snapshot (formatter-mandated docstring quote-initial normalization documented if it occurs).
5. `ruff format --check` must pass on every assigned path; `ruff check` findings must not increase vs the pre-format baseline.
6. Run the focused test surface recorded by TASK-26000 (bounded subset if the surface is whole-tree; never a bare pytest), plus `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`.
7. Tick ACs, record Implementation Notes (lineage, commands, results), set status Done.

## Implementation Notes

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `e92b01515f` (isolated worktree `.worktrees/ruff-debt-batch-5`, branch `chore/ruff-debt-batch-5`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All assigned paths exist unchanged at the base except as recorded under Lineage. Mechanically re-verified by recomputing `sha256(json.dumps(paths, separators=(",", ":")))` over the Assigned Paths: it matches both the `TASK-26000-PATHS-SHA256` marker in this file and the `paths_sha256` of the corresponding `cleanup_record` in `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`.

**Formatting (AC#2, AC#5).** 67 of 69 assigned paths formatted. `ruff format --check` passes on every existing assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's assigned paths: 122 findings before and 122 after, per-file counts identical — the formatter introduced zero findings; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after `ast.parse(src, type_comments=True)` with `TypeIgnore.lineno` normalized to 0 and `ast.dump(include_attributes=False)`: hashes equal on every assigned path. (For files whose `# type:` comments carry prose that `type_comments=True` cannot parse as a type expression — and only where BOTH sides fail identically — the comparison falls back to plain `ast.parse` on both sides; the fallback applies symmetrically and is recorded per file.)

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: `Tests/Image_Generation` + `Tests/Media_Creation` + `Tests/Media_Playback` + `Tests/Video_Generation` (recorded surface, bounded directories). Result on the formatted tree: 4 failed, 951 passed, 15 skipped in 27.42s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 4 failed, 951 passed, 15 skipped in 20.96s — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Batch-wide verification: the modified working set is exactly the assigned paths that needed formatting plus the 8 batch task files; zero unassigned paths touched. Every content diff is Ruff formatter output.

**Lineage.** `tldw_chatbook/Image_Generation/adapter_registry.py` and `tldw_chatbook/Video_Generation/adapter_registry.py` were already formatter-clean at the base and were deliberately left untouched; the other 67 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
