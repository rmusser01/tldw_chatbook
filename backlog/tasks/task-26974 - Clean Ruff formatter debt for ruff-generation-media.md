---
id: TASK-26974
title: Clean Ruff formatter debt for ruff-generation-media
status: To Do
assignee: []
created_date: '2026-08-31 18:31'
updated_date: '2026-08-31 18:31'
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
- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->
