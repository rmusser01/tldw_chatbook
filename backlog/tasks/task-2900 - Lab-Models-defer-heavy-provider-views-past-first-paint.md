---
id: TASK-2900
title: Lab ▸ Models — defer heavy provider views past first paint
status: Done
assignee: []
created_date: '2026-08-07 02:00'
labels:
  - lab
  - performance
  - defer-past-first-paint
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Screen survey (follow-up to task-2725): Lab ▸ Models mounts 388 widgets per visit, with the heavy weight in views that arrive hidden — `#llm-view-download-models` (76), `#llm-view-ollama` (58), plus the curated/installed/remote library views. All eleven views compose eagerly; visibility is CSS `-active` classes and switching funnels through `watch_active_view`, which already tolerates absent views (`QueryError` → warning). Per-view work is already deferred until shown (task-887), so the screen's own architecture supports the 2725 pattern.

Apply defer-past-first-paint: compose only the shell + initial view eagerly; mount the five heavy deferred views (ollama + the four library views, which are thin wrappers over existing widget classes) right after first paint, then run the one-shot initializers that touch them (`_autofill_ollama_path` — UX-078 — would otherwise silently never fire).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: with the deferred-mount seam stubbed, pushing the LLM screen mounts none of the five deferred views and the widget total drops; guard test (green before+after): after settle, all eleven views exist and llama-cpp is active.
2. GREEN: extract the ollama block into a small `VerticalScroll` subclass (single `self` reference: the prereq text, passed in); move the four library-view constructions into `_mount_deferred_views()`; `on_mount` chains mount → `_initialize_view` → re-run `_autofill_ollama_path`/`_update_ollama_api_state`.
3. Full LLM/Lab test files + live latency + live view-switch exercise.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria

<!-- SECTION:ACCEPTANCE_CRITERIA:BEGIN -->
- [x] Lab ▸ Models first paint excludes the five deferred views; all eleven are reachable after load, llama-cpp active by default.
- [x] Tab-switch latency improves measurably live (same tmux method); no view renders broken when activated.
- [x] One-shot initializers that touch deferred views still take effect (Ollama path autofill, API-state gating).
- [x] Existing LLM/Lab tests green.
<!-- SECTION:ACCEPTANCE_CRITERIA:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped the 2725 pattern with two screen-specific discoveries, both caught by the existing test surface during RED/GREEN:

1. **Failure coupling**: sequencing the previously-independent `call_after_refresh` callbacks (initialize / autofill / API-gate) into one coroutine meant a failure in one killed the rest — `_finish_deferred_mount` now guards each step individually (this incidentally FIXED the dev-baseline `test_ollama_path_autofills_when_found` failure).
2. **Recompose hydration race**: `LLMScreen`'s install-progress hydration fires from `on_lab_body_ready` via one `call_after_refresh`, which was only sufficient because compose built every view synchronously; with deferral it raced the mount and lost (three recompose-survival tests went red). Fix: the window posts `DeferredViewsMounted` when the views are queryable and the screen re-hydrates on it — hydration is internally guarded and idempotent, so both invocations are safe.

Mechanics: the four library views (curated/installed/remote/download-models) were already thin wrappers over widget classes — moved verbatim into `_mount_deferred_views`; the ollama view (58 widgets, ONE self-reference) extracted into `OllamaServiceView` with the prereq text passed in. No order anchoring needed (CSS `-active` shows exactly one view). Also repaired the two remaining stale Lab tests in this surface (batch4's construction-default pin predating the ""-with-init=False race fix AND the llama-cpp default from 9dd2374b5; batch6's retired-sidebar hint test — cycling now lives in the Lab rail, covered by the adoption tests).

Results: Lab switch 0.68s → **0.47–0.49s** live; ollama/curated/download views render with full content when activated, zero errors. Lab/LLM 12-file surface: **149 passed, 0 failed** (pristine dev baseline: 4 failed). Files: tldw_chatbook/UI/LLM_Management_Window.py, tldw_chatbook/UI/Screens/llm_screen.py, Tests/UI/test_llm_deferred_views.py, Tests/UI/test_ux_batch4.py, Tests/UI/test_ux_batch6.py.
<!-- SECTION:NOTES:END -->
