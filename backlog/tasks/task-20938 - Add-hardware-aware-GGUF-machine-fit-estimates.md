---
id: TASK-20938
title: Add hardware-aware GGUF machine-fit estimates
status: Done
assignee: []
created_date: '2026-08-22 19:44'
updated_date: '2026-08-22 22:44'
labels:
  - models
  - ui
  - ux
dependencies:
  - TASK-20937
references:
  - backlog/decisions/080-model-machine-memory-fit-estimation.md
  - Docs/superpowers/specs/2026-08-22-remote-model-machine-fit-design.md
  - Docs/superpowers/plans/2026-08-22-remote-model-machine-fit-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build on deterministic Remote variant guidance with transparent 32,768- and 65,536-token memory scenarios that compare a GGUF allowance with local RAM without implying model-context support, runtime compatibility, or successful inference.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Machine facts are collected through a provider-neutral, bounded, off-loop capability seam with independent system-memory and accelerator evidence states, fixed reason codes, and exact input/output limits.
- [x] #2 Each candidate shows a text-labeled memory-scenario classification, both estimated loads, working-budget margin, and adjacent limitations; no label claims that the model supports 32K/64K or that a runtime will load successfully.
- [x] #3 Unsupported platforms and incomplete CPU, RAM, GPU, or unified-memory evidence fall back to deterministic guidance without blocking browsing or installation.
- [x] #4 LLMScreen owns accepted machine facts, observation time, worker, and generation across body recomposition; RemoteView requests rechecks and renders hydrated immutable state without stale generations replacing newer facts.
- [x] #5 The estimation policy and platform-specific probes have focused boundary, lifecycle, process-cleanup, failure, privacy, and Linux, macOS, and Windows evidence before the feature is enabled.
- [x] #6 Projections use exactly 32,768 and 65,536 tokens, lead with the 65,536-token scenario, expose both estimated loads and the RAM working budget, and show current available-memory pressure separately without changing the stable classification.
- [x] #7 Observed VRAM is shown per device when bounded platform evidence is available, Apple unified memory is shown once, multiple devices are never blindly summed, and the UI states that accelerator evidence does not change the runtime-neutral RAM rating.
- [x] #8 Below 72 RemoteView content cells the repository workflow becomes a keyboard-complete one-pane drill-down with Back and collapsed estimate details; production 80×24 evidence covers both rail states, long names, overflow, focus restoration, Recheck, and Install.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/080-model-machine-memory-fit-estimation.md
Reason: This feature establishes a long-lived provider-neutral capability boundary, privacy contract, bounded platform-probe contract, and recomposition-stable Models-screen ownership.

1. Add immutable machine-memory domain values and exact pure 32,768-/65,536-token projection tests.
2. Add injected, bounded macOS/Linux/Windows RAM and optional VRAM probes with cleanup/privacy tests.
3. Add pure presentation copy and LLMScreen-owned generation, refresh, and recomposition hydration.
4. Add RemoteView machine evidence, current-pressure warnings, stable in-place candidate updates, and the 72-cell drill-down.
5. Prove the feature in production 80x24 layout, run targeted verification, self-review against ADR-080, and record exact task evidence.

Detailed plan: Docs/superpowers/plans/2026-08-22-remote-model-machine-fit-implementation.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the ADR-080 provider-neutral machine-memory capability, bounded
macOS/Linux/Windows probe, pure 32,768-/65,536-token projection policy,
runtime-neutral presenter, LLMScreen-owned refresh/generation lifecycle, and
RemoteView machine evidence plus adaptive one-pane workflow. The no-header
tradeoff is intentional: estimates use exact catalog candidate bytes and a
visible heuristic instead of adding remote GGUF range reads or implying
model-context/runtime support.

Core implementation and evidence live in
`tldw_chatbook/Model_Artifacts/machine_memory.py`,
`machine_memory_probe.py`, `UI/Screens/model_memory_presenter.py`,
`model_remote_view.py`, `llm_screen.py`, their four focused feature test files,
and `Tests/UI/test_llm_screen_lab_adoption.py`. Production-width evidence also
surfaced and fixed three bounded integration defects: the governed Remote CSS
sheet was regenerated from existing source, narrow machine actions now stack,
and completion reapplies the existing pane visibility policy after its internal
recompose. No new architecture decision was required beyond accepted ADR-080.

Targeted verification: import provenance `1 passed`; the four focused feature
files unfiltered `187 passed`; selected LLMScreen cases `13 passed, 126
deselected`; canonical CSS consolidation `31 passed`; CSS bundle sync, Ruff
check, Ruff format check, compileall, and `git diff --check` all passed. The
planned `Tests/UI/test_ui_css_parse.py` path does not exist, so the canonical
`Tests/UI/test_widget_css_consolidation.py` full run plus bundle-sync guard was
used under controller ruling. A once-only isolated `Darwin-arm64` diagnostic
passed with unified memory, one Apple shared marker, and no discrete accelerator
command; no observed capacity/device values were persisted. Full evidence and
RED/GREEN diagnosis are in the Task 5 implementation report.
<!-- SECTION:NOTES:END -->
