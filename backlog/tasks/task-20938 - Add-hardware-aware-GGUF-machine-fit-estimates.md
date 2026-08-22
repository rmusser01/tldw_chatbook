---
id: TASK-20938
title: Add hardware-aware GGUF machine-fit estimates
status: In Progress
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
- [ ] #1 Machine facts are collected through a provider-neutral, bounded, off-loop capability seam with independent system-memory and accelerator evidence states, fixed reason codes, and exact input/output limits.
- [ ] #2 Each candidate shows a text-labeled memory-scenario classification, both estimated loads, working-budget margin, and adjacent limitations; no label claims that the model supports 32K/64K or that a runtime will load successfully.
- [ ] #3 Unsupported platforms and incomplete CPU, RAM, GPU, or unified-memory evidence fall back to deterministic guidance without blocking browsing or installation.
- [ ] #4 LLMScreen owns accepted machine facts, observation time, worker, and generation across body recomposition; RemoteView requests rechecks and renders hydrated immutable state without stale generations replacing newer facts.
- [ ] #5 The estimation policy and platform-specific probes have focused boundary, lifecycle, process-cleanup, failure, privacy, and Linux, macOS, and Windows evidence before the feature is enabled.
- [ ] #6 Projections use exactly 32,768 and 65,536 tokens, lead with the 65,536-token scenario, expose both estimated loads and the RAM working budget, and show current available-memory pressure separately without changing the stable classification.
- [ ] #7 Observed VRAM is shown per device when bounded platform evidence is available, Apple unified memory is shown once, multiple devices are never blindly summed, and the UI states that accelerator evidence does not change the runtime-neutral RAM rating.
- [ ] #8 Below 72 RemoteView content cells the repository workflow becomes a keyboard-complete one-pane drill-down with Back and collapsed estimate details; production 80×24 evidence covers both rail states, long names, overflow, focus restoration, Recheck, and Install.
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
