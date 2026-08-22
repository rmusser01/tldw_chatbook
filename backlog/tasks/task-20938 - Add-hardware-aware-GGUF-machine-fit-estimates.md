---
id: TASK-20938
title: Add hardware-aware GGUF machine-fit estimates
status: To Do
assignee: []
created_date: '2026-08-22 19:44'
updated_date: '2026-08-22 21:09'
labels:
  - models
  - ui
  - ux
dependencies:
  - TASK-20937
references:
  - backlog/decisions/080-model-machine-memory-fit-estimation.md
  - Docs/superpowers/specs/2026-08-22-remote-model-machine-fit-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Build on deterministic Remote variant guidance by estimating whether an eligible GGUF is likely to fit the current machine while keeping uncertainty, runtime compatibility, and unsupported hardware explicit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Machine facts are collected through a provider-neutral, bounded, off-loop capability seam with explicit supported, partial, unavailable, and permission-denied states.
- [ ] #2 Each candidate may show a text-labeled likely-fit estimate with the inputs and safety margin used; the UI never upgrades the estimate into a compatibility or successful-runtime claim.
- [ ] #3 Unsupported platforms and incomplete CPU, RAM, GPU, or unified-memory evidence fall back to deterministic guidance without blocking browsing or installation.
- [ ] #4 Users can inspect or refresh the machine facts that informed an estimate, and stale generations cannot overwrite newer facts or selections.
- [ ] #5 The estimation policy and platform-specific probes have focused unit, failure, privacy, and Linux, macOS, and Windows evidence before the feature is enabled.
- [ ] #6 Fit projections use 32K as the small context and 64K as the standard context, expose both estimated loads and the safe machine budget, and lead with the 64K outcome without hiding the 32K fallback.
- [ ] #7 Observed VRAM is shown per device when bounded platform evidence is available, Apple unified memory is shown once, multiple devices are never blindly summed, and accelerator evidence does not change the runtime-neutral system-memory rating.
<!-- AC:END -->
