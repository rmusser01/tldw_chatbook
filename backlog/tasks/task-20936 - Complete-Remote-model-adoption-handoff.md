---
id: TASK-20936
title: Complete Remote model adoption handoff
status: Done
assignee:
  - '@codex'
created_date: '2026-08-22 17:44'
updated_date: '2026-08-22 18:40'
labels:
  - models
  - ui
dependencies:
  - TASK-19906
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the Remote model journey after a verified managed download so users can find the exact installed model or configure it for a compatible local runtime without silent activation or startup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A successful Remote install presents a durable `Downloaded · Verified · Managed · Not active` state attributed to the current `Hugging Face` source, and this outcome survives Models screen recomposition until the user starts another Remote discovery or install.
- [x] #2 `Open Installed` switches to Installed and locates the exact managed model reference that completed, without activating it.
- [x] #3 `Configure and use…` presents only compatible managed-GGUF runtime choices, preselects the exact completed model for the chosen runtime, and opens that runtime without activating the managed artifact or starting a server.
- [x] #4 Failed or cancelled installs retain their existing recovery behavior and never expose successful-adoption actions.
- [x] #5 The provider-neutral `Remote` destination identifies `Hugging Face` as the current source without introducing a redundant one-option provider selector.
- [x] #6 The completion and handoff controls remain painted, contained, text-labeled, and keyboard reachable at 80 columns under the production stylesheet.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md
Reason: ADR-025 already governs provider-neutral managed GGUF identity, explicit activation, and llama.cpp/llamafile managed-source authority. This task adds a UI handoff over those existing boundaries without changing storage, runtime ownership, or provider contracts.

1. Add failing RemoteView tests for the provider-attributed completion state, success-only actions, reset behavior, and 80-column rendering.
2. Add failing host/window tests for exact-reference Installed navigation and explicit llama.cpp/llamafile managed-source configuration without activation or process startup.
3. Implement the minimal completion message/state boundary across RemoteView, LLMScreen, InstalledView, and LLMManagementWindow while preserving the existing install worker and managed-store ownership.
4. Regenerate consolidated CSS if required, run the focused Remote/Models suites, and visually verify idle, success, Installed handoff, and runtime handoff states under the production stylesheet.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a durable provider-attributed Remote completion state with exact Open Installed and explicit Llama.cpp/Llamafile configuration handoffs. The Models screen now owns pending runtime intent across recomposition; the management window validates and reports exact managed inventory outcomes without activation or server startup. Installed navigation highlights and focuses the exact reference, with distinct inline recovery when a successful refresh proves it missing versus when inventory loading fails. Added production-stylesheet compositor and keyboard coverage at 80 columns plus lifecycle, focus-order, failure, and recompose regressions. ADR required: no; existing ADR-025 continues to govern managed artifact identity and runtime authority. Verification: 73 Remote/widget tests passed; 115 Installed/GGUF tests passed; 9 focused real-host adoption tests passed; CSS bundle guard 4 passed; Ruff and git diff checks passed. A broader host selection reached 178 passed with one unrelated modal child-mount race; that existing test passed immediately in isolation. Full repository sweep was not run under the targeted-test policy.
<!-- SECTION:NOTES:END -->
