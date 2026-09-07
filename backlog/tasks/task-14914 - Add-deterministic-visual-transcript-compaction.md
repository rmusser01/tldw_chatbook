---
id: TASK-14914
title: Add deterministic visual-transcript compaction
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 03:50'
updated_date: '2026-08-11 15:04'
labels: []
dependencies: []
references:
  - backlog/decisions/052-console-conversation-memory-and-compaction-policy.md
  - backlog/decisions/054-deterministic-visual-transcript-compaction.md
  - Docs/superpowers/qa/task-14914-visual-transcript-compaction-uat.md
priority: medium
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an optional on-device compaction representation that deterministically renders older transcript units into pixel-font PNG pages for transfer to vision-capable models, with a hybrid summary-plus-image mode. The goal is to preserve more exact history while reducing model input-token usage where measured provider vision accounting makes that beneficial.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings offers Text summary, Visual transcript, and Hybrid representations with unsupported choices disabled for text-only models
- [x] #2 Rendering is deterministic and local, with fixed pagination, role boundaries, code and tool-result treatment, ordering, and content hashes
- [x] #3 Recent active turns remain text, the original transcript remains canonical, and generated images are derived context artifacts with provenance
- [x] #4 Provider serialization accounts for the exact image representation and falls back safely to text compaction when image input is unavailable
- [x] #5 Benchmarks report model-specific token cost, latency, OCR fidelity, code and math recovery, instruction recall, and adversarial-text behavior before default enablement
- [x] #6 A canonical ADR defines capability ownership, storage lifetime, privacy, fallback behavior, and the accepted visual-token tradeoffs before implementation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define ADR-054 for representation ownership, deterministic rendering, request-scoped artifact lifetime, privacy, capability gating, provider accounting, and fallback behavior. 2. Extend context policy and existing sparse persistence with a representation choice while preserving Text summary as the default. 3. Add a deterministic local pixel-font renderer and provenance manifest with fixed pagination and explicit role, code, and tool boundaries. 4. Extend prepared-request memory projection and provider serialization for visual and hybrid memory, gated by model vision capability and exact image accounting. 5. Add canonical Settings and current-conversation controls with disabled unsupported choices and explicit fallback copy. 6. Add offline benchmark reporting plus unit, integration, narrow-terminal, and live UAT evidence. ADR required: yes. ADR path: backlog/decisions/054-deterministic-visual-transcript-compaction.md. Reason: this introduces a new derived-data representation, privacy and lifetime policy, model-capability boundary, provider wire contract, and long-lived compaction behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-054's opt-in Text summary, Visual transcript, and Hybrid policy across global Console Behavior and sparse per-conversation overrides, including a v33-to-v34 local schema migration. Added a versioned on-device monochrome renderer with fixed pagination, canonical prefix and PNG hashes, request-scoped provenance, image-limit preflight, off-loop execution, exact multimodal prepared-request accounting, and safe text fallback. Hybrid retains the existing guarded durable summary transaction and adds visual pages only when the final request still fits. Added an honest offline benchmark/evaluator contract and senior UX/HCI UAT findings; the measured sample does not justify default enablement, so Text summary remains the default. Verification: 111 focused policy, provider, compaction, migration, lifecycle, and mounted UI tests passed; a post-review 68-test subset and final renderer 6-test pass also passed. Ruff check and format pass for all new renderer/benchmark code and tests; broader legacy-file lint still reports pre-existing E721 checks and unrelated Settings import findings. ADR: backlog/decisions/054-deterministic-visual-transcript-compaction.md. UAT: Docs/superpowers/qa/task-14914-visual-transcript-compaction-uat.md.
<!-- SECTION:NOTES:END -->
