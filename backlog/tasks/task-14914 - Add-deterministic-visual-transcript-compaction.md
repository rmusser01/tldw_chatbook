---
id: TASK-14914
title: Add deterministic visual-transcript compaction
status: To Do
assignee: []
created_date: '2026-08-11 03:50'
labels: []
dependencies: []
references:
  - backlog/decisions/052-console-conversation-memory-and-compaction-policy.md
priority: medium
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add an optional on-device compaction representation that deterministically renders older transcript units into pixel-font PNG pages for transfer to vision-capable models, with a hybrid summary-plus-image mode. The goal is to preserve more exact history while reducing model input-token usage where measured provider vision accounting makes that beneficial.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Settings offers Text summary, Visual transcript, and Hybrid representations with unsupported choices disabled for text-only models
- [ ] #2 Rendering is deterministic and local, with fixed pagination, role boundaries, code and tool-result treatment, ordering, and content hashes
- [ ] #3 Recent active turns remain text, the original transcript remains canonical, and generated images are derived context artifacts with provenance
- [ ] #4 Provider serialization accounts for the exact image representation and falls back safely to text compaction when image input is unavailable
- [ ] #5 Benchmarks report model-specific token cost, latency, OCR fidelity, code and math recovery, instruction recall, and adversarial-text behavior before default enablement
- [ ] #6 A canonical ADR defines capability ownership, storage lifetime, privacy, fallback behavior, and the accepted visual-token tradeoffs before implementation
<!-- AC:END -->
