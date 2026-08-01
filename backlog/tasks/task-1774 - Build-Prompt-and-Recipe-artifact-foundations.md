---
id: TASK-1774
title: Build Prompt and Recipe artifact foundations
status: To Do
assignee: []
created_date: '2026-08-01 23:27'
labels: []
dependencies:
  - TASK-1773
references:
  - Docs/superpowers/plans/2026-08-01-console-prompt-improvement-workbench.md
  - Docs/superpowers/specs/2026-08-01-console-prompt-improvement-design.md
  - >-
    backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the durable Prompt/Recipe artifact contract that lets Console and server-backed libraries distinguish reusable recipes from executable prompts while preserving legacy and schema-v1 behavior. This foundation is required before any editor or improvement flow can safely list, save, search, or exchange structured artifacts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Local Prompt storage, normalized services, and import/export preserve a first-class prompt-or-recipe discriminator and compiled System/User compatibility fields without changing legacy or schema-v1 semantics.
- [ ] #2 Console block schema v2 and server structured kinds are explicitly dispatched, validated, and compiled without ambiguity; malformed, mismatched, and foreign records remain safely distinguishable.
- [ ] #3 Library source capabilities expose supported artifact kinds, limits, conditional-update availability, and honest backend search behavior, with focused local/server compatibility tests passing.
<!-- AC:END -->
