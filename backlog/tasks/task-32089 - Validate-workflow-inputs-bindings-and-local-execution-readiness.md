---
id: TASK-32089
title: Validate workflow inputs bindings and local execution readiness
status: To Do
assignee: []
created_date: '2026-09-08 20:56'
labels:
  - workflows
  - safety
dependencies:
  - TASK-32088
references:
  - Docs/superpowers/plans/2026-09-08-workflows-local-file-to-note.md
documentation:
  - Docs/superpowers/specs/2026-09-08-workflows-local-first-parity-design.md
  - backlog/decisions/138-portable-workflow-definitions-and-local-execution.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Run readiness explain exactly which definition fields, resources, effects, and local limits are supported without rewriting preserved documents.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Ordinary defaults, explicit run inputs, and binding-owned keys follow the approved precedence and cannot acquire authority from imported data.
- [ ] #2 Typed earlier-step references resolve correctly; unsupported routing, callbacks, fields, resource mappings, and operation subsets block Local with field-level reasons.
- [ ] #3 Approved numeric safety limits and immutable local profile, actor, endpoint, and resource snapshots are enforced at admission and rechecked at dispatch.
<!-- AC:END -->
