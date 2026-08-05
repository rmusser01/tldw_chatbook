---
id: TASK-553.17
title: Provision local citation fingerprint key on first enabled use
status: Done
assignee:
  - '@codex'
created_date: '2026-07-27 18:32'
updated_date: '2026-07-27 19:05'
labels:
  - rag
  - citations
  - console
  - security
dependencies:
  - TASK-553.16
references:
  - Docs/superpowers/qa/2026-07-27-task-553-rag-citation-uat.md
  - Docs/superpowers/plans/2026-07-27-local-citation-key-provisioning.md
  - Docs/superpowers/plans/2026-07-27-minimal-console-rag-citations.md
  - backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
parent_task_id: TASK-553
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the already-implemented local citation persistence path usable on a fresh profile while preserving fail-closed behavior for copied or previously populated databases.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A fresh profile with canonical citation writes enabled provisions one valid fingerprint secret through the secure production keyring before the first canonical write.
- [x] #2 A database that already has fingerprint-bearing canonical rows never silently replaces a missing or unreadable key.
- [x] #3 A real local Console RAG answer persists its trace and exposes `Sources (N)` after restart when using production service composition.
- [x] #4 Disabled writes and unavailable or insecure keyring backends remain fail-closed, with scoped automated tests covering each boundary.
- [x] #5 No server, sync, export, import, or new provenance subsystem is added.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Detailed plan:
`Docs/superpowers/plans/2026-07-27-local-citation-key-provisioning.md`

1. Specify secure create-if-missing behavior on the concrete keyring adapter.
2. Invoke it only from enabled local production composition after the existing
   repository row guard proves replacement is safe.
3. Run scoped tests and repeat the fresh-profile rendered UAT through restart,
   Sources inspection, and exact Library open.

ADR required: no
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: Implements the existing ADR/foundation rule without changing schema or
architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added secure create-if-missing behavior to the existing concrete keyring
adapter and invoked it only from enabled local service composition. The factory
serializes first-use provisioning with the existing SQLite transaction and
refuses provisioning whenever fingerprint-bearing canonical rows already
exist, preserving fail-closed recovery behavior.

Scoped verification passed: 437 citation/Console tests, a final 29-test
identity/factory rerun after self-review, 8 filtered Library tests, Ruff on the
four touched Python/test files, and `git diff --check`.
Rendered UAT generated a real local answer with `Sources (1)`, stored one trace
and owner, displayed the exact chunk, opened the exact Library note, and
retained the footer/modal after an app restart. No schema, Console, Library,
server, sync, export, import, or new provenance subsystem was added.
<!-- SECTION:NOTES:END -->
