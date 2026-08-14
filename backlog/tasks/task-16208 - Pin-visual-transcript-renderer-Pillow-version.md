---
id: TASK-16208
title: Pin visual transcript renderer Pillow version
status: Done
assignee:
  - '@codex'
created_date: '2026-08-13 23:57'
updated_date: '2026-08-13 23:57'
labels:
  - test-health
  - dependencies
  - console
dependencies: []
references:
  - backlog/decisions/054-deterministic-visual-transcript-compaction.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep deterministic visual-transcript rendering on the Pillow version used by the reviewed Terra context-use evidence so ordinary dependency resolution cannot silently change renderer bytes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Core package and requirements declarations pin Pillow to the evaluated renderer version.
- [x] #2 A packaging contract fails when either declaration drifts from that exact version.
- [x] #3 The evaluated renderer version reproduces the checked page hashes without rewriting live model evidence.
- [x] #4 Focused, static, dependency, and diff evidence pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/054-deterministic-visual-transcript-compaction.md
Reason: ADR-054 already makes the Pillow version part of the deterministic renderer identity; this enforces that accepted boundary.

1. Preserve the Pillow-12.1.1 renderer-version and page-hash mismatch as RED evidence.
2. Add a packaging contract for identical exact pins in `pyproject.toml` and `requirements.txt`.
3. Pin both core dependency declarations to Pillow 11.2.1.
4. Verify the checked renderer hashes under an isolated 11.2.1 install, then run focused/static/diff gates without changing the shared environment or live evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Pinned the core Pillow dependency to 11.2.1 in both package metadata and `requirements.txt`, matching the renderer identity recorded by the reviewed Terra context-use evidence. Added a fail-closed packaging contract for both declarations. RED: the shared unpinned environment resolved Pillow 12.1.1 and produced a different renderer identity plus two different PNG hashes. GREEN: an isolated `/private/tmp` Pillow 11.2.1 install reproduced the checked renderer identity and both page hashes; the full evaluator file passed 36 tests. Removing either pin independently failed the new contract. Ruff check/format and diff checks passed. The shared environment and checked live model evidence were not modified. ADR required: no new ADR; ADR-054 already owns the deterministic renderer-version boundary.
<!-- SECTION:NOTES:END -->
