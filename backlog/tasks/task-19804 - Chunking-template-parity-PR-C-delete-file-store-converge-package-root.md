---
id: TASK-19804
title: 'Chunking template parity PR C: delete the file store, converge the package root'
status: Done
assignee: []
created_date: '2026-08-21'
updated_date: '2026-08-21'
labels:
  - chunking
dependencies: [TASK-19803]
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR C (convergence — the breaking change, isolated) of the Chunking Template Parity sub-project (ADR-078): delete the second (file) template store — `chunking_templates.py`, `Chunking/templates/` — remove their `tldw_chatbook.Chunking` package-root exports (a published-import API change under ADR-032's distribution obligations), move all five packaging sites in the same commit, and pin the new failure mode: a bare-name `template=` raises a named exception pointing at `resolve_template`.

Spec: `Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md` (§8, §12 PR-C ACs 30-33). Plan: `Docs/superpowers/plans/2026-08-21-chunking-template-parity.md` (PR C, Task 9).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The file store, `chunking_templates.py`, and `Chunking/templates/` are gone; `tldw_chatbook.Chunking`'s exports no longer name them; only the vendored `ChunkingTemplate` survives, and it is not re-exported at the package root (spec AC 30)
- [x] #2 All five packaging sites move in the same commit (§8.1.2, ADR-032) and the wheel/sdist contract is re-proven against freshly built artifacts — zero template paths in either, checker exit 0 (spec AC 31)
- [x] #3 A bare-name `template=` on `Chunker` / `improved_chunking_process` raises a **named exception type** pointing at `resolve_template`; a pre-resolved dict works; `template_manager=` is documented as accepted-and-ignored and pinned as such (spec AC 32)
- [x] #4 `test_app_import_weight.py` stays green — no DB dependency reaches the shim (spec AC 33)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Delete the file store + package-root exports; pin the named bare-name exception (plan Task 9)
2. Move the five packaging sites in the same commit; rebuild and re-prove wheel/sdist; verify import-weight stays green
<!-- SECTION:PLAN:END -->

## Implementation Notes

Approach: isolated, surgically revertible breaking PR — one commit deletes the store and its exports and moves every packaging site; the manifest checker was inverted from required→forbidden so it now enforces the new absence contract.

- Commits `428501457..2e087aecf` (PR-C marker `2e087aecf`); SDD task 9; wheel/sdist zero-template-paths independently re-verified by the reviewer.
- Deviations-with-rulings: spec §13.1 and `.superpowers/sdd/2026-08-21-chunking-template-parity/progress.md` — MIGRATION_GUIDE.md delete-vs-rewrite carried to PR D (rewritten there to the dict-based contract); packaging-immutability test needs an external venv inside worktrees (mechanism-proven pre-existing sys.path isolation, not branch-caused).
