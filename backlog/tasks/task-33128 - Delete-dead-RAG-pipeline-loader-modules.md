---
id: TASK-33128
title: Delete dead RAG pipeline loader modules
status: Done
assignee: []
created_date: '2026-09-27 19:46'
labels: [cleanup, rag]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
RAG_Search/pipeline_integration.py has zero references anywhere in the repo, and RAG_Search/pipeline_loader.py is referenced only by comments in two unrelated modules and by three test files (Packaging, RAG_Search, Backup_Recovery), one of which pins its source text. Both modules are dead weight in the RAG package: they add review and packaging surface while the live RAG implementation is the simplified/ subpackage behind the package facade.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 pipeline_integration.py and pipeline_loader.py are deleted together with their pinning tests.
- [ ] #2 No references to either module remain anywhere in the repo, including comments that imply they are load-bearing.
- [ ] #3 Targeted RAG test suites pass unchanged.
<!-- AC:END -->

## Implementation Plan

Verify both modules' status against origin/dev before deleting.

ADR required: no
ADR path: N/A
Reason: no code change resulted; retention is governed by existing TASK-17365/17600/32628 history.

## Implementation Notes

REFUTED on origin/dev — closed with no production change. The filing premise (formed against a stale branch) called both modules dead weight with only comment/test references. On dev:

- pipeline_integration.py was already deleted upstream — only pipeline_loader.py remains.
- pipeline_loader.py is deliberately retained with three live dependencies:
  - Tests/Backup_Recovery/test_rag_pipeline_admission.py (added with TASK-32628, commit b5251e9a6e) uses PipelineLoader as the subject of the backup/recovery feature's storage-admission coverage.
  - Tests/RAG_Search/test_pipeline_middleware_contract.py is an ast-based guard whose docstring calls itself "the deliverable, more than any single deletion" — it enforces declared/implemented middleware parity by reading pipeline_loader's source.
  - Tests/Packaging/test_installed_distribution.py asserts the module ships in the installed distribution.
- A prior cleanup pass (f8d2889a53, TASK-17365/17600) already pruned this namespace and consciously kept the loader and its guards.
- Deleting the module would mean rewriting TASK-32628's admission test onto another subject and removing a guard the project explicitly values — a reversal of intentional recent work, not dead-code removal. ACs are withdrawn.


## Renumbering provenance

Filed 2026-09-27 as task-33083. origin/dev minted its own task-33083 before this branch merged, so per the landed-keeps-id rule this task moved to task-33128; every inbound reference moved with it.
