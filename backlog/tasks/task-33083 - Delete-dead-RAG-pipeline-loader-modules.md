---
id: TASK-33083
title: Delete dead RAG pipeline loader modules
status: To Do
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
