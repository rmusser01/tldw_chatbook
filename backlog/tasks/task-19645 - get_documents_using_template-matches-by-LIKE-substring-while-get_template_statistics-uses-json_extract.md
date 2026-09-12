---
id: TASK-19645
title: get_documents_using_template matches by LIKE substring while get_template_statistics uses json_extract
status: To Do
assignee: []
created_date: '2026-08-21'
labels:
  - bug
  - rag
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In the same file, `ChunkingInteropService.get_documents_using_template` (`Chunking/chunking_interop_library.py:482`, query at `:498-500`) matches a template by raw substring — `chunking_config LIKE '%"template": "{name}"%'` — while `get_template_statistics` (`:621`) uses `json_extract(chunking_config, '$.template')` on the same column. The LIKE form is fragile (whitespace or key-order variance misses; a name that is a substring of another name can false-positive) and the two readers can disagree about which documents use a template.

Filed from the chunking template parity design spec §11 item 5 (`Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md`; ADR-078). Re-verified live 2026-08-21 on `origin/dev` (file untouched since the spec's pin). Note: the parity sub-project's PR D adds a new `Media.chunking_config` writer whose shape must satisfy **both** readers (spec §9.2); converging the readers removes that dual constraint.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both template lookups in the file use the same structural query (json_extract or equivalent) — no `LIKE '%"template"…'` substring match remains
- [ ] #2 A test pins both the match and the non-match: a document stored under template `general` is not returned for `general_v2`, and key-order/whitespace variance in the stored JSON does not change the result
- [ ] #3 The documented shape of any `Media.chunking_config` writer is validated against the converged reader (one shape, both queries agree)
<!-- AC:END -->
