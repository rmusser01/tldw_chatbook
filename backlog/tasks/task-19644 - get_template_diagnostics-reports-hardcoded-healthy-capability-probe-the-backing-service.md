---
id: TASK-19644
title: get_template_diagnostics reports hardcoded healthy capability — probe the backing service
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
`LocalRAGAdminService.get_template_diagnostics` (`RAG_Admin/local_rag_admin_service.py:264-270`) returns hardcoded `capability: "native"`, `missing_methods: []`, and `fallback_enabled: False` — it never probes the backing chunking service (it derives only `db_class` from the real object). A broken or degraded backend therefore reports healthy.

Filed from the chunking template parity design spec §11 item 4 (`Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md`; ADR-078). Re-verified live 2026-08-21 on `origin/dev` (file untouched since the spec's pin). Consequence already shipped: the parity sub-project's Library report renderer consumes **only** `legacy_chunk_report` precisely because the other fields are fabricated (spec §10.1).

Deleting the fabricated fields is as valid an outcome as implementing the probe — the defect is the false claim, not the absence of a probe.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No diagnostics consumer can read a hardcoded healthy claim: the fields are either derived from the backing service (probe) or removed
- [ ] #2 A test proves a broken/missing backend is not reported as `native` with empty `missing_methods` (mutate the backend, watch the output change — or assert the fields are gone)
- [ ] #3 Any renderer that surfaces these fields is updated in the same change so no UI regresses
<!-- AC:END -->
