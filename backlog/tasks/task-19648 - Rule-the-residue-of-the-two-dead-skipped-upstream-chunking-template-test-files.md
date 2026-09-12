---
id: TASK-19648
title: Rule the residue of the two dead-skipped upstream chunking-template test files
status: To Do
assignee: []
created_date: '2026-08-21'
labels:
  - tech-debt
  - chunking
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Chunking/test_upstream_chunking_templates.py` and `Tests/Chunking/test_chunking_templates_validate_schema.py` are ported-but-dead-skipped via `pytest.importorskip("tldw_chatbook.NoSuchDeferredModule")` (chunking-engine-parity task 4). The template-parity sub-project (ADR-078) **partially** revives them — PR A revives the processor/validate coverage against the vendored `templates.py` and the local validator (spec §12 AC 14's fixture table). The **residue** this task rules on:

- the initialization half (`test_upstream_chunking_templates.py` imports `template_initialization`, which the parity spec explicitly does **not** vendor — spec §6.1);
- the `_shims/DB_Management` and `_shims/AuthNZ` imports both files expect, which exist only to satisfy upstream server-side test imports;
- anything else still behind `importorskip` after the parity sub-project lands.

Filed from the chunking template parity design spec §11 item 8 (`Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md`). Re-verified live 2026-08-21: both files still carry the dead-skip on `origin/dev`. Execute after the parity sub-project's PR A, so the residue is measured against what actually landed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No `importorskip` dead-skip remains in either file: every test is live, rewritten against chatbook surfaces, or deleted — with the ruling recorded per category in the Implementation Notes
- [ ] #2 The `_shims/DB_Management` / `_shims/AuthNZ` expectations are explicitly ruled (keep-as-shim, rewrite, or remove) and the decision applied — no shim survives solely to keep a dead test importable
- [ ] #3 `Tests/Chunking/` is green and the sync script's ported-test contract (upstream provenance headers) stays truthful for whatever remains
<!-- AC:END -->
