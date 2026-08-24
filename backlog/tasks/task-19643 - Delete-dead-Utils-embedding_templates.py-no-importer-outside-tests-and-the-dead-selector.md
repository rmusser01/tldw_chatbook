---
id: TASK-19643
title: Delete dead Utils/embedding_templates.py (no importer outside tests and the dead selector)
status: To Do
assignee: []
created_date: '2026-08-21'
labels:
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Utils/embedding_templates.py` has no importer outside tests (`Tests/UI/Embeddings/`) and `Widgets/embedding_template_selector.py` — and the selector is itself dead (nothing imports `EmbeddingTemplateSelector` outside its own module; TASK-16472 AC #2 already owns that widget's fix-vs-retire decision and records the reachability evidence).

Filed from the chunking template parity design spec §11 item 3a (`Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md`; ADR-078). Re-verified live 2026-08-21: importer grep in a worktree at/after the spec's pin, file untouched on `origin/dev` since. The spec files the module and the selector as separate units of work; this task covers the module only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Importer graph re-recorded at implementation time; deletion lands only after (or together with) TASK-16472's retire branch, so no live import breaks
- [ ] #2 The module, its tests (`Tests/UI/Embeddings/test_embedding_templates.py` and the runner), and any packaging/CSS references are removed in one change with the targeted suites green
- [ ] #3 If TASK-16472 instead revives the selector, this task records why the module survives (its revived consumer) and closes — a decision, not a silent drop
<!-- AC:END -->
