---
id: TASK-19646
title: _parse_template_config silently swallows malformed JSON to {} yielding a words-default chunking
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
`LocalRAGAdminService._parse_template_config` (`RAG_Admin/local_rag_admin_service.py:79-88`) catches `(TypeError, ValueError)` from `json.loads` and returns `{}`. Downstream, `_chunking_options_from_template` (`:350-376`) falls through its `or "words"` chains on an empty dict, so a stored template with malformed JSON silently chunks as `("words", {})` instead of surfacing an error — data corruption's friendly face, the same class as the rolling-summarize markers ADR-078 rules against.

Filed from the chunking template parity design spec §11 item 6 (`Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md`). Re-verified live 2026-08-21 on `origin/dev` (file untouched since the spec's pin). Coordinate with the parity sub-project's stored-invalid ruling (spec §5.4): an invalid stored row is listed with a flag, refused at apply with a **named** error, and editable — this task is the parse-layer half of that behavior for rows that are not even parseable JSON.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Malformed template JSON surfaces as a named error identifying the row (or is refused at apply per the spec §5.4 ruling) — never a silent `("words", {})` default
- [ ] #2 A test pins the failure path: a row with unparseable `template_json` is refused/flagged, with the row's name in the error, and no chunk is produced from it
- [ ] #3 Legitimately-empty values (NULL/blank column) keep today's behavior by explicit decision, recorded in the Implementation Notes — distinguished from malformed JSON, not conflated with it
<!-- AC:END -->
