---
id: TASK-19868
title: >-
  Chatbook Import embeddings control is inert in local mode and defaults on
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - ux
  - honesty
  - chatbooks
priority: medium
dependencies:
  - TASK-19734
---

## Description

Source: surfaced by the reviewer of **TASK-19734**, as the third inert control
in the same options box — missed by that task's own sweep because it is not
fully dead. Re-verified at `3605bd52d` (after 19734 merged as `cf38ee6f8`).

The import wizard's options step offers:

> ☑ Import embeddings
> *Import vector embeddings for RAG search (if available)*

(`UI/Wizards/ChatbookImportWizard.py:722-730`.) It is **checked by default**
(`value=True`), so the user is told, on every single import, that vector
embeddings are being brought across for RAG search.

In local mode nothing of the kind happens. The value is collected at `:795`,
passed through `local_chatbook_service.py:946` into
`ChatbookImporter.import_chatbook`, accepted as a parameter
(`chatbook_importer.py:506`), written into a debug log line (`:528`) — and
never read again. There is no code path in the local importer that imports an
embedding.

The other half of the capability does not exist either: the **exporter** never
writes embeddings. `include_embeddings` in `chatbook_creator.py` reaches only
the manifest field (`chatbook_models.py:128`); no embedding data is ever
written into a chatbook. So even a local importer that wanted to honour the
option would find nothing to import.

Why TASK-19734's sweep of this same box did not catch it: the control is not
inert everywhere. It feeds `_update_server_mode_availability` (`:753`, `:759`)
and is genuinely consumed on the server path
(`server_chatbook_service.py:105` filters `ContentType.EMBEDDING` on it). A
"has no consumers" search finds consumers. The defect is narrower and worse to
find: a control that is real in one mode and decorative in the other, with the
default set to the decorative-and-affirmative position.

One further correction belongs with this: the comment added at `:789-791`
during TASK-19734 —

> Every key here is consumed by the import call below.

— is **false** as written. `import_embeddings` is one of the four keys in that
return, and the local import call does not consume it. A comment that asserts
an invariant the code does not hold is worse than no comment, because the next
sweep will trust it.

## Acceptance Criteria

- [ ] In local mode the user is not told that vector embeddings will be
      imported, whether by the control's presence, its label, or its default
      state
- [ ] The control's behaviour matches the mode it is shown in — it is either
      absent, disabled with a reason, or honoured
- [ ] The capability gap is recorded: chatbooks do not contain embeddings
      because the exporter never writes them, so the import side has nothing to
      consume even in principle
- [ ] A test asserts that the local import path's collected options contain no
      key that the local import call ignores, and is mutation-checked
      (re-adding an ignored key makes it red)
- [ ] The comment at `:789-791` is either true of the code or removed
- [ ] The server-mode path continues to filter embeddings on this value (no
      regression to the half that works)

## Notes

The reason to fix the *default* and not only the wiring: a control that is off
by default and does nothing is clutter, while a control that is **on** by
default and does nothing is a claim. This one is a claim, made on every import,
about a feature neither end of the pipeline implements.
