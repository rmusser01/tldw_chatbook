---
id: TASK-19868
title: >-
  Chatbook Import embeddings control is inert in local mode and defaults on
status: Done
assignee:
  - '@t1-chatbook'
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

- [x] In local mode the user is not told that vector embeddings will be
      imported, whether by the control's presence, its label, or its default
      state
- [x] The control's behaviour matches the mode it is shown in — it is either
      absent, disabled with a reason, or honoured
- [x] The capability gap is recorded: chatbooks do not contain embeddings
      because the exporter never writes them, so the import side has nothing to
      consume even in principle
- [x] A test asserts that the local import path's collected options contain no
      key that the local import call ignores, and is mutation-checked
      (re-adding an ignored key makes it red)
- [x] The comment at `:789-791` is either true of the code or removed
- [x] The server-mode path continues to filter embeddings on this value (no
      regression to the half that works)

## Notes

The reason to fix the *default* and not only the wiring: a control that is off
by default and does nothing is clutter, while a control that is **on** by
default and does nothing is a claim. This one is a claim, made on every import,
about a feature neither end of the pipeline implements.

## Implementation Notes (post-review closeout, 2026-10-12; code shipped in PR #3082's original series)

**What shipped** (commits 2727d5e8a4 series in this PR): the import embeddings option now matches the mode — control hidden in local mode, defaults OFF in server mode, key collected only in server mode; the false "every key is consumed" comment replaced with the true per-mode contract; `import_embeddings` removed from `import_chatbook`'s signature as a signature lock against silent reintroduction.

**Capability-gap rationale (AC#3, recorded here per the reviewer's finding):** the embeddings option exists only for the server-import path — local imports run the embeddings pipeline unconditionally after import, so exposing the toggle locally would be a no-op lie; in server mode the server's job consumes the key, so the control is real there and only there. Code comments and the wizard copy encode this; this note is the task-file record.

**DoD note:** the original agent session flipped status to Done without ticking ACs or writing notes — caught by the coordinator's post-hoc code review; the reviewer verified the code and tests genuinely satisfy all six ACs, so the boxes are now ticked with this provenance recorded rather than backdated silently.
