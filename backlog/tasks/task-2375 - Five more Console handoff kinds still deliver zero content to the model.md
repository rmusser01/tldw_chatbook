---
id: TASK-2375
title: Five more Console handoff kinds still deliver zero content to the model
status: To Do
assignee: []
created_date: '2026-08-04 20:07'
labels:
  - console
  - rag
  - honesty
dependencies: []
priority: high
---

## Description

PR-T1 Task 9 (task-2374) fixed the silent content-loss defect for media, notes, conversations, and search-web "Use in Console" handoffs by removing the `"rag" in source` gate in `console_chat_controller.py`'s capture path. Task 9's review found that five OTHER live handoff kinds — `skills`, `watchlists_collections`, `library-source-snapshot`, `study`/quiz, and `personas`-attach — now carry a well-formed `evidence_bundle` from their respective builders, but `capture_console_staged_evidence_for_chat` still returns `(None, None)` for all five, because `RAG_Search/local_citation_capture.py`'s `_SOURCE_ALIASES` allowlist rejects their source types. This was empirically confirmed during review (`context=None` returned for each).

This is the same silent content-loss class Task 9 closed for the other four kinds: the UI shows the content as staged, but the model receives nothing on send.

Unlike media, notes, and conversations — which are checkable against a real DB row before being trusted as staged evidence — these five kinds are not backed by a DB row in the same way. Closing this requires a design decision about how non-DB-backed evidence should be validated for staged-evidence capture, not just an allowlist edit.

## Acceptance Criteria

- [ ] A decision is made and documented for how non-DB-backed evidence (skills, watchlists_collections, library-source-snapshot, study/quiz, personas-attach) is validated before being captured as staged evidence
- [ ] Each of the five kinds delivers its staged content to the model on send, or the UI stops claiming the content is staged if validation cannot be established for a given kind
- [ ] A capture round-trip test covers each of the five kinds, following the pattern Task 9 used for media/notes/conversations
