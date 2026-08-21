---
id: TASK-19576
title: >-
  Two reachable user paths broken by retired-surface drift — STTS AudioBook
  import crashes, Console PDF/Word/ebook attach points at a retired tab
status: To Do
assignee: []
created_date: '2026-08-21 20:22'
labels:
  - bug
  - stts
  - console
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 1 (architecture & reachability) — its
**#1** and **#2**. These are the two findings the lane's reachability census
promoted rather than downgraded: unlike the media and Reading-Highlights
defects it explicitly set aside as unreachable, **both of these sit on live,
routed, user-reachable surfaces.** Both re-verified at this branch base.

**A — STTS AudioBook "Import from Notes/Conversation" crashes.** CONFIRMED by
the lane's runtime probe, **independently re-confirmed by the review
controller**, and again here. `UI/STTS_Window.py` imports four database
functions **that do not exist**:

```
592:  from tldw_chatbook.DB.ChaChaNotes_DB import fetch_all_notes
605:  from tldw_chatbook.DB.ChaChaNotes_DB import fetch_note_by_id
648:  from tldw_chatbook.DB.ChaChaNotes_DB import fetch_all_conversations
666:  from tldw_chatbook.DB.ChaChaNotes_DB import fetch_messages_by_conversation_id
```

A grep for `def fetch_all_notes` / `def fetch_all_conversations` in
`ChaChaNotes_DB.py` returns **zero**. The imports sit **outside the `try:`**,
so the `ImportError` propagates rather than being handled.

The full reachable path was traced end to end: `stts` route → `STTSScreen` →
rail entry "📚 AudioBook/Podcast" → widget mounted at `STTS_Window.py:1935` →
the `("Notes","notes")` Select at `:335` → `#import-content-btn` at `:470` →
`_import_content` at `:520` → `_import_from_notes`. Every step is live.

**Fix:** route through the existing `notes_scope_service` /
`chat_conversation_scope_service` rather than resurrecting module-level DB
functions — the scope services are the current, supported access path.

**B — attaching a PDF, Word document or ebook in Console tells the user to go
to a tab that no longer exists.** CONFIRMED.
`Utils/file_handlers.py` returns a placeholder instead of extracting content,
and the placeholder names a retired destination — at **five** sites, not the
three the review summary reported:

```
327:  f"To process this PDF file, please use the Media Ingestion tab."
348:  f"To process this document file, please use the Media Ingestion tab."
369:  f"To process this ebook file, please use the Media Ingestion tab."
397:  f"To process large text files, please use the Media Ingestion tab."
406:  f"To process this file for RAG search, please use the Media Ingestion tab."
```

The Media/ingest route is aliased to Library in
`UI/Navigation/screen_registry.py`, so the tab the message names cannot be
opened. `PDFFileHandler` is second in the handler chain
(`file_handlers.py:489-498`), so it wins for **every** `.pdf` — this is not an
edge case.

**The app already ships a real extractor.** `local_file_ingestion.py`
(`parse_local_file_for_ingest`) is what Library ▸ Import uses. Pointing the
stubs at it is the durable, pragmatic fix — the capability exists and is
already trusted on another surface.

## Acceptance Criteria

- [ ] STTS AudioBook "Import from Notes" and "Import from Conversation"
      complete without raising, using `notes_scope_service` /
      `chat_conversation_scope_service`
- [ ] The imports at `STTS_Window.py:592/605/648/666` no longer reference
      symbols that do not exist
- [ ] Attaching a PDF, Word document or ebook in Console extracts real content
      via `parse_local_file_for_ingest`, rather than returning a placeholder
- [ ] All five `file_handlers.py` placeholder sites are resolved, not just the
      three named in the review summary
- [ ] No user-facing string anywhere directs the user to the "Media Ingestion
      tab" or any other retired destination — a test greps for retired
      destination names in user-visible copy
- [ ] Both paths are verified by actually driving the app, not only by unit
      test — these are runtime reachability defects and a green import test
      would not have caught either
- [ ] `Docs/User_Guide/` pages covering STTS AudioBook import and Console
      attachments are updated to match the repaired behaviour
