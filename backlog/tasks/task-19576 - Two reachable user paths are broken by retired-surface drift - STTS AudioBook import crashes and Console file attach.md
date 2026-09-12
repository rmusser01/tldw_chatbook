---
id: TASK-19576
title: >-
  Two reachable user paths broken by retired-surface drift — STTS AudioBook
  import crashes, Console PDF/Word/ebook attach points at a retired tab
status: Done
assignee: [claude]
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

- [x] STTS AudioBook "Import from Notes" and "Import from Conversation"
      complete without raising, using `notes_scope_service` /
      `chat_conversation_scope_service`
- [x] The imports at `STTS_Window.py:592/605/648/666` no longer reference
      symbols that do not exist
- [x] Attaching a PDF, Word document or ebook in Console extracts real content
      via `parse_local_file_for_ingest`, rather than returning a placeholder
- [x] All five `file_handlers.py` placeholder sites are resolved, not just the
      three named in the review summary
- [x] No user-facing string anywhere directs the user to the "Media Ingestion
      tab" or any other retired destination — a test greps for retired
      destination names in user-visible copy
- [x] Both paths are verified by actually driving the app, not only by unit
      test — these are runtime reachability defects and a green import test
      would not have caught either
- [x] `Docs/User_Guide/` pages covering STTS AudioBook import and Console
      attachments are updated to match the repaired behaviour

## Implementation Plan

1. **Defect A (STTS AudioBook import).** Replace the two nonexistent-import
   sites in `UI/STTS_Window.py` (`_import_from_notes`,
   `_import_from_conversation`) with async workers that route through
   `self.app.notes_scope_service.list_notes(scope=ScopeType.LOCAL_NOTE, ...)`
   and `self.app.chat_conversation_scope_service.list_conversations(...)` /
   `.get_messages_with_context(...)` -- the same seams Home/Library already
   use (`getattr(self.app_instance, "notes_scope_service", None)` pattern).
   Dispatch via `@work` since the services are async; use
   `await self.app.push_screen(dialog, wait_for_dismiss=True)` for the
   existing `NoteSelectionDialog`/`ConversationSelectionDialog` modals
   instead of the old sync-callback form. Degrade to a notify (not a crash)
   when the service attribute is absent.
2. **Defect B (file_handlers.py census).** Read all five placeholder sites
   confirmed by the task description. Add a shared
   `_extract_local_ingest_text` helper that calls
   `parse_local_file_for_ingest` off-thread (`asyncio.to_thread`, no DB
   write, no LLM analysis) and wire `PDFFileHandler`/`DocumentFileHandler`/
   `EbookFileHandler` to it, each with an honest failure-path message on
   exception. For `PlaintextDatabaseHandler`'s two sites (a deliberate
   size-based deferral, not a missing-capability stub), keep the existing
   behavior and only fix the destination name to "the Library tab" (a live
   route; Library is not itself aliased away).
3. Add born-red tests for both defects, confirming failure at the
   pre-fix base (origin/dev) and success after the fix.
4. Run the STTS suites, Chat attachment tests, and a repo-wide
   `--collect-only -q` sweep; baseline any unrelated failure against a
   clean `origin/dev` checkout before attributing it to this change.
5. Live-verify both paths in the real, running TUI (`verify` skill,
   isolated scratch profile) rather than relying on unit tests alone.
6. Update `Docs/User_Guide/console/attachments-images-voice.md`'s
   "Verified against" stamp (its prose already described the target
   behavior; the fix makes that prose true rather than requiring a wording
   change).

## Implementation Notes

**Defect A.** `_import_from_notes`/`_import_from_conversation` in
`UI/STTS_Window.py` are now thin sync dispatchers to new async
`@work`-decorated methods (`_import_from_notes_worker`,
`_import_from_conversation_worker`). Notes: `notes_scope_service.list_notes`
already returns full note rows (title+content, not previews), so selected
notes are combined straight from that one page -- no second per-note fetch.
Conversations: `chat_conversation_scope_service.list_conversations(mode=
"local", scope_type="all", ...)` lists, then
`.get_messages_with_context(conversation_id, mode="local", ...)` pages
messages (200/page, capped at 5000 total) -- the removed
`fetch_messages_by_conversation_id` had no limit at all, so this is a
deliberate bound, not a behavior regression. Both dialogs are shown via
`await self.app.push_screen(dialog, wait_for_dismiss=True)` instead of the
old sync-callback form; both workers degrade to a notify (never a crash)
when their scope-service attribute is absent. Fixed a latent bug found
while rewriting the conversation path: the removed code compared
`role != "user"` against `sender` values the DB stores capitalized
("User"), so the "User/Assistant messages only" radio options could never
have matched anything even before the crash -- the new filter
case-normalizes.

**Defect B.** Census of `Utils/file_handlers.py` found exactly the five
sites the task named. Disposition:
- `PDFFileHandler`, `DocumentFileHandler`, `EbookFileHandler` (3 sites) --
  wired to a new shared `_extract_local_ingest_text` helper that calls
  `parse_local_file_for_ingest` (the same extractor Library ▸ Import uses)
  off-thread via `asyncio.to_thread`, with `options={}` (no chunking, no
  LLM analysis -- `perform_analysis` defaults False). An extraction
  failure or empty-text result produces an honest message naming no
  destination, instead of falling back to the old placeholder.
- `PlaintextDatabaseHandler`'s two sites (>100KB, >10MB) -- kept the
  existing deliberate size-based deferral (large text files are NOT
  inlined into a chat turn's context; this is a genuine cost/context-
  window concern, not a missing-capability stub) and only fixed the
  destination name, from the retired "Media Ingestion tab" to "the Library
  tab" (a live, un-aliased route -- `media`/`ingest` themselves alias to
  `library` in `screen_registry.py`).
- No new size cap was invented for PDF/document/ebook: the existing
  `Chat/attachment_core.py` `MAX_ATTACHMENT_BYTES` (100MB) already gates
  every attachment before any handler runs, and Library ▸ Import imposes
  no additional file-size cap of its own at this layer either.

**Born-red evidence.**
- `Tests/UI/test_stts_audiobook_import_scope_services.py` (5 tests): each
  calls `_import_from_notes()`/`_import_from_conversation()` directly (no
  `pytest.raises`); at `origin/dev` baseline all 5 fail with the exact
  `ImportError: cannot import name 'fetch_all_notes' from
  'tldw_chatbook.DB.ChaChaNotes_DB'` signature (and the three sibling
  names). All 5 pass after the fix, driving real fake-service note/
  conversation selection through to `widget.content_text`.
- `Tests/Chat/test_attachment_local_ingestion_extraction.py` (7 tests):
  the primary one builds a REAL PDF with fitz/pymupdf and drives it
  through the unmocked extraction chain, asserting genuinely extracted
  text comes back and the retired-tab string does not; a census test
  drives all five sites and greps their actual returned copy (not source
  comments) for the retired name. All 7 fail at `origin/dev` baseline
  (showing the literal old placeholder strings) and pass after the fix.
- Both baselines were confirmed against a clean, separate `origin/dev`
  worktree (not by reverting this branch), per repo convention.

**Live verification.** Used the `verify` skill with an isolated scratch
`TLDW_CONFIG_PATH` profile (real user config/DB confirmed untouched
afterward): clicked STTS ▸ Speech ▸ AudioBook/Podcast ▸ Import Content for
both "Notes" and "Conversation" sources on a real running app -- no crash,
app remained fully navigable, and the in-app Logs screen showed 0 errors
across both clicks (proving no ImportError, since the base code's crash
was synchronous and would have surfaced there). Separately, pasted a real
PDF's path into a real Console composer (a fresh provider-configured
scratch profile) and confirmed via the Logs screen that the full real
pipeline ran end to end: `Chat.attachment_core` → `Local_Ingestion.
local_file_ingestion` → `Local_Ingestion.PDF_Processing_Lib` ("Text
extracted successfully ... using pymupdf4llm" / "Successfully processed
PDF") -- not the old placeholder, which performed no I/O and logged
nothing.

**Surprise / out-of-scope finding.** `Tests/Architecture/
test_persistent_diagnostic_inventory.py::test_production_diagnostic_inventory_and_sink_topology_are_unchanged`
fails on a clean `origin/dev` checkout with zero of this task's changes
(confirmed in an isolated worktree) -- pre-existing drift between
`Docs/security/production-diagnostic-inventory.json` and the current
source, unrelated to task-19576 and spanning ~10 files this task never
touched (`ChaChaNotes_DB.py`, `console_chat_controller.py`,
`chat_screen.py`, `library_screen.py`, etc.). This task's own 3 new
`logger.error(...)` call sites (in the new PDF/Document/Ebook extraction
failure paths) were reviewed for safety -- they log only the filename and
exception message, matching the existing safe pattern already used by
every other handler in this file -- but regenerating the inventory
wholesale would have bundled ~10 unrelated files' drift into this
branch's diff, so that regeneration was reverted and left for whoever
owns the broader drift. Not fixed here; flagging for the owner.

**Files changed:** `tldw_chatbook/UI/STTS_Window.py`,
`tldw_chatbook/Utils/file_handlers.py`,
`Docs/User_Guide/console/attachments-images-voice.md` (stamp only),
`Tests/UI/test_stts_audiobook_import_scope_services.py` (new),
`Tests/Chat/test_attachment_local_ingestion_extraction.py` (new).
