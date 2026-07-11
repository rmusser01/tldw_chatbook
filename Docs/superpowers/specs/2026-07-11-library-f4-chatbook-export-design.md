# F4 — Bulk Library export via chatbook zips — Design

**Status:** design approved in conversation 2026-07-11; corrected after code review (media flag, snapshot caps, service routing). User constraints: exports use the existing chatbook file/zip format (no new format); selection model is current-scope (no multi-select UI).

## Problem

The Library has no bulk export. Users need to take what the Library holds — media, conversations, notes — out of the app as a portable artifact. The chatbook format (`Chatbooks/chatbook_creator.py`, zip archives with a manifest + content tree) already models exactly this and has an importer, so export must produce chatbook zips.

## Entry points

- A new **Export** rail row in the Library, sibling of "Import media"; the section header copy becomes "Import / Export". Selecting it opens the in-canvas export form scoped to **everything**.
- Each browse canvas (Media list, Conversations, Notes) gains a compact "Export…" action opening the same form **pre-scoped** to that section with its current filter (e.g. the media type filter).
- Server runtime source: the Export row renders disabled with tooltip copy "Export packages local content only." (the creator reads local DBs; consistent with the scope-service gating pattern).

## Export form (in-canvas, Console-rail style)

- Scope summary line with **full-query counts** (see Scope resolution): `Everything: 128 media · 542 conversations · 87 notes` or `Media (type: video) · 12 items`.
- Name input, prefilled `Library export YYYY-MM-DD`. Optional description input.
- Media quality select — thumbnail / compressed / original (the creator's existing options), default thumbnail, helper line stating the trade-off ("original copies full media files into the zip").
- Destination via the existing save-file picker; the chosen path is normalized to `.zip` BEFORE any overwrite confirmation (the creator silently coerces suffixes, which could otherwise overwrite a different file than the one confirmed); path validated with `validate_path_simple`; existing-file overwrite requires explicit confirmation.
- Primary action: "Export chatbook". Single-flight: the button disables while an export runs.

## Scope resolution — NEVER from rendered snapshots

The Library canvases render capped snapshots (`LIBRARY_SOURCE_PAGE_SIZES`: conversations 50, media 50, notes 100). Resolving an export from them would silently truncate large libraries. Instead, a pure resolver module queries **full id lists** at export time, in the worker:

- Everything: all non-deleted media ids (media DB), all conversation ids, all note ids (ChaChaNotes DB) for the local user scope.
- Section scope: the same full queries with the canvas's **filter definition** re-applied as a query predicate (e.g. media type), never the rendered rows.
- Ids are emitted as strings of ints (the creator's media path does `int(media_id)`).
- The form's summary counts come from the same resolver, computed in a short thread worker when the form opens (DB reads stay off the UI thread; the summary line shows a brief loading placeholder until the counts land), so the user sees what will actually be exported.

## Execution

A thread worker (`@work(thread=True, exclusive=True, group="library_export")`):

1. Resolves the scope to `{ContentType: [ids]}` via the resolver.
2. Calls `local_chatbook_service.export_chatbook(...)` — the app-wired service seam wrapping `ChatbookCreator` — under its own private event loop (`asyncio.run`; the service methods are async-signature but synchronous-body and never touch the app loop). **`include_media=True` is ALWAYS passed when media is in scope**: the creator silently skips every selected media item when the flag is False (its default and the wrapper's default) — a media export without it would "succeed" with zero media. Quality is the form's select.
3. On success, registers the artifact via `local_chatbook_service.create_chatbook(...)` (registry record with name/description/file_path/tags) so the export appears as a first-class chatbook in the Artifacts window and Home's Recent feed. Zip first, registry only on success — no orphan records for failed exports.
4. Marshals completion via `call_from_thread`: done → notification with the output path, plus "N characters auto-included" when the creator's `dependency_info` reports auto-included conversation dependencies; failure → error line in the form (message from the creator), Export re-enabled for retry.

While running, the form shows a quiet "Exporting… (N items)" line. No progress callback or cancel exists in the creator — documented v1 limitation; the exclusive worker prevents concurrent exports.

## Error handling

- Unwritable/invalid destination: creator failure surfaces in the form's error line; no crash; no registry record.
- Empty scope (0 items): the Export button disables with helper copy "Nothing to export in this scope."
- Deleted-mid-export records: the creator already skips fetch-misses per item; the completion message carries its counts.

## Testing

- Resolver units: full-query id resolution vs seeded DBs incl. beyond-snapshot-cap datasets (seed 60+ conversations; assert all ids resolved — the truncation regression lock); filter-scoped resolution; empty scope.
- Round-trip integration: seed fixtures → export "everything" to a tmp path → `ChatbookImporter` (or direct zip inspection) asserts manifest counts AND that media entries carry their textual content (pins the include_media/quality semantics), then import into a fresh DB and assert counts.
- Pilots: form opens scoped from rail and from a section action; export runs through a real (small) creator call and lands the done notification; failure pilot (unwritable path) shows the error line and recovers; single-flight (second press while running is a no-op); server-mode row disabled.
- Visual QA before merge: export form (everything + section-scoped), running state, done notification, Artifacts/Home showing the new chatbook record.

## Out of scope (logged)

Multi-select row export; export progress/cancel (needs creator hooks); scheduled/automatic exports; server-content export; import-from-Library entry point (the Artifacts window already imports).
