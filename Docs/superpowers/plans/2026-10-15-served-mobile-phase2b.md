# Mobile Reading Surface (Phase 2b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Read-only notes and library browsing on `/m` — a Library tab with paginated notes (preview list + full reader) and media items (list + detail with transcript), served by the existing mobile API surface.

**Architecture:** `MobileChatService` gains lazy read-only handles for the notes DB (`ChaChaNotes_DB.list_notes/get_note_by_id`) and the media DB (`Client_Media_DB_v2.search_media_db` + `get_media_transcripts`), exposed through four GET routes under `/api/v1/mobile/` and a new Library view in the existing vanilla-JS page. No write paths, no new deps.

**Tech Stack:** existing (aiohttp, vanilla JS); no new dependencies.

**Spec:** `Docs/superpowers/specs/2026-10-15-served-mode-auth-and-mobile-surface-design.md` §4.2 Phase-2 additions
**ADR:** ADR-234 scoped Phase 2 out; this plan adds a one-line amendment recording the reading surface landing (Task 3).

## Global Constraints

- READ-ONLY: no write, create, edit, delete, or ingest paths anywhere in this phase. Media/notes DBs opened by the parent service for reads only; children remain migration owners (lazy open on first request — established Phase 1b pattern).
- All routes GET, behind the existing auth middleware (session; CSRF not applicable to GET), `Cache-Control: no-store`, inside the `mobile_surface_enabled()` gate; 404 when off.
- Pagination: `?page=` (1-based, default 1, max 100/page) mirroring the library API convention; responses carry `{"items": [...], "page": n, "has_more": bool}`.
- No HTML injection: page renders via textContent only; note/transcript content is plain text with `white-space: pre-wrap`.
- Known DB paths: notes DB + media DB resolved exactly as the app resolves them (config's data-root chatbook DB paths — read how `app.py`/`Chat` constructs `ChaChaNotes_DB` and `Client_Media_DB_v2` paths and mirror it; do not invent a second path scheme). If opening either DB can run migrations, note it in the report (accepted Phase 1b pattern: lazy open post-startup).
- Tests avoid config-loading paths; environmental baselines (11+4+209 RecoveryRequired) name-identical; green sets stay green (Canvas 83, session_store 7, cli_pair 3, approval_constants 2, webauthn 7, mobile 7/23/16 as of the current stack top).
- Worktree `/Users/macbook-dev/Documents/GitHub/tldw-auth-phase1a`, branch `feat/served-mobile-phase2b` off the current `feat/served-passkeys-phase2a` head (stacked). Targeted pytest only.

---

### Task 1: Reading service methods

**Files:**
- Modify: `tldw_chatbook/Web_Server/mobile_service.py` (lazy DB handles + read methods)
- Test: `Tests/Web_Server/test_mobile_service.py` (extend)

**Interfaces:**
- Consumes: `ChaChaNotes_DB.list_notes(limit, offset, preview_chars=200)`, `ChaChaNotes_DB.get_note_by_id(note_id)`, `Client_Media_DB_v2.search_media_db(search_query=None, page=, results_per_page=, include_trash=False, include_deleted=False)`, `get_media_transcripts(db_instance, media_id)` (module fn in `DB/Client_Media_DB_v2.py:8863`).
- Produces (added to `MobileChatService`):

```python
def notes_page(self, *, page: int = 1, per_page: int = 50) -> dict:
    # {"items": [{id, title, preview, updated_at?}], "page": n, "has_more": bool}
    # list_notes(limit=per_page+1, offset=(page-1)*per_page, preview_chars=200);
    # has_more from the (per_page+1)th row; items strip to id/title/preview + whatever
    # row keys list_notes returns for timestamps (pass through as-is, JSON-safe).

def note_detail(self, note_id: str) -> dict | None:
    # get_note_by_id -> {"id", "title", "content", other scalar row fields}; None if missing.

def library_page(self, *, page: int = 1, per_page: int = 25) -> dict:
    # search_media_db(search_query=None, page=page, results_per_page=per_page) ->
    # {"items": [{media_id, title, media_type, created_at/updated_at passthrough}], "page": n,
    #  "has_more": from the response's pagination/total if present, else items==per_page}
    # NOTE: search_media_db return SHAPE must be read at implementation (dict with results
    # list + total? list?) — adapt extraction, keep the wire shape above stable.

def library_detail(self, media_id: int) -> dict | None:
    # search_media_db(media_ids_filter=[media_id]) row -> {"media_id", "title", "media_type", ...}
    #  + {"transcripts": get_media_transcripts(db, media_id)} (list of {transcript-ish dicts},
    #  pass through JSON-safe); None when no row.
```

DB handles: `self._notes_db` / `self._media_db` built lazily in `_ensure_reading_dbs()` (thread-safe via the existing service lock/idiom; close in `shutdown()` alongside existing handles). DB paths mirror the app's own construction (read how the Chat/app layer builds them — e.g. via config data root helpers — and use the same helpers).

- [ ] Step 1: Failing tests — seeded tmp DBs (construct `ChaChaNotes_DB(path)` + insert 3 notes via its own write API in the fixture; `Client_Media_DB_v2(path)` + one media row + transcript via its write API): notes_page page-1 + has_more/page-2; preview not full body; note_detail full body + None for unknown id; library_page list + pagination; library_detail with transcript + None. TDD red first.
- [ ] Step 2: Implement; Step 3: green (whole file, existing tests intact); Step 4: Commit `feat(web-server): mobile reading service — notes + library reads (ADR-234 amendment)`.

---

### Task 2: Reading API routes

**Files:**
- Modify: `tldw_chatbook/Web_Server/mobile_api.py` (+4 routes in `build_mobile_routes`)
- Test: `Tests/Web_Server/test_mobile_api.py` (extend, fake-service idiom)

**Interfaces:**
- Produces:

| Route | Response |
| --- | --- |
| `GET /api/v1/mobile/notes?page=` | `200 {"items":[...],"page":n,"has_more":bool}` |
| `GET /api/v1/mobile/notes/{note_id}` | `200 {"note": {...}}` or `404 {"error":"unknown_note"}` |
| `GET /api/v1/mobile/library?page=` | `200 {"items":[...],"page":n,"has_more":bool}` |
| `GET /api/v1/mobile/library/{media_id}` | `200 {"item": {...}}` or `404 {"error":"unknown_item"}` |

Validation: `page` int ≥1 else `400 invalid_page`; `media_id` decimal int else `400 invalid_item`; all no-store; unknown_session-style uniform `{"error": ...}` shapes.

- [ ] Step 1: Failing tests (404s first; auth 401; shapes via fake service; gate-off 404s join the existing gate test). Step 2: implement (thin handlers, `_error` idiom). Step 3: green + integration baseline name-identical. Step 4: Commit `feat(web-server): mobile reading API routes (GET notes/library)`.

---

### Task 3: Library view, docs, close-out

**Files:**
- Modify: `Web_Server/static/mobile/index.html` + `app.js` + `app.css` (Library tab), `Web_Server/README.md`, `backlog/decisions/234-served-mobile-companion-surface.md` (amendment line), backlog task file (create via backlog-py at start of this task: "Mobile reading surface on /m (Phase 2b)")
- Test: `Tests/Web_Server/test_mobile_page.py` (extend)

**Interfaces:**
- Consumes: Task 2 routes. Produces: `/m` gains a bottom-tab or header nav switch between Chats and Library (both `textContent`-only):
  - **Notes list**: title + preview rows, infinite "Load more" via `?page=n+1` when `has_more`; tap → reader view (title + full content, pre-wrap, Back).
  - **Library list**: title + media_type rows, Load more; tap → detail view (metadata + transcripts joined as plain text, Back).
  - Views integrate the existing `[hidden]` guard pattern and safe-area layout; no new JS dependencies; polling/visibility logic untouched (reading views refresh on open only).
- [ ] Step 1: Failing page tests (Library tab present authed; gate-off absent; JS/CSS served and contain the library markers). Step 2: implement UI. Step 3: targeted matrix green + baselines name-identical. Step 4: README paragraph (read-only scope, routes, no-write guarantee) + ADR-234 amendment line ("Reading surface (notes + library, read-only) landed 2026-10-15; write paths remain out of scope.") + task close-out (ACs, notes, Done via backlog-py). Step 5: Commit `feat(web-server): /m Library view — notes + media reading (ADR-234 amendment)`.

---

## Self-Review

**Spec coverage:** §4.2 Phase-2 additions (notes + library read-only GETs) → Tasks 1-2; surface consumption → Task 3; §4.5 gate untouched (routes inside it). No write paths anywhere (Global Constraint). Out of scope: editing, ingest, RAG search over notes (future).
**Placeholder scan:** the one open shape (search_media_db return structure) is anchored for at-implementation reading with a stable wire contract; list_notes/get_note_by_id/get_media_transcripts signatures verified against source (lines 18304/18370/8863).
**Type consistency:** service method names match Task-2 handlers match Task-3 fetch paths; error codes `unknown_note`/`unknown_item`/`invalid_page`/`invalid_item` consistent.
