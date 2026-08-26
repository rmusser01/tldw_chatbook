---
id: TASK-15474
title: 'DB sundries: lazy BLOB logging and image-free list projections'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-11 12:05'
labels:
  - perf
  - db
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two small verified DB items from the audit. (1) Logging regression: `DB/ChaChaNotes_DB.py:5916` `logger.debug(f"Params: {final_params}")` is a plain eager f-string and `image` is in the updatable fields — saving a character card with an image builds a multi-MB repr of the raw BLOB on every update regardless of log level. This is exactly the pattern `DB/sql_logging.py` exists to prevent; the correct lazy form is 3k lines up in the same file (`:3014-3019`). Smaller eager stragglers: `DB/Client_Media_DB_v2.py:2184/:2245/:5819`. (2) Query shape: `character_cards.image` is a BLOB, and `list_character_cards` (`:5626`) / `list_character_cards_page` (`:5713/:5719`) select `*` — so the Console character picker deserializes up to 500 images to build a NAME list (`chat_screen.py:9000`), and personas paging/evals pickers drag the same BLOBs.

Fix direction: `logger.opt(lazy=True)` + `preview_params` for all four sites; explicit column projections excluding `image` for list/picker paths (detail views keep the full row). Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No eager params stringification remains in ChaChaNotes or Client_Media execute paths (grep + unit evidence)
- [x] #2 Character list/picker paths fetch no image column (evidence); card save/list behavior unchanged (tests)
<!-- AC:END -->

## Implementation Plan

1. Lazy logging: convert `ChaChaNotes_DB.py`'s `update_character_card` query+params debug
   lines (current line ~5920-5923; `final_params` carries `image`, a BLOB) to
   `logger.opt(lazy=True)` + `preview_params`, matching the established pattern at
   `:3021-3028`. Sweep `ChaChaNotes_DB.py` for any other eager params-interpolating
   log line (grep for `param`/`value` near `.debug(f"`/`.info(f"`/etc.) -- audit shows
   this is the only remaining site in this file.
2. Convert `Client_Media_DB_v2.py`'s three eager "Params:" sites (in the media search
   function, ~2192/2253, and `get_all_document_versions`, ~5827) to the same lazy
   pattern -- this file already has the `logger.opt(lazy=True)` + `preview_params`
   precedent at its own `execute_query` (~:823-826). Sweep for the smaller
   query-part/LIKE-pattern debug/info lines in the same search function
   (~2075/2076/2136) and convert those too for consistency (list-not-BLOB risk, but
   still eager stringification of params-shaped data).
3. Write a unit test per file asserting a sentinel bytes-like/repr-raising param
   passed through the execute path is never rendered when the effective log level is
   above DEBUG (and IS rendered, length-only, when DEBUG is admitted) -- proving
   laziness and BLOB-safety, not just presence of `opt(lazy=True)` in the source.
4. Image-free list projections: add an explicit column list for `character_cards`
   excluding `image` (mirroring the CREATE TABLE column order minus `image`), and a
   small helper to render it with/without a table alias. Add `include_image: bool =
   False` (keyword-only) to `list_character_cards` and `list_character_cards_page`;
   default excludes `image`, opt-in flag restores `SELECT *`/`cc.*` for the one
   caller that legitimately needs it.
5. Classify every caller of both functions (9 call sites found across
   Tools_Settings_Window, Evals evals_state/card_picker, watchlists_collections_screen,
   chat_screen, ChatbookCreationWizard, Chat_Functions x2, MCP/tools, Character_Chat
   local_character_persona_service + Character_Chat_Lib x2) -- update the one genuine
   image consumer (`Chat_Functions.load_characters`, base64-encodes `image` for a
   test-covered compatibility helper) to pass `include_image=True`; leave the rest on
   the new image-free default.
6. Tests: a test asserting the list/picker query text and/or returned row keys never
   include `image` by default, and a round-trip test that `include_image=True` still
   returns the BLOB unchanged. Run existing character/picker/evals-state suites green.
7. Measure `list_character_cards_page` before/after on a seeded scratch DB (few
   hundred cards, ~100KB images) -- isolated scratch probe, not the live app.

## Implementation Notes

**Lazy logging (AC #1).** Converted the one true BLOB-risk site --
`ChaChaNotes_DB.update_character_card`'s combined
`"Executing SINGLE character update query" | Params: {final_params}"` debug
line (`final_params` carries `image` on every card save) -- to
`logger.opt(lazy=True)` + `preview_params`, matching the file's own
`:3014-3028` precedent. Swept `Client_Media_DB_v2.py` and converted six more
eager sites: three explicitly named in the brief (`search_media_db`'s count
and results Params lines, `get_all_document_versions`'s Params line) plus
three lower-risk stragglers found during the sweep (`search_media_db`'s
FTS-query-parts/combined-FTS-query/LIKE-patterns debug/info lines) -- none of
these six carry a raw BLOB today (media content bytes are never a bound
*search filter* param), but they were still unconditional `str(...)` builds
on every call. Reviewed and left alone: `execute_many`'s debug/error lines
(log only `len(params_list)` or the exception text, never the params
collection) and the module's pre-existing `execute_query` lazy guard
(already correct, task-246). Grep evidence: no `f"..{...param..}"` literal
remains in either file outside the two reviewed-clean `execute_many` lines.
Tests: `Tests/DB/test_sql_debug_logging.py` gained a
`TestCharacterCardUpdateNoEagerBlobLogging` class (a `CountingBytes` sentinel
proves `update_character_card` never reprs a 2MB image blob, with and
without a DEBUG sink attached, plus a source-inspection regression check).
New `Tests/DB/test_client_media_debug_logging.py` proves the six converted
`Client_Media_DB_v2.py` sites still fire under an explicit DEBUG sink and
contain no eager `f"...{params}"` literal in source -- call-counting on
`preview_params` was tried first but rejected: this test process has an
ambient loguru default sink (level DEBUG) live for the whole pytest session,
so "not called without a sink" isn't a meaningful invariant here; BLOB-safety
itself is already proven directly by the `CountingBytes` tests.

**Image-free list projections (AC #2).** Added `_CHARACTER_CARD_LIST_COLUMNS`
(the `character_cards` CREATE TABLE column list minus `image`) and a
`_character_card_select_columns(include_image, alias)` helper to
`ChaChaNotes_DB.py`. Both `list_character_cards` and
`list_character_cards_page` gained a keyword-only `include_image: bool =
False`; default now projects the explicit image-free column list, opt-in
restores `SELECT *`/`cc.*`. Hit and fixed a real SQLite quirk along the way:
`list_character_cards_page`'s search branch joins `character_cards_fts`
(which also has a `name` column) against `character_cards cc`; the old code
relied on `cc.*` wildcard expansion being exempt from SQLite's ORDER BY
ambiguity check, and swapping in an explicit `cc.name` column list broke
that exemption ("ambiguous column name: name"), caught immediately by the
existing `test_search_composes_with_tag_and_sort` test going red. Fixed by
parameterizing `_CHARACTER_SORT_CLAUSES` with an `{a}` alias placeholder
(`cc.` when searching, `""` otherwise) filled in by `_resolve_sort_clause`.

Checked every caller of both functions (9 production call sites):
`Tools_Settings_Window._export_characters_worker` (JSON backup dump --
already crashes on `json.dumps()` for any image-bearing card today since raw
`bytes` isn't JSON-serializable and nothing base64-encodes it first; the new
default silently drops the already-broken field instead of crashing --
improvement, not a regression, and out of scope to fully fix here),
`Evals/evals_state.character_cards` (bench-editor picker, id/name only),
`watchlists_collections_screen._load_character_options` (briefing preset
Select, `(name, id)` tuples), `chat_screen._console_character_picker_options`
(id/name/description, matches the task's own claim),
`ChatbookCreationWizard` (content-node title/subtitle),
`Chat_Functions.get_character_names` (names only; also dead code -- zero
production callers), `MCP/tools.list_available_characters`
(id/name/description/message_count), `Character_Chat_Lib.
get_character_list_for_ui`/`get_character_page_for_ui` (id/name[/description/
tags], the task's own "personas paging" reference) and
`local_character_persona_service.list_characters` (a generic passthrough
with zero production UI callers -- personas' own local-mode paging goes
through `get_character_page_for_ui` instead; only exercised by its own unit
tests, none of which touch `image`). Exactly one caller is a genuine image
consumer: `Chat_Functions.load_characters` base64-encodes `image` into
`image_base64` for a test-covered (if currently UI-dead) compatibility
helper -- updated to pass `include_image=True`.

Tests added: `Tests/DB/test_character_cards_paging.py` gained five tests
(image excluded by default in both the plain-browse and FTS-search branches,
`include_image=True` round-trips the exact bytes in both branches, and a
"no other field regressed" check covering every non-image column).
`Tests/ChaChaNotesDB/test_chachanotes_db.py` gained two tests for
`list_character_cards` (default excludes `image`, `include_image=True`
round-trips it).

**Performance (isolated scratch probe, not the live app).** 400 character
cards seeded with 100KB images in a scratch `CharactersRAGDB`;
`list_character_cards_page(limit=100)` timed 15x each for `include_image=True`
(reproduces the exact pre-task query shape) vs the new default, back to back
in the same process against the same DB. Two runs: 63.95ms -> 2.67ms mean
(23.97x) and 41.90ms -> 2.82ms mean (14.88x) -- consistent order-of-magnitude
win, as expected for skipping ~10MB of BLOB deserialization per 100-row page.

**Test runs (READ pass counts).** `Tests/DB/` full: 861 passed, 32 failed
(pre-existing ChaChaNotes V33->V34 migration suite, confirmed unrelated --
same file/line-independent schema-migration failures called out as
known-pre-existing in the task brief). `Tests/Media_DB/`: 100 passed, 6
skipped (unrelated sync-server integration tests). `Tests/Character_Chat/`:
545 passed. Combined caller-suite run (`test_chat_functions`,
`test_evals_card_picker`, `test_ccp_handlers`,
`test_watchlists_briefing_presets_ui`, `test_chatbook_integration`,
`test_library_export_roundtrip`, `test_character_cards_paging`,
`test_chachanotes_db`, `Character_Chat/`, `Media_DB/`): 881 passed, 1 known
pre-existing failure, 6 skipped. `Tests/UI/test_tools_settings_window.py` +
`test_settings_tools_section.py`: 74 passed. One order-dependent flake was
observed in a large ad hoc 9-file combo (`test_uat_first_time_character_chat`
failing only under randomized test order); confirmed unrelated to this
change -- passes in isolation, in every smaller combo tried, and with
`-p no:randomly`. `Tests/MCP/test_builtin_tool_imports.py` has 4 pre-existing
errors unrelated to this task (network-blocked HuggingFace embedding-model
download in the sandboxed test environment, from `MCPTools.__init__`'s RAG
service construction -- nothing to do with character list/logging code).

**Files modified:** `tldw_chatbook/DB/ChaChaNotes_DB.py`,
`tldw_chatbook/DB/Client_Media_DB_v2.py`, `tldw_chatbook/Chat/Chat_Functions.py`,
`Tests/DB/test_sql_debug_logging.py`, `Tests/DB/test_character_cards_paging.py`,
`Tests/ChaChaNotesDB/test_chachanotes_db.py`. **Files added:**
`Tests/DB/test_client_media_debug_logging.py`.
