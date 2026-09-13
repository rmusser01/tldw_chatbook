# Design note: row state markers for Library ▸ Media (task-31278)

Status: APPROVED 2026-09-04 (Option A; see §5). Implementation: fix wave 5 row-markers PR (tasks 28008/28009).
Implements: task-28008 (analysis presence on rows), task-28009 (read/reviewed markers).
Author's note: the absence of these markers is my own earlier decision (review sets shipped with set-local done marks; the summary-contract bump was parked twice as "invasive, own PR"). This note is the proposal I owe for it.

## 1. The problem

The Items list renders each row as `title` + `type · age` and nothing else (critique #4, B cap_02). Two facts the reading loop depends on are invisible until the item is open:

- **Has this item been analysed?** Decides whether "review every analysis" will find anything, whether a batch Analyze is needed, and which rows Generate should target.
- **Have I reviewed this item?** The only ✓ today is the banner for the loaded item; finding the one skipped item means re-walking the set.

Both are blocked by the media summary contract.

## 2. The contract today

`tldw_chatbook/Library/library_media_state.py:65-66`:
```python
_MEDIA_SUMMARY_KEYS = frozenset({"id", "backing_media_id", "title", "media_type", "updated_at"})
```
`validate_media_browse_items` (`:150-190`) rejects any item whose key set is not exactly these five (`"Media browse items must contain exactly five summary keys."`), then checks the canonical id, positive backing id and page-uniqueness, and freezes the mapping.

Producers of the five-key shape:
- `tldw_chatbook/DB/Client_Media_DB_v2.py:2475-3040` — `search_media(library_summary=True)` selects only the summary columns (`:2638`, `:3040`).
- `tldw_chatbook/Media/media_reading_scope_service.py:501` (`_normalize_local_library_summary`) and `:717-776` (the `library_summary` branch of the scope search).
- `tldw_chatbook/Media/local_media_reading_service.py` (local backend passthrough).
- `tldw_chatbook/UI/Library_Modules/library_media_browse_controller.py:148` (the browse request sets `library_summary=True`).
- `tldw_chatbook/UI/Screens/library_screen.py` (Review-these / Review-selected page the whole result through the same projection).

Tests that build or assert the shape (six files): `Tests/UI/test_library_media_browse_controller.py`, `Tests/UI/test_console_rag_settings_modal.py`, `Tests/DB/test_client_media_debug_logging.py`, `Tests/DB/test_client_media_pagination.py`, `Tests/Media/test_local_media_reading_service.py`, `Tests/Media/test_media_reading_scope_service.py` — plus every UI test whose fake service returns browse rows (`_two_media_items`, `_many_media_items` in `Tests/UI/test_library_shell.py`, and the `StaticLibraryMediaScopeService` family), which is why the bump was called invasive.

## 3. Options

### Option A — Bump the contract to seven keys (recommended)

Add two keys to `_MEDIA_SUMMARY_KEYS`: `has_analysis: bool` and `reviewed: bool | None`.

- `has_analysis` is projected in SQL: `EXISTS (SELECT 1 FROM DocumentVersions v WHERE v.media_id = Media.id AND v.version_number = (SELECT MAX(version_number) …) AND TRIM(COALESCE(v.analysis_content,'')) <> '')` — the same "newest version's analysis" rule the viewer uses (`library_media_viewer_state._latest_version_analysis_text`), so list and Reader can never disagree.
- `reviewed` is **not** a media-DB fact. It is the active review set's done mark for that backing id (set-local by design, `Library/review_set_state.py`). The projection leaves it `None`; the screen decorates rows from the active set (`ReviewSet.items` is already in memory for the banner) before handing rows to the canvas. `None` means "no active set"; `False`/`True` mean the item is in the active set and not-yet/reviewed.
- Validation: the validator checks the exact seven-key set, `has_analysis is bool`, `reviewed in (None, False, True)`.
- Rendering: rows get a one-cell state slot before the type: `☐`/`☑` is taken by select mode, so use `✓` for reviewed, `·` for in-set-not-reviewed, nothing when no set; and `A` (or `¶`) for has-analysis in the secondary line: `article · 5m · analysed`. Text, not colour, carries the meaning (PRODUCT.md accessibility). At the 38-col Items pane the secondary line has room: `document · 5m · analysed` is 24 cells.

Cost: one SQL projection, one validator change, `_normalize_local_library_summary`, the local reading service passthrough, and every fake that returns browse rows must add the two keys (mechanical: a helper `_summary_row(**overrides)` in `Tests/UI/test_library_shell.py` that all fakes use, so future keys are one-line changes). Estimated 6 production files, ~8 test files.

### Option B — Keep five keys, add a side lookup

Leave the contract alone; the screen asks the media DB `analysis_present_for(ids)` per page and the review set for done marks, then the canvas takes an extra `row_flags: Mapping[str, RowFlags]` argument.

Pro: no contract bump, no fake churn. Con: a second query per page (20 ids, cheap) and a second source of row truth that the reconciler must keep in step across pager/sort/filter, exactly the drift the five-key freeze exists to prevent. Not recommended.

### Option C — Per-item read marker in the media DB (task-28009 as originally filed)

A `MediaReadState` row per item (read/unread independent of sets). Solves "have I read this ever" but not "have I reviewed this in this set"; the review-set done mark already exists and is the honest source for the walk. Defer; if wanted later it is a third key, not a redesign.

## 4. Decisions needed from the user

1. **Contract bump (Option A) vs side lookup (Option B).** Recommendation: A.
2. **Source of `reviewed`: active review set only (recommended), or a per-item read marker (Option C) as well?** Recommendation: set only, in v1.
3. **Glyphs and placement.** Recommendation: `✓`/`·` in the row's leading state slot (select mode's `☑/☐` replaces it while active), `analysed` as a word on the secondary line rather than a glyph.
4. **Scope of the migration PR.** Recommendation: one PR — contract + projection + validator + fakes helper + row rendering + painted-text tests at 38 cols; no batch Analyze in it (that is task-28007's PR).

## 5. Approval

- [x] User approved **Option A** on 2026-09-04 (critique #5 close, AskUserQuestion): decision 2 = `reviewed` from the active review set only (v1); decision 3 = `✓`/`·` in the row's leading state slot (select mode's `☑/☐` replaces it while active) and the word `analysed` on the secondary line, plus the matched keyword when a hit came from keywords rather than the title (critique #5 P2); decision 4 = one PR inside fix wave 5 (contract + projection + validator + fakes helper + row rendering + painted tests at 38 cols), no batch Analyze in it (shipped separately by PR #2400).
