---
id: TASK-35032
title: Mobile reading surface on /m (Phase 2b)
status: Done
created_date: 2026-10-16 17:06
updated_date: 2026-10-16 17:19
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Notes + library read-only browsing on /m per spec 2026-10-15 §4.2 phase-2
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] /m offers a Chats/Library view switch (bottom tab bar) in the existing design language; safe-area layout and the `[hidden]` guard pattern intact
- [x] Library view lists notes (title + preview rows) with a Load-more button when `has_more` (`?page=n+1`) and opens a note reader (title + full content, pre-wrap, Back)
- [x] Library view lists media (title + media_type rows) with Load-more and opens an item detail (metadata + transcripts joined as plain text, Back)
- [x] All rendering is textContent/createElement-only; no new JS dependencies; polling/visibility logic untouched (reading views refresh on open only)
- [x] Fetches ride the existing api() helper against exactly the four Task-2 GET routes (CSRF header on GET harmless; credentials same-origin)
- [x] Route-layer page ceiling: `page > 10_000` answers 400 invalid_page (OverflowError guard), tested
- [x] Gate-off still 404s the page, manifest, and app.js (asserted)
- [x] Targeted matrix green: mobile service 11 / api 27 / page 19; integration baseline 11 failures name-identical; Canvas 83, session_store 7, cli_pair 3, approval_constants 2, webauthn 7 green
- [x] Web_Server/README.md Mobile companion section documents the reading surface (routes, read-only guarantee, pagination)
- [x] ADR-234 amended in Consequences with the Phase-2b landing line
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

Task 3 of plan `Docs/superpowers/plans/2026-10-15-served-mobile-phase2b.md` (spec §4.2
phase-2; ADR-234 amendment). Builds on Task 1 (`7f0554fa6847` reading service) and
Task 2 (`24cdb325ad` reading routes); this task adds the consuming UI, the route-layer
page ceiling rider, docs, and close-out. Task-3 commit: `0ae3ac7399`.

**Approach**

- TDD: three new page tests (Library tab present on the authed page; served app.js
  carries the library markers + the four literal fetch-path fragments; served app.css
  carries the tab-bar/pre-wrap reading rules) written red first, plus one rider line
  (`?page=10001` → 400 invalid_page) added to `test_invalid_page_and_item_answer_400`
  and one strengthening line in the gate-off test (app.js also 404s).
- UI: a bottom tab bar (Chats | Library) — not header buttons, because the header's
  narrow-viewport fit is measured and pinned by
  `test_mobile_css_header_fits_narrow_viewports` (~305px cluster in a 370px row);
  a tab bar keeps the header untouched. The bar lives after `#app-main` in the body
  column, visible only on the two tab roots; detail views (chat, approvals, note,
  item) hide it, so the composer keeps owning its bottom safe-area inset while a
  visible bar owns it elsewhere.
- Library view: a Notes/Media segmented switch (`aria-pressed`) over two `<ul>`s;
  each loader fetches page 1 on entry (refresh-on-open) and "Load more" appends
  `?page=n+1` while `has_more`. Note rows are title + preview (preview wraps to its
  own line via flex-basis 100%); media rows are title + type chip (shares the title
  line). Readers: note view = title + meta line + full content in a `pre-wrap` `pre`;
  item view = title + meta line (media_type/media_id/ingestion_date/last_modified)
  + transcripts' `transcription` texts joined with a separator (or explicit
  no-transcripts copy). All dynamic strings render via textContent/createElement.
- Back returns to the active tab's root and refreshes it; tab taps re-enter (and
  thus refresh) the root. No changes to the WS stream, 12s interval, or
  visibilitychange handlers.
- Rider (controller ruling from Task-2 review): `_MAX_PAGE = 10_000` in
  `mobile_api.py`; `_parse_page` returns None above it (uniform 400 invalid_page),
  so a pathological page can never reach the offset arithmetic / SQLite int64
  binding range (OverflowError guard). Validation still answers before the service
  is touched.

**Files modified**

- `tldw_chatbook/Web_Server/static/mobile/index.html` (tab bar + library/note/item sections)
- `tldw_chatbook/Web_Server/static/mobile/app.js` (els/state/showView extension + library section)
- `tldw_chatbook/Web_Server/static/mobile/app.css` (tab bar, switch, entry lists, load-more, reading rules)
- `tldw_chatbook/Web_Server/mobile_api.py` (`_MAX_PAGE` page ceiling)
- `tldw_chatbook/Web_Server/README.md` (Library-tab paragraph: routes, read-only guarantee, pagination)
- `backlog/decisions/234-served-mobile-companion-surface.md` (Consequences amendment line)
- `Tests/Web_Server/test_mobile_page.py` (+3 tests, gate-off test strengthened)
- `Tests/Web_Server/test_mobile_api.py` (one rider line)

**Verification** (targeted pytest only, per plan constraint)

- Red first: the 3 new page tests and the rider line failed pre-implementation.
- Green: mobile trio 57 passed (service 11 / api 27 / page 19); session_store 7 +
  cli_pair 3 + approval_constants 2 + webauthn 7 = 19 passed; Canvas
  `Tests/Canvas/test_web_auth.py` 83 passed.
- Baselines name-identical (verified by stashing the change and diffing FAILED
  lists): `test_web_auth_integration.py` 11 failures (all environmental
  RecoveryRequired raw_participant_not_installed); `test_canvas_control_spawn.py`
  2 failures (environmental RecoveryRequired, pre-existing in this environment).
- `node --check` on app.js and `py_compile` on mobile_api.py OK. ruff/mypy not
  installed in this worktree venv (same as Tasks 1-2).

**Deviations**

- None from the brief. Design choice recorded: bottom tab bar over header nav
  (header fit is pinned by an existing CSS test; see Approach).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
