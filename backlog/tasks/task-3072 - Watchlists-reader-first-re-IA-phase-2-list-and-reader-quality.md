---
id: TASK-3072
title: 'Watchlists reader-first re-IA, phase 2: list and reader quality'
status: Done
assignee: []
created_date: '2026-08-07 17:50'
updated_date: '2026-08-08 04:49'
labels:
  - watchlists
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement phase 2 of the reader-first design (Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md, ADR-042): reader rows with snippet/relative-date/date-groups/unread-bold/star/ingested+queued markers, s key + Starred smart feed, Subscriptions/content_render.py, o open-in-browser, reading-pane action row via shared helpers, position footer, Subscriptions/item_dates.py date helper.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Article list rows show source+relative time, bold-when-unread title, 1-2 line snippet, unread dot, star/queued/ingested markers, and Today/Yesterday/locale-date group headers,s toggles star and a Starred smart feed with count appears in the rail; flags persist across re-fetch,Reading-pane action row (Star/Mark unread/Open in browser/Ingest/Queue) calls the same shared helpers as the inspector,o opens the item in the browser,Item body renders via content_render.py with stdlib fallback; hostile HTML renders safely as text and failure never blanks the pane,Position footer shows N of M within the displayed list plus a Next Unread control,Tests/Watchlists green
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-08-07-watchlists-reader-first-phase-2-list-reader-quality.md

ADR required: no (already exists)
ADR path: backlog/decisions/042-watchlists-reader-first-ia.md
Reason: ADR-042 covers the re-IA; phase 2 is a direct implementation of it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase 2 executed TDD per the plan (Docs/superpowers/plans/2026-08-07-watchlists-reader-first-phase-2-list-reader-quality.md), one commit per plan task, following the phase-1 playbook.

**Per task:** af2b1ca97 docs (plan + this task); 89ff11856 item_dates helper (effective date, relative time, local-day buckets, reusing humane_time's parse); 3a9a353b6 is_flagged plumbing (Subscriptions_DB.set_item_flagged / get_flagged_items_count, get_new_items predicate, list_items forwarding, normalizer key); af0ea888b effective-date ordering (COALESCE published/created DESC) + html_text.body_snippet; b96060e63 ArticleListPane (ListView reader rows under disabled day-header rows, Unread/All filter pushed into the query, glyphs); cd1fee027 drive-by fixture fix; cb1e8ece5 Starred rail feed (TreeScope "starred", badge via the existing counts path, {"is_flagged": True} scope mapping); 60d6b0c00 `s` star toggle + reader Star button (controller→scope→local→DB set_item_flagged chain, local-only like every item write); b4c524835 `o` open-in-browser + reader action row; 1e3f31855 position footer with Next Unread; plus the help-text/hostile-pin/docs commit closing this task.

**Key decisions:**
- The spec's `Subscriptions/content_render.py` was recorded as ALREADY satisfied by `Subscriptions/html_text.py` (TASK-2307): dependency-free stdlib HTMLParser, `readable_body_text` in `render_article` today, html2text an optional extra that is not required. No new renderer module; phase 2 added only `body_snippet`.
- `_repaint_row(item_id, **writes)` is display-only (per-row `display_overrides`, never written back to the shared item dict) — exact ItemsPane parity, and load-bearing: the dicts are shared with `_selected_content_item`/`ContentPane.item` and their staleness is the mark-unread guard's pinned invariant.
- Queued glyph diverges from ItemsPane (◆ not ●) because ● is the unread dot in the reader rows.
- Legacy filter values ("new"/"all statuses") are normalized to the reader vocabulary (Unread/All) at the pane-seed and filter-changed boundaries.
- The Starred badge rides the existing debounced counts path (STARRED_BUCKET in the counts mapping) — no new refresh plumbing; it is status-agnostic so it never shrinks as starred items are read (which would read as data loss).
- The reader's verbs share the Inspector's at the message layer (same `IngestRequested`/`ToggleBriefingQueueRequested` classes) — the no-drift mechanism the spec asked a "shared module" for, without a new module.
- `o` validates the scheme (http/https only) before `webbrowser.open`: the URL is a remote-derived string reaching an OS primitive.

**Disclosure (pre-existing on origin/dev, not from this branch):** the briefing overflow test had aged out of its hard-coded 7-day fixture window — fixed here as cd1fee027 (now-relative date); 6 other parity failures (schedules/mcp/main-nav) and 43 CI environment failures (42 Notes git-SSH + 1 real-server flake) reproduce on origin/dev unchanged.

**Tests:** Tests/Watchlists/ and Tests/Subscriptions/ green; coupled Tests/UI watchlists files green; ruff clean on every touched file. New coverage includes per-layer star-write routing, the reader filter's literal vocabulary contract, display-only repaint invariants, scheme-refusal for `o`, footer position tracking, and the hostile-HTML end-to-end pin (star + queue + inert render + flags survive re-persist).

**Files:** new `UI/Watchlists_Modules/article_list.py`, `Subscriptions/item_dates.py`, `Tests/Watchlists/test_watchlists_article_list.py`, `Tests/Subscriptions/test_item_dates.py`; modified `UI/Watchlists_Modules/{watchlist_tree,content_pane,watchlists_backend_controller,humane_time}.py`, `UI/Screens/watchlists_collections_screen.py`, `Subscriptions/{watchlist_bundle_service,local_watchlists_service,watchlist_scope_service,html_text}.py`, `DB/Subscriptions_DB.py`, `Subscriptions/watchlist_normalizers.py`, and the migrated Tests/{Watchlists,UI}/ suites listed in the per-task commits.
<!-- SECTION:NOTES:END -->
