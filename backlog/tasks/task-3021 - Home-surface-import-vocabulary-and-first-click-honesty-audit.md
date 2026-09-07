---
id: TASK-3021
title: Home-surface import vocabulary and first-click honesty audit
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 12:20'
updated_date: '2026-08-09 19:04'
labels:
  - home
  - ux-copy
  - uat-2026-08-06
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-2857 unified the Library's user-facing vocabulary on Import/Export, deliberately leaving
Home-surface strings out of scope. Recorded during that arc (positions at `6672ed276`):

1. `Home/active_work_adapter.py:410` — "Opening Library ingest job details."
2. `app.py:4403` — `HomeControlResult` message "This ingest job can no longer be retried."
   (rendered on the Home screen's Retry action)
3. `Docs/User_Guide/home.md` ("opens Study at flashcards", ~line 75) — Home's Study rows need the
   same first-click-honesty audit task-2854 applied to the Library rail (does the first click land
   on Study, or on a staging surface?). Verify live before rewording.

Scope: bring Home's user-facing strings in line with the Import/Export vocabulary where they name
the same concept, and make Home's Study glosses honest about their first-click destination.
"Chatbook"-as-app-name usages (File Notes panels) are a separate, larger naming decision — out of
scope here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Home-surface strings naming Library import jobs use the Import vocabulary
- [x] #2 Home's Study row glosses/docs describe the actual first-click destination (verified live)
- [x] #3 Changed strings inventoried in the task notes; affected user-guide pages updated or re-stamped
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Grep sweep tldw_chatbook/Home/ + Home-rendered strings in app.py for 'ingest' vocabulary; classify every hit fixed-or-justified.
2. Fix the two user-facing Home strings (active_work_adapter.py retry-details message, app.py HomeControlResult retry-unavailable message) to Import vocabulary; update the two tests asserting the old strings.
3. Live-verify Home's 'Review flashcards' first-click destination (seed a due flashcard, click through in a real tmux run).
4. Sync Docs/User_Guide/home.md's quoted strings/prose to the new vocabulary and re-stamp; file any live navigation bug found as a separate task rather than fixing it here.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: redid the plain-grep sweep standard (task-2857's own method) over tldw_chatbook/Home/*.py and Home-rendered strings in app.py, since the prior (crashed) session's sweep evidence hadn't survived.

Sweep result — 'ingest' hits classified:
- Home/active_work_adapter.py: 2 user-facing strings found, both were already the known findings from the task description — FIXED to Import vocabulary: 'Opening Library ingest job details.' -> 'Opening Library import job details.' (line ~410); all other ~50 hits in this file are internal identifiers (LOCAL_INGEST_ITEM_ID_PREFIX = "local:ingest:"), function/variable names (_ingest_job_title, _local_ingest_job_items), or code comments/docstrings — not rendered to users, justified as out of scope.
- Home/dashboard_state.py: all ~15 hits are the same item_id-prefix checks / comments / docstrings — no user-facing text, justified.
- Home/home_rail_state.py, Home/__init__.py: zero hits.
- app.py: one user-facing string — FIXED: HomeControlResult 'This ingest job can no longer be retried.' -> 'This import job can no longer be retried.' (~line 5241, the Home Retry-unavailable toast). The other ~90 hits in app.py are import statements, TAB_INGEST constant, log/exception messages, worker/pool names, and internal method names — none rendered on the Home screen; the one adjacent user-facing string in the same handler ('Retry queued for {basename}.') never contained the word.

Tests: updated the 3 assertions in Tests/UI/test_home_screen.py pinned to the two old strings. Full file run: 60 passed, 0 failed (created a venv for this worktree first — it had none; VIRTUAL_ENV=.venv uv pip install -e '.[dev]').

First-click honesty: live-verified in a fresh scratch profile (TLDW_CONFIG_PATH, tmux) — seeded one due flashcard directly via sqlite3 (ChaChaNotes_DB has no bare-python seeding path per repo convention), clicked Home's Review flashcards row via injected mouse click. Confirmed: the first click IS honest — it lands directly on Study's flashcards section, matching home.md's existing 'opens Study at flashcards' gloss (no reword needed there). But pressing Escape from that Study screen lands on Library's 'Study decks' staging canvas, not back on Home — breadcrumb/Escape are hardcoded to a Library origin (task-2854 only considered that one origin). This matches the prior session's task-4011 draft exactly; verified it independently and left the filing as-is (no changes needed) plus added a matching Quirks bullet to home.md. Per the brief, did not fix the navigation bug itself in this branch.

Docs: Docs/User_Guide/home.md — synced 4 prose/quoted-string spots to Import vocabulary (Needs Attention bullet, Retry/Open details button table rows, Related settings & docs line), added a Quirks bullet documenting the Escape-to-Library-not-Home gap (task-4011), and re-stamped 'Verified against dev @ 4d0232358 — 2026-08-09'.

Files: tldw_chatbook/Home/active_work_adapter.py, tldw_chatbook/app.py, Tests/UI/test_home_screen.py, Docs/User_Guide/home.md, backlog/tasks/task-3021, backlog/tasks/task-4011 (new filing, left in To Do).
<!-- SECTION:NOTES:END -->
