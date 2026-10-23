# Study and flashcards

This document describes the Study destination: its sections and their honest pane status, the learning tables and SM-2 scheduling, the Library handoff and Home due-row entries, review discipline, and the local/server asymmetries.

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Screen | `UI/Screens/study_screen.py`, `study_scope_models.py` | `StudyScreen(BaseAppScreen)`; `STUDY_INITIAL_SECTIONS` (dashboard, paths, flashcards, quizzes, mindmaps, course, guides, learning_map); origins `{home, library}` |
| Surface | `UI/Study_Window.py`, `UI/Study_Modules/flashcards_handler.py`, `quizzes_handler.py`, `Widgets/Study/study_dashboard.py`, `quiz_session_widget.py` | `StudyWindow`, `StudyFlashcardsController` (SM-2 submission discipline), `StudyQuizzesController` |
| Services | `Study_Interop/` | `LocalStudyService` / `ServerStudyService` / mode-aware `StudyScopeService` (+ quiz variants), normalizers, recovery adapter |
| Persistence | `DB/ChaChaNotes_DB.py` | the learning tables (see below); SRS updates; due queries |
| Library handoff | `Library/library_shell_state.py`, `Widgets/Library/library_entry_canvases.py`, `UI/Library_Modules/screen_constants.py` | the Study rail section + the "Continue in Study" staging canvas |
| Entries | `app_destinations.py::open_study_screen`, `app.py` due-count provider, `Home/dashboard_state.py` | three staged handoff channels; Home's flashcards-due row |

Study is a full screen **without** Library chrome: no nav tab highlights (`nav_bar_active = ""`), a breadcrumb header as the only location statement, and a screen-scoped Escape binding.

## Honest pane status

- **Functional**: flashcards (deck/card CRUD, review, search, move/delete) and quizzes (quiz/question CRUD, attempts, history).
- **Placeholders rendering honest notices**: structured learning paths, mindmaps, course creation, study guides, learning map — the underlying `learning_paths`/`topics`/`mindmaps` tables are write-only (no list/get readers exist), and no course/guide/milestone schema exists. Dead buttons were removed rather than left dangling.

## Schema

The learning tables predate the file-backed-migration era — they are inline SQL: `_MIGRATE_V10_TO_V11_SQL` adds `learning_paths`, `topics`, `decks`, `flashcards`, `review_history`, `mindmaps`, `mindmap_nodes`, `study_sessions` (+ FTS and sync triggers); `_MIGRATE_V14_TO_V15_SQL` adds the quiz parity tables (`quizzes`, `quiz_questions`, `quiz_attempts`, `flashcard_templates`, `flashcard_assets`); v15→v16 repairs flashcard FTS5 delete triggers. (The v31–v35 migration files are about message metadata/console policy — unrelated.)

SRS fields on `flashcards`: `interval` (days), `repetitions`, `ease_factor` (default 2.5), `next_review`, `last_review` (canonical UTC ISO per ADR-224's normalization, which made `next_review` comparisons sargable and index-served), plus suspension/soft-delete/version columns. `decks.card_count` is maintained by recount; `review_history` records rating/interval/ease before-and-after and time taken.

## Review flow and SM-2 discipline (the load-bearing path)

1. Start review → next due candidate (`next_review IS NULL OR next_review <= now` — due-ness is **computed at query time**; nothing in the `Scheduling/` workbench schedules flashcards).
2. Show answer → six rating buttons (0–5).
3. Rating press → a worker group **deliberately not exclusive** (exclusivity would cancel the in-flight durable save) → submission under an `asyncio.Lock` with a once-per-presentation marker — SM-2 compounds, so a second press on the same presentation is dropped and logged; an arrival guard re-checks the presentation token and panel liveness after awaits (view switches destroy widgets); cancellation leaves the marker unclaimed so the rating is retryable.
4. `update_flashcard_review` applies textbook SM-2 (rating < 3 resets repetitions; else 1/6/interval×ease; ease updated with the standard formula floored at 1.3; `next_review = now + interval`) and appends history. Server mode returns a durable review-session id; local reviews create none.
5. Deck changes and refreshes share one exclusive group so list rebuilds never interleave.

## Entry points (dataflow)

- **Library**: the Study rail section rows (keeping their legacy `create-*` deep-link ids) open the staging canvas → `app.open_study_screen(scope_context, initial_section=…)` staging scope/section handoffs. The scope context is global-with-samples (material source, summarized titles capped, up to 25 source items). Handoff staging failures show a recovery toast and do **not** navigate; failed claims are released (retryable).
- **Home**: a due-count provider feeds the synthetic "Flashcards due" attention row → Study with `initial_section="flashcards"`, `origin="home"` (breadcrumb "Home ▸ Study"; Escape returns Home; library-origin Escape returns to Library's Study handoff row).
- **Chat handoff**: quiz sessions offer "Review in Chat", staging a `ChatHandoffPayload` with scope summary, workspace isolation contract, and a suggested prompt to the Console.

## Local vs server

- Local decks/cards/quizzes are manually authored; **LLM flashcard generation and source study packs are server-only** (the local service raises "Study packs are server-only"; local Library copy says so explicitly). The dashboard's source-generation action queues a server job and polls bounded times.
- Workspace study scopes require server mode; the UI disables deck deletion in server mode by design; flashcard listing is deck-scoped by construction (scope kwargs are not accepted).
- Every mutation passes a runtime-policy action id (`study.deck.create.local|server`, review actions, …).
- Dashboard snapshots fall back from the scope service to direct ChaChaNotes reads, run off the event loop for file-backed DBs.

Quiz/question creation in `UI/Study_Modules/quizzes_handler.py` captures raw authoring fields before normalizing the service payload. Successful completion clears each field only when its current raw value still matches that submission; later typing and its editing context survive. Failed creates retain drafts and existing retry semantics (TASK-36099). Deck/card creation in `UI/Study_Modules/flashcards_handler.py` follows the same submitted-only clearing rule independently for deck name, front, back and tags: raw whitespace edits, caret/selection and current focus survive successful completion; definite pre-write failure preserves the current draft for retry (TASK-36132).

Quiz final-answer submission in `StudyQuizzesController.submit_current_answer` builds its payload from earlier completed answers plus the current answer without adding that final answer to pending state before persistence succeeds. A definite pre-write failure therefore keeps the question editable; retry uses the current answer once, retaining earlier answers and ordinary grading/history behavior (TASK-36360).

## Config

No study-specific config section: local storage is the ChaChaNotes DB path (`chachanotes_db_path`, profile-overridable); server services build from the runtime API client provider. Services are wired once at app scope.

## Governing decisions

ADR-015 (Study folded under the Library destination for label/search), ADR-173/224 (UTC timestamp storage and normalization — sargable `next_review`), ADR-033 (screen state ownership), ADR-055 (soft-delete reversibility for decks/cards). Specs: `2026-04-20-study-flashcard-lifecycle-parity-design.md`, `2026-04-20-study-workspace-scope-behavior-design.md`, `2026-08-23-student-workflow-design.md` (the agent-side notes workflow — deliberately not this screen; its flashcard output rides Notes). Key tasks: task-10.x (maturity phase), task-2854 (navigation identity), task-4011 (Home breadcrumb), task-34000.6 (handoff crash + shared title truncation), task-32803.2 (due-date off-by-one), task-16195/… (dead-pane removal).

## Verified gotchas

1. SM-2 is non-idempotent — the lock + presentation marker exist because a double-submit would compound the schedule.
2. Due-ness is purely on-read; Home/Library counts re-query on their refresh cadences, not on reminders.
3. Most `ccp_messages`-style compat definitions have Study parallels: five of the eight advertised sections are honest placeholders — don't document them as features.
4. `list_flashcards` takes no scope kwargs — spreading scope args crashes (fixed once; keep it that way).

## Related docs

- [library-shell.md](./library-shell.md) — the Study handoff rows and staging canvas
- [home.md](./home.md) — the flashcards-due attention row
- [database-layer.md](./database-layer.md) — the learning tables
