---
id: TASK-1780
title: Persist briefings and scripts into ChaChaNotes for permanence and later regeneration
status: Done
assignee: []
created_date: '2026-08-01 21:30'
updated_date: '2026-08-02 00:30'
labels:
  - watchlists
  - briefings
  - chachanotes
  - persistence
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed at the project owner's direction (2026-08-01), after spec #2 phases 1-3 shipped.

Generated briefings and cast scripts currently live only in `Subscriptions_DB`
(`briefings.body_markdown`, `briefing_scripts.turns_json` + `roster_snapshot_json`), and their
lifecycle is chained to the watchlist: `briefings.watchlist_id` is `ON DELETE CASCADE`, and
`briefing_scripts`/`briefing_audio` cascade off briefings in turn. Deleting a watchlist —
or any future retention/pruning policy on the subscriptions side — silently destroys every
briefing and script it ever produced. That is the right lifecycle for *watchlist working data*
and the wrong one for *content the user made*.

The owner wants generated briefings/scripts to also land in the user's ChaChaNotes DB (the
main conversations/notes/characters DB — the one chatbook export, notes sync, and long-term
user data already revolve around) so that:

- they are **permanent**: they survive watchlist deletion, source churn, and subscriptions-side
  pruning;
- they are **re-generatable later, on whatever the user decides then**: the briefing body is
  the canonical, cast-independent artifact (spec #2's core design decision), so a persisted
  briefing can be re-cast into a new script with a *future* roster/preset, and a persisted
  script re-synthesized with *future* voices/providers — none of the original watchlist
  machinery required.

Key facts for whoever designs this (verified as of `22fb8693e`):

- The briefing row carries everything needed to stay self-interpreting: `body_markdown`,
  coverage window (`covers_through_item_id`, `covers_from_ts`), `selection_mode`, `preset_id`,
  `model_used`, counts, timestamps. Scripts carry `preset_name` + write-once
  `roster_snapshot_json` + `turns_json` + `model_used`.
- Citations (`[item N]` markers) reference subscription items that may be pruned — the persisted
  copy inherits the honest-degradation story, not a guarantee the items still exist.
- ChaChaNotes schema changes must increment the schema version and add a migration
  (CLAUDE.md; schema collisions with concurrent sessions have happened five times — renumber-check
  at merge). Storing as Notes (existing entity: sync, templates, chatbook export for free) vs. a
  dedicated artifact kind is the central design decision and is deliberately NOT decided here.
- Whether persistence is automatic on `complete`, or an explicit "Keep" action, is likewise a
  design decision for the implementer to bring to the owner — auto-write doubles storage for
  every throwaway generation; manual-only risks the user losing the one briefing they cared
  about. (Audio files are large and live on disk under `briefing_audio_dir()`; persisting the
  *pointer* vs. copying the file needs its own decision.)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A generated briefing can be persisted into the user's ChaChaNotes DB, carrying its body and enough provenance (watchlist name, coverage window, generation time, model) to be self-interpreting on its own
- [x] #2 A cast script can be persisted likewise, including its roster snapshot and turns
- [x] #3 Persisted copies survive deletion of the originating watchlist (proven by a test that deletes the watchlist and re-reads the persisted artifact) — `Tests/Subscriptions/test_briefing_keep.py::test_kept_rows_survive_watchlist_deletion`
- [x] #4 From a persisted briefing, the user can generate a NEW script later using whatever preset/roster exists at that time — without the original watchlist, subscriptions rows, or preset needing to still exist — `Tests/Subscriptions/test_briefing_cast.py::test_recast_needs_no_subscriptions_rows`
- [x] #5 The persist-vs-keep policy (automatic on complete vs. explicit user action) is decided with the project owner and recorded in the spec before implementation. Decided 2026-08-01 (recorded in `Docs/superpowers/specs/2026-08-01-kept-briefings-design.md`, "Owner decisions"): (1) dedicated `kept_briefings`/`kept_scripts` tables in ChaChaNotes, not Notes — sync/chatbook-export coverage of these tables is explicitly deferred (follow-up task-1870); (2) auto-keep for scheduled runs (nobody is present to press Keep), explicit Keep action for manual generations; (3) briefings + scripts only — audio persistence stays out of scope for v1 (files already survive watchlist deletion on disk; only the DB row cascades).
- [x] #6 ChaChaNotes schema changes (if any) follow the migration rules: version increment, migration file, collision re-check at merge — `_CURRENT_SCHEMA_VERSION = 29` (`tldw_chatbook/DB/ChaChaNotes_DB.py`), migration `tldw_chatbook/DB/migrations/chachanotes_v28_to_v29_kept_briefings.sql`; re-verified against `origin/dev` at close-out (2026-08-02): `origin/dev` is still at v28 with no v28→v29 migration of its own — no collision.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Delivered across 6 sub-tasks (plan: `.superpowers/plans/2026-08-01-kept-briefings.md`; design: `Docs/superpowers/specs/2026-08-01-kept-briefings-design.md`, now Status: implemented). Dedicated `kept_briefings`/`kept_scripts` tables in ChaChaNotes (not Notes), schema v28→v29, per the owner decisions recorded against AC #5.

- **T1 — schema:** `kept_briefings`/`kept_scripts` tables + CRUD (`ChaChaNotes_DB.create_kept_briefing`/`create_kept_script`/`get_kept_briefing_by_source`/`kept_script_source_ids`/list/delete), migration `chachanotes_v28_to_v29_kept_briefings.sql`, `sql_validation.VALID_TABLES` entries. `kept_scripts.kept_briefing_id` is a real intra-ChaChaNotes FK (`ON DELETE CASCADE`); `source_briefing_id`/`source_script_id` are plain ints for tracing only, never cross-DB FKs. No sync columns — a deliberate v1 divergence (follow-up task-1870 filed at close-out).
- **T2 — keep service** (`Subscriptions/briefing_keep.py`): `keep_briefing(subs_db, chacha_db, briefing_id, *, origin)`, additive-idempotent (re-keeping only adds missing scripts by `source_script_id`, never duplicates/overwrites), refuses (no orphan row) for non-`complete` or empty-bodied briefings. Review round 1 hardened a real TOCTOU: two concurrent keepers can both pass the "already kept?" check before either inserts; the loser's `ConflictError` against `source_briefing_id UNIQUE` is now caught and folded into the normal re-keep path instead of surfacing raw.
- **T3 — scheduled auto-keep** (`scheduler/handlers/briefing_handler.py`, `app.py`): scheduled generations auto-keep on `complete` with `origin='scheduled'`; missing handle or keep failure logs type-only and never fails the generation. Review round 1 found the first pass wired `self.chachanotes_db` as a frozen constructor param — `BriefingJobHandler` is built in `app.py.__init__` before `self.chachanotes_db` itself is assigned, so auto-keep was wired but permanently inert in production. Fixed by replacing it with a zero-arg `chachanotes_db_getter` resolved fresh inside `_auto_keep` on every completion (construction-order bug, not a logic bug).
- **T4 — re-cast from kept** (`Subscriptions/briefing_cast.py`): `generate_script_from_text(chacha_db, kept_briefing_id, *, preset_id, subs_db, chat=..., load_character=None)` reuses `build_cast_prompt`/`parse_script_turns` against a kept body, writing into `kept_scripts` (`source_script_id=NULL`). Its own claim set (`_ACTIVE_KEPT_CAST_CLAIMS`, keyed by `kept_briefing_id`) is deliberately separate from the live-briefing claim space (different id spaces, different DBs — reusing phase-4's claims would collide). `preset_id=None` casts a single-speaker "Narrator" narration on the app's default provider (`APP_DEFAULT_PRESET_NAME = "(app default)"`) — the design fill for AC #4's "whatever they decide then" when no preset exists.
- **T5 — UI** (`UI/Watchlists_Modules/kept_briefings_modal.py`, `artifacts_pane.py`, `watchlists_collections_screen.py`): Keep button on the Artifacts toolbar (additive-idempotent toast: created vs. already-kept-N-added); `KeptBriefingsModal` (list + detail + hard-delete-with-confirmation + cast-from-kept via a preset `Select` whose "App default (single narrator)" label mirrors T4's `APP_DEFAULT_PRESET_NAME` copy), reachable regardless of watchlist scope. Found and fixed a real bug during testing: `_run_cast`'s always-recompose `finally` was silently wiping `_show_error`'s own message because `compose()` hard-coded the error `Static` to blank on every rebuild — error text is now held as instance state that `compose()` itself reads, closing a recompose-wipes-error bug class the wider program has hit before.
- **T6 — close-out (this task):** ACs verified, `origin/dev` schema-version collision re-checked (still v28, no collision), spec Status → implemented with delivery notes, sync/chatbook-export gap follow-up filed as task-1870.

Full sweep at close-out: `Tests/Subscriptions/ Tests/Scheduling/ Tests/Watchlists/ Tests/DB/test_chachanotes_kept_briefings.py Tests/Character_Chat/ Tests/UI/ -k watchlist` — see the T6 close-out report for counts; no regressions beyond pre-existing baselines (tree-chevron ×2, the numpy `test_chat_image_db_compatibility` failure, TASK-1345 rotating-victim flakes).

Modified/added files: `tldw_chatbook/DB/ChaChaNotes_DB.py`, `tldw_chatbook/DB/migrations/chachanotes_v28_to_v29_kept_briefings.sql`, `tldw_chatbook/DB/sql_validation.py`, `tldw_chatbook/Subscriptions/briefing_keep.py`, `tldw_chatbook/Subscriptions/briefing_cast.py`, `tldw_chatbook/scheduler/handlers/briefing_handler.py`, `tldw_chatbook/app.py`, `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`, `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py`, `tldw_chatbook/UI/Watchlists_Modules/kept_briefings_modal.py`, `tldw_chatbook/css/features/_watchlists.tcss`, `tldw_chatbook/css/tldw_cli_modular.tcss`, plus test files under `Tests/DB/`, `Tests/Subscriptions/`, `Tests/Scheduling/`, `Tests/Watchlists/`.
<!-- SECTION:NOTES:END -->
