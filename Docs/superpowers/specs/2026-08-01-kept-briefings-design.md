# Kept Briefings — persistence into ChaChaNotes (task-1780)

**Date:** 2026-08-01
**Status:** proposed
**Task:** `backlog/tasks/task-1780` (owner-directed, 2026-08-01). Successor to spec #2
(`2026-07-30-watchlists-briefings-design.md`, all four phases shipped: PRs #1115/#1145/#1164/#1177/#1187).

## Why

Generated briefings and scripts live in `Subscriptions_DB` with their lifecycle chained to the
watchlist (`ON DELETE CASCADE` through `briefings` → `briefing_scripts` → `briefing_audio`).
Deleting a watchlist silently destroys every artifact it produced — right for working data, wrong
for content the user made. Kept copies must survive watchlist deletion and support **re-casting
later with whatever presets/rosters exist at that time**.

## Owner decisions (2026-08-01)

1. **Dedicated artifact tables in ChaChaNotes** — not Notes. Cost stated plainly: ChaChaNotes
   sync and chatbook export do NOT cover these tables in v1 (a follow-up task records that);
   requires a ChaChaNotes schema version bump + migration (five historical version collisions —
   re-verify at merge).
2. **Auto for scheduled, Keep for manual.** Scheduled runs mirror automatically on completion
   (nobody is present to press Keep); manual generations persist only via an explicit Keep action.
3. **Briefings + scripts only.** Audio deferred: files are large, already survive watchlist
   deletion on disk (only the DB row cascades), and feed export is the shipped way out. The
   audio-pointer decision is recorded as deliberately open.

## Schema (ChaChaNotes, version bump + migration file per CLAUDE.md)

- `kept_briefings`: id; `source_briefing_id INTEGER NOT NULL UNIQUE` (idempotency key — the
  subscriptions-DB id as a plain int, **no cross-DB FK**); `watchlist_name TEXT` (denormalized —
  the watchlist may die); `body_markdown TEXT NOT NULL`; coverage window (`covers_through_item_id`,
  `covers_from_ts`); `selection_mode`, `model_used`, `item_count`, `featured_count`,
  `overflow_count`; `origin TEXT CHECK(origin IN ('manual','scheduled'))` — how the KEEP happened;
  `original_created_at`; `kept_at DEFAULT CURRENT_TIMESTAMP`.
- `kept_scripts`: id; `kept_briefing_id` FK → `kept_briefings` `ON DELETE CASCADE` (cascade
  *inside* ChaChaNotes only); `source_script_id INTEGER UNIQUE` (**nullable** — scripts cast
  directly from a kept briefing have no subscriptions-side source); `preset_name`,
  `roster_snapshot_json`, `turns_json`, `model_used`, `original_created_at`, `kept_at`.
- Every kept row is **self-interpreting with the subscriptions DB gone entirely** — all
  provenance denormalized, source ids kept for tracing only.
- **No sync columns** (`client_id`/`version`/`deleted`): these tables do not participate in
  ChaChaNotes sync in v1 — a deliberate, recorded divergence; deletion is hard-delete.

## Keep service (`Subscriptions/briefing_keep.py`)

`keep_briefing(subs_db, chacha_db, briefing_id, *, origin) -> dict` — copies the briefing plus
its **complete** scripts. **Additive-idempotent**: re-keeping adds scripts missing by
`source_script_id`, never duplicates, never overwrites existing rows (a scheduled briefing is
auto-kept scriptless; a script cast later from the original is picked up by pressing Keep again).
Refuses (raises, no row) for non-`complete` or empty-bodied briefings — auto-keep fires only on
`complete` with a non-empty body, so `empty` scheduled rows are never mirrored.

**Auto path:** the phase-4 `BriefingJobHandler` gains an *optional* ChaChaNotes handle; after a
scheduled generation completes it calls the keep service with `origin='scheduled'`. A missing
handle or a keep failure logs type-only and **never fails the generation** — a lost mirror costs
nothing permanent; the next scheduled run retries naturally. Plan-time verification required:
whether `CharactersRAGDB` is thread-safe under `asyncio.to_thread` the way `SubscriptionsDB` is
(per-thread connections); if not, keep writes are marshalled accordingly.

## UI

- **Keep button** on the Artifacts toolbar, enabled when a complete briefing is selected;
  `origin='manual'`; toast reports kept-with-N-scripts (or already-kept + newly-added count).
- **Kept briefings modal** (preset-manager shape: list + detail + hard-delete-with-confirmation),
  opened from the same toolbar — a modal, not a screen section, so it is reachable regardless of
  watchlist scope, including after the watchlist is gone. Detail renders body via
  `Markdown(..., hyperlinks=False)`; all remote-derived text through `rich.text.Text`.
- **Cast from kept**: inside the modal, a preset `Select` (current presets + app default — AC #4's
  "whatever they decide then") and a Cast button.

## Re-casting without the watchlist

`briefing_cast.generate_script_from_text(chacha_db, kept_briefing_id, *, preset_id, chat=...)`
reuses the existing pure pieces (`build_cast_prompt`, `parse_script_turns`) against the kept
body and writes the result into `kept_scripts` (`source_script_id = NULL`). It takes **its own
claim set keyed by kept_briefing_id** — kept ids are a different id space from live briefing ids,
so reusing the phase-4 cast claims would collide across spaces. Same error ethos as
`generate_script`: pre-flight refusals raise before any row; in-band failures write honest state;
the kept briefing row is never touched by a cast outcome.

## Error handling & testing

Phase-1-through-4 conventions bind throughout: pre-flight-raises/no-orphan-row, type-only logging
(never body/roster/turn content), toasts `markup=False`, all UI-thread DB via `asyncio.to_thread`,
mutation checks per behavioural change. Named invariant tests: **kept-rows-survive-watchlist-
deletion** (AC #3 — delete the watchlist, re-read the kept artifact); **re-cast-needs-no-
subscriptions-rows** (AC #4 — cast from a kept briefing after deleting watchlist AND presets used
originally); additive-idempotency both directions; auto-keep-skips-empty; keep-failure-never-
fails-generation. Real ChaChaNotes DB on tmp_path; migration collision re-check at merge.

## Non-goals (v1)

Audio pointers/copies; sync/chatbook-export coverage of kept tables (follow-up task filed at
close-out); any browsing surface beyond the modal; editing kept content.
