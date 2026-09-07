# Kept Briefings — persistence into ChaChaNotes (task-1780)

**Date:** 2026-08-01
**Status:** implemented
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

## Delivery notes (2026-08-02, close-out)

All 6 tasks shipped on `feat/task-1780-persist-briefings`; `backlog/tasks/task-1780` ACs #1-#6
all checked. Four things worth recording for anyone re-reading this design later:

- **The Narrator design fill.** The spec's "whatever preset exists at that time" (AC #4) leaves
  a gap when NO preset exists at all. Task 4 (`briefing_cast.generate_script_from_text`) fills it
  with `preset_id=None` casting a single-speaker "Narrator" narration on the app's default
  provider, no style notes, `preset_name = "(app default)"`. Task 5's kept-briefings modal picker
  mirrors the exact copy (`"App default (single narrator)"` label for the same `None` choice) so
  the two surfaces never drift into different wording for the same behavior.
- **The lazy-getter liveness fix.** Task 3's first pass wired `BriefingJobHandler` with a frozen
  `self.chachanotes_db` instance param — but `app.py.__init__` constructs the handler before
  `self.chachanotes_db` is itself assigned, so auto-keep was wired end-to-end yet permanently
  inert in production regardless of what the attribute later became. Fixed by a zero-arg
  `chachanotes_db_getter` resolved fresh inside `_auto_keep` on every completion, closing over the
  live app instance instead of a construction-time snapshot. A construction-order bug, not a logic
  bug — the kind this program has hit before and will likely hit again wherever a handler is built
  ahead of the attribute it needs.
- **The raced-keep-lands-friendly hardening.** Task 2 review found that two concurrent
  `keep_briefing` callers (a scheduled auto-keep and a manual Keep press landing near-simultaneously
  is the realistic case) can both pass the "already kept?" check before either inserts. The loser's
  `create_kept_briefing` call now has its `ConflictError` (unambiguous given the table's only
  unique constraint is `source_briefing_id`) caught and folded into the ordinary additive re-keep
  path, rather than surfacing a raw exception to whichever caller lost the race.
- **The recompose-wiped-error bug class, another instance.** Task 5 found `KeptBriefingsModal`'s
  `_run_cast` always-recompose `finally` clause was silently erasing `_show_error`'s own message,
  because `compose()` hard-coded the error `Static` back to blank on every rebuild. Fixed by
  holding the error text as instance state that `compose()` itself reads on rebuild. This is the
  same bug shape the wider program has caught more than once: a `finally`/recompose path that
  rebuilds the UI from scratch will silently discard any state set moments earlier unless that
  state is threaded through `compose()` rather than assumed to survive it.

## Follow-up decision (2026-08-02, task-1870): sync excluded, chatbook export included

Task-1870 was filed at this spec's close-out (see "Non-goals (v1)" above) to close the silent gap
on both of this design's deferred fronts — ChaChaNotes sync coverage and chatbook-export coverage
of `kept_briefings`/`kept_scripts`. Both are now resolved, asymmetrically:

- **ChaChaNotes sync: still excluded, by extension of this spec's original ruling.** The "Schema"
  section above already decided `kept_briefings`/`kept_scripts` carry no sync columns
  (`client_id`/`version`/`deleted`) and do not participate in ChaChaNotes's bidirectional sync
  machinery. Task-1870 does not revisit that call — it stays a deliberate v1 decision, not an
  oversight. The reason it holds up under the "should a follow-up close this?" question: the sync
  engine's unit of replication is a row with a `client_id`/`version`/tombstone lineage, and kept
  rows were designed from the start (this spec's "Schema" section) to be self-interpreting,
  hard-deleted, denormalized copies with no such lineage. Retrofitting sync columns would be a
  schema change with its own migration and conflict-resolution design, not a small follow-up: if a
  future task wants cross-device sync of kept content, it should treat that as a new schema
  proposal against this spec, not an extension of task-1870.
- **Chatbook export: now included.** `ContentType.KEPT_BRIEFING` (`Chatbooks/chatbook_models.py`)
  is a new chatbook content type. `ChatbookCreator._collect_kept_briefings`
  (`Chatbooks/chatbook_creator.py`) walks a user's selected kept briefings, and — mirroring how a
  conversation's messages are nested inside the conversation's own exported JSON rather than being
  independently selectable — nests each briefing's kept scripts inside its own payload. Every
  provenance column denormalized onto the kept rows by this spec's original schema (source ids,
  coverage window, model/preset identifiers, origin, original/kept timestamps) is carried into the
  export unchanged, so an exported kept briefing remains self-interpreting even without the source
  ChaChaNotes database, matching this spec's "self-interpreting with the subscriptions DB gone
  entirely" principle. Each briefing exports as both a JSON file (machine-round-trippable) and a
  companion Markdown file (human-readable), the same split already used for conversations
  (JSON + citation report) and notes (frontmatter + body).

  Import policy: `source_briefing_id`/`source_script_id` are device-local ids (this spec's
  "Schema" section is explicit that they are "kept only for tracing", never cross-DB foreign
  keys), so an incoming row can collide with an unrelated local row that happens to reuse the same
  source id on a different device. `ChatbookImporter._import_kept_briefings` handles this by
  trying the insert and catching `ConflictError` — the same "raced keep" pattern this spec's
  delivery notes already documented for `briefing_keep.py` — then comparing content: byte-identical
  is treated as an ordinary already-imported row (skipped silently, since re-importing the same
  chatbook must never duplicate it); differing content is reported as a named conflict in the
  import summary and the existing local row is never overwritten. `ConflictResolver`'s existing
  ask/skip/rename/replace machinery (built for display-name-keyed conversations/notes/characters)
  was deliberately not extended to kept rows — a UNIQUE source-id-keyed idempotent artifact doesn't
  fit "rename" or "ask per item" semantics, and this codebase already had a working precedent for
  the right shape in the keep service itself. NULL-source kept scripts (cast directly from a kept
  briefing, no subscriptions-side source) have no identity to key a `ConflictError` off of, so they
  are deduped by content match against the parent's pre-existing scripts instead, one-for-one
  (a matched candidate is consumed so it cannot absorb a second, different incoming script).

  Chatbooks created before this task have no `kept_briefings` section at all; importing one is
  unaffected — the new code path never runs when `ContentType.KEPT_BRIEFING` is absent from the
  manifest, and `ChatbookManifest.from_dict` defaults `total_kept_briefings` to 0 for old bundles.
