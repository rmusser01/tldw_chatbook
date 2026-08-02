---
id: TASK-1780
title: Persist briefings and scripts into ChaChaNotes for permanence and later regeneration
status: To Do
assignee: []
created_date: '2026-08-01 21:30'
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
- [ ] #1 A generated briefing can be persisted into the user's ChaChaNotes DB, carrying its body and enough provenance (watchlist name, coverage window, generation time, model) to be self-interpreting on its own
- [ ] #2 A cast script can be persisted likewise, including its roster snapshot and turns
- [ ] #3 Persisted copies survive deletion of the originating watchlist (proven by a test that deletes the watchlist and re-reads the persisted artifact)
- [ ] #4 From a persisted briefing, the user can generate a NEW script later using whatever preset/roster exists at that time — without the original watchlist, subscriptions rows, or preset needing to still exist
- [ ] #5 The persist-vs-keep policy (automatic on complete vs. explicit user action) is decided with the project owner and recorded in the spec before implementation
- [ ] #6 ChaChaNotes schema changes (if any) follow the migration rules: version increment, migration file, collision re-check at merge
<!-- AC:END -->
