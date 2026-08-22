---
id: TASK-19559
title: >-
  exclusive=True without group= makes every worker on a node cancel its
  siblings (145 sites)
status: Done
assignee: []
created_date: '2026-08-21 20:09'
labels:
  - concurrency
  - workers
  - ui
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 4 (concurrency / async / workers) —
its **#2**, and the lane's own **single highest-leverage remedy**. Framework
behaviour was verified against the installed Textual 8.2.8, not assumed.
Re-verified at this branch base.

**The mechanism.** `exclusive=True` with no `group=` lands the worker in the
group literally named `"default"`. `cancel_group` filters on `(node, group)`
and **never consults `name=`**. So every ungrouped exclusive worker on a node
mutually cancels every other one. There are **145 such sites across 35 files**.

Note for whoever picks this up: a naive `grep exclusive=True | grep -v group=`
over-counts badly (multi-line calls carry `group=` on the following line);
count properly before quoting a number.

`CancelledError` derives from `BaseException`, so the `except Exception:`
blocks these handlers use **cannot catch it** — the work vanishes silently.

**Named user-visible instances, all verified present:**

- **(a) Study loses spaced-repetition ratings.** `UI/Study_Window.py:914,
  1007, 1011` plus `UI/Study_Modules/flashcards_handler.py:894`
  (`submit_flashcard_review`). Every fast consecutive rating press — *the
  normal use of the feature* — can cancel the in-flight save.
- **(b) Media strands "Generating analysis…".** `UI/MediaWindow_v2.py:1984`
  against its siblings at `1555`, `1991`, `2080`, `2131`.
- **(c) Watchlists: the fix was applied to 2 of 6 siblings.**
  `UI/Screens/watchlists_collections_screen.py` — `4391` (`group="wc_items"`)
  and `4405` (`group="wl-briefings-load"`) are fixed; **`4393`, `4395`, `4397`,
  `4399`** (rules / runs / sources / notifications) are not — and the hazard
  comment explaining exactly why the group is needed sits **directly above the
  unfixed ones**.
- **(d) Settings load-backup overwrites what the user is typing.**
  `UI/Screens/settings_screen.py:8602` — `@work(exclusive=True, thread=True)`
  on `_advanced_load_backup_worker`, whose completion callback at `8613-8628`
  writes into the config editor `TextArea`. That screen has 12 thread workers.

Related framework facts the lane established, useful to whoever fixes this:
Textual **does** auto-cancel a node's workers on unmount, so "the screen never
cancels" is largely a non-issue for *async* workers (leaks survive for
**thread** workers and app-parented workers); and `Worker.cancel()` does **not**
stop a thread worker — its body runs to completion in an executor and its
`call_from_thread` callbacks still land.

**Reference implementations already in-repo:** `UI/Screens/chat_screen.py`
(zero ungrouped exclusives), `video_player_screen.py`,
`personas_screen.py:10634`, `library_screen.py:5696-5743`.

Per the owner's standing ruling, the durable remedy is the guard, not 145
hand-edits: a failing lint/test is what stops the 146th.

## Acceptance Criteria

- [x] A lint or test fails on `exclusive=True` scheduled without an explicit
      `group=`, covering both `@work(...)` decorators and `run_worker(...)`
      calls, including multi-line forms
- [x] The guard has an explicit, documented allowlist mechanism for any site
      that genuinely wants default-group mutual exclusion, so suppressing it is
      a deliberate recorded choice rather than a silent one
- [x] The four named user-visible instances are fixed: Study rating submission,
      Media analysis generation, the four unfixed Watchlists section loaders,
      and the Settings advanced-config backup load
- [x] Fast consecutive flashcard ratings all persist — pinned by a test that
      drives consecutive submissions, not by inspection
- [x] The Settings advanced-config editor is never overwritten by a background
      load while the user has unsaved typing in it
- [x] Handlers in this family no longer rely on `except Exception:` to observe
      cancellation, given `CancelledError` is a `BaseException`
- [x] The remaining sites are triaged with the user-visible ones fixed and the
      rest either grouped or allowlisted — the count is driven to a number the
      guard can hold

## Implementation Plan

1. Re-confirm the framework mechanism against the installed Textual 8.2.8
   (`Worker.__init__` default group, `WorkerManager.add_worker`/`cancel_group`
   source, `Worker.cancel` on a thread worker).
2. Census the real site count with an AST walk, not a grep, and record both
   numbers so the over-count is documented rather than repeated.
3. Land the durable remedy first: an `Tests/Architecture/` inventory guard with
   a documented, reason-carrying allowlist, proven to bite on both the
   decorator form and the multi-line call form.
4. Fix the four named user-visible instances.
5. Triage every remaining site to an explicit group named after the *work*.

## Implementation Notes

**The guard is the deliverable.** `Tests/Architecture/test_worker_exclusive_group_inventory.py`
walks the AST of every module under `tldw_chatbook/` and fails on any
`@work(...)` or `run_worker(...)` that requests exclusivity without naming a
group. It handles the multi-line form, the positional `run_worker(work, name,
group, ...)` form, and fails **closed** on a `**kwargs` spread or a non-literal
`exclusive=` (the stance `Tests/UI/test_chat_screen_worker_groups.py` already
took for Console). `DEFAULT_GROUP_ALLOWLIST` is keyed by
`"<path>::<owning function>"` — a name, not a line number, so an entry survives
edits above it — and carries the reasoning that earns it; two further tests
assert no entry is stale and no entry ships without prose.

**Measured, not estimated.** At the branch base the AST census found **133
sites across 32 files**. The naive `grep exclusive=True | grep -v group=` the
task warned about reported **523** — a ~4x over-count, because a multi-line call
carries `group=` on a following line (`chat_screen.py` alone accounts for 29 of
the phantom hits while having zero real ones). After this change: **1 site**,
the vendored `textual-fspicker` widget, allowlisted with its reasoning.

**Group naming rule.** A group is named after the *work*, not the caller. Two
call sites that start the same load share a group (a refresh supersedes a
refresh: `wc_rules` is used by all four `_load_rules()` dispatchers); two
different operations do not (a section load must never kill a save). That rule
drove all 131 mechanical edits and is stated in the guard's docstring.

**The four named instances.**

- **(a) Study ratings.** Grouping alone is *not* sufficient here and the born-red
  test proves it: with an explicit group but `exclusive=True` still set, the
  second press still destroys the first save. A spaced-repetition rating is a
  durable write, so `handle_review_rating` now dispatches into
  `group="study-flashcard-rating"` with **no exclusivity at all**, and
  `FlashcardsHandler.submit_rating` serialises presses behind an `asyncio.Lock`,
  capturing the card under review synchronously before its first `await` so a
  fast second press rates the card the user was looking at. It also grows an
  explicit `except asyncio.CancelledError:` that logs and re-raises — the
  `except Exception:` beneath it is a `BaseException` blind spot by construction.
- **(b) Media analysis.** `perform_analysis` and its four siblings each name
  their own group, so generating an analysis can no longer be cancelled by a
  save/overwrite/delete/read-it-later toggle.
- **(c) Watchlists.** `wc_rules` / `wc_runs` / `wc_sources` / `wc_notifications`
  join the two already-fixed branches, matching the adjacent `wc_items`
  convention; the method docstring now records that the hazard comment sat
  directly above four siblings it did not protect.
- **(d) Settings backup load.** Grouping is again insufficient: the worker is a
  *thread* worker, whose body `Worker.cancel()` cannot stop. The editor text is
  now captured at dispatch and compared on arrival; if the user typed while the
  backup was being read the write is refused and the refusal is reported
  ("unsaved edits were kept") instead of silently replacing their work.

**Born-red evidence.** Both behavioural tests were driven, then re-run against a
neutered fix and observed to fail with the exact production symptom:
`persisted=[('card-local-1', 5)]` (the rating-3 save gone) and a
`provider = "OpenAI"` → `provider = "Ollama"` editor diff. The guard was proven
to bite by injecting one single-line decorator violation and one multi-line
`name=`-only call into real package files; both were reported, and the files
restored to byte-identical SHA-256.

**Modified:** `Tests/Architecture/test_worker_exclusive_group_inventory.py`
(new); `UI/Study_Window.py`, `UI/Study_Modules/flashcards_handler.py`,
`UI/MediaWindow_v2.py`, `UI/Screens/watchlists_collections_screen.py`,
`UI/Screens/settings_screen.py`, plus 26 further modules taking mechanical
`group=` additions; `Tests/UI/test_settings_configuration_hub.py`,
`Tests/UI/test_media_window_v2_parity.py`,
`Tests/UI/test_study_flashcards_screen.py`.

## Review response (do-not-ship findings R1/R2/R3)

Independent review confirmed the guard, the allowlist and the census (133
sites / 32 files; the task's own "145 / 35" header is wrong) but returned
**do-not-ship** on the Study path: removing exclusivity there was correct, and
was not paired with the arrival-time guard this same change applies everywhere
else. All three findings reproduced, were fixed, and are now pinned.

Evidence is a single probe run identically against base `e98076411`, the
pre-fix branch `3f539c2d4`, and the fix — real Textual workers, real DB,
reading persisted state:

| | base | pre-fix branch | fixed |
|---|---|---|---|
| R1 `_handle_exception` | `[]` | `WorkerFailed(NoMatches('#review-status'))` | `[]` |
| R1 rating persisted | `[]` (lost) | `[('card-local-1', 3)]` | `[('card-local-1', 3)]` |
| R1 `current_review_session_id` | `None` | `41` (resurrected) | `None` |
| R2 `#card-list` rows / `current_cards` | 2 / 2 | 4 / 2 | 2 / 2 |
| R3 `repetitions` / `interval` | 1 / 1d | 2 / 6d | 1 / 1d |

**R1 — arrival guard.** `StudyWindow.watch_current_view` calls
`remove_children()`, so leaving the sub-view mid-save destroys the widgets
every `_set_review_*` helper reaches with a bare `query_one`. Exclusivity used
to swallow this by cancelling the save first. `StudyFlashcardsController` now
carries a `_review_presentation` token, bumped wherever the presented card
changes or the panel is torn down (`reset_review_panel`,
`_load_next_review_candidate`, `end_review_session_if_needed`).
`submit_rating` captures it before its first `await` and re-checks it — plus
`_review_panel_is_live()` — after the write lands, before touching session
state or UI. Note the fixed column beats base on the row that matters: the
rating **persists** where base lost it, and nothing raises.

**R2 — the missed writer.** `handle_deck_select_changed` was left ungrouped
*and* non-exclusive, so it sat in `"default"` while `handle_refresh_cards`
moved to `study-refresh-cards`; base's ungrouped-exclusive refresh had been
cancelling it. Both rebuild `#card-list`, so by this task's own rule (group
after the *work*) they now share `study-refresh-cards`, exclusive.

**R3 — SM-2 is compounding; semantic choice flagged for the owner.**
`ChaChaNotes_DB.update_flashcard_review` is not idempotent: two reviews of one
card move it `repetitions` 0->1->2 and `interval` 1d->6d. Converting a lost
write into a doubled schedule is a different data defect, not a fix, and the
old gated test *pinned* the doubling by holding one card fixed.

Resolved as **one review per card presentation**: `submit_rating` records the
presentation it has written for and drops a second submission for the same
one, and rating buttons are now disabled the moment a press is accepted (re-
enabled on the failure path), so the UI cannot produce the second press at all.

The controller proposed "apply the **latest** rating once"; this ships
**first-press-wins**, and the difference needs an owner call. Reasoning: by the
time a second press arrives the first write is already awaiting the service, so
the only ways to honour "latest" are to un-apply SM-2 (no such API) or to
debounce the first write, which adds latency to *every* rating on the hot path.
Anki, the reference implementation for this interaction, also disables the
answer buttons on press and routes the next press to the *next* card — which is
what disabling the buttons now reproduces. Both options agree on the part that
matters (SM-2 applied exactly once); only the tie-break differs, and it is only
reachable programmatically now. If the owner prefers latest-wins, the mechanism
is a short debounce window and it should be filed as its own task.

**Tests.** The doubling-pinning test is gone. Five now stand, each red at the
baseline that owns it: `..._survives_a_sibling_study_worker` (red at base —
this is the AC's actual named bug, `Study_Window.py:1007/1011` eating the
save), `..._on_distinct_cards_all_persist` (the AC as-meant), `..._survives_
leaving_the_flashcards_sub_view` (red at **both** baselines: lost write at
base, `WorkerFailed` pre-fix), `..._do_not_interleave_the_card_list` (red
pre-fix, 4 vs 2), `..._applies_sm2_once` (red pre-fix, real DB, 2/6d).

**Guard hole closed by the reviewer (`3f539c2d4`, kept):** the walk checked
only that a `group=` node was *present*. `group="default"` is byte-identical to
omitting it, and `group=""`/`None` are falsy so `add_worker`'s
`if exclusive and worker.group:` skips `cancel_group` entirely — the site asks
for exclusivity and silently gets none.

**Residuals to file, not fixed here** (both are the inverse shape — work that
should be inside an existing group running outside it):
1. `schedules_workbench` mutation workers (`_delete_and_refresh`,
   `_save_and_refresh`, `_run_and_refresh`, `_update_and_refresh`,
   `_bulk_delete`, `_bulk_toggle`) `await load_tasks()` inline, so that reload
   runs outside `schedules-load-tasks` and cannot be superseded by a newer one.
2. Watchlists notification mark-read/dismiss reload notifications inline,
   outside `wc_notifications`.

**Also recorded:** `Tests/Watchlists/test_watchlists_pane_filter_in_place.py::
test_article_search_hides_a_day_header_whose_whole_group_is_filtered_out` is a
**timezone flake**, not a regression — at base `e98076411` it passes under
`TZ=UTC` and fails under `America/Los_Angeles`. The other two residual
failures (`test_persistent_diagnostic_inventory`, `test_screen_size_ratchet
[chat_screen.py]`) were confirmed pre-existing at that same base.
