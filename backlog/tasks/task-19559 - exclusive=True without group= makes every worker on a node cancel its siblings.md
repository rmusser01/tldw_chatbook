---
id: TASK-19559
title: >-
  exclusive=True without group= makes every worker on a node cancel its
  siblings (145 sites)
status: To Do
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

- [ ] A lint or test fails on `exclusive=True` scheduled without an explicit
      `group=`, covering both `@work(...)` decorators and `run_worker(...)`
      calls, including multi-line forms
- [ ] The guard has an explicit, documented allowlist mechanism for any site
      that genuinely wants default-group mutual exclusion, so suppressing it is
      a deliberate recorded choice rather than a silent one
- [ ] The four named user-visible instances are fixed: Study rating submission,
      Media analysis generation, the four unfixed Watchlists section loaders,
      and the Settings advanced-config backup load
- [ ] Fast consecutive flashcard ratings all persist — pinned by a test that
      drives consecutive submissions, not by inspection
- [ ] The Settings advanced-config editor is never overwritten by a background
      load while the user has unsaved typing in it
- [ ] Handlers in this family no longer rely on `except Exception:` to observe
      cancellation, given `CancelledError` is a `BaseException`
- [ ] The remaining sites are triaged with the user-visible ones fixed and the
      rest either grouped or allowlisted — the count is driven to a number the
      guard can hold
