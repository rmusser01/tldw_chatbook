---
id: TASK-19563
title: >-
  Three surfaces show stale results because name= was mistaken for worker
  scoping and no generation guard exists
status: Done
assignee: []
created_date: '2026-08-21 20:13'
labels:
  - concurrency
  - ui
  - workers
priority: medium
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 4 (concurrency / async / workers) —
its **#5**. Re-verified at this branch base.

Three surfaces dispatch a worker per keystroke or per selection and then render
whatever result arrives last, with no check that the result still corresponds
to the current input. In each case the author passed `name=` believing it
scoped the exclusivity — it does not; `cancel_group` filters on `(node, group)`
and never consults `name=`.

The clearest instance, `UI/CCP_Modules/ccp_conversation_handler.py:110-118`:

```python
self.window.run_worker(
    self._search_conversations_sync,
    search_term,
    search_type,
    thread=True,
    exclusive=True,
    name="conversation_search",   # ← does not scope anything
)
```

Because these are **thread** workers, `Worker.cancel()` does not stop the body
— it runs to completion in an executor and its `call_from_thread` callback
still lands. So even correct grouping would not be sufficient here: the result
must be discarded at arrival time.

The three sites, with honest severity:

- **(a) CCP conversation search** — `ccp_conversation_handler.py:110-172`. The
  list shows results for `"a"` while the search box already reads `"ab"`. User-
  visible wrong data.
- **(b) Personas character load** — `ccp_character_handler.py:458-463`.
  **Display corruption only** — the modern save path has its own generation
  guard, so this does not corrupt stored data.
- **(c) Model install view** — `model_installed_view.py:655-688`. Stray toasts
  landing over an unrelated screen.

**The canonical fix is already in this repo**: `UI/Screens/chat_screen.py`
uses a generation counter captured at dispatch and compared at arrival. Copy
that rather than inventing a new mechanism — this is exactly the
durable-over-clever case in the owner's standing ruling.

Related, and deliberately *not* inflated: the lane also found `is_mounted` used
as a post-await detach guard in 42 places where it can never be `False`
(`_is_mounted` is never reset). It rated this **lower severity than it looks**,
because async workers raise at the await first. `is_attached` is the valid
check, and `library_screen.py:5711-5715` documents the true semantics. Fix the
instances touched by this work; do not open a 42-site sweep on its own account.

## Acceptance Criteria

- [x] Each of the three surfaces discards results that no longer match the
      current input, using the generation-counter pattern already established
      in `chat_screen.py`
- [x] The guard is applied at **result arrival**, not only at dispatch — thread
      worker bodies cannot be cancelled, so arrival-time rejection is required
- [x] Typing quickly in CCP conversation search never leaves the list showing
      results for a prefix of the current query — pinned by a test that
      interleaves two dispatches and delivers them out of order
- [x] The Personas character load no longer renders a superseded character
- [x] `model_installed_view` toasts cannot land on an unrelated screen
- [x] Any `is_mounted` post-await guard touched by this change is corrected to
      `is_attached`; the remaining instances are left for separate triage
      rather than swept blind

## Implementation Plan

1. Copy `chat_screen.py`'s generation counter rather than inventing a mechanism:
   bump at dispatch, carry the value with the work, compare on arrival.
2. Put the comparison at **result arrival** on each of the three surfaces,
   because a thread worker's body cannot be cancelled.
3. Correct only the `is_mounted` guards this change actually touches.

## Implementation Notes

**Headline: CCP conversation search never ran at all.** Reading the named site
closely turned up two independent defects underneath the stale-results
symptom, neither observable from the handler:

* `run_worker(self._search_conversations_sync, search_term, search_type,
  thread=True, exclusive=True, name="conversation_search")` binds
  `search_term` to `run_worker`'s own `name` parameter and `search_type` to its
  `group`, then collides with the explicit `name=` keyword —
  `TypeError: run_worker() got multiple values for argument 'name'`. Proved by
  binding the real `DOMNode.run_worker` signature, not by reading.
* `_search_conversations_sync` carried `@work(thread=True)`, and Textual's
  decorator does `assert isinstance(self, DOMNode)`. `CCPConversationHandler`
  is a plain object that merely *holds* a window, so the decorator could only
  ever have raised `AssertionError`. Proved by running a two-line repro.

The same `run_worker` misuse was present in `refresh_conversation_list`,
`load_conversation`, and `ccp_dictionary_handler.load_dictionary` — all now use
the `functools.partial` shape `CCPCharacterHandler.load_character` already used,
and the two `@work` decorators on non-DOMNode helpers are gone. **An AST sweep
found five more sites of this exact shape in `UI/Tools_Settings_Window.py`
(6685, 6747, 6894, 7000, 7325); left untouched as that screen is
`DEPRECATED (TASK-1346)` and nav-unreachable, and it is outside this task.**

**The three surfaces.** Each gained a monotonic counter bumped at dispatch and
compared in an arrival callback that runs on the event loop:

- **(a) CCP conversation search** — the worker no longer writes
  `self.search_results` from the worker thread. It hands rows to
  `_apply_search_results(generation, results)`, which adopts them only while the
  generation is current.
- **(b) Personas character load** — `_load_character_sync` no longer mutates
  handler state or posts messages from the thread; `_apply_loaded_character`
  does both, behind the generation check. Display corruption only, as the task
  states: the modern save path has its own guard.
- **(c) Model install view** — `ensure_loaded` bumps `_inventory_generation` and
  passes it to `_load_inventory`; `_apply_inventory` drops a superseded read
  outright, and gates the UI-driving branch (recompose, observation refresh,
  focus restoration) on `is_attached` so a read that lands after the user
  navigated away cannot drive another screen.

**`is_mounted`:** the one guard added here uses `is_attached`, which is the
check that can actually be `False`; no 42-site sweep was opened.

**Born-red evidence.** Both new arrival-guard tests were re-run against a
neutered guard and failed with the exact production symptom: the CCP list
rendered `[['conv-ab'], ['conv-a']]` (the stale prefix winning) and the
character surface displayed `['char.local.bob', 'char.local.alice']`. The
inventory test failed with `('stale',) == ()`.

**Modified:** `UI/CCP_Modules/ccp_conversation_handler.py`,
`UI/CCP_Modules/ccp_character_handler.py`,
`UI/CCP_Modules/ccp_dictionary_handler.py`,
`UI/Screens/model_installed_view.py`, `Tests/UI/test_ccp_handlers.py`,
`Tests/UI/test_model_installed_view.py`.
