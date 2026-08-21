---
id: TASK-19563
title: >-
  Three surfaces show stale results because name= was mistaken for worker
  scoping and no generation guard exists
status: To Do
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

- [ ] Each of the three surfaces discards results that no longer match the
      current input, using the generation-counter pattern already established
      in `chat_screen.py`
- [ ] The guard is applied at **result arrival**, not only at dispatch — thread
      worker bodies cannot be cancelled, so arrival-time rejection is required
- [ ] Typing quickly in CCP conversation search never leaves the list showing
      results for a prefix of the current query — pinned by a test that
      interleaves two dispatches and delivers them out of order
- [ ] The Personas character load no longer renders a superseded character
- [ ] `model_installed_view` toasts cannot land on an unrelated screen
- [ ] Any `is_mounted` post-await guard touched by this change is corrected to
      `is_attached`; the remaining instances are left for separate triage
      rather than swept blind
