---
id: TASK-15701
title: >-
  SimpleRAGCache sync get/put render a key that ignores three search-defining
  dimensions
status: Done
assignee: []
created_date: '2026-08-12 23:05'
labels:
  - rag
  - cache
dependencies: []
priority: medium
---

## Description

`SimpleRAGCache._make_key` takes eight inputs, three of which describe
*what search was actually run*: `keyword_source_types` (which sub-legs the
keyword leg queried), `hybrid_fusion` (alpha / rrf_k / pool), and
`fts_match_construction` (how the FTS MATCH expression was built).

The **async** path passes all three. The **synchronous** public API does
not — and cannot: `SimpleRAGCache.get()` / `.put()` do not accept those
parameters at all, so `_sync_get_impl` (`simple_cache.py:488`) and
`_sync_put_impl` (`:716`) call `_make_key` with five arguments and render
the legacy, construction-less key.

**Today this is latent: no production code calls the sync methods** (grep
across `tldw_chatbook/`; the only constructor is `rag_service.py:726` and
every live read/write goes through the async path). The task is to close
the trap before something wires to it, not to fix a live bug.

**Why it got worse, and why it is worth a task now.** Before TASK-15400 the
legacy key was *accidentally correct* for a default-config search: `and`
was both the legacy value and the shipped default, so an omitted key part
described the search truthfully. Since the default flipped to
`and_stopword_trim` (2026-08-12) that is no longer true — a sync-path entry
would be stored and served under a key asserting the **full AND** produced
it, while `and_stopword_trim` actually did. The failure mode is not a
missed cache hit (which is harmless); it is a **wrong hit**: two searches
that differ in a dimension the key omits collide, and the second one is
served the first one's rows. The same argument applies to the other two
omitted dimensions — a Notes-only keyword search and an all-sources one
collide, as do two searches at different fusion weights.

The same class of failure has already been paid for once in this area
(TASK-4110's sweep would have measured nothing had fusion parameters stayed
out of the key), which is why TASK-15400 Task 1 put the construction into
the async key and Task 2 pinned the sync twins rather than leaving them
unremarked.

**Restated after TASK-15700 (2026-08-13).** The shipped default moved again,
`and_stopword_trim -> and_then_prefix`, so the paragraph above is unchanged
in force and only its name is stale: the sync key still asserts the **full
AND** while the leg now runs a full-AND primary *plus a prefix fallback*.
The gap between what the key claims and what produced the rows is therefore
wider than it was, not narrower. The async key is value-keyed and renders
`fts:and_then_prefix`, so entries cached under the previous construction are
keyed apart rather than invalidated — correct, at the cost of one run of
cold misses after the upgrade.

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 The synchronous `get`/`put` either accept and forward all three search-defining dimensions to `_make_key`, or are removed / made private if the async path is genuinely the only supported entry point — the decision is stated in the module docstring with its reason
- [x] #2 A test fails on today's behaviour: two sync `put`/`get` round trips that differ ONLY in `fts_match_construction` (and, separately, only in `keyword_source_types`, and only in `hybrid_fusion`) must not collide on one key
- [x] #3 If the sync API is kept, the async and sync paths are shown to render the SAME key for the same search, rather than each being asserted separately
- [x] #4 The "no production caller" claim is re-verified at the time of the fix and recorded, so a future reader knows whether this was still latent when it was closed
<!-- AC:END -->

## Related

- **TASK-15400** — put `fts_match_construction` into the async cache key and
  flipped the shipped default, which is what turned this from a missed-hit
  risk into a mislabelled-entry risk. `config.py`'s field comment and
  `simple_cache._make_key`'s docstring both qualify their guarantee to the
  async path and point here.
- **TASK-4110** — the precedent: fusion parameters absent from a cache key
  would have flattened that arc's sweep silently.

## Implementation Notes

**AC#1 — the sync API is KEPT and corrected, not removed.** It now accepts
and forwards all three search-defining dimensions, so it renders the same key
as the async path. The reason is stated in `get`'s docstring: the sync
methods are the cache's ergonomic **test** surface — 58 call sites across
five test modules — while production reads and writes go through the async
path only. Removing a correct-but-unused API to fix a key bug would have
traded a latent trap for 58 rewrites and no behavioural gain.

**AC#4 — the "no production caller" claim re-verified at fix time
(2026-08-18) and it still holds.** The only `SimpleRAGCache` traffic in
`tldw_chatbook/` is `rag_service.py:1301` (`get_async`), `:1395`
(`put_async`) and `:4229` (`get_metrics`). The `_global_cache` singleton has
no consumer outside `simple_cache.py` itself. So this was closed while still
latent, which is the state the task was filed to preserve.

**AC#2 — three RED-first collision tests**, one per omitted dimension
(`fts_match_construction`, `keyword_source_types`, `hybrid_fusion`),
parametrized so each names the dimension it protects. Each asserts the
*wrong-hit* failure specifically: the identical search must still hit, and
the differing one must not be served the first search's rows. A missed hit
would be harmless; being served another search's results is not.

**AC#3 — shown, not asserted separately:** a sync `put` followed by an async
`get_async` for the same search. That round trip cannot pass while the two
paths render different keys, which is exactly what asserting each path's key
in isolation would have failed to catch.

`Tests/RAG/simplified/` + the three sibling suites: 354 passed. The one red
(`test_config.py::test_config_loading_from_file`) is pre-existing on clean
dev — baselined in a throwaway worktree at `e49c5dba8` before being called
unrelated.
