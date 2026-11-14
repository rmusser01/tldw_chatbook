# Bounded lazy native-render caches

Goal: remove unused native FIFO allocations without changing rendering or budgets.
Design: [approved ADR226](../../../backlog/decisions/226-lazy-textual-strip-cache-allocation.md).
Execute inline; do not delegate to or contact other sessions.

ADR required: yes
ADR path: backlog/decisions/226-lazy-textual-strip-cache-allocation.md
Reason: exact-version guarded pinned-dependency allocation adaptation, explicitly approved.

1. Add `Tests/CI/test_textual_strip_cache_allocation.py` so the native contract
   enters the existing PR fast lane without a workflow edit. Import the
   normal Widgets compatibility entry point; assert a newly constructed native
   Strip owns zero FIFO objects before use and one after a single cache access.
   Observe the current eager constructor fail that allocation requirement.
2. Cover every native cache's capacity4/16/16/4/4/4/4, independence, hits/misses
   and FIFO eviction. Use native segments, filters, Unicode crops/divisions and
   ANSI rendering, comparing results with an isolated interpreter that never
   imports the adapter. Test existing Strips, repeat install and version/shape
   mismatch without mutating installed files or calling collectors.
3. Add `Utils/textual_strip_cache.py`: validate exact8.2.8/slot descriptors and
   constructor ownership first; create per-slot lazy properties and initialize
   only the native Strip data fields and empty cache slots. On first access use
   an initialization lock and recheck the slot before constructing FIFOCache.
   All native operations remain in Textual. Preserve pre-existing cache values.
4. Call that installer from the existing `_install_textual_compatibility_shims`
   after Static compatibility. No package dependency/GC/environment edits.
5. Verify native RED/GREEN and parity controls, relevant Console rendering and
   boot-GC contracts, lint/format and artifact/Backlog guards; self-review and
   commit locally. Build a new exact-head corpus and Keyword receipt, then run
   the unchanged original52x20/120x50 matrix. Keep50ms loop/100ms busy limits,
   exact activation/focus/cancellation and native resource evidence unchanged.
6. Continue isolating residual non-GC callbacks if the matrix remains non-green.
   Preserve all failures and publish audited outcomes/hashes only, not raw dumps.
