---
id: TASK-16839
title: 'Fix and bound calculate_change_percentage (quadratic cost and autojunk-degenerate ratios)'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-16'
labels:
  - bug
  - perf
  - watchlists
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two entangled defects in one function, from the TASK-15764 review (PR #1679, finding 5);
merged into one task because any fix to one must account for the other (they are opposite
regimes of the same `SequenceMatcher` call). Verified at dev `ee741cf10`:
`Subscriptions/monitoring_engine.py:255-273` still runs
`SequenceMatcher(None, old_content, new_content)` — character-level, default
`autojunk=True`, unbounded input.

1. **Correctness — autojunk makes the percentage meaningless for large Latin pages.**
With `autojunk=True`, any element occurring in >1% of a ≥200-element sequence is junked;
for Latin text at ~160 KB every letter clears that bar, the matcher degenerates, and the
review measured **pct=1.0 ("100% changed") for a 5%-edited page**. Downstream this feeds
`change_info["change_percentage"]` and every noise/threshold decision built on it.

2. **Cost — quadratic in the large-repertoire regime.** For content where nothing is
junked (CJK, heavy unicode), the review measured a clean 4x-per-doubling curve: 7.1 ms at
10k chars → 257.9 ms at 80k chars, extrapolating to **~7 minutes at the 10 MB
`MAX_FETCH_BYTES_PAGE` cap** — on a worker thread that holds the GIL the whole time
(difflib is pure Python), so it starves the app even off-loop (task-15764 moved it off
the event loop; it did not bound it).

Fixing the cost by capping input changes the reported percentage; disabling autojunk to
fix correctness explodes the cost — hence one task. A segment-level (line/sentence) diff
with an input-size bound is the likely shape; whatever ships must state what the
percentage now means. Note the 15764 lesson entry in `lessons-testing-evidence.md`
carries the old (wrong) mechanism claim for this function's cost — correct it if this
task changes the story it tells.
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

1. Consumer census: the ratio feeds exactly (a) `check_url`'s `change_percentage <
   threshold` withhold comparison (default threshold 0.0), (b) the withheld disposition
   (`withheld_percentage`, x100), (c) the stored item's `change_percentage` (x100,
   rendered `f"{pct:.0f}% changed"` by `content_pane.render_change` and echoed by
   briefings/tool service). All consumers need a coarse, monotonic-ish 0..1 magnitude,
   not char-exact ratios.
2. Rebase the percentage on the SAME `_segment_for_diff` segmentation the stored diff,
   `diff_summary` and `added_and_removed_text` already use, so all outputs of a change
   tell one story: primary = `SequenceMatcher(None, old_segments, new_segments,
   autojunk=False).ratio()` (autojunk over near-unique segments is the defect's
   mechanism, so it is off and cost is bounded explicitly instead); fallback = a
   segment multiset ratio (order-insensitive, O(n)) whenever the alignment tier's
   measured cost model says alignment could be slow (total segments or
   equal-segment collision count over a budget).
3. Bound `_segment_for_diff` itself: `textwrap.wrap` is quadratic on a unit containing
   a giant unbreakable run (a 10 MB single-line CJK page costs minutes) — detect
   `[^\s-]{1001,}` and fixed-slice that unit instead.
4. Born-red first: degeneracy pin (5%-edited large Latin page must report ~5%, not
   ~100%) and a loose wall-clock cost pin at a large-repertoire input; then implement,
   then measure at 160 KB Latin / 160 K-char CJK / the 10 MB cap.
5. Re-run the threshold-consumer families (noise-not-volume, content-kind producer,
   off-loop thread identity x2, local service); update the three pins the new basis
   legitimately shifts, each with a disclosed reason.
6. Correct the task-15764 lesson entry (its mechanism/number claims describe code this
   task retires) and the stale module comments; ruff on touched files.

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A 5%-edited large Latin page reports a small change percentage, not 1.0 (regression test)
- [x] #2 Worst-case runtime at the 10 MB fetch cap is bounded to sub-second, measured and stated
- [x] #3 Existing change-classification behavior for ordinary pages is preserved or the delta is characterized (byte-diff over the 15764 semantic-identity cases or equivalent)
- [x] #4 The stale mechanism/number claims in the 15764 lesson entry are corrected to match the shipped behavior
<!-- AC:END -->

## Implementation Notes

**Design.** `calculate_change_percentage` now computes its 0..1 ratio over
`_segment_for_diff` segments -- the same sentence/line-sized basis the stored
diff body, `diff_summary` and `added_and_removed_text` already use -- so the
percentage, the diff and the added/removed haystacks all describe the change at
one granularity. Two tiers in `_segment_change_ratio` (**superseded by the
fix round below -- the alignment tier is retired; the multiset ratio is the
sole mechanism at every size**): alignment
(`SequenceMatcher` over segments, `autojunk=False` -- popularity junking IS the
degeneracy mechanism, so it is off and cost is bounded explicitly instead) for
pages up to `_ALIGNMENT_MAX_TOTAL_SEGMENTS`=4,000 segments with
`_ALIGNMENT_MAX_SEGMENT_COLLISIONS`=200K equal-segment pairs; past either
bound, an O(n) multiset ratio (`2*matches/total`, order-insensitive --
documented as the honest coarse answer for the noise-threshold consumer). Both
bounds chosen from adversarial measurements (edit-every-2nd-segment: 119 ms at
the 2,000/side boundary; repetitive pages route to multiset). Alternatives
rejected: input truncation (silently zeroes edits past the cap), sampling
(nondeterministic), keeping char-level for small pages (the autojunk cliff
starts at a few KB, so "small" had no safe region).

**Consumer evidence.** The ratio has exactly three consumers -- `check_url`'s
threshold withhold (default 0.0), the withheld disposition, and the x100
display value rendered `f"{pct:.0f}% changed"` -- all coarse-magnitude readers.

**Cost, measured** (M-series laptop; was ~39 s at 128 KB Latin / quadratic 4x
per doubling for CJK / ~7 min extrapolated at the cap): 160 KB Latin 5%-edit
5.2 ms; 160 K-char CJK one-sentence edit 7.5 ms; at the 10 MB fetch cap: Latin
647 ms, CJK prose 236 ms, spaceless blob 246 ms -- worst measured shape at the
cap is 647 ms, stated bound sub-second, in-test regression bound 5 s (loose
wall-clock, marked as such).

**Value sanity**: identical 0.0; disjoint ~1.0; 5%-edited large Latin page
0.0500 (was 0.47-1.0 degenerate); 1-of-41-sentence edit 2.4%; empty-vs-content
1.0; whitespace-only difference 0.0 (agrees with the "no textual change after
normalization" diff path).

**Two supporting fixes in the same seam** (both disclosed): (a)
`_SENTENCE_BOUNDARY` now also splits after CJK sentence enders (。．！？)
without requiring whitespace -- previously an entire CJK page was ONE unit
fixed-wrapped into 110-char slices whose boundaries all shift under any edit;
(b) `_segment_for_diff` fixed-slices units containing a whitespace-free run
>1,000 chars (`_UNWRAPPABLE_RUN`) because `textwrap.wrap` re-slices the whole
remainder per emitted line (quadratic; minutes at 10 MB spaceless) -- for a
fully unbreakable unit the slices are byte-identical to what wrap produced
(verified empirically).

**Threshold-consumer disposition.** End-to-end byte-diff vs origin/dev over the
eleven 15764 semantic-identity cases (`semantic_delta_16839.py`, adapted from
the review harness): 10/11 byte-identical in every non-percentage field
(dispositions, diff bodies, summaries, rule-match text, snapshots); the one
divergence is the intended CJK-boundary improvement (finer diff lines, same
content). Percentages shift basis as characterized per case -- e.g. "oversized"
(all 400 sentences replaced) 10.16%->100% (the old value was the degenerate
lie), "withheld below threshold" 2.35%->2.50% still withheld. Three test pins
updated with disclosed reasons: the noise-not-volume `< 1.0` precondition is
now `< 10.0` (1 segment of 41 is 2.4%; still below the retired 0.1 default the
pin exists to guard), the two off-loop `_segment_for_diff == 2` counts are 4
(the percentage hop segments once per side too; the details hop still shares),
and the content-kind producer's `"0% changed" not in` substring check gained
`(?<!\d)` (it matched the tail of the new honest "60% changed").

**Born-red evidence** (run at pre-fix HEAD): degeneracy pin FAILED with
pct=0.471 for the 5%-edited page after ~39 s; cost pin FAILED with 7.2 s
against the 2 s bound at 640 K chars (a first-draft deterministic-cycling CJK
generator did NOT reproduce the quadratic regime -- periodic text aligns
cheaply -- and was replaced with seeded-random draws; recorded in the test).

**Tests**: new `Tests/Subscriptions/test_change_percentage_bounds.py` (8);
`Tests/Subscriptions/` 805 passed / 1 pre-existing skip; wider watchlist suites
(Tools/UI/Scheduling/DB) 334 passed. Ruff clean on touched files; no new
format dirt (the one `ruff format` hunk left in `monitoring_engine.py` is the
known pre-existing dirt in the untouched FeedMonitor region).

**Files**: `tldw_chatbook/Subscriptions/monitoring_engine.py`;
`Tests/Subscriptions/test_change_percentage_bounds.py` (new);
`test_url_monitor_off_loop.py`, `test_watchlists_db_instance_and_off_loop.py`,
`test_watchlist_noise_not_volume.py`, `test_watchlist_content_kind_producer.py`
(pin updates); `backlog/docs/lessons-testing-evidence.md` (AC#4 correction).

### Fix round (2026-08-20, independent review FIX-FIRST verdict)

**Finding 1 (blocking) -- ALIGNMENT/MULTISET tier cliff for reorders, FIXED
by retiring the alignment tier.** The review reproduced a sign-flip
discontinuity at the 4,000-total-segment boundary: a purely reordered page
(same segments, shuffled, zero content change) reported pct=0.9925 at 4,000
total segments (order-sensitive `SequenceMatcher` tier) and pct=0.0000 at
4,002 (order-insensitive multiset tier) -- one extra sentence per side
flipped "99% changed" to "0% changed". Born-red evidence: the new
`test_pure_reorder_reports_zero_change_with_no_size_cliff` was run against
6d22de89f and failed with exactly that signature ("a pure reorder reports
0.9925 at 4,000 total segments but 0.0000 at 4,002"). Resolution: the
**two-tier design described above is retired**; `_segment_change_ratio` is
the O(n) order-insensitive multiset ratio at EVERY size, and both alignment
constants are deleted. This is the only continuous option: any hard boundary
between an order-sensitive and an order-insensitive mechanism flips on
reorder-shaped edits wherever it is placed, and order-sensitivity at every
size is the unaffordable quadratic this task retired. **Semantic decision,
documented in `_segment_change_ratio`'s docstring: a segment that merely
moved is not a content change -- a pure reorder reports 0.0 at any size.**
Rationale: the consumers read the value as "how much content changed"; a
re-sorted listing page is exactly the noise a raised threshold exists to
withhold, and near-100% for zero content change is the misleading-percentage
defect this task exists to eliminate. A pure reorder is still detected (hash
differs) and at the default threshold 0.0 still produces an item, with the
moves visible in the (position-aware) diff body. Behavior delta vs the
reviewed commit: none for non-move edit shapes (multiset equals the
alignment value there -- the review's own probe: 0.050000/0.049975 across
the boundary for a scattered 5% edit); moves now consistently count 0
instead of order-dependently 0-or-~1.

**Finding 2 (non-blocking) -- doubled segmentation across `check_url`'s two
hops, FIXED by sharing segments.** The percentage hop
(`_change_percentage_with_segments`, new) segments each side once and
returns the lists; `check_url` passes them to
`_build_significant_change_details`, which (like `build_change_diff` and
`added_and_removed_text` already did) accepts optional pre-built segments.
`calculate_change_percentage` gained matching optional
`old_segments`/`new_segments` kwargs; all default to None so every other
caller is unchanged. Measured at the 10 MB cap: end-to-end changed-path CPU
947 ms -> 528 ms (one `_segment_for_diff` pass over a 10 MB side is ~200 ms,
paid 4x before, 2x now), outputs byte-identical across the two shapes.

**Re-measured post-fix** (same M-series laptop): 160 KB Latin 5%-edit
6.2 ms; at the 10 MB cap: Latin 431 ms, CJK prose 162 ms, spaceless blob
166 ms -- worst measured shape at the cap is now 431 ms (the multiset-only
path is faster than the reviewed two-tier's 647 ms; the 5 s in-test loose
bound is unchanged). Pure-reorder probe: 0.0 at 12/300/2,000/2,001/6,000
segments per side -- no cliff.

**Test pin updates in this round**, each disclosed in-file: the two off-loop
`_segment_for_diff` counts go 4 -> 2 (the cross-hop sharing is the fix;
an extra call now means the sharing broke); the off-loop/service spies on
`calculate_change_percentage` and `_build_significant_change_details`
forward the new segment kwargs;
`test_multiset_regime_above_the_alignment_bound_keeps_sane_values` is now
`test_very_large_page_keeps_sane_values` (the bound it referenced no longer
exists). `Tests/Subscriptions/`: 806 passed, 1 pre-existing skip.

**Lessons**: corrected the 15764/16839 entry's mechanism sentence (multiset
is now the sole tier) and added a new entry: a tiered design's boundary must
be probed with a shape the tiers DISAGREE on -- the smooth scattered-edit
step proved nothing because the tiers agree on that shape by construction.
