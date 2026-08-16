---
id: TASK-16839
title: 'Fix and bound calculate_change_percentage (quadratic cost and autojunk-degenerate ratios)'
status: To Do
assignee: []
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

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A 5%-edited large Latin page reports a small change percentage, not 1.0 (regression test)
- [ ] #2 Worst-case runtime at the 10 MB fetch cap is bounded to sub-second, measured and stated
- [ ] #3 Existing change-classification behavior for ordinary pages is preserved or the delta is characterized (byte-diff over the 15764 semantic-identity cases or equivalent)
- [ ] #4 The stale mechanism/number claims in the 15764 lesson entry are corrected to match the shipped behavior
<!-- AC:END -->
