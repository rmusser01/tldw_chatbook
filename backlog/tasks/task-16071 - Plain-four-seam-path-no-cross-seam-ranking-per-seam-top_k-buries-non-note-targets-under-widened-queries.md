---
id: TASK-16071
title: >-
  Plain four-seam path: no cross-seam ranking + per-seam top_k buries non-note
  targets under widened queries
status: To Do
assignee: []
created_date: '2026-08-14 01:52'
labels:
  - rag
  - library
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The plain profile's four-seam keyword path runs each source type as its own
seam with its own `top_k`, then concatenates the seams in a fixed source
order. No row carries a score (every row builder sets `"score": None`) and
nothing sorts the concatenation, so **cross-seam position is decided by
source type and by how many rows the earlier seams returned — never by match
quality**. The consequence: any pass that matches `top_k` or more NOTES
buries every media and conversation target behind them, however well those
targets match.

This is not a PRF finding. It was measured while probing PRF (TASK-15965),
which is why the evidence below is stated in that probe's terms, but it
prices **any** technique that widens a query on this path — expansion,
stemming, prefix fallbacks, a future clarification gate's rewrite — because
widening is what makes a pass match many notes. How hard it bites depends on
**expansion breadth**, which is a property of the widening technique, not of
the path.

Both halves of that sentence are load-bearing and both are measured. Feeding
the retrieval its own target document (an oracle feed — the best expansion
any technique could produce) and changing **only the term-ranking key** moves
the number by nearly a factor of two:

| selector (oracle feed, same path, same k, same composition) | N | target reaches top-10 | note | media | conversation |
|---|---|---|---|---|---|
| TF `tf/\|D\|` | 8 | 8 / 22 | 7/7 | 1/9 | 0/6 |
| rarest-by-corpus-DF (ranking key only) | 8 | **15 / 22** | 7/7 | 6/9 | 2/6 |

22 of 22 oracle expressions match their target at k=200 in every row, so
every miss above is displacement, not a failure to reach the document. Note
targets are unaffected in both rows (7/7) because notes are the first seam;
media and conversation targets are where the whole effect lives.

**Do not restate this as a fixed ceiling.** An earlier write-up of the same
data claimed the path "caps any query-widening technique at 8/22" and that
"14 of 22 cells are never observable". A review refuted that by re-running
the control with the ranking key swapped, and the two rows above are the
refutation. The path property is real; the number attached to it is a joint
property of the path and the widening.

**Collateral damage is on the same axis.** In the same probe run, widening
the query cost currently-hitting queries their rank-1 answers: **10 of 21
hitters lost under the broad (TF-8) selector, 3 under the narrower
(rarest-8-by-DF) one.** Diagnosed by re-running the same expression at k=200:
of the 10, **8 were seam-displaced and 2 merge-displaced — 0 unmatched.**
Every lost document was still matched; it lost its seam's 10-row budget to
expansion-term rows. The damage channel is pure dilution against a per-seam
cap, and it is exactly the mechanism this task is about.

Code, as it stands (`tldw_chatbook/Library/library_local_rag_search_service.py`):

- `:67` — `_KNOWN_KEYWORD_SOURCE_TYPES = ("notes", "media", "conversations", "prompts")`, the fixed seam order.
- `:449-452` — the merge: `rows: list[...] = []` then
  `for source_type in _KNOWN_KEYWORD_SOURCE_TYPES: ... rows.extend(outcomes[source_type][1])`.
  A fixed-order concatenation, and the only `sort` in the module (`:677`) is
  on the SEMANTIC path, not this one.
- `:1080-1118` — `_note_row` / `_media_row` / `_conversation_row`, all setting
  `"score": None`, so there is no cross-seam key to sort on even if a sort
  were added today.
- Each seam is called with `limit=top_k` (e.g. `_search_notes`, `:467+`), so
  the cap is per seam and the merged list can be up to 4×`top_k` long with
  the first seam owning the first `top_k` slots.

Two smaller observations from the same run, recorded here so they are not
re-derived:

- **`_FTS5_STOPWORDS` (67 words) is too short for TF-based term derivation on
  this corpus.** `rather`, `once`, `each`, `taken`, `through`, `back`, `same`,
  `before` survive into derived expansion lists and do the expanding — which
  is also the mechanism behind the broad selector's poor showing above.
- **Media and conversation seam rows carry no document text** (`"Matched
  media · {type}"`, `"Matched conversation · N messages"`). In the probe's
  feed, 39 of 211 rows (18%) were label-only, so anything deriving terms from
  seam rows must pay one content fetch per row or its input skews silently to
  notes.

Committed record: TASK-15965's Implementation Notes and the "fourth retired
P2c premise" section of `Tests/RAG_Eval/README.md`. Full measurement with the
probe's verbatim output:
`.superpowers/sdd/2026-08-13-rag-p2c-prf-fail-first/task-2-report.md` (§3,
§4a, §5) — an untracked SDD record; the run is reproducible with
`RAG_EVAL=1 .venv/bin/python -m pytest Tests/RAG_Eval/test_prf_probe_run.py -s -q`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test reproduces the behaviour on the real path: with a query whose keyword pass matches at least `top_k` notes, a media or conversation document that matches well is absent from the merged top-`top_k`, and its absence is shown to be positional (present at a deeper k) rather than a non-match
- [ ] #2 The decision on whether to change the merge is recorded with a measurement, not an argument: for at least one concrete alternative (cross-seam ranking on a comparable key, or a budget split across seams), the golden set's per-query gains AND losses by query id are reported before anything ships
- [ ] #3 Any shipped change keeps every currently-hitting golden query's target in the plain top-10 (the lost-column discipline: zero regressions by query id, reported per id, not as an average)
- [ ] #4 If the decision is to keep the fixed-order concatenation, the reason and its price are written into the module beside the merge, so the next reader finds the measurement instead of re-deriving it
<!-- AC:END -->
