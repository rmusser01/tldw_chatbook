---
id: TASK-16071
title: >-
  Plain four-seam path: no cross-seam ranking + per-seam top_k buries non-note
  targets under widened queries
status: Done
assignee: []
created_date: '2026-08-14 01:52'
updated_date: '2026-08-14 07:20'
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
- `:1080-1118` (`_note_row`/`_media_row`/`_conversation_row`) **and `_prompt_row` at `:1123` (`"score": None` at `:1136`) — FOUR row builders, prompts the last and most buried seam; a cross-seam key added to only the cited three leaves prompt rows silently un-rankable** — `_note_row` / `_media_row` / `_conversation_row`, all setting
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

### Worked examples (from the probe's own printed table, so AC#1 is writable from this file alone)

- `kw-quillon-mast` -> `media-quillon-antenna`: media-seam rank 1, merged position 14 under the narrow selector (13 notes ahead of it by seam order alone).
- `kw-ashgrove-pump` -> `conv-ashgrove-pump`: conversation-seam hit, merged position 21 (two full seams ahead of it).
- `sc-storm-overflow-record` -> `media-kelsingham-angling` displacement case: position 13 — the widened pass filled the notes seam and the media target never reached the merged top-10.

(Full per-query tables: re-run `RAG_EVAL=1 pytest Tests/RAG_Eval/test_prf_probe_run.py -s`.)

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test reproduces the behaviour on the real path: with a query whose keyword pass matches at least `top_k` notes, a media or conversation document that matches well is absent from the merged top-`top_k`, and its absence is shown to be positional (present at a deeper k) rather than a non-match
- [x] #2 The decision on whether to change the merge is recorded with a measurement, not an argument: for at least one concrete alternative (cross-seam ranking on a comparable key, or a budget split across seams), the golden set's per-query gains AND losses by query id are reported before anything ships
- [x] #3 Any shipped change keeps every currently-hitting golden query's target in the plain top-10 (the lost-column discipline: zero regressions by query id, reported per id, not as an average)
- [x] #4 If the decision is to keep the fixed-order concatenation, the reason and its price are written into the module beside the merge, so the next reader finds the measurement instead of re-deriving it
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-14-rag-four-seam-cross-ranking-design.md
Plan: Docs/superpowers/plans/2026-08-14-rag-four-seam-cross-ranking.md
Ledger: .superpowers/sdd/2026-08-14-rag-four-seam-cross-ranking/progress.md

1. Venv (pinned recipe) + import provenance; measure the two plain-keyword misses BEFORE the fix (gated harness, RAG_EVAL=1) and record the mechanism (seam-burial vs genuine) — this decides the recall prediction.
2. Verify display-side consumption of row order (evidence panel) and enumerate any visible Search-mode consequence as a disclosed change.
3. RED-first pins in Tests/Library/test_library_keyword_cross_seam.py: displacement, rank-fairness, single-seam byte-identity, no-truncation, prompts-seam participation.
4. Replace the fixed-order rows.extend merge (library_local_rag_search_service.py:449-452) with interleave_rankings from RAG_Search/fusion.py keyed on (provenance.source_type, source_id); write the rule at the site citing the 16071 worked examples; no-tiering comment with the 15700 pointer; NO truncation added.
5. Gated re-measure + zero-movement proof + the PRF oracle control re-run (Task 2); User Guide/doc stamp + close-out (Task 3).
<!-- SECTION:PLAN:END -->

## Implementation Notes

**Production consumer named at final review (2026-08-14).** The consumer
enumeration ran on the private `_search_keyword` and so missed the public
`search()`'s second production caller: `Agents/library_rag_tool_provider.py`
(`:216-219` issues `mode="rag"`, `:250-252` cuts to `_MAX_TOP_K` = 10) — the
DEFAULT Console Library retrieval tool when `direct_library_tools` is off.
Under a plain profile it is the only production site where the reorder changes
SET MEMBERSHIP behind a hard cut, not just order: up to ten text-bearing note
rows become a ~4/3/3 rotation whose media/conversation rows carry label
snippets only (`"Matched media · {type}"`). Same mechanism as the probe's
18% → 54% label-only inversion, on an LLM's evidence. Anticipated cost of
rank-fairness on a path with non-text-bearing rows; the fetch case for
TASK-16174. Recorded in `Tests/RAG_Eval/README.md`'s arc section.

<!-- SECTION:NOTES:BEGIN -->
Task 2 (gated capture + PRF oracle control, HEAD 0b512f24a): ZERO MOVEMENT — gate PASSED, 105/105 cells (+0.000), grep-counted, and bit-exact (0 of 105 metrics differ by float equality); NO re-stamp, baselines untouched; plain keyword recall exactly 0.84375 / scoped 1.000 as Task 1 predicted. The PRF oracle CONTROL split the defect: TF-8 8/22->14/22 and rarest-DF 15/22->19/22, with the ENTIRE gain in the conversation column (0/6->6/6, 2/6->6/6) and media moving +0 under both selectors (1/9, 6/9) — conversation burial was seam ORDER (all six now at position 3, the conv seam's rank-1 slot; within-seam rank was always 1), media shortfall is per-seam VOLUME (misses sit 6-28 deep in their own seam, unreachable by any ordering change); notes 7/7 verified at position 1, not assumed. Bound recalibrates >=15/22 -> >=19/22 observable. Real feed also moved: TF base point 0/22->2/22 rescued (both conversation targets), which licensed the 9-point sweep for the first time (max 2/22 vs bar >=5) and the axis control 0/22->1/22 at both points — VERDICT still NULL, PRF STAYS RETIRED, null now stronger. Fed rows went 18%->54% label-only: the merge changes WHAT a top-M consumer sees. Batteries unchanged: 5 / 340 / 315p+13s / 306 gated / 1 PRF; scope_pipeline 4F/73P still dev's pre-existing red. HANDOVER: Tests/RAG_Eval/harness/runner.py:740-743 prints a now-false 'fixed seam order' claim on every gated run, plus README:773 and :1079; the README PRF section's rows :714, :736-741 and :754-760 are stale too, not just the oracle table.

CLOSE-OUT (Task 3). SHIPPED: the plain four-seam merge in `Library/library_local_rag_search_service.py` is `interleave_rankings` (the engine's own rank-fair primitive) keyed on `(provenance.source_type, source_id)` — each seam's rank-1 row, then each seam's rank-2 row, with `_KNOWN_KEYWORD_SOURCE_TYPES` order breaking ties WITHIN a position (pinned, not accidental). ORDER only: no truncation added, no tiering (all four rankings are all-primary, so 15700's tier design has nothing to tier here; it applies the day this path gains fallback forms). The rule, the worked examples and both decisions are written at the merge site.

AC#1 — `Tests/Library/test_library_keyword_cross_seam.py`, 5 pins on the REAL path over four REAL databases written by production's own writers, RED 3/5 on the concatenation with the verbatim failures ("the media seam's rank-1 row is merged BEHIND the notes seam's rank-5 row"; the prompts seam's only hit at merged position 7). Positional-not-non-match is shown twice: the buried row is present in the same merged list at a deeper position (pin (d) also pins that nothing is truncated), and the PRF probe's oracle control has 22/22 of its expressions matching their target at k=200 in every row, so every miss it counts is displacement. Three mutations red the pins (extend-loop restored; seam order permuted within a position; each seam's ranking reversed) — the last one only after the references were re-sourced from the seam methods instead of through `search()`.

AC#2 — measured before shipping, both directions. BEFORE the fix (Task 1, gated): plain keyword recall 0.84375 = 13.5/16 with the three lost targets diagnosed at DEEP_K=200 as GENUINE non-matches (0-1 rows total; morphology, not seam burial), plus the census that decided the whole contract — all 60 golden queries return ≤1 row under the shipped plain pass at k=10 AND k=200 (rows>1: 0; multi-seam: 0), so an order-only change is the IDENTITY on this instrument and a moved cell would be a STOP. AFTER (Task 2): gate PASSED, 105/105 cells (+0.000), and bit-exact — 0 of 105 metrics differ by float equality. No re-stamp; baselines untouched. Gains and losses by query id exist only in the widened (oracle-fed) regime and are reported there: rescues `pr-platform-offline` and `vm-nearsightedness` (both conversation targets); losses swapped, below.

AC#3 — per id, measured post-fix rather than argued: the plain per-query capture at HEAD is byte-identical to Task 1's pre-fix table on all 16 keyword ids (same targets, same retrieved lists, same per-query recall; mean 0.84375, precision 0.875, scoped 1.000), and the census re-reads rows>1: 0, multi-seam: 0 over all 60. Zero regressions by query id.

AC#4 — the antecedent did not hold (the concatenation was replaced, not kept), and the obligation behind it was honoured in the branch actually taken: the reason, the 16071 worked examples, the order-within-position convention, the no-tiering decision with its 15700 pointer, the vacuous-dedup note and "no truncation added" are all written beside the merge, so the next reader finds the measurement instead of re-deriving it.

THE CONTROL SPLIT ONE NUMBER INTO TWO DEFECTS. Oracle table, pre-fix → post-fix (same feed, path, k, corpus; the pre-fix column re-measured in this environment by reverting the merge, byte-identical to the committed table on every row, so the whole delta is attributable to this change): TF-8 8/22 → 14/22, rarest-DF 15/22 → 19/22, rarest-1 22/22 → 22/22. Column by column: note 7/7 → 7/7 (verified at position 1, not assumed), conversation 0/6 → 6/6 and 2/6 → 6/6, media 1/9 → 1/9 and 6/9 → 6/9 (+0 under BOTH selectors). So conversation burial was seam ORDER and is gone completely (all six now read merged position 3, the conv seam's rank-1 slot in the harness's three-seam fan-out); the media shortfall is per-seam VOLUME — those misses sit at within-seam rank ≥7 (`r ≥ (pos+1)/3` from merged positions 18-82) against a reachable depth of 3 in a ten-row window, so no ordering change reaches them. Bound recalibrated ≥15/22 → ≥19/22 observable. THE RESIDUAL HEADROOM IS THE MEDIA COLUMN AND IT IS NOT AN ORDERING PROBLEM: the levers are the per-seam budget or the widening construction (TASK-3997/15400 territory), never another merge change.

THE DISCLOSED COST, both halves measured. (1) The collateral swap: in the PRF probe's widened-feed regime the loss count is 10 of 21 hitters either side, but two of the ten are DIFFERENT queries — pre-fix lost `kw-ashgrove-pump`/`kw-drayton-conveyor` (conversation targets merge-displaced at position 21, now pulled back inside k), post-fix loses `kw-plant-maintenance-record`/`sc-meter-box-key` (both NOTE targets). The rotation hands media/conversation rows slots a full notes seam used to monopolise, so the displacement cost lands on the notes seam; net-neutral on the clause, named so it is not a quiet count. (2) The fed top-M window went 39/211 (18%) → 113/211 (54%) label-only at the same 211 fetches: the merge changes WHAT a top-M consumer sees, not merely the order — landed in TASK-16174, whose AC#4 premise it triples.

LIVE CHECK — DEMONSTRATED, both arms in the real TUI on the same data, one code constant apart. Scratch profile (own HOME/XDG_*/`TLDW_CONFIG_PATH` + `[paths] data_dir`, `HF_HUB_OFFLINE=1`); isolation confirmed at the running PID (`lsof`: 0 handles under the real profile, 52 under the scratch) and the real `config.toml` byte-identical before and after (sha256 42e2f42d…). Library seeded through the app's own writers because the precondition (one query matching several notes AND a media AND a conversation) does not exist in a real library: 12 shift-log notes that mention "kestrel gearbox inspection" in passing, one media document that IS the manual, one conversation about scheduling it. Search mode, default profile ("Evidence · top 15 per source", 14 results both arms). SHIPPED: 1 note, **2 Kestrel gearbox inspection manual (Media)**, **3 Maintenance: kestrel gearbox inspection scheduling (conversation)**, then notes 4-14. REVERTED merge (Edit, restored with Edit, `git diff --quiet` clean afterwards): notes 1-12, **13 the media manual, 14 the conversation**. The TASK-15810 stall was not hit — it belongs to RAG Answer's first embedding-backed query, and Search mode is keyword-only.

BATTERIES at close: cross-seam pins 5; + the six Library rag files 340; `Tests/RAG_Eval` ungated + inventory 315 passed/13 skipped (320 with the pins); gated `Tests/RAG_Eval` excl. the PRF run 306 with the gate PASSED at 105/105 (+0.000); gated PRF run 1 (VERDICT NULL over 9 points, PRF stays retired). Pre-existing dev red, unrelated and byte-identical: `Tests/RAG/test_scope_pipeline_enforcement.py` 4F/73P; `Tests/UI/test_library_shell.py` excluded (known monolith hang). DOCS: `Tests/RAG_Eval/README.md` gains the arc section ("The sixth arc, and the second re-stamp that did NOT happen") and its PRF section is rewritten as post-16071 fact with the pre-fix figures kept as dated history (the staleness marker is retired); `Docs/User_Guide/library/search-and-rag.md` discloses the Search-mode row-order change with a live stamp; two lessons added to `backlog/docs/lessons-testing-evidence.md` (an expected value computed through the code under test cannot fail; a fix recorded only in a gitignored file is not a fix).
<!-- SECTION:NOTES:END -->
