---
id: TASK-15965
title: 'P2c: PRF probed before built'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-14 00:21'
updated_date: '2026-08-14 01:55'
labels:
  - rag
  - p2c
  - fail-first
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first P2c FEATURE candidate (pseudo-relevance feedback) gets the same fail-first discipline that retired the expansion, acronym and compositional premises: the premise is probed and priced on the golden corpus under pre-registered admission criteria BEFORE any production code exists. PRF's honest target is the plain profile's 22 paraphrase + vocabulary_mismatch queries sitting at 0.000 recall; its structural risk is that a first pass returning zero or topically-wrong rows feeds PRF nothing or poison. The arc ends in a recorded NULL just as legitimately as in a shipped feature.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The fireability census over the 22 plain-failing queries runs FIRST, before any grid point, and its per-query rows-returned table is reported
- [x] #2 The pre-registered admission bar is applied mechanically in writing, line by line, against the measured numbers -- the verdict is computed, never argued
- [x] #3 Every guard population (currently-hitting plain queries, negation, negatives) is derived from a baseline pass at probe time, never hardcoded
- [x] #4 The probe report states gains AND losses by query id (the lost-column discipline)
- [x] #5 A below-bar result is recorded as the NULL beside the three retired P2c premises, with the next candidate's task filed carrying a pointer to the probe machinery
- [x] #6 An above-bar result ships PRF as plain-profile-only (off by default on hybrid/semantic), disclosed in the route-note vocabulary, priced in extra queries and wall-time, and stamped exactly once -- **N/A BECAUSE THE VERDICT IS NULL**: no grid point met the bar (clause 1: 0/22 rescued against a floor of 5; clause 2: 10 of 21 hitters lost), so there was no above-bar result to ship. No PRF production code exists, no profile field was added, no baseline was re-stamped
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Spec: Docs/superpowers/specs/2026-08-13-rag-p2c-prf-fail-first-design.md
Plan: Docs/superpowers/plans/2026-08-13-rag-p2c-prf-fail-first.md
Ledger: .superpowers/sdd/2026-08-13-rag-p2c-prf-fail-first/progress.md

1. Task 1 (unconditional): worktree venv on the pinned recipe; the probe machinery as pure, pinned functions -- term derivation (TF, stopword/query-term exclusion, deterministic tie-break) and expression composition reusing the engine's own quoting helper; always-on RED-first tests incl. hostile-token inertness against real FTS5; mutations.
2. Task 2 (unconditional): STEP 0 fireability census FIRST; the ONE licensed OR-feedback variant only if fireability < 5/22; the base grid point (N=8/M=5), the full grid only on signal; guards derived in the same run; THE VERDICT computed against the bar line by line.
3. Task 3 (conditional): ADMIT -> build Phase B plain-profile-only through the four-seam seams, disclosed and priced. NULL -> record the fourth retired P2c premise in Tests/RAG_Eval/README.md and file the next candidate.
4. Task 4 (ADMIT only): ONE re-stamp with the zero-movement proof on hybrid/semantic, live check, closure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 2 (probe run) DONE — VERDICT NULL, plus a review round that REFUTED one of my own claims. Step 0 fireability: 0/22 on the SHIPPED four-seam AND-strict first pass (floor 5) -> the one licensed OR-feedback variant activated (18/22, feedback selection only, disclosed in every table). Base point N=8/M=5 (TF, pre-registered): 0/22 rescued, 10 of 21 hitters lost (all rank-1 today), 0 new negative docs, negation 0->30 rows +10 new docs on all 3. Bar: [1] FAIL [2] FAIL [3] PASS-but-structural [4] binds. Grid stopped at the base point by pre-registration (no signal = no sweep). CORRECTION (review, reproduced independently): the oracle rescue 'ceiling' of 8/22 is a property of the TF SELECTOR'S BREADTH, not of the four-seam path — same path, same oracle feed, ranking key only: TF-8 8/22 (note 7/7, media 1/9, conv 0/6), rarest-8-by-corpus-DF 15/22 (7/7, 6/9, 2/6), rarest-1 with the query side dropped 22/22; 22/22 oracle expressions match at k=200 in every row. Defensible form: the plain path has no cross-seam ranking and a per-seam top_k, so any pass matching K+ notes buries every media/conversation target regardless of match quality, and how hard that bites depends on expansion breadth. The correction makes the null STRONGER: >=15 observable cells, PRF still rescued zero. AXIS CONTROL on the real feed (outside the pre-registration, cannot ADMIT): rarest-by-DF rescued 0 at N=8 AND 0 at N=4, losses 10->3. Loss mechanism at k=200: 0 of 10 unmatched (pure dilution). Hitter population 21 = plain.json precision/MRR 0.875 x 16 = 14 keyword + 7 scoped. Gated run: Tests/RAG_Eval/test_prf_probe_run.py; report .superpowers/sdd/2026-08-13-rag-p2c-prf-fail-first/task-2-report.md. Task 3 records the null and files the four-seam finding in its BREADTH-DEPENDENT form only; Task 4 skipped.

Task 3 (THE NULL BRANCH) DONE — the arc ends with no production code, which is the outcome the discipline exists to make available.

THE VERDICT, as recorded:

| # | pre-registered clause (spec, verbatim) | measured | result |
|---|---|---|---|
| 1 | ">=5 of the 22 plain-failing queries reach their target in the second pass's top-10" | 0 / 22 rescued | FAIL |
| 2 | "zero currently-hitting plain queries (any category) lose their target" | 10 of 21 hitters lost, every one at rank 1 today | FAIL |
| 3 | "zero new rows on negatives" | 0 new rows and 0 new docs on all 7 -- but STRUCTURAL, not live | PASS, worth nothing as safety evidence |
| 4 | "the negation guard: no negation query's row set grows with assertion-side junk (measured, reported; expected to bind)" | all 3 went 0 -> 30 rows, +10 new docs | BINDS, as pre-registered |

Fireability decided the regime in one command: 0/22 on the SHIPPED four-seam AND-strict first pass (floor 5) -> the ONE licensed OR-feedback variant activated (feedback selection only; 18/22), with every before-column left as the shipped pass. Base point N=8/M=5 was the only point run -- the sweep is licensed only on signal, and there was none. Loss diagnosis at k=200: 8 seam-displaced, 2 merge-displaced, 0 unmatched -- pure dilution against a per-seam top_k, which REFUTED my own prior that the missing plural/singular widening would strand documents.

OBSERVABILITY, in the corrected form only (the ceiling framing is retired): the plain path has no cross-seam ranking and a per-seam top_k, so any pass matching K+ notes buries every media/conversation target regardless of match quality -- and how hard that bites depends on EXPANSION BREADTH, the selector's property, not the path's. Same path, same oracle feed, ranking key only: TF-8 -> 8/22 observable, rarest-8-by-corpus-DF -> 15/22, rarest-1 with the query side dropped -> 22/22; 22/22 oracle expressions match at k=200 in every row. The honest bound on this null is therefore >=15 of 22 cells observable, PRF rescued 0.

THE TWO CONTROLS BEYOND THE BRIEF, AND WHY THE REVIEW IS PART OF THE RECORD. (1) The rescue-channel ORACLE was added mid-run because 0/22 cannot be read without knowing how many cells a rescue could have been seen in. I first ran it under ONE selector, got 8/22, and reported it as a ceiling imposed by the four-seam path; the probe printed "could not have been rescued by any real feedback set" on every run. THE REVIEW REFUTED THAT by re-running the control with only the ranking key swapped. The refutation is now the instrument: _format_oracle prints the selector-comparison table instead of the universal claim, and an assertion fails the run if a non-pre-registered selector reaches the verdict. Direction matters -- the correction makes the null STRONGER (>=15 observable, still zero), and the old wording softened a null that needs no softening. (2) An AXIS CONTROL on the REAL feed (outside the pre-registration, so it can never ADMIT) answered "would a narrower selector rescue anything?" with a measurement rather than an argument: rarest-by-corpus-DF rescued 0/22 at N=8 AND 0/22 at N=4, with collateral damage cut 10 -> 3 and every survivor merge-displaced rather than evicted.

Two spec corrections recorded rather than smoothed: the negatives guard never becomes live even under the variant (negative queries' content words are absent from the corpus, so the OR feed returns 0 rows too), and the price is real (211 content fetches per grid point; 39 of 211 fed rows -- 18% -- are label-only, so an unfetched feed skews silently to notes).

Task 3 deliverables: the null recorded beside the three retired premises in Tests/RAG_Eval/README.md ("The fourth retired P2c premise", plus pointers from the intro blockquote and the headroom section, and the probe machinery added to the Layout block); TASK-16071 filed for the four-seam path finding in the BREADTH-DEPENDENT form only, with both oracle rows, the code citations (tldw_chatbook/Library/library_local_rag_search_service.py:67 seam order, :449-452 the fixed-order concatenation, :1080-1118 row builders all "score": None, the only sort in the module at :677 being on the semantic path) and the dilution figures; TASK-16072 filed for the next candidate (clarification gate) pointing at the probe machinery and stating its premise honestly -- probably nothing measurable on this corpus without user-interaction fixtures, so a paper analysis is an acceptable deliverable; one lessons entry ("A control that holds a second variable fixed measures the PAIR, not the thing you named"). Report: .superpowers/sdd/2026-08-13-rag-p2c-prf-fail-first/task-3-report.md.
<!-- SECTION:NOTES:END -->
