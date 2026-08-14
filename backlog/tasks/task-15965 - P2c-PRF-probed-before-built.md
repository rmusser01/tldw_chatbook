---
id: TASK-15965
title: 'P2c: PRF probed before built'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-14 00:21'
updated_date: '2026-08-14 01:37'
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
- [ ] #5 A below-bar result is recorded as the NULL beside the three retired P2c premises, with the next candidate's task filed carrying a pointer to the probe machinery
- [ ] #6 An above-bar result ships PRF as plain-profile-only (off by default on hybrid/semantic), disclosed in the route-note vocabulary, priced in extra queries and wall-time, and stamped exactly once
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
<!-- SECTION:NOTES:END -->
