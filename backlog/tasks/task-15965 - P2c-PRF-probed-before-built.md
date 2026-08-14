---
id: TASK-15965
title: 'P2c: PRF probed before built'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-14 00:21'
updated_date: '2026-08-14 00:21'
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
- [ ] #1 The fireability census over the 22 plain-failing queries runs FIRST, before any grid point, and its per-query rows-returned table is reported
- [ ] #2 The pre-registered admission bar is applied mechanically in writing, line by line, against the measured numbers -- the verdict is computed, never argued
- [ ] #3 Every guard population (currently-hitting plain queries, negation, negatives) is derived from a baseline pass at probe time, never hardcoded
- [ ] #4 The probe report states gains AND losses by query id (the lost-column discipline)
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
