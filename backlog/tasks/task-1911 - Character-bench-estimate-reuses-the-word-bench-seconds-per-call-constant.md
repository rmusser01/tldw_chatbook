---
id: TASK-1911
title: Character bench estimate reuses the word-bench seconds-per-call constant
status: To Do
assignee: []
created_date: '2026-08-02 04:15'
labels:
  - evals
  - character-probe
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`CharacterBenchEstimate` reports a projected duration using the word bench's
`_ASSUMED_SECONDS_PER_CALL`, which was measured against a single-token logprob
probe. A character-probe call generates up to `max_tokens` of free text, so the
estimate is off by roughly an order of magnitude and understates how long a run
will take. The estimate exists specifically so a user can decide whether to press
Run; an estimate that wrong defeats its own purpose.

Found by the whole-branch review of the character-probe Phase 2 authoring UI
(`feat/character-probe-phase2-1691`), triaged as follow-up rather than a merge
blocker because the call COUNT — the load-bearing half of the estimate — is exact.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The character-probe duration estimate uses a generation-based rate, not the logprob-probe rate
- [ ] #2 The rate's provenance is documented where it is defined
- [ ] #3 The word bench's own estimate is unchanged
<!-- AC:END -->
