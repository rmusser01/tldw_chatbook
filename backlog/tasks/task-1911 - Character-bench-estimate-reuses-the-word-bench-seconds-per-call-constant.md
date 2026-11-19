---
id: TASK-1911
title: Character bench estimate reuses the word-bench seconds-per-call constant
status: Done
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
- [x] #1 The character-probe duration estimate uses a generation-based rate, not the logprob-probe rate
- [x] #2 The rate's provenance is documented where it is defined
- [x] #3 The word bench's own estimate is unchanged
<!-- AC:END -->

## Implementation Notes

Implemented in the Evals UX follow-up batch (fix/t1-evals, 2026-10-11).

New `_CHARACTER_ASSUMED_SECONDS_PER_TOKEN = 0.05` in `inspector.py`,
defined beside the word-bench constant with its provenance documented
in place (AC2): an explicitly-labelled placeholder in the same sense as
`_ASSUMED_SECONDS_PER_CALL` (no provider is called from that module),
scaled per generated TOKEN because that is the axis a generation call's
cost varies on -- ~20 tokens/second, a conservative round figure for a
small model on a local llama.cpp server on consumer hardware. The
estimate is `call_count * max(config.max_tokens, 1) * rate`, so a user
who lowers `max_tokens` in the character-bench editor sees the estimate
fall with it (the fixture bench: one call at the default 512 projects
~00:26 where the logprob rate projected ~00:00). The word bench's own
estimate keeps `_ASSUMED_SECONDS_PER_CALL` untouched (AC3) -- pinned by
an assertion in the new test that the two rates produce different
durations for the same call count, plus all pre-existing word-bench
estimate tests passing unchanged.

The call-count half stays exact (`_character_bench_call_count`
unchanged). Also folds in task-1913 AC1 while in the same function:
a zero-count bench (no characters/targets picked) now refuses to print
an estimate, matching the missing-probe-set refusal.

Modified files: `tldw_chatbook/UI/Evals/inspector.py`,
`Tests/UI/test_evals_screen.py`.
