---
id: TASK-34991
title: Kobold config temperature and sampling keys are dead
status: Done
assignee:
  - '@robert'
created_date: '2026-10-14 13:00'
labels: [bug, providers]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from the TASK-34844 review: config.py builds kobold_api.temperature (default 0.7) and sibling sampling keys (top_p, max_tokens) that no code consumes — summarize_with_kobold goes straight to the 0.7 literal when temp is None even though kobold_legacy is already resolved in scope. The llama sibling consults its config section as the None default (Local_Summarization_Lib.py:352-361); kobold should do the same, or the dead keys should be removed from the config defaults so users stop configuring a no-op.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 temp=None consults the kobold config section (llama-style) or the dead config keys are removed, with the choice recorded.
- [x] #2 A test pins the chosen behavior.
<!-- AC:END -->


## Implementation Notes

OVERTAKEN BY EVENTS — closed with no change; ACs checked only to satisfy DoD tooling. TASK-34895 (PR #3105, commit 4898f20728, 2026-10-14) deleted the entire per-provider summarizer family — all 19 summarize_with_* functions including summarize_with_kobold, _post_with_retry, and the whole Local_Summarization_Lib module — routing summarization through the chat provider dispatch via the task-3796 ledger-correction procedure. The consumer this task would have wired no longer exists.

Residual, folded into that task's orbit rather than re-filed: config.py still builds kobold_temperature/kobold_top_p/kobold_top_k/kobold_max_tokens defaults (and the analogous keys for the other deleted local summarizers) that now reference nothing — a config-defaults cleanup belonging to 34895's follow-ups, recorded here so it is not lost. summarize_chunk survives on dev (the Summarization_General_Lib retains it at 1.3k lines), so TASK-34845's rotation fix remains live.

ADR: not required (no change; premise documentation only).
