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

CLOSED with no production change; ACs checked only to satisfy DoD tooling. **Review round 1 corrected the record — the first draft trusted commit 4898f20728's message instead of its diff.** Verified state of origin/dev (2026-10-19):

- TASK-34895's tranche 4 (PR #3105, commit 4898f20728) deleted the NINE hosted summarize_with_* functions, _post_with_retry, and summarize_chunk from Summarization_General_Lib (3.3k→1.3k lines), with the task-3796 ledger re-freeze. Its message claims more than it did.
- **Local_Summarization_Lib.py SURVIVES on dev, orphaned** (2,321 lines; nothing in tldw_chatbook/ imports it; summarize_with_kobold still defined at :544). The ledger records its sites as outcome=deleted while the module remains on disk — 34895's deletion was recorded but not performed (filed as TASK-35553).
- The close verdict stands on the LIVE path: summarization routes through chat dispatch; nothing live consumes kobold_temperature/top_p/top_k/max_tokens (the kobold summarizer spec lacks config_temp=True, and the orphaned module is unreachable). The analogous llama/ooba/vllm/ollama temperature keys ARE live via config_temp=True specs — the first draft's "all reference nothing" generalization was wrong.
- **summarize_chunk was deleted by 4898f20728 itself**, so TASK-34845's rotation fix is not live; its routing test dangled red on dev (calling the deleted function) and is removed in this PR.

ADR: not required (no change; premise documentation only).
