---
id: TASK-34980
title: Fix pre-existing positional-argument mismatches in summarizer dispatch (kobold, tabbyapi, ollama)
status: In Progress
assignee:
  - '@robert'
created_date: '2026-10-12 10:05'
labels: [bug, providers]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-33245 dispatch-table review (verified mechanically against the pre-refactor elif chain, which carried the same mismatches): three local summarizer routes pass arguments positionally onto signatures whose parameter order does not match, so values land on the wrong parameters. The TASK-33245 refactor preserved them bug-for-bug deliberately; the new `_SUMMARIZER_DISPATCH` table is now the single, testable place to correct them.

- **kobold**: dispatch passes `(text, api_key, cp, temp, sm, st)` onto `summarize_with_kobold(input_data, api_key, custom_prompt_input, system_message=None, temp=None, kobold_api_ip=..., streaming=False)` — so `system_message=temp`, `temp=system_message`, the streaming bool lands on `kobold_api_ip`, and `streaming` stays False.
- **tabbyapi**: passes `(text, cp, temp, sm, st)` onto `summarize_with_tabbyapi(input_data, custom_prompt_input, system_message=None, api_key=None, temp=None, ...)` — so `system_message=temp`, `api_key=system_message`, `temp=streaming`.
- **ollama**: the 7-positional call's last value binds `model=streaming`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All three routes pass every argument by its correct parameter (keyword or correctly-ordered positional), verified against each function's signature.
- [x] #2 Regression tests pin the corrected routing for all three providers through `_dispatch_to_api`/`analyze`.
- [x] #3 Targeted local-summarizer suites pass (kobold/tabby config tests included).
<!-- AC:END -->


## Implementation Plan

Discovered by the PR-3092 review (mechanically verified against the pre-refactor elif chain); fixed at the single site — the _SUMMARIZER_DISPATCH adapters (stacked on PR #3092's branch, which carries the table).

1. Verify each signature against Local_Summarization_Lib (done: kobold system_message-before-temp; tabbyapi system_message-before-api_key; ollama api_url/model ordering).
2. Correct the three adapters: kobold and tabbyapi positional-through-temp + streaming keyword; ollama fully keyword-driven with api_url/model left to their defaults.
3. Pin routing with a parametrized regression test asserting exact (args, kwargs) per provider through _dispatch_to_api.
4. Run the analyze suite plus the kobold/tabby config suites.

ADR required: no
ADR path: N/A
Reason: bug fix at an existing dispatch site; no architectural decision.


## Implementation Notes

PR #3094 (stacked on #3092's table). All three adapters corrected and pinned by exact-(args, kwargs) regression tests through the real _dispatch_to_api entry. Review-verified downstream behavior: ollama now resolves model from config instead of receiving the streaming bool; kobold's streaming path is now actually reachable (its temp/system_message swap was behaviorally inert — the function never reads them; filed as follow-up); tabbyapi's api_key/system_message/temp now land where the payload reads them. No in-repo analyze() caller streams, so the newly-live streaming routes have no existing consumers. Two follow-ups filed from the review: kobold's dead parameters, and summarize_chunk's same-species argument rotation into analyze() (FIXME-marked, test-only caller).

ADR: not required (bug fix at an existing site; no architectural decision).

## Renumbering provenance

Renumbered from task-34812 to TASK-34980 (2026-10-13): this ID collided
with a concurrently filed task after the branches merged; per the
2026-08-21 owner rule (TASK-19601) the older arrival kept the id.
