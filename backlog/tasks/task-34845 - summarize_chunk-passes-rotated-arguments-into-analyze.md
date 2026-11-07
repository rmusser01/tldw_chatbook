---
id: TASK-34845
title: summarize_chunk passes rotated arguments into analyze (FIXME)
status: Done
assignee:
  - '@robert'
created_date: '2026-10-12 12:10'
labels: [bug, providers]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-34812 review (same species): summarize_chunk (Summarization_General_Lib.py ~2595, FIXME-marked, currently only called by tests) rotates the first three arguments into analyze() — text lands on api_name — and swaps temp/system_message relative to analyze's (api_name, input_data, custom_prompt_arg, api_key, system_message, temp, ...) order. Verify every live call path (or confirm it is dead), fix the rotation, and remove the FIXME.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 summarize_chunk's call into analyze passes every argument in the correct position, verified against analyze's signature.
- [x] #2 Callers audited; a test covers the corrected path (or the function is deleted with the decision recorded if truly dead).
<!-- AC:END -->


## Implementation Notes

Fixed (not deleted): the call now passes (api_name, text, custom_prompt_input) positionally with keyword api_key/system_message/temp, matching analyze's signature exactly; the FIXME removed. Caller audit confirmed the earlier review's finding: zero production callers (analyze's own recursive/chunked paths dispatch through _dispatch_to_api directly with correct argument order) — summarize_chunk is exercised only by the diagnostic-privacy suites, which bind by signature and pass unchanged (4/4). New pin asserts the exact corrected (args, kwargs). Routing + analyze suites: 14/14.

ADR: not required (argument-order bug fix at one call site).

Review round 1 (yes-verdict) follow-ups recorded, not filed: (1) summarize_chunk never passes streaming, so its generator branch is production-dead (reachable only via the monkeypatched privacy test) — wire a streaming parameter or delete the branch if this helper ever gains a production caller; (2) the function lacks a docstring. Also noted: analyze is api_name-first while _dispatch_to_api is text-first — keyword arguments at every call into either (the trap that produced this bug and the TASK-34812 siblings).
