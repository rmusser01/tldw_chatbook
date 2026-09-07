---
id: TASK-3221
title: Deep-search sub-query generation fails silently after paid attempts
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 16:30'
updated_date: '2026-08-07 20:26'
labels:
  - web-tools
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When search_enable_subquery is on, generate_and_search makes up to 3 LLM attempts to generate sub-queries; if all fail it proceeds with just the original query and no signal — indistinguishable from the feature being off, despite 3 paid calls. In a tool whose whole contract is cost transparency (footer states sub-query count, description states spend shape), three billed attempts with zero user-visible trace is a gap. Deferred as a minor in Task 5's review; the final whole-branch review (2026-08-07) promoted it to a follow-up: a warnings entry closes it cheaply and would surface in the tool footer's existing warnings path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exhausting all sub-query generation attempts appends a warning to web_search_results_dict["warnings"] stating generation failed after N attempts
- [x] #2 The web_deep_search footer surfaces that warning like any other provider warning
- [x] #3 A test drives all attempts to failure and asserts the warning text and footer passthrough
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In generate_and_search, append a warning to web_search_results_dict['warnings'] when all sub-query generation attempts fail\n2. Test warning text + web_deep_search footer passthrough (existing warnings path)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: added a module constant _SUBQUERY_GENERATION_MAX_ATTEMPTS = 3
(shared by analyze_question's `for attempt in range(...)` loop and the new
warning text) so the "N attempts" figure can never drift from the loop
bound that actually produced it. generate_and_search now checks, right
after calling analyze_question (only inside the subquery_generation=True
branch), whether sub_questions came back empty -- the only way that
happens given analyze_question's own loop (it only ever breaks early on a
non-empty parse) is total exhaustion of all paid attempts. When so, a
knowable-only warning ("sub-query generation failed after 3 attempts;
searched only the original query") is appended to
web_search_results_dict["warnings"] once that dict exists (the check
itself runs before the dict is initialized, so the verdict is stashed in a
local and appended right after `warnings = []`). No warning is added when
subquery_generation is off (analyze_question is never called) or when
sub-queries were dropped later by the question-dedup step (that's a
distinct, already-warning-free path).

The web_deep_search footer already surfaces `warnings` as a bare COUNT
("· N search warning(s)") -- not per-message text -- for every provider
warning already; this new warning rides that exact same, pre-existing
path, so no footer code changed.

Tests:
- Tests/Web_Scraping/test_deep_search_pipeline.py::
  test_generate_and_search_warns_when_subquery_generation_exhausts_attempts
  drives the REAL analyze_question/generate_and_search path (only
  chat_api_call/perform_websearch faked) to a total failure and asserts
  the exact warning text lands in web_search_results_dict['warnings'] and
  that all 3 paid attempts were actually made.
- ...test_generate_and_search_no_warning_when_subquery_generation_disabled:
  sanity check the warning never appears when the feature is off.
- Tests/Tools/test_web_deep_search.py::
  test_deep_search_footer_surfaces_subquery_generation_failure_warning:
  phase-boundary-fake passthrough test (same style as the existing
  test_deep_search_footer_warning_note) proving the warning's count
  reaches the tool's footer output.

Files: tldw_chatbook/Web_Scraping/WebSearch_APIs.py,
Tests/Web_Scraping/test_deep_search_pipeline.py,
Tests/Tools/test_web_deep_search.py.
<!-- SECTION:NOTES:END -->
