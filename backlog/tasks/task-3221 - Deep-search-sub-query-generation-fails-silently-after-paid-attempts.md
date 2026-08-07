---
id: TASK-3221
title: Deep-search sub-query generation fails silently after paid attempts
status: To Do
assignee: []
created_date: '2026-08-07 16:30'
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
- [ ] #1 Exhausting all sub-query generation attempts appends a warning to web_search_results_dict["warnings"] stating generation failed after N attempts
- [ ] #2 The web_deep_search footer surfaces that warning like any other provider warning
- [ ] #3 A test drives all attempts to failure and asserts the warning text and footer passthrough
<!-- AC:END -->
