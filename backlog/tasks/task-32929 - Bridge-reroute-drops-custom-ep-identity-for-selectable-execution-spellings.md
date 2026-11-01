---
id: TASK-32929
title: Bridge reroute drops custom-ep identity for selectable execution spellings
status: Done
assignee:
  - '@codex'
created_date: '2026-09-26 04:54'
updated_date: '2026-09-29 19:35'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fifth spelling seam (found during PR #2828 Qodo follow-ups, reported not fixed): custom-ep parents whose execution spelling IS selectable (engine-off custom-openai-api, llama_cpp/ollama families) reroute to the bare built-in provider, dropping the registry base_url. Fix requires a deliberate decision between PR-2651's built-in-family protection and threading identity onto the run-turn call.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Decision recorded between PR-2651 protection and identity threading
- [x] #2 Custom-ep sends under engine-off and llama/ollama families reach their registry base_url
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/147-agent-provider-routing.md
Reason: direct repair and clarification of existing selected-identity versus execution-key routing.
1. Reproduce primary sends using selectable execution keys and preserve explicit built-in child routing.
2. Keep raw identity in AgentConfig.provider for model calls, leaving service capability/protocol preparation on api_endpoint execution keys.
3. Verify real bridge dispatch and adjacent routing/fallback regressions and static checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Preserved raw selected provider identity in AgentConfig.provider and used that identity for the eventual model call while keeping execution-family capability preparation separate. ADR-147 now records this clarification. Engine-off custom OpenAI, llama.cpp and Ollama identity regressions failed at the real gateway before the repair; the final primary/child/custom-hosted selection passed 9 tests. Existing bridge tests now retain their collection-time private bootstrap profile rather than changing the bound config source mid-test. Evidence: /private/tmp/agent-burndown-endpoint-red2.log and /private/tmp/agent-burndown-endpoint-green3.log. Independent combined review pending; no full-suite or live-provider result claimed.

Final disposition 2026-09-29: The primary Console resolution URL is also captured in AgentConfig.base_url for fallback index-zero freezing. A real bridge-to-service spy reproduced three missing snapshots, then all nine endpoint/child/custom-hosted checks passed in 18.45s (/private/tmp/agent-burndown-primary-url-red.log and -green.log). Independent fallback/target review approved this coordinated handoff. All acceptance criteria are checked; scoped tests, changed-code static checks, documentation and independent review are complete. Task is Done. Inherited source formatting debt is preserved; no full-suite/live-provider result is claimed. This disposition supersedes earlier pending-review notes.
<!-- SECTION:NOTES:END -->
