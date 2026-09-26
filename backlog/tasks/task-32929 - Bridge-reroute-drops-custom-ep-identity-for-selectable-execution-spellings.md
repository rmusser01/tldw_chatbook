---
id: TASK-32929
title: Bridge reroute drops custom-ep identity for selectable execution spellings
status: To Do
assignee:
  - '@Robert'
created_date: '2026-09-26 04:54'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fifth spelling seam (found during PR #2828 Qodo follow-ups, reported not fixed): custom-ep parents whose execution spelling IS selectable (engine-off custom-openai-api, llama_cpp/ollama families) reroute to the bare built-in provider, dropping the registry base_url. Fix requires a deliberate decision between PR-2651's built-in-family protection and threading identity onto the run-turn call.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Decision recorded between PR-2651 protection and identity threading
- [ ] #2 Custom-ep sends under engine-off and llama/ollama families reach their registry base_url
<!-- AC:END -->
