---
id: TASK-37762
title: >-
  Console: a named OpenAI-compatible custom endpoint's traced request never
  aligns, so its stream retry is off
status: To Do
assignee: []
created_date: '2026-10-29 17:45'
labels:
  - console
  - trace
  - streaming
  - local-models
  - p2
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A named custom endpoint of the OpenAI-compatible family (`custom-ep:<slug>`, executed by the hosted engine under the `custom-hosted` key) records every provider call's traced request surface as one `provider_overlay` omission with reason `alignment_mismatch`, instead of the saved user revision that Custom, vLLM and llama.cpp calls record. Measured 2026-10-29 with trace capture on (the default), reading `console_trace_surface_nodes` in an isolated profile: the same on origin/dev 38c9d2edcf (a plain send) and on TASK-33620.10's branch. So the trace never holds what was actually sent to such an endpoint.

It also blocks any second traced call that has to compare against that surface. TASK-37421's one-time non-streaming retry did: live on the branch, a custom-ep endpoint whose server refused `"stream": true` ended "Run: Blocked — capture failed", because the retry's trace reservation raised `ValueError("surface_replacement_checkpoint_unavailable")` in `prepare_current_surface_delta` (the incoming saved revision cannot replace the omission without a checkpoint for the `llama_fallback` route). TASK-37421 therefore leaves the `custom-hosted` key out of the retry (`refused_stream.retries_refused_stream`); such an endpoint's refused stream fails once with the Streaming advice, and an empty or unreadable stream fails as "unexpected provider error".

Evidence: the live captures, front logs and DB reads were kept with that session's working files, not in the repo.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A custom-ep (OpenAI-compatible family) call traced with capture on records the saved user revision in its surface, as a Custom slot call does, or the reason it cannot is documented and the retry's reservation tolerates it
- [ ] #2 With that in place, the custom-hosted key is retried once without streaming like the Custom slots, and a refused or empty custom-ep stream completes with capture on and its retry traced on the llama_fallback route
- [ ] #3 A test with the real trace boundary factory drives a custom-ep call through the hosted engine and asserts the recorded surface; it fails before the change
- [ ] #4 The User Guide's chat-basics Streaming section and the CHANGELOG no longer carve named custom endpoints out of the retry
<!-- AC:END -->
