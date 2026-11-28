---
id: TASK-42527
title: Verify the first-chat catalog with valid OpenRouter and Hugging Face keys
status: To Do
assignee: []
created_date: '2026-11-24 00:53'
labels:
  - first-run-wizard
  - providers
  - live-verification
dependencies: []
priority: medium
updated_date: '2026-11-28 13:00'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-34100.2 (PR #3517, ported onto dev in round 24) closed its live-verification criterion, AC#13, without two cases: a valid OpenRouter key and a valid Hugging Face key. Both need real providers (round 24 forbids contacting them; owner decision 2026-11-23, "close without valid-key cases"). Every other AC#13 case (real OpenAI, Anthropic and Gemini keys, the invalid OpenRouter and Hugging Face keys, env-key-only launches and a real llama.cpp server as a Custom OpenAI-compatible endpoint) was verified live on the branch and is recorded in TASK-34100.2's notes.

This task closes that gap: with a valid key for each provider, setup's authenticated key check accepts the key (OpenRouter's /api/v1/key, Hugging Face's whoami-v2), and the first chat on that provider gets a real reply, on a fresh isolated profile through the real app.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On a fresh isolated profile through the real app, setup with a valid OpenRouter key, pasted and exported as OPENROUTER_API_KEY, reports the key verified (not 'key not checked'), and the shipped OpenRouter default and the wizard-recommended OpenRouter model each produce a real first reply in Console ("exported as OPENROUTER_API_KEY" added 2026-11-28, owner ruling)
- [ ] #2 On a fresh isolated profile through the real app, setup with a valid Hugging Face token reports the token verified by whoami-v2, and a listed Hugging Face chat model produces a real first reply in Console
- [ ] #3 The same profiles read 'Key verified HH:MM' in Settings' Key check row after Test, and never 'Not ready' for those keys
- [ ] #4 The evidence (captures and the provider replies' first lines, never the keys) is recorded in this task's Implementation Notes, and any failure is filed as its own task
- [ ] #5 The OpenRouter legs moved here are run in the same session: a streaming and a non-streaming Console reply on the valid key (TASK-34100.5.2 AC#1), and the Model step's OpenRouter listing with a valid key (TASK-34100.7 AC#11's OpenRouter leg)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Owner ruling 2026-11-28: this task now holds every OpenRouter valid-key leg.** The owner's 2026-11-23 ruling for TASK-34100.2 AC#13 ("close without valid-key cases") was applied to the other tasks that waited on a valid OpenRouter key. The repo-root key answers 401 ("API key expired"). By that ruling these legs no longer hold their tasks open; they are carried here:
- TASK-34100.6 AC#32: a valid OpenRouter key, pasted and exported as OPENROUTER_API_KEY, in the Provider step (ticked with this exception; TASK-34100.6 is Done).
- TASK-34100.5.2 AC#1: a streaming and a non-streaming Console reply on a valid OpenRouter key.
- TASK-34100.7 AC#11: the OpenRouter leg of its live run.
- This task's own AC#1 to AC#4.

AC#1 now also asks for the exported case. Run them all in one live session when a valid key arrives.
<!-- SECTION:NOTES:END -->
