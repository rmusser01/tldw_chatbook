---
id: TASK-37423
title: >-
  Google: the shipped default model gemini-2.5-flash answers 404 'no longer
  available to new users'
status: To Do
assignee: []
created_date: '2026-10-29 12:22'
labels:
  - google
  - config
  - providers
  - p2
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The shipped [api_settings.google] model and the first entry of the Google model list are gemini-2.5-flash. Live on 2026-10-29 (during TASK-33620.10's verification), a Console send with a current Gemini API key failed with: 'Provider error from Google Gemini: model or endpoint not found. Status: 404. Google Gemini says: This model models/gemini-2.5-flash is no longer available to new users. Please update your code to use models/gemini-3.8-flash ...'. The model listing for the same key still names gemini-2.5-flash, so a new user who picks Google gets a default that cannot answer and a model picker that offers it. The suggested replacement, gemini-3.8-flash, sent no bytes within 100 s to a plain curl request (streamed or not) in the same session, and gemini-3.1-flash-lite and gemini-2.5-flash-lite answered 404 with an empty body, so which default to ship needs a fresh live check. TASK-34100.5 saw the same 404 for gemini-2.0-flash.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A new profile that chooses Google gets a default model that answers a live Console send with the shipped template
- [ ] #2 The Google model list no longer offers models the API refuses for new keys, or marks them
- [ ] #3 A test pins the shipped Google default model to the chosen value
<!-- AC:END -->
