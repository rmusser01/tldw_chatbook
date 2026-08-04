---
id: TASK-2261
title: Document pointing chatbook at OpenAI-compatible TTS servers
status: To Do
assignee: []
created_date: '2026-08-04 12:00'
labels:
  - tts
  - docs
dependencies:
  - task-2260
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The complaint that produced TASK-2260 was a discoverability failure as much as a behavior one:
a user wanting to use a local OpenAI-compatible TTS server (pocket-tts) had no documentation
telling them the OpenAI provider's Base URL setting is the way to do it. There is no Speech/TTS
page under `Docs/User_Guide/` at all (the User Guide programme G1-G5 has not reached Speech).
Write a short user-facing doc covering: choosing the OpenAI provider with a custom Base URL for
any OpenAI-compatible server (keyless local servers included, with custom model/voice names
passed through — behavior shipped in TASK-2260), and the AllTalk provider as the alternative
for AllTalk servers specifically. Include a worked pocket-tts example.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A user-guide page explains how to point TTS at an OpenAI-compatible server via the OpenAI provider's Base URL setting, including a keyless local-server example
- [ ] #2 The page states that custom model and voice names are passed through unmodified when a custom Base URL is set
- [ ] #3 The page is reachable from wherever the User Guide indexes speech/TTS topics
<!-- AC:END -->
