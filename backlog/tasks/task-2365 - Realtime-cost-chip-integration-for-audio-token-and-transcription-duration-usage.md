---
id: TASK-2365
title: >-
  Realtime: cost-chip integration for audio-token and transcription-duration
  usage
status: To Do
assignee: []
created_date: '2026-08-05 04:16'
labels:
  - realtime
  - cost
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-2363 captured realtime's audio/text token split (ProviderUsage.audio_input/audio_output) and input-audio transcription duration (ProviderUsage.transcription_seconds) onto Console turns, but deliberately left them unbilled: pricing_catalog.py's cost math only reads the plain uncached/cache/output token buckets, and realtime is billed per audio MINUTE, not per token, which the current per-mtok pricing model cannot represent as-is. This task is the follow-up that makes the Console cost chip honest about a realtime session's actual cost.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Realtime sessions' cost estimate/display accounts for audio-minute billing using ProviderUsage.audio_input/audio_output and/or transcription_seconds, not just the token buckets pricing_catalog.py already reads
- [ ] #2 Pricing catalog entries exist (or a documented decision to omit them) for the realtime model(s) this app supports
- [ ] #3 Existing token-based cost math for non-realtime providers is unaffected
<!-- AC:END -->
