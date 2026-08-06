---
id: TASK-2390
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

## Research (verified 2026-08-06, developers.openai.com/api/docs/pricing)

**This task's Description premise is WRONG and should not be built on.** Realtime audio is
billed **per 1M tokens**, not per audio minute — so `ModelPricing`'s existing per-mtok shape
extends additively (new optional audio fields, exactly how `cache_read_per_mtok` expresses
"no published rate") rather than needing a redesign. Only *transcription* is per-minute.

Rates read from the official pricing page, units "per 1M tokens unless noted":

| Model | Text in | Cached text in | Text out | Audio in | Cached audio in | Audio out |
|---|---|---|---|---|---|---|
| gpt-realtime (the app's default) | $4.00 | $0.40 | $16.00 | $32.00 | $0.40 | $64.00 |
| gpt-realtime-mini | $0.60 | $0.06 | $2.40 | $10.00 | $0.30 | $20.00 |
| gpt-realtime-2.1 | $4.00 | $0.40 | $24.00 | $32.00 | $0.40 | $64.00 |
| gpt-realtime-2 | $4.00 | $0.40 | $24.00 | $32.00 | $0.40 | $64.00 |
| gpt-realtime-1.5 | $4.00 | $0.40 | $16.00 | $32.00 | $0.40 | $64.00 |

Whisper transcription: **$0.006 per minute** (the unit `ProviderUsage.transcription_seconds`
feeds).

Design notes for whoever implements this:
- Cached AUDIO input is a *separate rate* from cached text input — they coincide at $0.40 for
  `gpt-realtime` but diverge for `-mini` ($0.30 vs $0.06), so one shared cache field would be
  wrong. Check first whether `ProviderUsage` can even attribute cache reads to audio vs text
  (`input_token_details`), and if it cannot, say so and price conservatively rather than
  guessing.
- Confirm whether `ProviderUsage.audio_input` is inclusive of cached audio tokens before
  summing, or the bill will double-count.
- Re-verify these rates before committing them: the catalog carries `as_of` precisely because
  published rates go stale.
