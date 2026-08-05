---
id: task-2365
title: 'realtime_vad_threshold reader accepts TOML true as 1.0'
status: To Do
assignee: []
created_date: '2026-08-04'
labels: [realtime, config]
dependencies: []
priority: low
---

## Description (the why)

`realtime_vad_threshold()` lacks the `isinstance(raw, bool)` guard its sibling
`realtime_vad_silence_ms()` has, so `vad_threshold = true` in TOML silently coerces to a
legal-but-unintended 1.0 instead of being rejected as non-numeric (VAD-change review minor).

## Acceptance Criteria (the what)

- [ ] Boolean TOML values are rejected (log + provider default) like other non-numerics.
- [ ] A reader test pins it, mirroring the silence_ms sibling's test.
