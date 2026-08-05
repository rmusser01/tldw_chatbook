---
id: task-2363
title: 'Realtime: usage completeness — transcription duration and audio-token split'
status: To Do
assignee: []
created_date: '2026-08-04'
labels: [realtime, cost]
dependencies: []
priority: medium
---

## Description (the why)

Input-audio transcription events carry `usage: {type: duration, seconds: N}` that never
reaches `on_usage` (T2-F12), and realtime `response.done` usage folds audio tokens into
text counts in `ProviderUsage` (final review F9 fixed the cached-token half only). Realtime
is billed per audio minute; the Console cost chip cannot be honest about it until these are
captured distinctly.

## Acceptance Criteria (the what)

- [ ] Transcription duration usage is captured onto the turn (or explicitly recorded as
      unbillable metadata) rather than dropped.
- [ ] Audio vs text token counts from realtime responses are recorded distinctly.
- [ ] Cost-chip integration for realtime sessions is either wired or filed as its own task
      with the captured fields it needs.
