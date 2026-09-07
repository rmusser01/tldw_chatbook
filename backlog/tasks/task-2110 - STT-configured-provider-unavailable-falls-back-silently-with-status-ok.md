---
id: TASK-2110
title: 'STT: configured provider unavailable falls back silently with status=ok'
status: To Do
assignee: []
created_date: '2026-08-03'
labels: [speech, diagnostics]
dependencies: []
priority: high
---

## Description (the why)

During the hands-free live gate (2026-08-03), a worktree venv without parakeet_mlx made
dictation silently substitute faster-whisper base for the user's configured
`[transcription] default_provider = "parakeet-mlx"`. The startup diagnostic reported
`event=speech_stack_available model=base provider=faster-whisper status=ok` — status OK
while running a provider the user explicitly did not choose. The user's standard is
parakeet, whisper explicit-only; a silent substitution is indistinguishable from working
config until transcription quality collapses. The degraded-VAD path already has an honesty
surface; provider fallback needs the same.

## Acceptance Criteria (the what)

- [ ] When the configured STT provider cannot be used and another is substituted, the
      startup diagnostic reports a degraded/fallback status naming BOTH the configured and
      the substituted provider (not `status=ok`).
- [ ] The first capture after such a substitution surfaces a user-visible notice naming the
      configured provider that was unavailable and what is being used instead.
- [ ] A configured provider that is available produces no new notice (no noise in the
      healthy path).
