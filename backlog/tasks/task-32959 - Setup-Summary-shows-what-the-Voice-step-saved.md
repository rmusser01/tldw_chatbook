---
id: TASK-32959
title: Setup Summary shows what the Voice step saved
status: To Do
assignee: []
created_date: '2026-09-26 01:30'
updated_date: '2026-09-26 08:00'
labels:
  - wizard
  - tts
dependencies:
  - TASK-32958
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The first-run setup Summary reads back a ✓/✗/– line per area from saved config, but has no Voice line for any service (PocketTTS, OpenAI, Custom or OmniVoice), so a user gets no confirmation of what the Voice step saved or whether it became the default. Found during TASK-32958 live UAT; pre-existing on dev. (The other two follow-ups first filed here — a broken configured OmniVoice path offering a useless download, and a stale sample overwriting a newer status — were fixed in TASK-32958's PR after Qodo review.)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The setup Summary shows a Voice line naming the saved service and whether it is the default TTS provider, read back from saved config
- [ ] #2 A skipped Voice step shows as not set up (optional), like the other optional areas
<!-- AC:END -->
