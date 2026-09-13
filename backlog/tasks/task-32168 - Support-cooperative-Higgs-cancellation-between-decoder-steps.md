---
id: TASK-32168
title: Support cooperative Higgs cancellation between decoder steps
status: To Do
assignee: []
created_date: '2026-09-09 08:19'
labels: []
dependencies:
  - TASK-32151
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Higgs CPU cancellation retains and joins generation safely, but the real qualified request completed 512 decoder forwards before Stop returned after 96.4 seconds. The pinned serve-engine API does not expose the model-supported per-step stopping criterion. Extend an explicit reviewed provider boundary so cancelled replies stop decoding promptly while retained cleanup and successor playback remain correct. Exact API and empty-audio unwind constraints are documented in Docs/QA/tts-macos-burndown-2026-09-09/providers/higgs/STOPPING-CRITERIA-CONTRACT.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An ADR selects a supported per-request serve-engine or model API without global callback mutation or an implicit dependency patch.
- [ ] #2 Stop during observed decoding prevents subsequent decoder steps after the permitted in-flight step and emits no cancelled-request audio.
- [ ] #3 Early cancellation including cancellation before any audio token safely unwinds tokenizer and engine resources and a successor produces complete speech.
- [ ] #4 Real CPU cancellation timing and exact package and model provenance are recorded with targeted regression and playback evidence.
<!-- AC:END -->
