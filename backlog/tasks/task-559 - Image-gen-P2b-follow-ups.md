---
id: TASK-559
title: >-
  Image-gen P2b follow-ups
status: To Do
assignee: []
created_date: '2026-07-24 13:30'
updated_date: '2026-07-24 13:30'
labels:
  - image-generation
  - console
  - followup
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred/polish items from the image-gen P2b slice (PR #850: speak 🔊, @style presets + picker, generate-from-conversation; spec `Docs/superpowers/specs/2026-07-24-image-gen-p2b-tts-style-context-design.md` §4). Post-merge live smoke 2026-07-24 verified all wiring end-to-end in the real app (style refusals, picker draft composition, draft restore, speak's graceful no-TTS toast, context-path dispatch) — these are enhancements, not defects. Distinct from [[task-497]] (P1 polish), [[task-498]] (egress adoption), and [[task-558]] (P2a polish).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] Richer conversation-context extraction for `/generate-image` with no prompt: the current `extract_context_from_messages` is keyword-shallow (mood via keyword match; `mentioned_characters`/`mentioned_settings` never populated). Design and implement a better context builder (e.g. LLM-composed prompt from the last N turns), keeping the composed-prompt-visible-in-card behavior.
- [ ] Console TTS playback controls: speak is fire-and-forget today; add stop (and optionally pause/save) for Console-originated speech, reusing the legacy widgets' `TTSPlaybackEvent` actions.
- [ ] Style picker offers template previews (base-prompt/negative snippet) in the row or a detail pane, not just name — category — id.
- [ ] Per-style user-defined templates (beyond the 13 built-ins) loadable from config or a templates dir.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->
