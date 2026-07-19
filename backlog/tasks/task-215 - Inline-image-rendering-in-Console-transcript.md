---
id: TASK-215
title: Inline image rendering in Console transcript
status: Done
assignee: ['@claude']
created_date: '2026-07-13 09:30'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 1 of Console attachments (PR #621) ships a placeholder chip (🖼 label) for image messages. This fast-follow renders images inline in the transcript: pixel mode (rich-pixels) and terminal-graphics mode (textual-image), with a Toggle View action, porting the proven rendering from ChatMessageEnhanced (_render_pixelated/_render_regular/_render_fallback + [chat.images] render-mode config and terminal_overrides). Must respect the transcript's row-signature reconcile loop (0.2s streaming timer) — image rows must not re-render every tick.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Image messages render inline (pixels or TGP per config/terminal) with the chip as fallback
- [x] #2 Toggle View switches modes per message without remounting the transcript
- [x] #3 Streaming reconcile does not rebuild image rows on unrelated ticks (row-signature stable)
- [x] #4 Unmounted-row rendering stays safe (no NoActiveAppError; Content-based fallback path)
<!-- AC:END -->

## Implementation Plan

1. Pure module (Chat/console_image_view.py): modes, config+terminal resolution, view state, bounded render cache. 2. Transcript image row kind fed by prebuilt spec maps. 3. Toggle View message action. 4. Screen wiring: off-loop prep worker, spec-map build, screen-state serialization, session eviction. 5. Full verification + live QA + user gate. (Docs/superpowers/plans/2026-07-13-console-inline-image-rendering.md)

## Implementation Notes

Inline rendering ships behind the existing [chat.images] keys — this feature is their first real consumer (legacy defined but never read them; a QA-found config-shape bug meant even our first wiring missed the live COMPREHENSIVE_CONFIG_RAW nesting — fixed and live-proven). Architecture: pure module owns modes/state/cache (decode capped 1024px, LRU 16, negative-cache, off-loop prep in dedicated worker group); transcript renders keyed image rows only for prebuilt specs so streaming reconcile never rebuilds them; View action cycles pixels/graphics/hidden per message; overrides ride the screen-state allowlist (strings only). Inline rendering is bounded to the most recent 16 image messages per session (cache capacity, mirrors provider payload policy) — older stay chip-only; this was the final-review churn fix. Known trade-offs: default mode resolves once per screen life; action row sits below tall images; resumed chips show mime-size labels. Verification: 1046-test affected gate green, legacy image suites untouched, 7 live captures incl. config-drives-mode proof (Docs/superpowers/qa/console-inline-images-2026-07/). Key files: Chat/console_image_view.py, Widgets/Console/console_transcript.py, Chat/console_message_actions.py, UI/Screens/chat_screen.py, Utils/terminal_utils.py (additive terminal_type).
