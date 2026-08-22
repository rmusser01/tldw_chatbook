---
id: TASK-21000
title: Distill Persona Buddy to pet-only chrome
status: To Do
assignee: []
created_date: '2026-08-22 23:31'
labels: []
dependencies:
  - TASK-20938
references:
  - Docs/superpowers/specs/2026-08-22-persona-buddy-pet-only-chrome-design.md
  - backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove resting chrome and dead space from Persona Buddy so the floating companion is visually just the pet, while preserving direct icon controls, actionable alert text, folding, accessibility, persistence, and runtime authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 At rest, the floating Buddy contains only the complete prepared pet animation box and its one-cell boundary: no title, header, state label, movement hint, default prose, decorative padding, or unused chrome is painted; an ordinary single-frame visual touches every inner edge, while undersized assets may use only the explicit 10x4-cell operable-control footprint and varying animation frames use one stable maximum box without window jitter.
- [ ] #2 Fold/Open and Close are one-glyph controls inside compact native buttons overlaid in opposite top corners with exact tooltips, non-obscuring keyboard focus, transient focus-visible labels, click targets, and existing keyboard actions; the remaining pet surface stays draggable and the existing resize authority remains usable.
- [ ] #3 Idle, thinking, speaking, listening, tool-running, wake, explicit, and authored states communicate through the pet visual without default words; only actionable approval-needed, error, and offline states temporarily replace the portrait with fixed concise text and restore the current pet when resolved.
- [ ] #4 Folded mode paints a real reduced pet thumbnail with Open and Close icons under a distinct folded render authority rather than a text strip; only an effective render area constrained by the preferred geometry and viewport below the 10x4-cell thumbnail/control minimum may degrade to the two operable compact icon buttons.
- [ ] #5 The accepted visual generation's maximum prepared-frame dimensions determine stable displayed bounds without cropping or stretching, while saved geometry remains the render budget; derived fit dimensions are neither persisted nor fed back into resolution, and prior-budget, prior-viewport, or stale-generation results cannot resize or repaint the current view.
- [ ] #6 Navigation, remount, cancellation drain, reduced motion, animation timing, fallback, unavailable recovery, geometry persistence, viewport clamping, and targeted invalidation remain behaviorally intact.
- [ ] #7 Born-RED-to-GREEN real-CSS/Pilot tests, mutations for text removal, fit authority, alert replacement, thumbnail folding, and hit regions, scoped static/governance gates, one Impeccable review, and isolated actual-terminal screenshots prove the pet-only normal, alert, folded, and constrained-viewport states.
<!-- AC:END -->
