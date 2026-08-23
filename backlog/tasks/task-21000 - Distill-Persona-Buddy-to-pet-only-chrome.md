---
id: TASK-21000
title: Distill Persona Buddy to pet-only chrome
status: Done
assignee: []
created_date: '2026-08-22 23:31'
updated_date: '2026-08-22 23:48'
labels: []
dependencies:
  - TASK-20938
references:
  - Docs/superpowers/specs/2026-08-22-persona-buddy-pet-only-chrome-design.md
  - >-
    backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove resting chrome and dead space from Persona Buddy so the floating companion is visually just the pet, while preserving direct icon controls, actionable alert text, folding, accessibility, persistence, and runtime authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At rest, the floating Buddy contains only the complete prepared pet animation box and its one-cell boundary: no title, header, state label, movement hint, default prose, decorative padding, or unused chrome is painted; an ordinary single-frame visual touches every inner edge, while undersized assets may use only the explicit 10x4-cell operable-control footprint and varying animation frames use one stable maximum box without window jitter.
- [x] #2 Fold/Open and Close are one-glyph controls inside compact native buttons overlaid in opposite top corners with exact tooltips, non-obscuring keyboard focus, transient focus-visible labels when the 10x4 pet box fits, glyph-only focus in the below-minimum fallback, click targets, and existing keyboard actions; the remaining pet surface stays draggable and the existing resize authority remains usable.
- [x] #3 Idle, thinking, speaking, listening, tool-running, wake, explicit, and authored states communicate through the pet visual without default words; only actionable approval-needed, error, and offline states temporarily replace the portrait with fixed concise text and restore the current pet when resolved.
- [x] #4 Folded mode paints a real reduced pet thumbnail with Open and Close icons under a distinct folded render authority rather than a text strip; only an effective render area constrained by the preferred geometry and viewport below the 10x4-cell thumbnail/control minimum may degrade to the two operable compact icon buttons.
- [x] #5 The accepted visual generation's maximum prepared-frame dimensions determine stable displayed bounds without cropping or stretching, while saved geometry remains the render budget; derived fit dimensions are neither persisted nor fed back into resolution, and prior-budget, prior-viewport, or stale-generation results cannot resize or repaint the current view.
- [x] #6 Navigation, remount, cancellation drain, reduced motion, animation timing, fallback, unavailable recovery, geometry persistence, viewport clamping, and targeted invalidation remain behaviorally intact.
- [x] #7 Born-RED-to-GREEN real-CSS/Pilot tests, mutations for text removal, fit authority, alert replacement, thumbnail folding, and hit regions, scoped static/governance gates, one Impeccable review, and isolated actual-terminal screenshots prove the pet-only normal, alert, folded, and constrained-viewport states.
<!-- AC:END -->

## Implementation Plan

- Follow [the approved implementation plan](../../Docs/superpowers/plans/2026-08-22-task-21000-persona-buddy-pet-only-chrome.md) using born-RED TDD and verification before every commit.
- Pin a direct-result accepted-render record and stable fit geometry without persisting or feeding derived dimensions back into resolution.
- Replace resting panel chrome with the pet, overlaid icon controls, fixed actionable alerts, and a real folded thumbnail under exact current authority.
- Rebuild and verify generated CSS, preserve app-owned lifecycle/persistence fences, and prove the four visual states through real-CSS/Pilot tests and isolated PTY captures.
- Complete the one-shot Impeccable review and close the task only after every scoped gate and human screenshot UAT is green.

## Implementation Notes

Replaced the labelled floating panel with a fitted pet-owned surface: one immutable direct-result accepted-render record controls stable animation bounds, derived fit dimensions remain display-only, and stale budget/viewport results cannot repaint or resize the view. Resting chrome and prose were removed; only `approval_needed`, `error`, and `offline` replace the pet with fixed path-free labels. Folded mode resolves a distinct frozen thumbnail, while below-minimum geometry uses the two-button fallback until a current smaller render is accepted.

Fold/Open and Close are overlaid native buttons with exact tooltips and keyboard actions. The pet surface remains draggable, and compact Close is disjoint from a reserved lower-right resize grip. Final review exposed that expanded focus words, two three-cell controls, and that grip cannot coexist in ten columns; the approved spec and AC #2 now make the existing 10x4 threshold explicit: normal/folded focus exposes action words, while the below-minimum fallback retains focused glyphs plus the incumbent bold/underline treatment, tooltips, and keyboard activation. The incident is recorded in `backlog/docs/lessons-testing-evidence.md`.

Verification on final implementation head `5058bc610` passed the seven-file real-CSS/Pilot, app-mount, resolution, architecture, live-PTY, CSS-sync, and consolidation gate: 187 passed with only inherited requests-version and `audioop` warnings. Scoped Ruff, format, compile, privacy, CSS, and diff checks passed; the one permitted Impeccable detector returned exact `[]`; final cumulative review reported no Critical, Important, or Minor findings. Born-RED/mutation work covered default text, direct fit authority, stale accepted frames, alerts, folded authority, hit regions, compact focus, and disjoint Close/resize behavior.

Final sanitized PTY evidence is under `/private/tmp/persona-buddy-task21000-final.SD0M3a` with every report check true and normal/alert/folded/constrained ANSI, SVG, and PNG artifacts. Actual full-application UAT ran in a maximized 212x63 MacBook Terminal from a copied disposable profile; the accepted screenshot is `/private/tmp/task21000-fullscreen.yHOVZF/fullscreen-terminal-window.png`, and its disposable process was stopped. Human screenshot UAT was accepted. No new ADR was required: the implementation follows the approved design and existing ADR-074 ownership, rendering, persistence, and authority boundaries.
