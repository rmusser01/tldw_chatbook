---
id: TASK-22301
title: Bound Library rail width across modes
status: Done
assignee:
  - '@codex'
created_date: '2026-08-26 03:31'
updated_date: '2026-08-26 22:50'
labels:
  - library
  - ux
  - layout
dependencies: []
references:
  - Docs/superpowers/specs/2026-08-25-library-rail-bounded-width-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the persistent Library navigation rail visually stable across every Library mode by retaining fractional sizing while bounding it around the approved Collections reference width.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With custom widths disabled, every Library rail displayed alongside content uses the shared 3:13 default projection and renders within an exact 24–34-cell range; rail-only, hidden/collapsed, and compressed-priority states remain explicit exceptions.
- [x] #2 Switching among Media, Chats, Notes, Prompts, Skills, Collections, Search / RAG, Import, Export, Study handoffs, and the landing canvas at the same settled width keeps a co-present rail edge stable within one compositor cell.
- [x] #3 The existing explicit custom-width preference remains valid from 24 through 48 cells and applies to co-present rails across ordinary and adaptive Library destinations; ordinary layout transiently compresses it to preserve a 40-cell canvas, and the unchanged saved value restores after compression, responsive collapse, or a rail-only compact takeover.
- [x] #4 Adaptive auto-collapse, five-cell grips, focus recovery, automatic/explicit priority, hysteresis, ordinary manual collapse, and existing Notes-specific takeovers retain their behavior; below 64 columns every ordinary route uses a reversible rail-only/canvas-only emergency stage with one pinned guarded `‹ Library`/`< Library` action, truthful state-derived Escape/footer/F1 affordance, and no blank reserved space.
- [x] #5 At supported widths, the rail, canvas, adaptive panes, and footer remain contained without intersection; extreme-width escape behavior may hide panes, compress an explicitly prioritized adaptive pane below its protected minimum, or invoke the approved ordinary `<64` emergency stage rather than overflow.
- [x] #6 Production-styled tests cover the exact projection and sub-24 escape oracles, wide/compact box-model inputs, initial/settled mount, every enumerated route and state-aware emergency return, Notes Navigator/work/explicit-priority adaptive branches, scoped recompose, equality-guarded live resize, custom compression/restoration boundaries, and specified 235-, 170-, 120-, 100-, 80-, and 60-column geometry states without resize-time data/config work.
- [x] #7 Library Settings copy, defaults, ADR-086, the adaptive-reader design, and user documentation describe bounded fractional defaults separately from explicit 24–48-cell custom widths.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a pure shared 3:13 bounded rail-width policy and exact custom-width resolver.
2. Make adaptive readers consume one projected or custom requested rail width in every branch.
3. Apply reversible equality-guarded ordinary rail contracts from one normalized settings snapshot.
4. Add the route-general below-64 single-stage layout and one guarded pinned Library return action.
5. Lock production box-model geometry, accessibility, and resize no-work behavior with mounted tests.
6. Align fresh/reset defaults, Settings copy, and user documentation without migrating stored widths.
7. Run targeted static/test verification and approved real-PTY UAT, then close the task.

ADR required: yes
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md
Reason: ADR-086 owns the long-lived cross-route Library responsive shell contract and has been amended for this change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented one shared bounded-fractional Library rail policy: automatic sizing
uses 3:13 within 24–34 cells, explicit custom sizing remains 24–48, and ordinary
routes temporarily compress only to preserve a 40-cell canvas. Ordinary and
adaptive shells consume the same normalized preference snapshot without
resize-time configuration, persistence, data-load, worker, or polling work.

Added the reversible `<64` ordinary emergency stage with a retained guarded
`‹ Library`/`< Library` action, state-derived Escape/footer/F1 truth, real
focus/scroll restoration, and app-global F6 keyboard reachability. Adaptive
Media, Chats, and Notes retain their five-cell grips, priority, collapse, focus,
and hysteresis behavior. Transient Textual allocation frames are rejected until
the production box model settles, without self-rescheduling.

Updated fresh/reset defaults to 31 while preserving stored legacy/shared values
without migration or load-time writes. Settings and user guides now distinguish
automatic 24–34 sizing, explicit 24–48 sizing, temporary compression, adaptive
collapse, and emergency behavior. ADR-086 and the approved adaptive-reader
design were updated; no new ADR was required beyond that amendment.

Verification includes the final static gate, 257 changed/new tests, focused
policy/reader/settings suites, production geometry matrices, and detached tmux
PTY UAT at 235/170/120/100/80/60 columns. PTY testing also covered custom 35 and
48 boundaries, ordinary/adaptive route switches, 60↔64 restoration, keyboard
Enter return, and ASCII substitution through production Settings. The optional
full repository sweep was not run because no user opt-in was received. Exact
commands, baseline classifications, pre-fix failures, post-fix captures, and
cleanup evidence are recorded in
`Docs/superpowers/qa/library-rail-bounded-width-2026-08/`.

The live run exposed three issues not covered by the initial mounted matrix:
untouched 100/80-column landing takeover, keyboard-inaccessible emergency
return, and stale ASCII Settings truth. All were fixed and retained as before/
after evidence. This incident is recorded in
`backlog/docs/lessons-live-verification.md`.
<!-- SECTION:NOTES:END -->
