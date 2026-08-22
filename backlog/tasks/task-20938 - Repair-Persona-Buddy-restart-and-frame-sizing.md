---
id: TASK-20938
title: Repair Persona Buddy restart and frame sizing
status: Done
assignee: []
created_date: '2026-08-22 21:52'
updated_date: '2026-08-22 22:54'
labels: []
dependencies:
  - TASK-19055
references:
  - Docs/superpowers/specs/2026-08-22-persona-buddy-uat-repairs-design.md
  - Docs/superpowers/plans/2026-08-22-task-20938-persona-buddy-uat-repairs.md
  - >-
    backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the restart, portrait-cropping, and setup-recovery defects found during full-application Persona Buddy UAT so an explicitly configured Buddy restores faithfully, paints its complete resolved frame, and can reach trusted live states.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The effective `[persona_buddy]` table reaches the app-owned controller at startup, restoring the exact enabled state, local Persona selection, open/collapsed state, and saved geometry without a startup write.
- [x] #2 Missing or malformed Buddy fields retain the existing strict independent safe defaults, and no unrelated configuration table or secret is added to the projected settings surface.
- [x] #3 The first Workbench Buddy action after restart cannot replace valid saved geometry with the never-positioned sentinel merely because startup omitted persisted preferences.
- [x] #4 Full-size Buddy resolution uses the visible `#persona-buddy-frame` content dimensions rather than the containing window dimensions; the complete prepared frame fits the painted slot without vertical cropping, and frame-slot size changes invalidate the exact resolution authority.
- [x] #5 Collapsed, compact, hidden, detached, and stale views do not start an invalid zero-size resolution or repaint current views; existing animation, reduced-motion, fallback, cancellation, and navigation behavior remains unchanged.
- [x] #6 Born-RED-to-GREEN focused tests, mutation proof for both root-cause guards, scoped static checks, and an isolated latest-dev full-app UAT without `NO_COLOR` prove color rendering, full-frame legibility, persistence across restart, and trusted operational state changes through a disposable local provider.
- [x] #7 Disabling unavailable Console project instructions terminalizes the rejected attempt instead of leaving Console and Buddy stuck in `Running`/`thinking`; a subsequent prompt can reach the configured disposable local provider.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md`
Reason: ADR-074 already defines Buddy preference persistence, native Textual rendering, exact runtime authority, and verification boundaries; this task repairs two implementation defects without changing those decisions.

Executable plan: [Persona Buddy UAT Repairs Implementation Plan](../../Docs/superpowers/plans/2026-08-22-task-20938-persona-buddy-uat-repairs.md)

1. Project the effective Buddy table at the existing normalized-config boundary, with real-TOML startup and first-write TDD.
2. Use the visible frame Static as the exact shared resolution/authority size, with real-CSS crop, resize, and hidden-state TDD.
3. Run scoped regression, mutation, static, privacy, and architecture gates.
4. Pin and repair the project-instructions disable path discovered during actual-app UAT so the rejected attempt terminalizes and a retry can proceed.
5. Repeat isolated full-application UAT without `NO_COLOR`, then record exact closeout evidence and deviations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented three focused repairs. `config.py` now projects only the effective
`[persona_buddy]` table into normalized app settings, preserving strict parser
defaults and the incumbent writer. `PersonaBuddyWidget` now resolves against the
visible frame content region and includes that exact size in resolution authority.
Console project-instructions recovery now terminalizes a disabled setup attempt
through the existing blocked-run path, allowing the next prompt to proceed.

TDD evidence: the config projection began with four failures and finished with 25
passing tests; the frame-slot work began with five failures and finished with 28
widget tests plus 82 adjacent Buddy/runtime tests; the live-discovered Console
recovery began with `VALIDATING` instead of `BLOCKED`, then finished with 29/29
project-instructions tests. Removing each production guard reproduced its focused
failure. The final pre-import isolated component gate passed 154 tests. Scoped Ruff,
format (excluding the unchanged legacy Console monolith baseline), compile, and diff
checks passed.

Actual-app UAT used a copied private profile under
`/private/tmp/task20938-uat.30R0o3`, a 120x40 tmux terminal, no `NO_COLOR`, and a
loopback-only OpenAI-compatible provider. The selected local Buddy appeared before
input at the persisted geometry, painted the complete five-row portrait, emitted
143 RGB sequences in the portrait block (119 non-grayscale), changed its portrait
paint digest while idling, preserved move/Fold/Open/navigation state, and restored
selection/geometry/full-frame color after restart. The real Console showed
`thinking`, `speaking`, `approval needed`, `error`, recovery to `idle`, and a
successful provider response after project instructions were disabled. A requested
repository file read was denied; a content-free internal-tool attempt completed too
quickly or was policy-blocked, so no standalone `tool running` frame is claimed.

Containment: all disposable app/provider processes were stopped and the worktree was
clean except for intended closeout files. The real config was not modified during
the final isolated phase or a direct override-writer probe, but its hash had already
changed at 15:14 during the broader forked-session window from the recorded pre-UAT
value; the current file was preserved rather than guessing at a rollback. The UAT
also exposed the terminal-state recovery lesson recorded in
`backlog/docs/lessons-live-verification.md`. No ADR change was required because the
repairs preserve ADR-074's existing ownership and runtime boundaries.
<!-- SECTION:NOTES:END -->
