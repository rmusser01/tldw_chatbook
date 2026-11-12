# Textual Settings Patterns Implementation Plan

> Execute inline with superpowers:executing-plans, preserving the approved screen-decomposition design and TASK-35676/35677/35678 acceptance criteria.

**Goal:** prevent programmatic paste-control edits, give that region local widget ownership, and share Settings action eligibility.
**Architecture:** the screen retains its category draft, validation and save authority. One small widget owns paste controls and emits edits. Pure category predicates feed actions and the custom help/footer; guidance-only actions remain bound.
**Tech stack:** Python 3.12, pinned Textual 8.2.8, pytest, existing TCSS tokens.
**Spec:** Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md and the three Backlog tasks.

ADR required: no
ADR path: backlog/decisions/033-settings-commit-models-three-honestly-labeled.md; backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md; backlog/decisions/161-component-pattern-library.md
Reason: preserve existing UI, commit and runtime boundaries.

## Constraints and review focus

- Targeted tests only; private test profiles and null keyring; keep existing CSS classes and field IDs.
- A value-changing refresh must not count as an edit after queued messages arrive.
- Mount/re-entry must not swallow the first genuine edit.
- Typing retains Input identity, cursor, selection and focus.
- Save/Revert and instant-apply controls retain their existing ownership.
- Guidance-only shortcuts remain available; buttons retain direct-call focus behavior.

## TASK-35676

- [x] Add a mounted regression for both controls and observe the original failure.
- [x] Apply prevent(...) to programmatic assignments and prove real edits still stage.
- [x] Verify save/revert and category-return cases against the original writer.

## TASK-35677

- [x] Census and move only composer-paste composition/events into Widgets/Settings_Widgets/composer_paste_panel.py.
- [x] Preserve screen-owned staging and validation through widget-local edit messages.
- [x] Prove targeted refresh preserves focus/cursor/selection and field search.
- [x] Update architecture and measured size/import governance where needed.

## TASK-35678

- [x] Pin category/focus/advertisement behavior before consolidation.
- [x] Reuse category eligibility for dispatch, footer/help and relevant controls without a generic registry.
- [x] Verify guidance-only, text-entry, busy/dirty, keyboard/button and narrow-width cases.

## Completion

- [x] Run focused regressions, affected existing checks, scoped lint/format and derived artifacts.
- [x] Run private-profile keyboard journeys and inspect rendered screens/logs.
- [x] Self-review, update task notes and ACs only with evidence, commit/push, and create/attach PR against dev.
- [x] Reassess the updated source against the book and report remaining evidence-backed opportunities.
