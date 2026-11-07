---
id: TASK-38542
title: Author and browse dedicated mix-ins in Library Prompts
status: Done
created_date: 2026-11-01 19:30
dependencies:
- TASK-38537
- TASK-38539
labels:
- prompts
- console
references:
- backlog/decisions/286-console-prompt-mix-ins.md
documentation:
- Docs/superpowers/specs/2026-11-01-prompt-mix-ins-design.md
- Docs/superpowers/plans/2026-11-01-prompt-mix-ins-implementation.md
- backlog/decisions/286-console-prompt-mix-ins.md
updated_date: 2026-11-02 00:49
assignee:
- '@codex'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users create, edit, find and enable literal mix-ins through the existing prompt-library surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library type filtering distinguishes Prompts, Recipes and Mix-ins, with a simple Name/Instructions/Before-or-After editor that preserves literal text and existing save/version/dirty-state behavior.
- [x] #2 Enable in current chat uses the existing guarded exact-target handoff, never inserts into the composer or replaces System, and refuses stale or absent destinations.
- [x] #3 Updated/deleted/unsupported-source states are visible, keyboard controls use existing token-backed patterns, and targeted UI/export regressions preserve normal Prompt/Recipe flows.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add editor/filter and exact-target enable regressions, then run them red.
2. Reuse Library CRUD, dirty guards, source policy and typed pending handoff for literal mix-ins.
3. Verify targeted Library/Console/export regressions, production-style native UI captures, scoped lint and docs.
ADR required: no
ADR path: backlog/decisions/286-console-prompt-mix-ins.md
Reason: direct implementation of approved ADR-286.
Interface notes: approved ADR-286; nullable definition is create/draft-only and read_definition always constructs a strict literal value. Existing Library screen delegates, screen-state projection and Console wiring are necessary owners for the body-free shared-slot request and runtime enable_reference seam.
Final exact-origin ruling: ConsoleMixinEnableRequest additionally retains a repr-hidden, identity-copied tuple of the exact PromptScopeService, local backend and DB objects. Runtime enable_reference compares those fences before awaiting and after the read, rejecting same-path reopen/backend replacement and scope-wrapper replacement; no source objects are serialized or persisted. Cover same-path replacement before consumption and held-consumer source replacement.
Review fix 1: contain strict literal validation at Duplicate and conflict Save-as-new action boundaries, preserve raw draft/conflict/source/version and expose the existing actionable Instructions message; verify invalid actions and subsequent valid copy persistence with real DB/production CSS. ADR required: no; routine boundary bug fix under ADR-286. Source fences/handoff remain unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the literal Name/Instructions/Before-or-After editor with parent-owned raw drafts, metadata, expected versions and exact source identity. Search/type filtering retains exact count/page scopes, including restore and empty-filter recovery. Saved clean mix-ins stage a typed body-free request in the existing Prompt slot; Console validates source/session/incarnation/preference revision and enables through runtime CAS without touching composer or System. Existing shared work-pane composition and token-backed forms are reused; no stylesheet changes.
ADR required: no; direct implementation of backlog/decisions/286-console-prompt-mix-ins.md. Extra wiring/projection owners and nullable create-only editor interface follow controller rulings.
Verification: final focused confirmation 29 passed; broader reader/state/handoff run 372 passed; explicit Console constructor/ordinary-handoff checks 4 passed; Markdown/Chatbook/source regression paths passed in the targeted regression run. Three unrelated Console improvement tests still fail with NoMatches(#console-left-rail), reproduced on exact BASE 6f5bfae and current. Whole touched-file BASE lint comparison: 269 inherited diagnostics on each side, none introduced; new widget/tests pass Ruff and formatter. Six real-DB production-style captures at 80x24, 120x40 and 170x48 retained for review. Full suite not run.
Updated Library guide/architecture. Implementation complete; status remains In Progress for controller review.
Final exact-origin correction retains repr-hidden identity-preserved (scope, backend, DB) fences, rejecting same-path reopen and scope-wrapper replacement before or during consumption. New Create is synchronously disabled while the existing browse/Items refresh settles and enabled only through the existing post-recompose callback; deterministic held browse/source test proves the old control stays disabled through read completion, then the distinct mounted control accepts the click and retains an armed editor. Final bounded UI/source/ordinary-reader/Prompt-handoff qualification: 33 passed, 335 deselected. Final source-fence/ordering control: 5 passed; deterministic boundary control: 1 passed. No new timer or permission/source bypass.
Review fix 1: Duplicate and conflict Save-as-new now contain strict literal validation before any detachment, reuse ordinary Save's actionable Instructions message/focus, and keep raw name/body/placement, dirty/conflict state, source identity and expected version unchanged. Valid Duplicate clears only the obsolete raw draft after its exact detached copy is constructed/adopted, preserving the generated (copy) name and literal definition. Real-DB production-CSS invalid-action/recovery tests verify exact kind/body/placement/source and parent metadata. Focused verification: 4 passed, 18 deselected; FIX_BASE 398e79d50b whole touched-Python-file lint: 17 inherited diagnostics on each side, none introduced. Source fences, pending handoff and styles unchanged; status remains In Progress for controller review.
Independent spec/quality/native-finish review cleared after f0e1d640c4 contained invalid Save-as-new/Duplicate strict validation and preserved raw draft/conflict/source/version; corrected valid-copy adoption is tested.4 targeted fix cases pass,17/17 lint. Earlier33 final qualification and headless production-style6captures are separately recorded;3 exact-BASE Console NoMatches remain qualified. ADR286; nullable explicit create only, sharedtypedhandoffslot and exact(scope,backend,DB) transient fences. No CSS/system redesign or OS/live-provider claim.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
