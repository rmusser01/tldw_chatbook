---
id: TASK-2766
title: 'Decompose _open_console_prompts_modal (397 lines, 17 nested closures)'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 06:41'
updated_date: '2026-08-07 19:09'
labels:
  - refactor
  - console
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 3 moved this method verbatim into ConsolePromptsController. Review judged its 15-of-18 dependency fan-out inherent rather than a wrong controller boundary, but identified the method itself as a class in disguise: a modal callback-bundle factory whose 17 nested closures each own one dependency. Decomposing it was deliberately kept out of a byte-fidelity wave.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The modal's callbacks are objects or methods with named dependencies rather than closures over a 397-line scope
- [x] #2 ConsolePromptsController's constructor dependency count falls below 18
- [x] #3 No behaviour change: the modal-open path's characterisation tests pass unchanged
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Characterise the modal-open path (5 source callables, pin lifecycle, focus restore) and mutation-check the new assertions.
2. Extract the 5 data-source closures into a `_ConsolePromptSource` adapter over the app's prompt_scope_service (its only capture).
3. Extract the 11 improvement-flow closures into a `_ConsolePromptImprovementFlow` collaborator whose shared per-open state (session id, composer, opening snapshot/fingerprint, gateway, pinned resolution) becomes fields and whose app-level needs stay named callables.
4. Promote `restore_focus` to a controller method and `_resolution_identity` to a module-level pure function; leave nothing inline that only a closure could express.
5. Collapse the three post-apply re-sync bridges (always called as an ordered trio, only here) into one named dependency, taking the constructor from 18 to 16.
6. Re-run the characterisation + the 304-test native chat flow suite and the wiring/architecture gates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`_open_console_prompts_modal` is 82 lines with zero nested closures (was 397 with 17). Its callbacks are now the bound methods of two collaborators built per modal open.

**`_ConsolePromptSource`** (5 closures). Those five captured exactly one thing -- `app_instance.prompt_scope_service` -- and did one thing with it, so the cluster's whole source contract now lives in one 96-line class: pages of 10, searches capped at 25, `save`'s `source` routed into the service's `mode`, and a `_require` helper carrying the per-callable refusal copy a duck-typed source triggers. `record_prompt_usage` joined it, since the improvement flow read the same service.

**`_ConsolePromptImprovementFlow`** (11 closures). The stateful half: `nonlocal pinned_improvement_resolution` becomes a field, and the eight values the closures each re-captured (store, session id, composer, opening snapshot, System prompt + fingerprint, gateway, disclosure context) become named constructor arguments. Its two staleness guards, duplicated across three call sites, collapse into `_stale_reason()`, which still raises or returns the exact copy each site always did.

`restore_focus` (captured only `self`) became `ConsolePromptsController._restore_console_composer_focus`; `_resolution_identity` (a pure function of its argument) became module-level `_provider_resolution_identity`. Nothing was left inline.

**Constructor 18 -> 16 named dependencies.** The three post-apply re-sync bridges were only ever called as an ordered trio, at the two moments the store accepted a new System prompt, never individually -- so `sync_console_system_prompt_surfaces` replaces them. `wiring.py` resolves each screen method by name at CALL time inside a nested function, so every instance-level replacement the suite makes is still observed. Nothing else could be shed without manufacturing an abstraction: the five provider dependencies are genuinely distinct services and the rest are single-use-but-irreducible.

**Behaviour.** An AST equivalence check against HEAD~1 shows every moved body identical modulo the four documented transformations (`_stale_reason`, `_require`, the trio collapse, `self.` prefixes); `capture_manual_resolution`, `validate_improvement`, both saved-artifact guards and `restore_focus` are byte-identical. Call order, the two `_console_provider_blocker_copy()` reads and the modal's kwarg evaluation order are unchanged; the controller still owns zero DOM. One deliberate non-identity: `record_prompt_usage`'s availability lookup now sits inside the caller's `try`, so a scope service whose attribute ACCESS raised would warn instead of propagating into an already-applied apply -- strictly safer, and unreachable for any plain object.

**Evidence.** 7 new characterisation tests written and pushed green BEFORE the refactor, mutation-checked (per_page 10->11 and dropping the capture-once guard each fail exactly one). After: 44 prompts controller + wiring, 304 native chat flow (identical to baseline), 268 across the prompt-cluster files, 195 responsiveness/agent-swap/internals, 89 architecture + ownership, 9073 collected clean over Tests/UI, ruff clean. Only failure is the pre-existing `test_persistent_diagnostic_inventory` (task-2768), which names neither changed file.

**Files**: `tldw_chatbook/UI/Console_Modules/prompts.py`, `tldw_chatbook/UI/Console_Modules/wiring.py`, `Tests/UI/test_console_prompts_controller.py`.
<!-- SECTION:NOTES:END -->
