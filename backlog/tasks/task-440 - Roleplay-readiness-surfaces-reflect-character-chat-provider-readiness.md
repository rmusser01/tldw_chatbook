---
id: TASK-440
title: Roleplay readiness surfaces reflect character-chat provider readiness
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 09:38'
updated_date: '2026-07-23 19:45'
labels:
  - roleplay
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live: the workbench header showed "Ready" and the inspector "Console ready" while character replies were impossible (character provider unready - see task-425). Readiness badges currently reflect internal wiring, not "a character reply will work". They should incorporate the resolved character provider's readiness or say specifically what is and is not ready.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When the character-chat provider is not ready, Roleplay readiness surfaces say so (and what to do), rather than showing Ready
- [x] #2 Inspector Console-readiness reflects the actual send path for the staged intent
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate the two readiness surfaces (inspector "Console ready" line via `_sync_inspector_console_actions`, header badge via `_update_title`) and the existing provider-resolution seams (task-425's `_resolve_selection_with_fallback`, config-only `provider_readout`, and the shared `get_provider_readiness` helper Chat/Settings already use).
2. TDD: add a RED test (character selected, no usable provider in app_config) asserting the inspector does NOT show "Console ready" and the header does NOT show "Ready"; add a GREEN-parity test (configured API key) asserting the existing copy is unchanged.
3. Add `PersonasPreviewController.resolved_send_readiness()`: a cheap, config/env-only mirror of the task-425 character_defaults -> chat_defaults fallback built on `get_provider_readiness` (no network probe).
4. Feed it into both surfaces via a screen-level `_provider_send_block_reason()`; extend `PersonasInspectorPane.set_console_actions_enabled` with a copy-only `provider_block_reason` that never disables the buttons.
5. Verify: targeted new tests, the full personas UI suites, and `import tldw_chatbook.app`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Seam reused: `get_provider_readiness` (tldw_chatbook/Chat/provider_readiness.py) - the same side-effect-free readiness helper Chat and Settings badges already use - wrapped in a new `PersonasPreviewController.resolved_send_readiness()` that mirrors task-425's character_defaults -> chat_defaults fallback exactly (same same-target short-circuit, same "surface the chat_defaults blocker when a distinct fallback was attempted" precedent from `_resolve_selection_with_fallback`). Deliberately NOT the async `ConsoleProviderGateway.resolve_for_send` probe: readiness recomputes on every selection sync so it must stay cheap and synchronous. Documented trade-off: no llama.cpp reachability check here, so a configured-but-unreachable local endpoint reads ready on the badge and still fails honestly at send time via the preview status line.

Surfaces: (1) inspector - `_sync_inspector_console_actions` now passes `provider_block_reason` into `PersonasInspectorPane.set_console_actions_enabled`; when set (and the selection gate is otherwise open) the readiness line renders `Console blocked: {ProviderReadiness.user_message}` (e.g. "anthropic is not ready: Missing API key. Set ANTHROPIC_API_KEY or add api_key under [api_settings.anthropic].") instead of "Console ready", and the Attach/Start buttons get a "Reply may fail: ..." tooltip. (2) header - `_update_title` derives the badge status: unready provider with a staged character/persona => `blocked` (the shell's existing degraded-status pattern, cf. stats_screen), else `ready`.

Gating decision: copy-only, buttons stay ENABLED. Task-427 made Start Chat open a real conversation and task-425 made sends fall back - blocking the buttons would regress that established UX; the AC asks for honest copy, which this delivers while leaving the actions usable (the user can still stage/start and fix the provider from Settings).

Files: tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py (new `resolved_send_readiness`), tldw_chatbook/UI/Screens/personas_screen.py (`_provider_send_block_reason`, `_update_title`, `_sync_inspector_console_actions`), tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py (`provider_block_reason` copy channel), Tests/UI/test_personas_workbench.py (2 new tests in TestConsoleActions).

Verification: new unready test RED against pre-change code (failed with `assert 'Console ready' != 'Console ready'`, 3/3 reruns), both new tests GREEN after; Tests/UI/test_personas_preview.py + test_personas_workbench.py + test_personas_inspector_pane.py + test_personas_dictionaries.py = 306 passed; test_destination_shells.py 100 passed/1 skipped with 2 failures (library/schedules param cases) reproduced identically on a clean tree = pre-existing baseline; `import tldw_chatbook.app` clean; ruff clean on the changed hunks (the 2 F821s in personas_screen.py exist at HEAD).
<!-- SECTION:NOTES:END -->
