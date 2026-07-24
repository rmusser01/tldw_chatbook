---
id: TASK-440
title: Roleplay readiness surfaces reflect character-chat provider readiness
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 09:38'
updated_date: '2026-07-23 20:15'
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
3. Add a cheap, config/env-only readiness helper on `PersonasPreviewController` built on `get_provider_readiness` (no network probe).
4. Feed it into both surfaces via a screen-level `_provider_send_block_reason()`; extend `PersonasInspectorPane.set_console_actions_enabled` with a copy-only `provider_block_reason` that never disables the buttons.
5. Verify: targeted new tests, the full personas UI suites, and `import tldw_chatbook.app`.
6. (Review fix) Re-point the helper at the Start-Chat handoff's ACTUAL resolution (chat_defaults-only, no character_defaults short-circuit); add divergent-case tests both ways.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Seam reused: `get_provider_readiness` (tldw_chatbook/Chat/provider_readiness.py) - the same side-effect-free readiness helper Chat and Settings badges already use - wrapped in a new `PersonasPreviewController.console_handoff_readiness()`. Deliberately NOT the async `ConsoleProviderGateway.resolve_for_send` probe: readiness recomputes on every selection sync so it must stay cheap and synchronous. Documented trade-off: no llama.cpp reachability check here, so a configured-but-unreachable local endpoint reads ready on the badge and still fails honestly at send time via the preview status line.

AC#2 evidence (corrected by review): the surfaces gate Attach/Start Chat, whose REAL send path is a fresh native-Console session (`chat_screen._start_character_console_session` -> `_default_console_session_settings` -> `_effective_console_provider_model`) resolved from chat_defaults - the native Console never reads character_defaults. The first cut mirrored the PREVIEW's character_defaults-first fallback (task-425) and therefore claimed Ready with shipped defaults (chat_defaults=OpenAI unready, character_defaults=Anthropic keyed) while the real Start-Chat send would fail. Fixed: `console_handoff_readiness()` checks ONLY the chat_defaults selection (via the existing `_selection_from_defaults` / `build_default_console_session_settings` pure helper - the same defaults a fresh handoff session is built from) as the config-side approximation of `_effective_console_provider_model` (this screen cannot see the Console screen's live provider reactives; a fresh handoff session is created from these config defaults, so config-side is the honest cheap answer). The preview pane's own provider readout keeps its character_defaults -> chat_defaults fallback behavior untouched - that IS the preview path.

Surfaces: (1) inspector - `_sync_inspector_console_actions` now passes `provider_block_reason` into `PersonasInspectorPane.set_console_actions_enabled`; when set (and the selection gate is otherwise open) the readiness line renders `Console blocked: {ProviderReadiness.user_message}` (e.g. "openai is not ready: Missing API key. Set OPENAI_API_KEY or add api_key under [api_settings.openai].") instead of "Console ready", and the Attach/Start buttons get a "Reply may fail: ..." tooltip. (2) header - `_update_title` derives the badge status: unready handoff provider with a staged character/persona => `blocked` (the shell's existing degraded-status pattern, cf. stats_screen), else `ready`.

Gating decision: copy-only, buttons stay ENABLED. Task-427 made Start Chat open a real conversation and task-425 made preview sends fall back - blocking the buttons would regress that established UX; the AC asks for honest copy, which this delivers while leaving the actions usable (the user can still stage/start and fix the provider from Settings).

Files: tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py (`console_handoff_readiness`), tldw_chatbook/UI/Screens/personas_screen.py (`_provider_send_block_reason`, `_update_title`, `_sync_inspector_console_actions`), tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py (`provider_block_reason` copy channel), Tests/UI/test_personas_workbench.py (4 tests in TestConsoleActions).

Verification: initial unready test RED against pre-change code (failed with `assert 'Console ready' != 'Console ready'`, 3/3 reruns); review-fix divergent test (chat_defaults unready + character_defaults ready) RED against the first-cut commit with the same `assert 'Console ready' != 'Console ready'` failure, GREEN after re-pointing the helper; opposite divergent case (chat_defaults ready + character_defaults unready) shows "Console ready"/"Ready". Suites: test_personas_workbench.py + test_personas_preview.py = 215 passed post-fix (306 across the 4 personas UI files pre-fix); test_destination_shells.py 100 passed/1 skipped with 2 failures (library/schedules param cases) reproduced identically on a clean tree = pre-existing baseline; `import tldw_chatbook.app` clean; ruff clean on all changed files except the 2 pre-existing F821s in personas_screen.py (exist at HEAD, untouched regions).
<!-- SECTION:NOTES:END -->
