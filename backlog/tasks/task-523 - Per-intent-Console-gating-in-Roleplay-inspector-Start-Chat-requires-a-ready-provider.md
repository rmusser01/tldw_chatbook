---
id: TASK-523
title: >-
  Per-intent Console gating in Roleplay inspector (Start Chat requires a ready
  provider)
status: In Progress
assignee: []
created_date: '2026-07-24 13:17'
updated_date: '2026-07-24 13:17'
labels:
  - roleplay
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-440 (merged with a string-only readiness approach: both Console buttons stay enabled with a 'Reply may fail' warning). This task applies the per-intent gating chosen during that work: Attach stays enabled on a valid selection (it stages context; the reply is deferred), while Start Chat is disabled when the resolved character-chat provider is not ready (it needs an immediate reply). Also adds the red 'Blocked' header colour cue the merged version lacks, and a defensive Start-Chat action guard. Reuses the merged console_handoff_readiness()/_provider_send_block_reason() readiness signal — no new provider-resolution logic.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Start Chat is disabled when the staged Console handoff provider is not ready; Attach stays enabled on a valid selection
- [x] #2 The Start-Chat button/readiness names the remedy; a start_chat action guard blocks staging when unready
- [x] #3 The header status badge shows a red visual cue (not just the word) when Blocked
<!-- AC:END -->

## Implementation Plan

1. Split the inspector's `_apply_action_state` so `provider_block_reason` disables Start Chat only (Attach stays gated on selection alone); readiness copy becomes "Start Chat blocked: ...".
2. Add a defensive `start_chat` action guard in `_attach_selection_to_console` (re-checks `_provider_send_block_reason()`).
3. Add the red-cue CSS (`#personas-header.status-blocked .workbench-header-status { color: $error }`) — the header already flips to `status="blocked"` via `_update_title`.
4. Update the 3 dev tests that asserted the old string-only behavior; add per-intent, action-guard, and header-class tests.
5. Full workbench + inspector-pane regression; whole-branch review; merge cycle.

## Implementation Notes

Per-intent Console gating on top of dev's merged TASK-440 (which shipped a string-only "Reply may fail" warning with both buttons enabled).

**Approach:** Reused the merged `_provider_send_block_reason()` / `console_handoff_readiness()` signal — no new provider-resolution logic. The inspector now treats the two Console actions by their send semantics: Attach only stages context (reply deferred), so it stays enabled on a valid selection; Start Chat needs an immediate provider reply, so it is disabled and its readiness reads "Start Chat blocked: <remedy>" when the resolved chat_defaults provider is unready. Precedence is unchanged (Qodo #824-2): the provider axis is only operative once the action gate passes, so a closed gate still shows the single "Console blocked: <reason>" story.

**Trade-off:** Kept dev's `set_console_actions_enabled(enabled, *, reason, provider_block_reason)` signature (its callers already pass `provider_block_reason`) rather than the wider `attach_enabled/start_chat_enabled` signature from the abandoned pre-merge branch — smaller diff, same behavior, since `provider_block_reason` fully determines the Start-Chat-only disable.

**Whole-branch review fix (AC #3):** the red-cue rule was first placed in `PersonasScreen.DEFAULT_CSS`, but Textual ranks app-bundle CSS above any widget `DEFAULT_CSS` regardless of selector specificity, so the bundle's `.ds-status-badge { color: $ds-text-primary }` would keep the "Blocked" badge primary-coloured and the cue would never render in the real app. Moved the rule to the app-tier source `css/components/_workbench.tcss` (`#personas-header.status-blocked .workbench-header-status { color: $ds-status-blocked }` — id+class+descendant outranks `.ds-status-badge` at the same tier) and rebuilt the bundle via `build_css.py`. Added a `StyledPersonasTestApp` (real-bundle) regression test asserting the blocked-state badge colour differs from the ready-state colour — which fails if the rule is ever outranked again.

**Modified files:** `Widgets/Persona_Widgets/personas_inspector_pane.py` (per-intent `_apply_action_state`, updated docstring), `UI/Screens/personas_screen.py` (start_chat action guard; red-cue moved out of DEFAULT_CSS with a pointer comment), `css/components/_workbench.tcss` + `css/tldw_cli_modular.tcss` (app-tier red-cue rule + rebuilt bundle), `Tests/UI/test_personas_workbench.py` (3 dev tests updated to per-intent expectations + 4 new tests: action guard, Attach-not-gated, header `status-blocked` class, real-bundle colour differential). 195 workbench + 20 inspector/foundation tests pass.
